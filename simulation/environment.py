"""
Main environment class for the RC car simulation.
Follows OpenAI Gym interface for easy RL integration.
"""

import pygame
import numpy as np
import math
import os

from utils.map_utils import extract_and_scale_contours, generate_racing_line, load_distance_field
from utils.geometry import segment_intersection, distance, contour_to_segments
from simulation.car import DifferentialDriveCar
from simulation.sensors import SensorArray

class Environment:
    def __init__(self, map_path, window_size=(1024, 1024), show_collision_box=True, 
                 show_sensors=True, show_racing_line=True, show_track_edges=False, 
                 show_distance_heatmap=False, display_map_path=None, headless=False):
        """
        Initialize the RC car environment.
        
        Args:
            map_path: Path to the track map image
            window_size: Pygame window size
            show_collision_box: Whether to show car collision box
            show_sensors: Whether to show sensor rays
            show_racing_line: Whether to show racing line
            show_track_edges: Whether to show track boundary edges
            headless: If True, skip pygame window creation for faster headless training
        """
        self.window_size = window_size
        self.show_collision_box = show_collision_box
        self.show_sensors = show_sensors
        self.show_racing_line = show_racing_line
        self.show_track_edges = show_track_edges
        self.show_distance_heatmap = show_distance_heatmap
        self.headless = headless
        
        # init pygame - only create window if not headless
        pygame.init()
        if not headless:
            self.screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption("RC Car Simulation")
            # Cache font object to avoid recreating every frame
            self.debug_font = pygame.font.Font(None, 24)
        else:
            self.screen = None
            self.debug_font = None
        
        # map and track data
        self.map_path = map_path
        self.display_map_path = display_map_path
        self.load_map()
        
        # init car
        self.car = DifferentialDriveCar()
        
        # init sensors
        self.sensors = SensorArray()
        
        # episode management
        self.max_steps = 3000
        self.current_step = 0
        
        # reward tracking
        self.prev_position = None
        self.prev_heading = None
        self._last_dt = 1.0/60.0
        
        # Cache for sensor readings to avoid multiple calculations per frame
        self._sensor_cache = {
            'readings': None,
            'car_state': None,  # (x, y, heading) to track when cache is valid
            'frame_step': -1    # which simulation step the cache is from
        }
        
        self.reset()
    
    def load_map(self):
        """Load the track map and extract boundaries."""
        display_path = self.display_map_path
        if not display_path:
            # auto-derive from provided map_path
            dirname, filename = os.path.split(self.map_path)
            candidate = None
            if "map_start" in filename:
                candidate = filename.replace("map_start", "map")
            elif "map_mask" in filename:
                candidate = filename.replace("map_mask", "map")
            # fallback: if no pattern matched, use original
            display_path = os.path.join(dirname, candidate) if candidate else self.map_path
            if not os.path.exists(display_path):
                display_path = self.map_path
        
        # Only load map image for rendering (not needed in headless mode)
        if not self.headless:
            self.map_image = pygame.image.load(display_path).convert()
        else:
            self.map_image = None
        
        # map data
        boundaries_outer, boundaries_inner, start_points, headings = extract_and_scale_contours(self.map_path)
        
        self.track_boundaries = [boundaries_outer, boundaries_inner]
        
        # Convert boundaries to line segments for collision detection
        self.boundary_segments = []
        if boundaries_outer:
            self.boundary_segments.extend(contour_to_segments(boundaries_outer))
        if boundaries_inner:
            self.boundary_segments.extend(contour_to_segments(boundaries_inner))
        
        self.start_positions = start_points if start_points else [(512, 512)]
        self.start_headings = headings if headings else [0.0]
        
        # Always generate racing line at startup (regardless of show_racing_line flag)
        # This avoids repeated expensive calculations when toggling visibility
        try:
            racing_lines = generate_racing_line(self.map_path)
            self.racing_lines = racing_lines
        except Exception as e:
            print(f"Could not generate racing line: {e}")
            self.racing_lines = []
        
        # Load distance field for proximity calculations
        self.distance_field = load_distance_field(self.map_path)
        if self.distance_field is not None:
            print(f"✓ Loaded distance field: {self.distance_field.shape}")
            # Precompute heatmap visualization surface
            self.distance_heatmap_surface = self._create_distance_heatmap_surface()
        else:
            print("⚠ No distance field found - proximity penalties disabled")
            self.distance_heatmap_surface = None
    
    def reset(self):
        """Reset the environment to initial state."""
        # choose a random start position
        if self.start_positions:
            start_idx = np.random.randint(len(self.start_positions))
            start_x, start_y = self.start_positions[start_idx]
            start_heading = self.start_headings[start_idx] if start_idx < len(self.start_headings) else 0.0
            # face the opposite direction (rotate by 180 degrees)
            start_heading = math.atan2(math.sin(start_heading + math.pi), math.cos(start_heading + math.pi))
        else:
            start_x, start_y = 200, 200
            start_heading = 0.0
        
        self.car.reset(start_x, start_y, start_heading)
        self.current_step = 0
        
        # Initialize reward tracking
        self.prev_position = self.car.get_position()
        self.prev_heading = self.car.get_heading()
        
        # Invalidate sensor cache since car position changed
        self._sensor_cache['readings'] = None
        self._sensor_cache['car_state'] = None
        self._sensor_cache['frame_step'] = -1
        
        # initial observation
        observation = self.get_observation()
        
        return observation
    
    def step(self, action, dt=1.0/60.0):
        """
        Take a step in the environment.
        
        Args:
            action: [left_wheel_velocity, right_wheel_velocity] both in range [-1, 1]
            dt: Time delta for physics update (default: 1/60 seconds)
            
        Returns:
            observation: Current sensor readings
            reward: Reward for this step
            terminated: Whether episode ended due to collision/failure
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        # apply action to car
        left_vel, right_vel = action
        self.car.set_wheel_velocities(left_vel, right_vel)
        self._last_dt = dt
        self.car.update(dt)
        
        observation = self.get_observation()
        collision = self.check_collision()
        
        reward = self.calculate_reward(collision)
        
        # check episode termination conditions
        self.current_step += 1
        terminated = collision  # Episode ends due to collision/failure
        truncated = self.current_step >= self.max_steps  # Episode ends due to time limit
        
        info = {
            'collision': collision,
            'position': self.car.get_position(),
            'heading': self.car.get_heading(),
            'step': self.current_step,
            'max_steps': self.max_steps
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_cached_sensor_readings(self):
        """Get sensor readings with caching to avoid redundant calculations."""
        car_x, car_y = self.car.get_position()
        car_heading = self.car.get_heading()
        current_state = (car_x, car_y, car_heading)
        
        # Check if cache is valid (same car state and same simulation step)
        if (self._sensor_cache['car_state'] == current_state and 
            self._sensor_cache['frame_step'] == self.current_step and
            self._sensor_cache['readings'] is not None):
            return self._sensor_cache['readings']
        
        # Cache is invalid, recalculate
        sensor_readings = self.sensors.get_readings(car_x, car_y, car_heading, self.boundary_segments)
        
        # Update cache
        self._sensor_cache['readings'] = sensor_readings
        self._sensor_cache['car_state'] = current_state
        self._sensor_cache['frame_step'] = self.current_step
        
        return sensor_readings
    
    def get_observation(self):
        """Get current observation (sensor readings)."""
        sensor_readings = self._get_cached_sensor_readings()
        return np.array(sensor_readings, dtype=np.float32)
    
    def check_collision(self):
        """Check if car collides with track boundaries using rotated rectangle vs segments."""
        # Get car bounding box corners
        car_corners = self.car.get_bounding_box()
        
        # Create car edges from corners
        car_edges = []
        for i in range(len(car_corners)):
            start = car_corners[i]
            end = car_corners[(i + 1) % len(car_corners)]
            car_edges.append((start, end))
        
        # Check each car edge against each boundary segment
        for car_edge in car_edges:
            for boundary_segment in self.boundary_segments:
                intersection = segment_intersection(car_edge[0], car_edge[1], 
                                                 boundary_segment[0], boundary_segment[1])
                if intersection:
                    return True
        
        # Also check if car is outside map bounds
        car_x, car_y = self.car.get_position()
        if car_x < 0 or car_x >= self.window_size[0] or car_y < 0 or car_y >= self.window_size[1]:
            return True
        
        return False
    
    def calculate_reward(self, collision):
        """Calculate reward for current step."""
        if collision:
            return -100.0
        # Signed forward progress: project displacement onto heading vector
        current_pos = self.car.get_position()
        if self.prev_position is not None:
            dx = current_pos[0] - self.prev_position[0]
            dy = current_pos[1] - self.prev_position[1]
            heading = self.car.get_heading()
            forward_progress = dx * math.cos(heading) + dy * math.sin(heading)
        else:
            forward_progress = 0.0

        # Update previous position for next step
        self.prev_position = current_pos

        # Heading-change penalty: allow reasonable turn rate based on car dynamics.
        # One-wheel-stationary yaw rate (rad/s): omega_allow = max_wheel_speed / wheelbase
        # Allow per-step heading change: delta_allow = omega_allow * dt
        current_heading = self.car.get_heading()
        if self.prev_heading is not None:
            dtheta = math.atan2(math.sin(current_heading - self.prev_heading),
                                 math.cos(current_heading - self.prev_heading))
            delta_allow = (self.car.max_wheel_speed / self.car.wheelbase) * self._last_dt
            excess = max(0.0, abs(dtheta) - delta_allow)
            # Convert excess rotation to an equivalent linear "wobble" distance at half wheelbase
            # and scale with the same factor as forward progress to keep units comparable.
            heading_penalty = 10.0 * (0.5 * self.car.wheelbase) * excess
        else:
            heading_penalty = 0.0

        # Update previous heading for next step
        self.prev_heading = current_heading

        # Proximity penalty based on distance field
        proximity_penalty = self._calculate_proximity_penalty()

        # Reward forward motion minus penalties
        return 10.0 * forward_progress - heading_penalty - proximity_penalty
    
    def _calculate_proximity_penalty(self):
        """Calculate proximity penalty based on distance field."""
        if self.distance_field is None:
            return 0.0
        
        # Get car center position
        car_x, car_y = self.car.get_position()
        
        # Convert to pixel coordinates (clamp to image bounds)
        px = int(np.clip(car_x, 0, self.distance_field.shape[1] - 1))
        py = int(np.clip(car_y, 0, self.distance_field.shape[0] - 1))
        
        # Get distance value (0=wall, 1=safe area)
        distance_value = self.distance_field[py, px]
        
        # Convert to proximity penalty
        # Higher penalty for lower distance values (closer to walls)
        # Smooth penalty that ramps up as distance_value decreases
        safety_threshold = 0.7  # Below this value, start applying penalty (accounting for car width)
        if distance_value < safety_threshold:
            # Quadratic penalty that increases as we get closer to walls
            penalty_strength = 5.0  # Adjust this to make car more/less "afraid"
            normalized_proximity = (safety_threshold - distance_value) / safety_threshold
            penalty = penalty_strength * (normalized_proximity ** 2)
            return penalty
        
        return 0.0
    
    def _create_distance_heatmap_surface(self):
        """Create a transparent colored heatmap surface from the distance field."""
        if self.distance_field is None or self.headless:
            return None
        
        # Get distance field dimensions
        height, width = self.distance_field.shape
        
        # Create RGBA array for efficient batch processing
        rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Vectorized colormap generation
        d = self.distance_field.astype(np.float32)
        
        # Initialize RGB channels
        r = np.zeros_like(d, dtype=np.uint8)
        g = np.zeros_like(d, dtype=np.uint8) 
        b = np.zeros_like(d, dtype=np.uint8)
        
        # Zone 1: Very close to walls (0-0.25): blue to purple
        mask1 = d < 0.25
        t1 = np.where(mask1, d / 0.25, 0)
        r[mask1] = (128 * t1[mask1]).astype(np.uint8)
        g[mask1] = 0
        b[mask1] = 255
        
        # Zone 2: Close to walls (0.25-0.5): purple to red
        mask2 = (d >= 0.25) & (d < 0.5)
        t2 = np.where(mask2, (d - 0.25) / 0.25, 0)
        r[mask2] = (128 + 127 * t2[mask2]).astype(np.uint8)
        g[mask2] = 0
        b[mask2] = (255 * (1 - t2[mask2])).astype(np.uint8)
        
        # Zone 3: Medium distance (0.5-0.75): red to yellow
        mask3 = (d >= 0.5) & (d < 0.75)
        t3 = np.where(mask3, (d - 0.5) / 0.25, 0)
        r[mask3] = 255
        g[mask3] = (255 * t3[mask3]).astype(np.uint8)
        b[mask3] = 0
        
        # Zone 4: Far from walls (0.75-1.0): yellow to green
        mask4 = d >= 0.75
        t4 = np.where(mask4, (d - 0.75) / 0.25, 0)
        r[mask4] = (255 * (1 - t4[mask4])).astype(np.uint8)
        g[mask4] = 255
        b[mask4] = 0
        
        # Set alpha based on distance (closer to walls = more opaque)
        alpha = np.clip(150 * (1 - d) + 50, 30, 180).astype(np.uint8)
        
        # Assemble RGBA array
        rgba_array[:, :, 0] = r
        rgba_array[:, :, 1] = g
        rgba_array[:, :, 2] = b
        rgba_array[:, :, 3] = alpha
        
        # Create pygame surface and set pixels efficiently
        heatmap_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Convert to format pygame expects: (width, height, 3) for RGB
        rgb_array = np.stack([r, g, b], axis=2).swapaxes(0, 1)  # (width, height, 3)
        
        # Use surfarray for efficient pixel setting
        pygame.surfarray.blit_array(heatmap_surface, rgb_array[:, :, :3])
        
        # Apply overall transparency
        heatmap_surface.set_alpha(120)  # Make the whole surface semi-transparent
        
        return heatmap_surface
    
    def render(self):
        """Render the environment."""
        if self.headless or self.screen is None:
            return  # Skip rendering in headless mode
            
        self.screen.fill((50, 50, 50))
        
        if self.map_image is not None:
            self.screen.blit(self.map_image, (0, 0))
        
        # distance heatmap overlay (render after track but before other elements)
        if self.show_distance_heatmap and self.distance_heatmap_surface is not None:
            self.screen.blit(self.distance_heatmap_surface, (0, 0))
        
        # racing line
        if self.show_racing_line and self.racing_lines:
            for racing_line, start_point in self.racing_lines:
                if racing_line:
                    for i, point in enumerate(racing_line):
                        color = (255, 255 - i * 255 // len(racing_line), 0)
                        pygame.draw.circle(self.screen, color, (int(point[0]), int(point[1])), 2)
        
        # car - rotate sprite based on current heading for rendering
        angle_degrees = -math.degrees(self.car.get_heading())
        rotated_sprite = pygame.transform.rotate(self.car.original_sprite, angle_degrees)
        car_rect = self.car.get_render_rect(rotated_sprite)
        self.screen.blit(rotated_sprite, car_rect)
        
        # collision box
        if self.show_collision_box:
            corners = self.car.get_bounding_box()
            # Convert float coordinates to integers for pygame
            int_corners = [(int(x), int(y)) for x, y in corners]
            pygame.draw.polygon(self.screen, (255, 0, 0), int_corners, 2)
        
        # sensors
        if self.show_sensors:
            car_x, car_y = self.car.get_position()
            car_heading = self.car.get_heading()
            sensor_readings = self._get_cached_sensor_readings()
            sensor_rays = self.sensors.get_sensor_rays(car_x, car_y, car_heading, sensor_readings)
            
            colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]  # Green, Yellow, Cyan
            for i, (start, end) in enumerate(sensor_rays):
                pygame.draw.line(self.screen, colors[i], 
                               (int(start[0]), int(start[1])), 
                               (int(end[0]), int(end[1])), 2)
        
        # track edges (optional)
        if self.show_track_edges:
            for segment in self.boundary_segments:
                start, end = segment
                pygame.draw.line(self.screen, (255, 0, 255), 
                               (int(start[0]), int(start[1])), 
                               (int(end[0]), int(end[1])), 1)
        
        # debug info (fps will be passed from main loop)
        self.draw_debug_info()
    
    def draw_debug_info(self, fps=None):
        """Draw debug information on screen."""
        if self.headless or self.screen is None or self.debug_font is None:
            return  # Skip drawing in headless mode
        
        # car info
        car_x, car_y = self.car.get_position()
        info_lines = [
            f"FPS: {fps:.1f}" if fps else "FPS: --",
            f"Position: ({car_x:.1f}, {car_y:.1f})",
            f"Heading: {math.degrees(self.car.get_heading()):.1f}°",
            f"Left Wheel: {self.car.left_wheel_velocity:.2f}",
            f"Right Wheel: {self.car.right_wheel_velocity:.2f}",
            f"Step: {self.current_step}/{self.max_steps}"
        ]
        
        # get sensor readings for display (use cached to avoid redundant computation)
        sensor_readings = self._get_cached_sensor_readings()
        info_lines.append(f"Sensors: {[f'{r:.2f}' for r in sensor_readings]}")
        
        for i, line in enumerate(info_lines):
            text = self.debug_font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, 10 + i * 25))
