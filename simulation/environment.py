"""
Main environment class for the RC car simulation.
Follows OpenAI Gym interface for easy RL integration.
"""

import pygame
import numpy as np
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from utils.map_utils import prep_mask, edges_from_mask, edge_polylines_from_masks, compute_centerline, distance_fields, create_stylized_track_image, project_and_reorder_centerline
from utils.geometry import segment_intersection, distance, contour_to_segments
from simulation.car import DifferentialDriveCar
from simulation.sensors import SensorArray

@dataclass
class MapBundle:
    """Container for all map-specific data."""
    map_path: str
    display_map_path: Optional[str]
    map_image: Optional[pygame.Surface]
    distance_field: Optional[np.ndarray]
    distance_heatmap_surface: Optional[pygame.Surface]
    track_boundaries: List
    boundary_segments: List
    start_positions: List[Tuple[float, float]]
    start_headings: List[float]
    racing_line: Optional[Tuple[np.ndarray, Tuple[float, float]]]

class Environment:
    def __init__(self, map_path: Union[str, List[str]], window_size=(1024, 1024), show_collision_box=True, 
                 show_sensors=True, show_racing_line=True, show_track_edges=False, 
                 show_distance_heatmap=False, display_map_path: Optional[Union[str, List[str]]] = None, headless=False):
        """
        Initialize the RC car environment.
        
        Args:
            map_path: Path(s) to the track map image(s) - can be single string or list
            window_size: Pygame window size
            show_collision_box: Whether to show car collision box
            show_sensors: Whether to show sensor rays
            show_racing_line: Whether to show racing line
            show_track_edges: Whether to show track boundary edges
            display_map_path: Optional display map path(s) - can be single string or list
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
        
        # Handle map paths - convert single string to list for consistency
        if isinstance(map_path, str):
            map_paths = [map_path]
        else:
            map_paths = map_path
            
        # Handle display map paths
        if display_map_path is None:
            display_map_paths = [None] * len(map_paths)
        elif isinstance(display_map_path, str):
            display_map_paths = [display_map_path] * len(map_paths)
        else:
            display_map_paths = display_map_path
            
        # Pre-calculate all map bundles
        self.map_bundles = []
        for i, (mp, dmp) in enumerate(zip(map_paths, display_map_paths)):
            print(f"Pre-calculating map {i+1}/{len(map_paths)}: {mp}")
            bundle = self._create_map_bundle(mp, dmp)
            self.map_bundles.append(bundle)
            
        # Set current map to first one
        self.current_map_bundle = None
        self.set_map(0)
        
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
        self.last_reward = 0.0  # Track last reward for display
        self.last_reward_components = {}  # Track individual reward components
        
        # direction tracking for wrong-way detection
        self.wrong_direction_steps = 0
        self.max_wrong_direction_steps = 20
        self.wrong_direction_penalty = -100.0
        
        # Cache for sensor readings to avoid multiple calculations per frame
        self._sensor_cache = {
            'readings': None,
            'car_state': None,  # (x, y, heading) to track when cache is valid
            'frame_step': -1    # which simulation step the cache is from
        }
        
        self.reset()
    
    def _create_map_bundle(self, map_path: str, display_map_path: Optional[str] = None) -> MapBundle:
        """Create a MapBundle containing all pre-calculated map data."""
        
        # Only load map image for rendering (not needed in headless mode)
        if not self.headless:
            # Generate stylized track image
            track_mask_for_stylized, _ = prep_mask(map_path)
            stylized_pil = create_stylized_track_image(track_mask_for_stylized)
            # Convert PIL to pygame surface
            stylized_array = np.array(stylized_pil)
            map_image = pygame.surfarray.make_surface(stylized_array.swapaxes(0, 1))
        else:
            map_image = None
        
        # map data
        track_mask, start_points = prep_mask(map_path)
        
        # Get all map data from compute_centerline (includes distance fields and edge polylines)
        centerline_result = compute_centerline(track_mask, return_edge_polys=True)
        
        # Extract distance field from compute_centerline results
        distance_field = centerline_result.get('dist_u8')
        distance_heatmap_surface = None
        if distance_field is not None and not self.headless:
            # Precompute heatmap visualization surface
            distance_heatmap_surface = self._create_distance_heatmap_surface_from_field(distance_field)
        
        # Extract track boundaries from compute_centerline results
        boundaries_outer = centerline_result.get('outer_poly')
        boundaries_inner = centerline_result.get('inner_poly')
        track_boundaries = [boundaries_outer, boundaries_inner]
        
        # Convert boundaries to line segments for collision detection
        boundary_segments = []
        if boundaries_outer is not None and len(boundaries_outer) > 0:
            boundary_segments.extend(contour_to_segments(boundaries_outer))
        if boundaries_inner is not None and len(boundaries_inner) > 0:
            boundary_segments.extend(contour_to_segments(boundaries_inner))
        
        start_positions = start_points if start_points else [(512, 512)]
        start_headings = [math.pi] * len(start_positions)  # Default headings
        
        # Always generate racing line
        racing_line = None
        try:
            if 'centerline' in centerline_result and centerline_result['centerline'] is not None:
                # Project start point onto centerline and reorder to start from there
                ordered_centerline, proj_pt, seg_idx, t, dist2 = project_and_reorder_centerline(
                    centerline_result['centerline'],
                    np.array(start_positions[0]) if start_positions else np.array([512, 512])
                )
                
                # Note: Direction adjustment will be done when bundle is loaded via set_map
                # This avoids accessing instance variables during bundle creation
                
                # Format as expected by render function: racing_line tuple
                racing_line = (ordered_centerline, proj_pt)
        except Exception as e:
            print(f"Could not generate racing line for {map_path}: {e}")
        
        return MapBundle(
            map_path=map_path,
            display_map_path=display_map_path,
            map_image=map_image,
            distance_field=distance_field,
            distance_heatmap_surface=distance_heatmap_surface,
            track_boundaries=track_boundaries,
            boundary_segments=boundary_segments,
            start_positions=start_positions,
            start_headings=start_headings,
            racing_line=racing_line
        )
    
    def set_map(self, bundle_index: int):
        """Switch to a different map bundle."""
        if bundle_index < 0 or bundle_index >= len(self.map_bundles):
            raise ValueError(f"Invalid bundle index {bundle_index}. Must be between 0 and {len(self.map_bundles)-1}")
            
        bundle = self.map_bundles[bundle_index]
        self.current_map_bundle = bundle
        
        # Set all map-specific variables from the bundle
        self.map_path = bundle.map_path
        self.display_map_path = bundle.display_map_path
        self.map_image = bundle.map_image
        self.distance_field = bundle.distance_field
        self.distance_heatmap_surface = bundle.distance_heatmap_surface
        self.track_boundaries = bundle.track_boundaries
        self.boundary_segments = bundle.boundary_segments
        self.start_positions = bundle.start_positions
        self.start_headings = bundle.start_headings
        
        # Apply direction adjustment to racing line now that instance variables are set
        if bundle.racing_line:
            centerline, proj_pt = bundle.racing_line
            # Find the segment index for the projected point
            # This is a simplified approach - we'll use segment 0 as approximation
            adjusted_centerline = self._adjust_centerline_direction(centerline, proj_pt, 0)
            self.racing_line = (adjusted_centerline, proj_pt)
        else:
            self.racing_line = None
        
        print(f"✓ Switched to map: {bundle.map_path}")
    
    def _create_distance_heatmap_surface_from_field(self, distance_field: np.ndarray) -> Optional[pygame.Surface]:
        """Create a transparent colored heatmap surface from a distance field."""
        if distance_field is None or self.headless:
            return None
        
        # Get distance field dimensions
        height, width = distance_field.shape
        
        # Create RGBA array for efficient batch processing
        rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Normalize distance field to 0-1 range for coloring
        if distance_field.max() > distance_field.min():
            normalized = (distance_field - distance_field.min()) / (distance_field.max() - distance_field.min())
        else:
            normalized = np.zeros_like(distance_field)
        
        # Create heatmap colors (blue for close, red for far)
        # Invert so closer distances are more intense
        intensity = (1.0 - normalized) * 255
        
        # Blue channel (close = high, far = low)
        rgba_array[:, :, 2] = intensity.astype(np.uint8)  # Blue
        # Red channel (close = low, far = high) 
        rgba_array[:, :, 0] = (normalized * 255).astype(np.uint8)  # Red
        # Green channel (medium values)
        rgba_array[:, :, 1] = (np.abs(normalized - 0.5) * 2 * 255).astype(np.uint8)  # Green
        
        # Alpha channel - make it semi-transparent
        rgba_array[:, :, 3] = 120  # Semi-transparent
        
        # Convert to pygame surface (RGB only, no alpha channel)
        rgb_array = rgba_array[:, :, :3]  # Remove alpha channel for pygame compatibility
        heatmap_surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
        
        return heatmap_surface
    
    def load_map(self):
        """Deprecated: Use set_map() instead. This method is kept for backwards compatibility."""
        if len(self.map_bundles) > 0:
            self.set_map(0)
        else:
            raise RuntimeError("No map bundles available. Environment was not properly initialized.")
    
    def reset(self):
        """Reset the environment to initial state."""
        # choose a random start position
        start_idx = np.random.randint(len(self.start_positions))
        start_x, start_y = self.start_positions[start_idx]
        
        # Determine correct heading from centerline direction
        if self.racing_line and len(self.racing_line[0]) > 1:
            centerline, start_pt = self.racing_line
            # Get direction from first two points of centerline
            direction = centerline[1] - centerline[0]
            correct_heading = math.atan2(direction[1], direction[0])
        else:
            # Fallback to original logic if no racing line
            start_heading = self.start_headings[start_idx] if start_idx < len(self.start_headings) else 0.0
            correct_heading = math.atan2(math.sin(start_heading + math.pi), math.cos(start_heading + math.pi))
        
        self.car.reset(start_x, start_y, correct_heading)
        self.current_step = 0
        
        # Initialize reward tracking
        self.prev_position = self.car.get_position()
        self.prev_heading = self.car.get_heading()
        
        # Reset direction tracking
        self.wrong_direction_steps = 0
        
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
        
        reward, terminate_due_to_wrong_direction = self.calculate_reward(collision)
        self.last_reward = reward  # Store for display
        
        # check episode termination conditions
        self.current_step += 1
        terminated = collision or terminate_due_to_wrong_direction  # Episode ends due to collision or wrong direction
        truncated = self.current_step >= self.max_steps  # Episode ends due to time limit
        
        info = {
            'collision': collision,
            'position': self.car.get_position(),
            'heading': self.car.get_heading(),
            'step': self.current_step,
            'max_steps': self.max_steps,
            'map_path': self.map_path,
            'map_index': self.map_bundles.index(self.current_map_bundle) if self.current_map_bundle else 0
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
        # Initialize reward components dictionary
        reward_components = {
            'collision': 0.0,
            'forward_progress': 0.0,
            'heading_penalty': 0.0,
            'proximity_penalty': 0.0
        }
        
        if collision:
            reward_components['collision'] = -500.0
            total_reward = sum(reward_components.values())
            self.last_reward_components = reward_components
            # Return tuple format: (reward, terminate_due_to_wrong_direction)
            return total_reward, False
            
        # Signed forward progress: project displacement onto heading vector
        current_pos = self.car.get_position()
        if self.prev_position is not None:
            dx = current_pos[0] - self.prev_position[0]
            dy = current_pos[1] - self.prev_position[1]
            heading = self.car.get_heading()
            forward_progress = dx * math.cos(heading) + dy * math.sin(heading)
            reward_components['forward_progress'] = 10.0 * forward_progress
        else:
            reward_components['forward_progress'] = 0.0

        # Update previous position for next step
        self.prev_position = current_pos

        # Heading-change penalty: allow reasonable turn rate based on car dynamics.
        # One-wheel-stationary yaw rate (rad/s): omega_allow = max_wheel_speed / effective_track_width
        # Allow per-step heading change: delta_allow = omega_allow * dt
        current_heading = self.car.get_heading()
        if self.prev_heading is not None:
            dtheta = math.atan2(math.sin(current_heading - self.prev_heading),
                                 math.cos(current_heading - self.prev_heading))
            delta_allow = (self.car.max_wheel_speed / self.car.effective_track_width) * self._last_dt
            excess = max(0.0, abs(dtheta) - delta_allow)
            # Convert excess rotation to an equivalent linear "wobble" distance at half effective_track_width
            # and scale with the same factor as forward progress to keep units comparable.
            reward_components['heading_penalty'] = -10.0 * (0.5 * self.car.effective_track_width) * excess
        else:
            reward_components['heading_penalty'] = 0.0

        # Update previous heading for next step
        self.prev_heading = current_heading

        # Proximity penalty based on distance from centerline
        reward_components['proximity_penalty'] = -self._calculate_proximity_penalty()*0.75

        # Check car direction relative to centerline
        if self.racing_line:
            centerline, _ = self.racing_line
            car_pos = (current_pos[0], current_pos[1])
            direction_dot = self._project_heading_to_centerline(car_pos, current_heading, centerline)
            
            # If dot product is negative, car is facing wrong direction
            if direction_dot < 0:
                self.wrong_direction_steps += 1
                if self.wrong_direction_steps >= self.max_wrong_direction_steps:
                    # Big penalty and end episode
                    reward_components['wrong_direction_penalty'] = self.wrong_direction_penalty
                    terminated = True  # End episode due to wrong direction
                else:
                    reward_components['wrong_direction_penalty'] = 0.0
            else:
                # Reset counter when facing correct direction
                self.wrong_direction_steps = 0
                reward_components['wrong_direction_penalty'] = 0.0
        else:
            reward_components['wrong_direction_penalty'] = 0.0

        # Calculate total reward and store components for display
        total_reward = sum(reward_components.values())
        self.last_reward_components = reward_components
        
        # Check if episode should terminate due to wrong direction
        terminate_due_to_wrong_direction = (self.wrong_direction_steps >= self.max_wrong_direction_steps)
        
        return total_reward, terminate_due_to_wrong_direction
    
    def _project_point_to_centerline(self, point, centerline):
        """
        Project a point onto the centerline and return the distance.
        
        Args:
            point: (x, y) tuple or array
            centerline: Nx2 array of centerline points
            
        Returns:
            distance: Distance from point to projected point on centerline
        """
        if centerline is None or len(centerline) < 2:
            return float('inf')
            
        C = np.asarray(centerline, dtype=float)
        P = C
        Q = np.roll(C, -1, axis=0)  # segment ends (wrap around)
        v = Q - P  # segment vectors
        s = np.asarray(point, float)
        
        # Projection parameter t for each segment (clamped to [0,1])
        vv = (v[:, 0]**2 + v[:, 1]**2)
        vv = np.where(vv < 1e-12, 1e-12, vv)  # avoid division by zero
        t = np.clip(((s[0]-P[:,0])*v[:,0] + (s[1]-P[:,1])*v[:,1]) / vv, 0.0, 1.0)
        
        # Projected points on each segment
        proj = P + t[:, None] * v
        dist2 = (proj[:, 0] - s[0])**2 + (proj[:, 1] - s[1])**2
        
        # Find the minimum distance
        min_dist2 = np.min(dist2)
        return np.sqrt(min_dist2)
    
    def _project_heading_to_centerline(self, car_pos, car_heading, centerline):
        """
        Project car's heading vector onto the centerline to determine direction.
        
        Args:
            car_pos: (x, y) car center position
            car_heading: angle in radians
            centerline: Nx2 array of centerline points
            
        Returns:
            dot_product: positive if heading aligns with centerline direction, negative if opposite
        """
        if centerline is None or len(centerline) < 2:
            return 0.0
            
        # First project car position onto centerline
        C = np.asarray(centerline, dtype=float)
        P = C
        Q = np.roll(C, -1, axis=0)
        v = Q - P
        s = np.asarray(car_pos, float)
        
        # Projection parameter t for each segment
        vv = (v[:, 0]**2 + v[:, 1]**2)
        vv = np.where(vv < 1e-12, 1e-12, vv)
        t = np.clip(((s[0]-P[:,0])*v[:,0] + (s[1]-P[:,1])*v[:,1]) / vv, 0.0, 1.0)
        
        # Find the closest segment
        proj = P + t[:, None] * v
        dist2 = (proj[:, 0] - s[0])**2 + (proj[:, 1] - s[1])**2
        i = int(np.argmin(dist2))
        
        # Get the centerline direction at this segment
        centerline_dir = v[i]
        centerline_dir = centerline_dir / np.linalg.norm(centerline_dir)
        
        # Get car's heading direction
        car_dir = np.array([np.cos(car_heading), np.sin(car_heading)])
        
        # Calculate dot product
        dot_product = np.dot(car_dir, centerline_dir)
        
        return dot_product
    
    def _calculate_proximity_penalty(self):
        """Calculate proximity penalty based on distance from centerline."""
        if not self.racing_line:
            return 0.0
        
        # Get car center position
        car_x, car_y = self.car.get_position()
        car_pos = (car_x, car_y)
        
        # Get the centerline
        centerline, _ = self.racing_line
        
        # Project car position onto centerline and get distance
        distance_from_centerline = self._project_point_to_centerline(car_pos, centerline)
        
        # Convert to proximity penalty
        # Higher penalty for larger distance values (farther from centerline)
        # Start applying penalty when car is more than optimal_distance from centerline
        optimal_distance = 10.0  # pixels - distance considered "on track"
        if distance_from_centerline > optimal_distance:
            # Linear penalty that increases as distance from centerline increases
            penalty_strength = 2.0  # Adjust this to control penalty severity
            excess_distance = distance_from_centerline - optimal_distance
            penalty = penalty_strength * excess_distance
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
        d = self.distance_field.astype(np.float32)/255.0  # Normalize to [0, 1]
        
        # Handle invalid values (NaN, inf) that might come from the new distance field computation
        d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=0.0)
        
        # Ensure values are clamped to valid range [0, 1]
        d = np.clip(d, 0.0, 1.0)
        
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
    
    def _adjust_centerline_direction(self, centerline, proj_pt, seg_idx):
        """
        Adjust centerline direction to match car's initial heading.
        
        Args:
            centerline: Nx2 array of centerline points starting from proj_pt
            proj_pt: (x,y) projection point on centerline
            seg_idx: Index of segment containing the projection
            
        Returns:
            Adjusted centerline array with correct direction
        """
        if len(centerline) < 3:
            return centerline  # Not enough points to determine direction
            
        # Get car's initial heading (use the first start heading)
        car_heading = self.start_headings[0] if self.start_headings else math.pi
        
        # Create heading vector from car's heading angle
        car_dir = np.array([math.cos(car_heading), math.sin(car_heading)])
        
        # Find the start point index in the centerline (should be index 0)
        start_idx = 0
        
        # Get points around the start for direction calculation
        n_points = len(centerline)
        
        # Get the next point (forward direction)
        next_idx = (start_idx + 1) % n_points
        forward_dir = centerline[next_idx] - centerline[start_idx]
        forward_dir = forward_dir / np.linalg.norm(forward_dir)  # Normalize
        
        # Get the previous point (backward direction) 
        prev_idx = (start_idx - 1) % n_points
        backward_dir = centerline[prev_idx] - centerline[start_idx]
        backward_dir = backward_dir / np.linalg.norm(backward_dir)  # Normalize
        
        # Calculate dot products
        forward_dot = np.dot(car_dir, forward_dir)
        backward_dot = np.dot(car_dir, backward_dir)
        
        # Add some tolerance to avoid flipping on very small differences
        tolerance = 0.1
        
        # If backward direction has significantly smaller dot product, reverse the centerline
        if backward_dot < forward_dot - tolerance:
            # print(f"Reversing centerline direction (car heading: {math.degrees(car_heading):.1f}°, forward_dot: {forward_dot:.3f}, backward_dot: {backward_dot:.3f})")
            # Reverse the centerline but keep the start point first
            reversed_centerline = np.flip(centerline, axis=0)
            # Rotate so start point is first again
            start_pos = np.where(np.allclose(reversed_centerline, proj_pt, atol=1e-6))[0]
            if len(start_pos) > 0:
                start_pos = start_pos[0]
                reversed_centerline = np.roll(reversed_centerline, -start_pos, axis=0)
            return reversed_centerline
        else:
            print(f"Keeping centerline direction (car heading: {math.degrees(car_heading):.1f}°, forward_dot: {forward_dot:.3f}, backward_dot: {backward_dot:.3f})")
            return centerline
    
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
        if self.show_racing_line and self.racing_line:
            racing_line, start_point = self.racing_line
            if racing_line is not None and len(racing_line) > 0:
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
        angular_velocity_dps = math.degrees(self.car.angular_velocity)
        info_lines = [
            f"FPS: {fps:.1f}" if fps else "FPS: --",
            f"Position: ({car_x:.1f}, {car_y:.1f})",
            f"Heading: {math.degrees(self.car.get_heading()):.1f}°",
            f"Angular Velocity: {angular_velocity_dps:.1f}°/s",
            f"Left Wheel: {self.car.left_wheel_velocity:.2f}",
            f"Right Wheel: {self.car.right_wheel_velocity:.2f}",
            f"Step: {self.current_step}/{self.max_steps}"
        ]
        
        # get sensor readings for display (use cached to avoid redundant computation)
        sensor_readings = self._get_cached_sensor_readings()
        info_lines.append(f"Sensors: {[f'{r:.2f}' for r in sensor_readings]}")
        
        # Add reward breakdown
        info_lines.append(f"Total Reward: {self.last_reward:.3f}")
        if hasattr(self, 'last_reward_components') and self.last_reward_components:
            for component, value in self.last_reward_components.items():
                if value != 0.0:  # Only show non-zero components
                    info_lines.append(f"  {component.replace('_', ' ').title()}: {value:.3f}")
        
        for i, line in enumerate(info_lines):
            text = self.debug_font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, 10 + i * 25))
