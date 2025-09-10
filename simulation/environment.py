"""
Main environment class for the RC car simulation.
Follows OpenAI Gym interface for easy RL integration.
"""

import pygame
import numpy as np
import math

from utils.map_generator import extract_and_scale_contours, generate_racing_line
from utils.geometry import segment_intersection, distance, contour_to_segments
from simulation.car import DifferentialDriveCar
from simulation.sensors import SensorArray

class Environment:
    def __init__(self, map_path, window_size=(1024, 1024), show_collision_box=True, 
                 show_sensors=True, show_racing_line=True, show_track_edges=False, display_map_path=None):
        """
        Initialize the RC car environment.
        
        Args:
            map_path: Path to the track map image
            window_size: Pygame window size
            show_collision_box: Whether to show car collision box
            show_sensors: Whether to show sensor rays
            show_racing_line: Whether to show racing line
            show_track_edges: Whether to show track boundary edges
        """
        self.window_size = window_size
        self.show_collision_box = show_collision_box
        self.show_sensors = show_sensors
        self.show_racing_line = show_racing_line
        self.show_track_edges = show_track_edges
        
        # init pygame
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("RC Car Simulation")
        
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
        
        self.map_image = pygame.image.load(display_path).convert()
        
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
        
        # generate racing line
        if self.show_racing_line:
            try:
                racing_lines = generate_racing_line(self.map_path)
                self.racing_lines = racing_lines
            except Exception as e:
                print(f"Could not generate racing line: {e}")
                self.racing_lines = []
        else:
            self.racing_lines = []
    
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
    
    def get_observation(self):
        """Get current observation (sensor readings)."""
        car_x, car_y = self.car.get_position()
        car_heading = self.car.get_heading()
        
        sensor_readings = self.sensors.get_readings(car_x, car_y, car_heading, self.boundary_segments)
        
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
        
        # Simple reward: small positive for staying alive
        return 1.0
    
    def render(self):
        """Render the environment."""
        self.screen.fill((50, 50, 50))
        
        self.screen.blit(self.map_image, (0, 0))
        
        # racing line
        if self.show_racing_line and self.racing_lines:
            for racing_line, start_point in self.racing_lines:
                if racing_line:
                    for i, point in enumerate(racing_line):
                        color = (255, 255 - i * 255 // len(racing_line), 0)
                        pygame.draw.circle(self.screen, color, (int(point[0]), int(point[1])), 2)
        
        # car
        car_rect = self.car.get_render_rect()
        self.screen.blit(self.car.sprite, car_rect)
        
        # collision box
        if self.show_collision_box:
            corners = self.car.get_bounding_box()
            pygame.draw.polygon(self.screen, (255, 0, 0), corners, 2)
        
        # sensors
        if self.show_sensors:
            car_x, car_y = self.car.get_position()
            car_heading = self.car.get_heading()
            sensor_readings = self.sensors.get_readings(car_x, car_y, car_heading, self.boundary_segments)
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
        font = pygame.font.Font(None, 24)
        
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
        
        # get sensor readings for display
        sensor_readings = self.sensors.get_readings(car_x, car_y, self.car.get_heading(), self.track_boundaries)
        info_lines.append(f"Sensors: {[f'{r:.2f}' for r in sensor_readings]}")
        
        for i, line in enumerate(info_lines):
            text = font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, 10 + i * 25))
