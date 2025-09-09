"""
Main environment class for the RC car simulation.
Follows OpenAI Gym interface for easy RL integration.
"""

import pygame
import numpy as np
import math
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.map_generator import extract_and_scale_contours, generate_racing_line
from simulation.car import DifferentialDriveCar
from simulation.sensors import SensorArray

class RCCarEnvironment:
    def __init__(self, map_path, window_size=(1024, 1024), show_collision_box=True, 
                 show_sensors=True, show_racing_line=True, display_map_path=None):
        """
        Initialize the RC car environment.
        
        Args:
            map_path: Path to the track map image
            window_size: Pygame window size
            show_collision_box: Whether to show car collision box
            show_sensors: Whether to show sensor rays
            show_racing_line: Whether to show racing line
        """
        self.window_size = window_size
        self.show_collision_box = show_collision_box
        self.show_sensors = show_sensors
        self.show_racing_line = show_racing_line
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("RC Car Simulation")
        
        # Load map and track data
        self.map_path = map_path  # expected to be the start-mask map (e.g., maps/map_start1.png)
        self.display_map_path = display_map_path  # optional explicit visual map (e.g., maps/map1.png)
        self.load_map()
        
        # Initialize car
        self.car = DifferentialDriveCar()
        
        # Initialize sensors
        self.sensors = SensorArray()
        
        # Episode management
        self.max_steps = 3000
        self.current_step = 0
        
        # Reset to starting position
        self.reset()
    
    def load_map(self):
        """Load the track map and extract boundaries."""
        try:
            # Determine which image to use for display (visual track)
            display_path = self.display_map_path
            if not display_path:
                # Try to auto-derive from provided map_path
                dirname, filename = os.path.split(self.map_path)
                candidate = None
                if "map_start" in filename:
                    candidate = filename.replace("map_start", "map")
                elif "map_mask" in filename:
                    candidate = filename.replace("map_mask", "map")
                # Fallback: if no pattern matched, use original
                display_path = os.path.join(dirname, candidate) if candidate else self.map_path
                if not os.path.exists(display_path):
                    # If derived path doesn't exist, fall back to original
                    display_path = self.map_path
            
            # Load map image for display
            self.map_image = pygame.image.load(display_path).convert()
            
            # Extract track boundaries using map_generator
            # Use the provided map_path (usually map_startX.png) so start positions (green circles) are detectable
            boundaries_outer, boundaries_inner, start_points, headings = extract_and_scale_contours(self.map_path)
            
            # Store boundaries for collision detection
            self.track_boundaries = [boundaries_outer, boundaries_inner]
            
            # Store start positions
            self.start_positions = start_points if start_points else [(512, 512)]
            self.start_headings = headings if headings else [0.0]
            
            # Generate racing line for visualization
            if self.show_racing_line:
                try:
                    racing_lines = generate_racing_line(self.map_path)
                    self.racing_lines = racing_lines
                except Exception as e:
                    print(f"Could not generate racing line: {e}")
                    self.racing_lines = []
            else:
                self.racing_lines = []
            
        except Exception as e:
            print(f"Error loading map: {e}")
            # Create a simple default map
            self.map_image = pygame.Surface(self.window_size)
            self.map_image.fill((255, 255, 255))  # White background
            pygame.draw.rect(self.map_image, (0, 0, 0), (100, 100, 824, 824), 50)  # Black border
            
            # Default boundaries (simple rectangular track)
            self.track_boundaries = [
                [(100, 100), (924, 100), (924, 924), (100, 924)],  # Outer boundary
                [(150, 150), (874, 150), (874, 874), (150, 874)]   # Inner boundary
            ]
            self.start_positions = [(200, 200)]
            self.start_headings = [0.0]
            self.racing_lines = []
    
    def reset(self):
        """Reset the environment to initial state."""
        # Choose a random start position
        if self.start_positions:
            start_idx = np.random.randint(len(self.start_positions))
            start_x, start_y = self.start_positions[start_idx]
            start_heading = self.start_headings[start_idx] if start_idx < len(self.start_headings) else 0.0
            # Face the opposite direction (rotate by 180 degrees)
            start_heading = math.atan2(math.sin(start_heading + math.pi), math.cos(start_heading + math.pi))
        else:
            start_x, start_y = 200, 200
            start_heading = 0.0
        
        # Reset car
        self.car.reset(start_x, start_y, start_heading)
        
        # Reset episode counters
        self.current_step = 0
        
        # Get initial observation
        observation = self.get_observation()
        
        return observation
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: [left_wheel_velocity, right_wheel_velocity] both in range [-1, 1]
            
        Returns:
            observation: Current sensor readings
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        # Apply action to car
        left_vel, right_vel = action
        self.car.set_wheel_velocities(left_vel, right_vel)
        
        # Update car physics (60 FPS)
        dt = 1.0 / 60.0
        self.car.update(dt)
        
        # Get observation
        observation = self.get_observation()
        
        # Check for collision
        collision = self.check_collision()
        
        # Calculate reward (placeholder for now)
        reward = self.calculate_reward(collision)
        
        # Check if episode is done
        self.current_step += 1
        done = collision or self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'collision': collision,
            'position': self.car.get_position(),
            'heading': self.car.get_heading(),
            'step': self.current_step
        }
        
        return observation, reward, done, info
    
    def get_observation(self):
        """Get current observation (sensor readings)."""
        car_x, car_y = self.car.get_position()
        car_heading = self.car.get_heading()
        
        # Get sensor readings
        sensor_readings = self.sensors.get_readings(car_x, car_y, car_heading, self.track_boundaries)
        
        return np.array(sensor_readings, dtype=np.float32)
    
    def check_collision(self):
        """Check if car collides with track boundaries."""
        # Simple implementation - check if car center is outside track bounds
        car_x, car_y = self.car.get_position()
        
        # Check if car is within map bounds
        if car_x < 0 or car_x >= self.window_size[0] or car_y < 0 or car_y >= self.window_size[1]:
            return True
        
        # For now, just check basic bounds - proper polygon collision can be added later
        # This is a placeholder that can be enhanced with proper boundary polygon checking
        return False
    
    def calculate_reward(self, collision):
        """Calculate reward for current step."""
        if collision:
            return -100.0
        
        # Simple reward: small positive for staying alive
        return 1.0
    
    def render(self):
        """Render the environment."""
        # Clear screen
        self.screen.fill((50, 50, 50))
        
        # Draw map
        self.screen.blit(self.map_image, (0, 0))
        
        # Draw racing lines
        if self.show_racing_line and self.racing_lines:
            for racing_line, start_point in self.racing_lines:
                if racing_line:
                    for i, point in enumerate(racing_line):
                        color = (255, 255 - i * 255 // len(racing_line), 0)  # Red to yellow gradient
                        pygame.draw.circle(self.screen, color, (int(point[0]), int(point[1])), 2)
        
        # Draw car
        car_rect = self.car.get_render_rect()
        self.screen.blit(self.car.sprite, car_rect)
        
        # Draw collision box
        if self.show_collision_box:
            corners = self.car.get_bounding_box()
            pygame.draw.polygon(self.screen, (255, 0, 0), corners, 2)
        
        # Draw sensors
        if self.show_sensors:
            car_x, car_y = self.car.get_position()
            car_heading = self.car.get_heading()
            sensor_readings = self.sensors.get_readings(car_x, car_y, car_heading, self.track_boundaries)
            sensor_rays = self.sensors.get_sensor_rays(car_x, car_y, car_heading, sensor_readings)
            
            colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]  # Green, Yellow, Cyan
            for i, (start, end) in enumerate(sensor_rays):
                pygame.draw.line(self.screen, colors[i], 
                               (int(start[0]), int(start[1])), 
                               (int(end[0]), int(end[1])), 2)
        
        # Draw debug info
        self.draw_debug_info()
    
    def draw_debug_info(self):
        """Draw debug information on screen."""
        font = pygame.font.Font(None, 24)
        
        # Car info
        car_x, car_y = self.car.get_position()
        info_lines = [
            f"Position: ({car_x:.1f}, {car_y:.1f})",
            f"Heading: {math.degrees(self.car.get_heading()):.1f}°",
            f"Left Wheel: {self.car.left_wheel_velocity:.2f}",
            f"Right Wheel: {self.car.right_wheel_velocity:.2f}",
            f"Step: {self.current_step}/{self.max_steps}"
        ]
        
        # Get sensor readings for display
        sensor_readings = self.sensors.get_readings(car_x, car_y, self.car.get_heading(), self.track_boundaries)
        info_lines.append(f"Sensors: {[f'{r:.2f}' for r in sensor_readings]}")
        
        for i, line in enumerate(info_lines):
            text = font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, 10 + i * 25))
    
    def close(self):
        """Close the environment."""
        pygame.quit()
