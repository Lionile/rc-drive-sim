"""
Manual control system for the RC car simulation.
"""

import pygame
import numpy as np

class ManualController:
    def __init__(self):
        """Initialize the manual controller."""
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0
        
        # Control sensitivity
        self.acceleration = 3.0  # How fast the car accelerates
        self.turn_rate = 2.0     # How fast the car turns
        self.decay_rate = 5.0    # How fast velocities decay when no input
        
    def get_action(self):
        """
        Get control action based on keyboard input.
        
        Returns:
            [left_wheel_velocity, right_wheel_velocity] in range [-1, 1]
        """
        dt = 1.0 / 60.0  # Assume 60 FPS
        
        # Get current key states
        keys = pygame.key.get_pressed()
        
        # Apply decay to current velocities
        self.left_wheel_vel *= (1.0 - self.decay_rate * dt)
        self.right_wheel_vel *= (1.0 - self.decay_rate * dt)
        
        # Method 1: WASD controls (car-like control)
        if keys[pygame.K_w]:  # Forward
            self.left_wheel_vel += self.acceleration * dt
            self.right_wheel_vel += self.acceleration * dt
            
        if keys[pygame.K_s]:  # Backward
            self.left_wheel_vel -= self.acceleration * dt
            self.right_wheel_vel -= self.acceleration * dt
            
        if keys[pygame.K_a]:  # Turn left (left wheel faster, right wheel slower)
            self.left_wheel_vel += self.turn_rate * dt
            self.right_wheel_vel -= self.turn_rate * dt
            
        if keys[pygame.K_d]:  # Turn right (left wheel slower, right wheel faster)
            self.left_wheel_vel -= self.turn_rate * dt
            self.right_wheel_vel += self.turn_rate * dt
        
        # Method 2: Arrow keys for direct wheel control
        if keys[pygame.K_UP]:  # Both wheels forward
            self.left_wheel_vel += self.acceleration * dt
            self.right_wheel_vel += self.acceleration * dt
            
        if keys[pygame.K_DOWN]:  # Both wheels backward
            self.left_wheel_vel -= self.acceleration * dt
            self.right_wheel_vel -= self.acceleration * dt
            
        if keys[pygame.K_LEFT]:  # Turn left (adjust differentially)
            self.left_wheel_vel += self.turn_rate * dt
            self.right_wheel_vel -= self.turn_rate * dt
            
        if keys[pygame.K_RIGHT]:  # Turn right (adjust differentially)
            self.left_wheel_vel -= self.turn_rate * dt
            self.right_wheel_vel += self.turn_rate * dt
        
        # Clamp velocities to [-1, 1]
        self.left_wheel_vel = np.clip(self.left_wheel_vel, -1.0, 1.0)
        self.right_wheel_vel = np.clip(self.right_wheel_vel, -1.0, 1.0)
        
        return [self.left_wheel_vel, self.right_wheel_vel]
    
    def reset(self):
        """Reset controller state."""
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0
