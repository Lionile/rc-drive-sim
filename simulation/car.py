"""
Car physics and dynamics for differential drive robot.
"""

import numpy as np
import pygame
import math

class DifferentialDriveCar:
    def __init__(self, x=0, y=0, heading=0, wheelbase=63):  # 63 pixels = 6.3cm
        """
        Initialize the differential drive car.
        
        Args:
            x: Initial x position (pixels)
            y: Initial y position (pixels) 
            heading: Initial heading in radians (0 = facing right)
            wheelbase: Distance between wheels in pixels (63 pixels = 6.3cm)
        """
        self.x = x
        self.y = y
        self.heading = heading
        self.wheelbase = wheelbase
        
        # Wheel velocities (-1 to 1)
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        
        # Car dimensions for rendering and collision (in pixels)
        # width_px: lateral size (wheel-to-wheel distance) -> 63 px
        # length_px: longitudinal size (car length) -> 0.8 * wheelbase
        self.width_px = int(wheelbase)              # lateral (Y-axis in local frame)
        self.length_px = int(wheelbase * 0.84)       # longitudinal (X-axis in local frame)
            
        # Maximum wheel speed in pixels/second (1 px = 1 mm)
        # 0.237 m/s = 237 mm/s = 237 px/s
        self.max_wheel_speed = 237.0
        
        # Load and scale the car sprite
        self.load_sprite()
        
    def load_sprite(self):
        """Load and scale the car sprite."""
        try:
            self.original_sprite = pygame.image.load("sprite_images/Player.png").convert_alpha()
            # Scale sprite to match car dimensions. Assume the sprite faces to the right by default,
            # so X dimension = car length, Y dimension = car width (wheelbase).
            self.original_sprite = pygame.transform.scale(
                self.original_sprite, (self.length_px, self.width_px)
            )
        except pygame.error:
            # Create a simple rectangle if sprite not found
            self.original_sprite = pygame.Surface((self.length_px, self.width_px), pygame.SRCALPHA)
            pygame.draw.rect(self.original_sprite, (255, 0, 0), (0, 0, self.length_px, self.width_px))
            # Add direction indicator (triangle at the front/right side)
            pygame.draw.polygon(
                self.original_sprite,
                (255, 255, 255),
                [
                    (self.length_px - 6, self.width_px // 2 - 5),
                    (self.length_px, self.width_px // 2),
                    (self.length_px - 6, self.width_px // 2 + 5),
                ],
            )
        
        self.sprite = self.original_sprite
        
    def set_wheel_velocities(self, left_vel, right_vel):
        """Set the wheel velocities (-1 to 1)."""
        self.left_wheel_velocity = np.clip(left_vel, -1.0, 1.0)
        self.right_wheel_velocity = np.clip(right_vel, -1.0, 1.0)
        
    def update(self, dt):
        """
        Update car position and orientation based on differential drive kinematics.
        
        Args:
            dt: Time step in seconds
        """
        # Convert normalized velocities to actual speeds
        left_speed = self.left_wheel_velocity * self.max_wheel_speed
        right_speed = self.right_wheel_velocity * self.max_wheel_speed
        
        # Differential drive kinematics (unicycle approximation)
        v = (left_speed + right_speed) / 2.0
        omega = (right_speed - left_speed) / self.wheelbase

        # Screen coordinates have Y increasing downward. With y-down, decreasing
        # the heading rotates visually to the left; increasing rotates to the right.
        # To make "left wheel faster" (left input) turn visually left, we ADD omega.
        self.x += v * math.cos(self.heading) * dt
        self.y += v * math.sin(self.heading) * dt
        self.heading += omega * dt
                    
        # Normalize heading to [-pi, pi]
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
        
        # Update sprite rotation
        self.update_sprite()
    
    def update_sprite(self):
        """Update the rotated sprite based on current heading."""
        # Convert heading to degrees (pygame uses degrees, and 0 degrees is right)
        angle_degrees = -math.degrees(self.heading)
        self.sprite = pygame.transform.rotate(self.original_sprite, angle_degrees)
    
    def get_position(self):
        """Get current position as (x, y) tuple."""
        return (self.x, self.y)
    
    def get_heading(self):
        """Get current heading in radians."""
        return self.heading
    
    def get_bounding_box(self):
        """
        Get the car's bounding box for collision detection.
        Returns the four corners of the rotated rectangle.
        """
        # Half dimensions in local car frame
        half_length = self.length_px / 2.0   # along local X (forward)
        half_width = self.width_px / 2.0     # along local Y (left-right)
            
        # Corners in local coordinates (center at origin)
        corners_local = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width),
        ]
        
        # Rotate corners based on heading
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        
        corners_world = []
        for lx, ly in corners_local:
            # Rotate and translate to world coordinates
            wx = self.x + lx * cos_h - ly * sin_h
            wy = self.y + lx * sin_h + ly * cos_h
            corners_world.append((wx, wy))
        
        return corners_world
    
    def get_render_rect(self):
        """Get the rectangle for rendering the sprite."""
        sprite_rect = self.sprite.get_rect()
        sprite_rect.center = (int(self.x), int(self.y))
        return sprite_rect
    
    def reset(self, x, y, heading):
        """Reset car to initial position and orientation."""
        self.x = x
        self.y = y
        self.heading = heading
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.update_sprite()
