"""
PID Controller for RC car simulation.
Uses sensor readings to maintain center line and avoid obstacles.

The PID controller:
- Uses sensor difference (left vs right) as error signal for steering
- Uses minimum sensor reading to modulate speed (slow down near obstacles)
- Implements separate PID loops for steering and speed control
"""

import numpy as np
from .base_controller import BaseController

class PIDController(BaseController):
    def __init__(self):
        super().__init__()
        
        # Steering PID parameters (controls left-right balance)
        self.steering_kp = 0.5    # Proportional gain for steering
        self.steering_ki = 0.5    # Integral gain for steering
        self.steering_kd = 0    # Derivative gain for steering
        
        # Speed PID parameters (controls forward motion based on obstacle distance)
        self.speed_kp = 1.0      # Proportional gain for speed
        self.speed_ki = 0.5      # Integral gain for speed
        self.speed_kd = 0.3       # Derivative gain for speed
        
        # PID state variables for steering
        self.steering_error_prev = 0.0
        self.steering_error_integral = 0.0
        
        # PID state variables for speed
        self.speed_error_prev = 0.0
        self.speed_error_integral = 0.0
        
        # Control parameters
        self.target_speed = 0.6           # Desired forward speed (0-1)
        self.min_safe_distance = 0.3     # Minimum safe distance to obstacles (0-1)
        self.max_steering = 0.8           # Maximum steering differential (0-1)
        self.base_speed = 0.4             # Minimum forward speed
        
        # Integral windup prevention
        self.max_integral = 1.0
        
    def reset(self):
        """Reset PID controller state."""
        self.steering_error_prev = 0.0
        self.steering_error_integral = 0.0
        self.speed_error_prev = 0.0
        self.speed_error_integral = 0.0
        
    def act(self, observation, dt=1.0/60.0):
        """
        Get control action using PID control based on sensor readings.
        
        Args:
            observation: Sensor readings [front, left_diagonal, right_diagonal]
            dt: Time delta for physics update
            
        Returns:
            [left_wheel_velocity, right_wheel_velocity] in range [-1, 1]
        """
        # Ensure we have the expected sensor readings
        if len(observation) < 3:
            # Fallback: stop if we don't have enough sensors
            return [0.0, 0.0]
        
        front_sensor = observation[0]     # Front sensor (45° right)
        left_sensor = observation[1]      # Left diagonal sensor  
        right_sensor = observation[2]     # Right diagonal sensor
        
        # Calculate steering error (positive = need to steer right, negative = steer left)
        # If right sensor is closer than left, we're too close to right wall
        steering_error = right_sensor - left_sensor
        
        # Update steering PID
        self.steering_error_integral += steering_error * dt
        # Prevent integral windup
        self.steering_error_integral = np.clip(self.steering_error_integral, 
                                             -self.max_integral, self.max_integral)
        
        steering_error_derivative = (steering_error - self.steering_error_prev) / dt
        
        # Calculate steering output
        steering_output = (self.steering_kp * steering_error + 
                          self.steering_ki * self.steering_error_integral + 
                          self.steering_kd * steering_error_derivative)
        
        # Limit steering output
        steering_output = np.clip(steering_output, -self.max_steering, self.max_steering)
        
        # Calculate speed based on minimum distance to obstacles
        min_distance = min(front_sensor, left_sensor, right_sensor)
        
        # Speed error (positive = need to speed up, negative = slow down)
        if min_distance < self.min_safe_distance:
            # Slow down when too close to obstacles
            target_speed = self.base_speed * (min_distance / self.min_safe_distance)
        else:
            # Maintain target speed when clear
            target_speed = self.target_speed
        
        # For speed control, we use the front sensor as primary feedback
        current_speed_feedback = front_sensor  # Higher distance = can go faster
        speed_error = target_speed - (1.0 - current_speed_feedback)  # Convert distance to speed feedback
        
        # Update speed PID
        self.speed_error_integral += speed_error * dt
        self.speed_error_integral = np.clip(self.speed_error_integral,
                                          -self.max_integral, self.max_integral)
        
        speed_error_derivative = (speed_error - self.speed_error_prev) / dt
        
        # Calculate speed output
        speed_output = (self.speed_kp * speed_error +
                       self.speed_ki * self.speed_error_integral +
                       self.speed_kd * speed_error_derivative)
        
        # Base forward motion + speed adjustment
        base_forward_speed = self.base_speed + speed_output
        base_forward_speed = np.clip(base_forward_speed, 0.0, 1.0)
        
        # Calculate differential wheel speeds
        # Positive steering_output means steer right (left wheel faster)
        left_wheel_speed = base_forward_speed + steering_output
        right_wheel_speed = base_forward_speed - steering_output
        
        # Normalize if any wheel exceeds limits while maintaining differential
        max_wheel = max(abs(left_wheel_speed), abs(right_wheel_speed))
        if max_wheel > 1.0:
            left_wheel_speed /= max_wheel
            right_wheel_speed /= max_wheel
        
        # Clamp to valid range
        left_wheel_speed = np.clip(left_wheel_speed, -1.0, 1.0)
        right_wheel_speed = np.clip(right_wheel_speed, -1.0, 1.0)
        
        # Update previous errors for next iteration
        self.steering_error_prev = steering_error
        self.speed_error_prev = speed_error
        
        return [left_wheel_speed, right_wheel_speed]
