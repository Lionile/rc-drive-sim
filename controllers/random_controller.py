"""
Random controller for testing and baseline comparison.
Outputs random actions within valid ranges.
"""

import numpy as np
from .base_controller import BaseController

class RandomController(BaseController):
    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        
    def act(self, observation, dt=1.0/60.0):
        """
        Generate random actions.
        
        Args:
            observation: Current sensor readings (not used)
            dt: Time delta for physics update (not used for random actions)
            
        Returns:
            [left_wheel_velocity, right_wheel_velocity] in range [-1, 1]
        """
        # Generate random wheel velocities
        left_vel = self.rng.uniform(-1.0, 1.0)
        right_vel = self.rng.uniform(-1.0, 1.0)
        
        return [left_vel, right_vel]
    
    def reset(self):
        """Reset controller state (nothing to reset for random controller)."""
        pass
