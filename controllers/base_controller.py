"""
Base controller interface for RC car simulation.
All controllers should inherit from this base class.
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseController(ABC):
    """
    Base class for all RC car controllers.
    
    Controllers take observations and return actions for the car.
    """
    
    def __init__(self):
        """Initialize the controller."""
        pass
    
    @abstractmethod
    def act(self, observation, dt=1.0/60.0):
        """
        Get action from observation.
        
        Args:
            observation: Current sensor readings/state (numpy array or list)
            dt: Time delta for physics update (default: 1/60 seconds)
            
        Returns:
            action: [left_wheel_velocity, right_wheel_velocity] in range [-1, 1]
        """
        pass
    
    def reset(self):
        """
        Reset controller state (optional).
        Called when environment is reset.
        """
        pass
    
    def close(self):
        """
        Cleanup controller resources (optional).
        Called when shutting down.
        """
        pass
