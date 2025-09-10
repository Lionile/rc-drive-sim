"""
Controllers package - imports all available controllers.
"""

from .base_controller import BaseController
from .manual_controller import ManualController

# Import other controllers when they exist
try:
    from .ppo_controller import PPOController
except ImportError:
    PPOController = None

try:
    from .td3_controller import TD3Controller
except ImportError:
    TD3Controller = None

try:
    from .random_controller import RandomController
except ImportError:
    RandomController = None

__all__ = ['BaseController', 'ManualController', 'PPOController', 'TD3Controller', 'RandomController']
