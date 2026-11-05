"""JAX Forward-Backward representations for continuous Four Rooms."""
from .environment import ContinuousFourRooms
from .agent import FBConfig, FBAgent

__all__ = ["ContinuousFourRooms", "FBConfig", "FBAgent"]
