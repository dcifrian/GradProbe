"""
Pruning strategies for GradProbe.

This package contains various strategies for selecting weights to tentatively prune.
"""

from .base import PruningStrategy

# Import optimized versions as the default
from .magnitude_optimized import MagnitudePruningOptimized as MagnitudePruning
from .wanda_optimized import WANDAPruningOptimized as WANDAPruning

# Also expose the original versions for comparison
from .magnitude import MagnitudePruning as MagnitudePruningOriginal
from .wanda import WANDAPruning as WANDAPruningOriginal

__all__ = [
    'PruningStrategy',
    'MagnitudePruning',
    'WANDAPruning',
    'MagnitudePruningOriginal',
    'WANDAPruningOriginal',
]
