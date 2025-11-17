"""
Pruning strategies for GradProbe.

This package contains various strategies for selecting weights to tentatively prune.
"""

from .base import PruningStrategy
from .magnitude import MagnitudePruning
from .wanda import WANDAPruning

__all__ = ['PruningStrategy', 'MagnitudePruning', 'WANDAPruning']
