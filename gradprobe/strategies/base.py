"""
Base pruning strategy interface for GradProbe.

This module defines the abstract base class for all pruning strategies.
Strategies determine which weights should be considered for tentative pruning
before the gradient comparison step.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class PruningStrategy(ABC):
    """
    Abstract base class for pruning strategies.

    A pruning strategy determines which weights in a model should be
    tentatively pruned based on some criterion (e.g., magnitude, activation, etc.).
    """

    @abstractmethod
    def select_weights_to_prune(
        self,
        model: nn.Module,
        sparsity: float
    ) -> Dict[str, torch.Tensor]:
        """
        Select weights to tentatively prune based on the strategy's criterion.

        Args:
            model: The neural network model to analyze
            sparsity: Target sparsity level (fraction of weights to prune, 0-1)

        Returns:
            Dictionary mapping parameter names to boolean masks where True indicates
            the weight should be tentatively pruned
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this pruning strategy."""
        pass
