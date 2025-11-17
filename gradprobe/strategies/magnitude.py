"""
Magnitude-based pruning strategy.

This strategy selects weights with the smallest absolute values for pruning,
which is one of the most common and straightforward pruning approaches.
"""

from typing import Dict
import torch
import torch.nn as nn

from .base import PruningStrategy


class MagnitudePruning(PruningStrategy):
    """
    Magnitude-based pruning strategy.

    Selects weights with the smallest absolute values for tentative pruning.
    This is a simple yet effective baseline pruning method.
    """

    def select_weights_to_prune(
        self,
        model: nn.Module,
        sparsity: float
    ) -> Dict[str, torch.Tensor]:
        """
        Select weights to prune based on their absolute magnitude.

        Args:
            model: The neural network model to analyze
            sparsity: Target sparsity level (fraction of weights to prune, 0-1)

        Returns:
            Dictionary mapping parameter names to boolean masks where True indicates
            the weight should be tentatively pruned
        """
        if not 0 <= sparsity <= 1:
            raise ValueError(f"Sparsity must be between 0 and 1, got {sparsity}")

        # Collect all weights and their absolute values
        all_weights = []
        param_shapes = {}
        param_names = []

        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:  # Only prune weight matrices
                all_weights.append(param.data.abs().flatten())
                param_shapes[name] = param.shape
                param_names.append(name)

        if not all_weights:
            return {}

        # Concatenate all weights and find the threshold
        all_weights_flat = torch.cat(all_weights)
        num_weights_to_prune = int(sparsity * len(all_weights_flat))

        if num_weights_to_prune == 0:
            # Return empty masks (no pruning)
            # Create on CPU if model is on CUDA to save GPU memory
            masks = {}
            for name in param_names:
                param = model.state_dict()[name]
                if hasattr(param, 'device') and param.device.type == 'cuda':
                    masks[name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')
                else:
                    masks[name] = torch.zeros_like(param, dtype=torch.bool)
            return masks

        # Find the threshold value (the magnitude at which we prune)
        threshold = torch.topk(
            all_weights_flat,
            num_weights_to_prune,
            largest=False
        ).values.max()

        # Create masks for each parameter
        # Create on CPU if model is on CUDA to save GPU memory
        masks = {}
        for name, param in model.named_parameters():
            if name in param_names:
                # True indicates this weight should be tentatively pruned
                mask = param.data.abs() <= threshold
                if param.device.type == 'cuda':
                    masks[name] = mask.cpu()
                else:
                    masks[name] = mask
            else:
                # Don't prune this parameter
                if param.device.type == 'cuda':
                    masks[name] = torch.zeros(param.data.shape, dtype=torch.bool, device='cpu')
                else:
                    masks[name] = torch.zeros_like(param.data, dtype=torch.bool)

        return masks

    def get_name(self) -> str:
        """Return the name of this pruning strategy."""
        return "magnitude"
