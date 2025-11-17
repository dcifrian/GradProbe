"""
Magnitude-based pruning strategy.

This strategy selects weights with the smallest absolute values for pruning,
which is one of the most common and straightforward pruning approaches.
"""

from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

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

        Uses PyTorch's built-in global_unstructured pruning which is memory-efficient
        and doesn't require concatenating all weights.

        Args:
            model: The neural network model to analyze
            sparsity: Target sparsity level (fraction of weights to prune, 0-1)

        Returns:
            Dictionary mapping parameter names to boolean masks where True indicates
            the weight should be tentatively pruned
        """
        if not 0 <= sparsity <= 1:
            raise ValueError(f"Sparsity must be between 0 and 1, got {sparsity}")

        # Collect parameters to prune (weight matrices only)
        parameters_to_prune = []
        param_name_to_module = {}

        for name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad and len(param.shape) >= 2:  # Only prune weight matrices
                    full_name = f"{name}.{param_name}" if name else param_name
                    parameters_to_prune.append((module, param_name))
                    param_name_to_module[full_name] = (module, param_name)

        if not parameters_to_prune:
            return {}

        if sparsity == 0:
            # No pruning - return empty masks
            masks = {}
            for full_name, (module, param_name) in param_name_to_module.items():
                param = getattr(module, param_name)
                masks[full_name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')
            return masks

        # Use PyTorch's global_unstructured which is memory-efficient
        # It computes the threshold without concatenating all weights
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )

        # Extract the masks that were created
        masks = {}
        for full_name, (module, param_name) in param_name_to_module.items():
            # PyTorch creates a mask attribute named {param_name}_mask
            mask_name = f"{param_name}_mask"
            if hasattr(module, mask_name):
                # Get the mask (True = keep, False = prune in PyTorch convention)
                pytorch_mask = getattr(module, mask_name)
                # Invert it for our convention (True = prune)
                our_mask = ~pytorch_mask
                # Move to CPU to save VRAM
                masks[full_name] = our_mask.cpu()
            else:
                # Fallback if no mask was created
                param = getattr(module, param_name)
                masks[full_name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')

        # Clean up: remove the pruning hooks and masks to restore original state
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        # Also include non-prunable parameters with zero masks
        for name, param in model.named_parameters():
            if name not in masks:
                masks[name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')

        return masks

    def get_name(self) -> str:
        """Return the name of this pruning strategy."""
        return "magnitude"
