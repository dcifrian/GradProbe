"""
Magnitude-based pruning strategy.

This strategy selects weights with the smallest absolute values for pruning,
which is one of the most common and straightforward pruning approaches.
"""

from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import psutil
import time

from .base import PruningStrategy


def log_memory(msg, log_file="/tmp/magnitude_memory.log"):
    """Log current memory usage to file."""
    process = psutil.Process()
    rss_gb = process.memory_info().rss / (1024**3)
    vms_gb = process.memory_info().vms / (1024**3)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / (1024**3)
        vram_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        log_line = f"[{time.strftime('%H:%M:%S')}] {msg} | RAM: {rss_gb:.2f}GB (VMS: {vms_gb:.2f}GB) | VRAM: {vram_gb:.2f}GB (Reserved: {vram_reserved_gb:.2f}GB)\n"
    else:
        log_line = f"[{time.strftime('%H:%M:%S')}] {msg} | RAM: {rss_gb:.2f}GB (VMS: {vms_gb:.2f}GB)\n"

    with open(log_file, 'a') as f:
        f.write(log_line)
    print(log_line.strip())


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

        Uses per-layer l1_unstructured to avoid memory explosion from global pruning.

        Args:
            model: The neural network model to analyze
            sparsity: Target sparsity level (fraction of weights to prune, 0-1)

        Returns:
            Dictionary mapping parameter names to boolean masks where True indicates
            the weight should be tentatively pruned
        """
        log_memory("START: Entering select_weights_to_prune")

        if not 0 <= sparsity <= 1:
            raise ValueError(f"Sparsity must be between 0 and 1, got {sparsity}")

        # Collect parameters to prune (weight matrices only)
        param_list = []

        log_memory("Before collecting parameters")
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:  # Only prune weight matrices
                param_list.append((name, param))

        log_memory(f"After collecting {len(param_list)} parameters")

        if not param_list:
            return {}

        if sparsity == 0:
            # No pruning - return empty masks
            masks = {}
            for name, param in param_list:
                masks[name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')
            # Add non-prunable params
            for name, param in model.named_parameters():
                if name not in masks:
                    masks[name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')
            return masks

        # Process each parameter individually using l1_unstructured
        # This avoids the memory explosion from global_unstructured
        log_memory("Starting per-layer pruning")
        masks = {}

        for i, (name, param) in enumerate(param_list):
            # Find the module and parameter name
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                module_name, param_name = parts
                module = model.get_submodule(module_name)
            else:
                # Top-level parameter
                module = model
                param_name = name

            # Move to CPU if needed to avoid VRAM issues
            original_device = param.device
            if param.device.type == 'cuda':
                param.data = param.data.cpu()

            # Apply l1_unstructured to this single parameter
            prune.l1_unstructured(module, param_name, amount=sparsity)

            # Extract the mask
            mask_name = f"{param_name}_mask"
            if hasattr(module, mask_name):
                pytorch_mask = getattr(module, mask_name)
                # Invert for our convention (True = prune)
                our_mask = ~pytorch_mask
                masks[name] = our_mask.cpu() if our_mask.device.type == 'cuda' else our_mask
            else:
                masks[name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')

            # Clean up: remove the pruning hook
            prune.remove(module, param_name)

            # Move back to original device
            if original_device.type == 'cuda':
                param.data = param.data.to(original_device)

            if i % 50 == 0:
                log_memory(f"Processed {i}/{len(param_list)} params")

        log_memory(f"After processing all {len(param_list)} params")

        # Add non-prunable parameters with zero masks
        for name, param in model.named_parameters():
            if name not in masks:
                masks[name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')

        log_memory(f"END: Returning {len(masks)} masks")
        return masks

    def get_name(self) -> str:
        """Return the name of this pruning strategy."""
        return "magnitude"
