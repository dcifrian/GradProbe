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

        Uses PyTorch's built-in global_unstructured pruning which is memory-efficient
        and doesn't require concatenating all weights.

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
        parameters_to_prune = []
        param_name_to_module = {}

        log_memory("Before collecting parameters")
        for name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad and len(param.shape) >= 2:  # Only prune weight matrices
                    full_name = f"{name}.{param_name}" if name else param_name
                    parameters_to_prune.append((module, param_name))
                    param_name_to_module[full_name] = (module, param_name)

        log_memory(f"After collecting {len(parameters_to_prune)} parameters")

        if not parameters_to_prune:
            return {}

        if sparsity == 0:
            # No pruning - return empty masks
            masks = {}
            for full_name, (module, param_name) in param_name_to_module.items():
                param = getattr(module, param_name)
                masks[full_name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')
            return masks

        # Move model to CPU temporarily to avoid CUDA OOM
        # PyTorch's pruning creates _orig and _mask buffers which would double VRAM usage
        log_memory("Before moving params to CPU")
        original_device = {}
        for i, (module, param_name) in enumerate(parameters_to_prune):
            param = getattr(module, param_name)
            original_device[(module, param_name)] = param.device
            if param.device.type == 'cuda':
                # Move parameter to CPU
                param.data = param.data.cpu()

            # Log every 50 parameters
            if i % 50 == 0:
                log_memory(f"Moved {i}/{len(parameters_to_prune)} params to CPU")

        log_memory(f"After moving all {len(parameters_to_prune)} params to CPU")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log_memory("After CUDA empty_cache")

        # Use PyTorch's global_unstructured which is memory-efficient
        # It computes the threshold without concatenating all weights
        # This now operates on CPU to avoid CUDA OOM
        log_memory("Before prune.global_unstructured")
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        log_memory("After prune.global_unstructured")

        # Extract the masks that were created (already on CPU)
        log_memory("Before extracting masks")
        masks = {}
        for i, (full_name, (module, param_name)) in enumerate(param_name_to_module.items()):
            # PyTorch creates a mask attribute named {param_name}_mask
            mask_name = f"{param_name}_mask"
            if hasattr(module, mask_name):
                # Get the mask (True = keep, False = prune in PyTorch convention)
                pytorch_mask = getattr(module, mask_name)
                # Invert it for our convention (True = prune)
                our_mask = ~pytorch_mask
                masks[full_name] = our_mask
            else:
                # Fallback if no mask was created
                param = getattr(module, param_name)
                masks[full_name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')

            # Log every 50 masks
            if i % 50 == 0:
                log_memory(f"Extracted {i}/{len(param_name_to_module)} masks")

        log_memory(f"After extracting all {len(masks)} masks")

        # Clean up: remove the pruning hooks and masks to restore original state
        log_memory("Before cleanup and moving params back to GPU")
        for i, (module, param_name) in enumerate(parameters_to_prune):
            prune.remove(module, param_name)
            # Move parameter back to original device
            param = getattr(module, param_name)
            original_dev = original_device[(module, param_name)]
            if original_dev.type == 'cuda':
                param.data = param.data.to(original_dev)

            # Log every 50 parameters
            if i % 50 == 0:
                log_memory(f"Cleaned up {i}/{len(parameters_to_prune)} params")

        log_memory(f"After cleanup and moving all params back")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log_memory("After final CUDA empty_cache")

        # Also include non-prunable parameters with zero masks
        log_memory("Before adding non-prunable param masks")
        for name, param in model.named_parameters():
            if name not in masks:
                masks[name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')

        log_memory(f"END: Returning {len(masks)} masks")
        return masks

    def get_name(self) -> str:
        """Return the name of this pruning strategy."""
        return "magnitude"
