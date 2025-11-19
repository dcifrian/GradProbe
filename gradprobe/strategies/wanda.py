"""
WANDA (Pruning by Weights And activations) pruning strategy.

This strategy computes importance scores as |weight| * ||activation||,
combining weight magnitude with activation norms to identify unimportant weights.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import psutil
import time
import gc

from .base import PruningStrategy
from ..logger import get_logger


def log_memory(msg, log_file="/tmp/wanda_memory.log"):
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
    get_logger().memory(log_line.strip())


class WANDAPruning(PruningStrategy):
    """
    WANDA pruning strategy.

    Selects weights based on the product of weight magnitude and activation norm.
    Weights with low |weight| * ||activation|| scores are considered unimportant.
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader, num_batches: int = 10):
        """
        Initialize WANDA pruning strategy.

        Args:
            dataloader: DataLoader to sample activations from
            num_batches: Number of batches to use for activation computation
        """
        self.dataloader = dataloader
        self.num_batches = num_batches
        self._cached_activation_norms = None  # Cache activations to avoid recomputation

    def select_weights_to_prune(
        self,
        model: nn.Module,
        sparsity: float
    ) -> Dict[str, torch.Tensor]:
        """
        Select weights to prune based on WANDA importance scores.

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

        # Collect activations for each layer (cache to avoid recomputation in iterative pruning)
        if self._cached_activation_norms is None:
            log_memory("Before collecting activation norms (first time)")
            activation_norms = self._collect_activation_norms(model)
            self._cached_activation_norms = activation_norms  # Cache for future calls
            log_memory(f"After collecting activation norms for {len(activation_norms)} layers (cached)")
        else:
            log_memory("Using cached activation norms (skipping re-collection)")
            activation_norms = self._cached_activation_norms

        if not activation_norms:
            # Fall back to magnitude-only if no activations collected
            return self._magnitude_only_fallback(model, sparsity)

        # Collect parameter info
        log_memory("Before collecting parameters")
        param_list = []
        param_names = []

        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:  # Weight matrices only
                param_list.append((name, param))
                param_names.append(name)

        log_memory(f"After collecting {len(param_list)} parameters")

        if not param_list:
            return {}

        if sparsity == 0:
            # Return empty masks (no pruning)
            masks = {}
            for name, param in param_list:
                masks[name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')
            for name, param in model.named_parameters():
                if name not in masks:
                    masks[name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')
            return masks

        # Compute threshold using memory-efficient sampling
        log_memory("Computing importance threshold")
        threshold = self._compute_threshold_efficient(model, param_list, activation_norms, sparsity)
        log_memory(f"Threshold computed: {threshold:.6f}")

        # Create masks for each parameter
        # Create on CPU if model is on CUDA to save GPU memory
        log_memory("Creating masks")
        masks = {}
        for i, (name, param) in enumerate(param_list):
            if name in activation_norms:
                # Compute importance: |W| * ||X||
                act_norm = activation_norms[name].to(param.device)
                if len(param.shape) == 2:
                    act_norm_expanded = act_norm.unsqueeze(0)
                    importance = param.data.abs() * act_norm_expanded
                else:
                    importance = param.data.abs()

                # True indicates this weight should be tentatively pruned
                mask = importance <= threshold
                masks[name] = mask.cpu() if param.device.type == 'cuda' else mask
            else:
                # No activation data, use magnitude only
                mask = param.data.abs() <= threshold
                masks[name] = mask.cpu() if param.device.type == 'cuda' else mask

            if i % 50 == 0:
                log_memory(f"Created {i}/{len(param_list)} masks")

        log_memory(f"After creating all {len(masks)} masks")

        # Add non-prunable parameters with zero masks
        for name, param in model.named_parameters():
            if name not in masks:
                masks[name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')

        log_memory(f"END: Returning {len(masks)} masks")
        return masks

    def _collect_activation_norms(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Collect activation norms for each layer by running forward passes.

        Returns:
            Dictionary mapping parameter names to activation norms
        """
        activations = {}
        hooks = []

        def create_hook(name, module):
            def hook(module, input, output):
                # input is a tuple, get the first element
                if isinstance(input, tuple):
                    inp = input[0]
                else:
                    inp = input

                # Handle different input shapes
                # For Linear: input can be [batch, ...any..., in_features]
                # We want norm per in_features dimension

                # Get the expected input dimension from the module
                if isinstance(module, nn.Linear):
                    in_features = module.in_features
                    # Reshape to [batch * other_dims, in_features]
                    inp_reshaped = inp.reshape(-1, in_features)
                    # Compute norm across batch dimension, get [in_features]
                    norm = torch.norm(inp_reshaped, p=2, dim=0)
                elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    # For conv layers, take norm across spatial dimensions
                    # Input: [batch, channels, ...spatial...]
                    batch_size = inp.shape[0]
                    channels = inp.shape[1]
                    inp_reshaped = inp.reshape(batch_size, channels, -1)
                    norm = torch.norm(inp_reshaped, p=2, dim=(0, 2))  # [channels]
                else:
                    # Fallback: flatten and take norm
                    inp_flat = inp.reshape(inp.shape[0], -1)
                    norm = torch.norm(inp_flat, p=2, dim=0)

                if name not in activations:
                    activations[name] = norm
                else:
                    # Accumulate (take max across batches)
                    activations[name] = torch.maximum(activations[name], norm)
            return hook

        # Register hooks for all linear/conv layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Map module name to parameter name
                # For nn.Linear, the weight is name.weight
                param_name = f"{name}.weight"
                hook = module.register_forward_hook(create_hook(param_name, module))
                hooks.append(hook)

        # Run forward passes
        model.eval()

        # Detect model device from first parameter
        device = next(model.parameters()).device

        batch_count = 0
        with torch.no_grad():
            for batch in self.dataloader:
                if batch_count >= self.num_batches:
                    break

                # Handle different batch formats and move to model device
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)

                # Forward pass
                try:
                    model(inputs)
                except Exception:
                    # If forward fails, skip this batch
                    pass

                batch_count += 1

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activations

    def _compute_threshold_efficient(
        self,
        model: nn.Module,
        param_list: List[Tuple[str, nn.Parameter]],
        activation_norms: Dict[str, torch.Tensor],
        sparsity: float
    ) -> float:
        """
        Compute importance threshold using memory-efficient sampling.

        For large models, samples a subset of weights to estimate the threshold
        instead of concatenating all weights.
        """
        # Count total weights
        total_weights = sum(p.numel() for _, p in param_list)
        num_weights_to_prune = int(sparsity * total_weights)

        if num_weights_to_prune == 0:
            return -1.0  # No pruning

        # Sample up to 100M importance scores to estimate threshold
        max_sample_size = min(100_000_000, total_weights)

        if total_weights <= max_sample_size:
            # Small enough to process all at once
            log_memory("Computing threshold from all weights")
            importance_scores = []
            for name, param in param_list:
                if name in activation_norms:
                    act_norm = activation_norms[name].to(param.device)
                    if len(param.shape) == 2:
                        act_norm_expanded = act_norm.unsqueeze(0)
                        importance = param.data.abs() * act_norm_expanded
                    else:
                        importance = param.data.abs()
                    importance_scores.append(importance.cpu().flatten())
                else:
                    importance_scores.append(param.data.abs().cpu().flatten())

            all_scores = torch.cat(importance_scores)
            threshold = torch.topk(
                all_scores,
                min(num_weights_to_prune, len(all_scores)),
                largest=False
            ).values.max()
            return threshold.item()

        # For very large models, use sampling
        log_memory("Computing threshold from sampled weights")
        sample_ratio = max_sample_size / total_weights
        samples = []

        for name, param in param_list:
            if name in activation_norms:
                act_norm = activation_norms[name].to(param.device)
                if len(param.shape) == 2:
                    act_norm_expanded = act_norm.unsqueeze(0)
                    importance = param.data.abs() * act_norm_expanded
                else:
                    importance = param.data.abs()
                importance_flat = importance.cpu().flatten()
            else:
                importance_flat = param.data.abs().cpu().flatten()

            # Sample from this parameter
            n_samples = int(len(importance_flat) * sample_ratio)
            if n_samples > 0:
                indices = torch.randperm(len(importance_flat))[:n_samples]
                samples.append(importance_flat[indices])

            # Explicitly delete importance tensors to free memory immediately
            del importance_flat
            if name in activation_norms and len(param.shape) == 2:
                del importance, act_norm_expanded
            elif name in activation_norms:
                del importance

        # Force garbage collection to free memory from deleted tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory("After sampling and garbage collection")

        # Estimate threshold from samples
        all_samples = torch.cat(samples)
        sample_threshold_idx = int(sparsity * len(all_samples))
        threshold = torch.topk(
            all_samples,
            min(sample_threshold_idx, len(all_samples)),
            largest=False
        ).values.max()

        return threshold.item()

    def _magnitude_only_fallback(self, model: nn.Module, sparsity: float) -> Dict[str, torch.Tensor]:
        """
        Fallback to magnitude-only pruning if activation collection fails.
        """
        all_weights = []
        param_names = []

        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                all_weights.append(param.data.abs().flatten())
                param_names.append(name)

        if not all_weights:
            return {}

        all_weights_flat = torch.cat(all_weights)
        num_weights_to_prune = int(sparsity * len(all_weights_flat))

        if num_weights_to_prune == 0:
            masks = {}
            for name in param_names:
                param = model.state_dict()[name]
                if hasattr(param, 'device') and param.device.type == 'cuda':
                    masks[name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')
                else:
                    masks[name] = torch.zeros_like(param, dtype=torch.bool)
            return masks

        threshold = torch.topk(
            all_weights_flat,
            num_weights_to_prune,
            largest=False
        ).values.max()

        masks = {}
        for name, param in model.named_parameters():
            if name in param_names:
                mask = param.data.abs() <= threshold
                if param.device.type == 'cuda':
                    masks[name] = mask.cpu()
                else:
                    masks[name] = mask
            else:
                if param.device.type == 'cuda':
                    masks[name] = torch.zeros(param.data.shape, dtype=torch.bool, device='cpu')
                else:
                    masks[name] = torch.zeros_like(param.data, dtype=torch.bool)

        return masks

    def get_name(self) -> str:
        """Return the name of this pruning strategy."""
        return "WANDA"
