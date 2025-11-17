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

        # Collect parameter info
        param_names = []
        param_list = []

        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:  # Only prune weight matrices
                param_names.append(name)
                param_list.append(param)

        if not param_names:
            return {}

        # Check if we should use GPU or CPU for threshold computation
        # Try GPU first if available and model is on GPU
        use_gpu = False
        if param_list[0].device.type == 'cuda':
            # Check if we have enough VRAM headroom (need ~3GB for operations)
            if torch.cuda.is_available():
                vram_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                vram_free_gb = vram_free / (1024**3)
                # Use GPU if we have at least 3GB free
                use_gpu = vram_free_gb >= 3.0

        if use_gpu:
            # Fast path: compute on GPU
            threshold = self._compute_threshold_gpu(param_list, sparsity)
            device = param_list[0].device
        else:
            # Memory-efficient path: compute on CPU without concatenating all weights
            threshold = self._compute_threshold_cpu_streaming(param_list, sparsity)
            device = 'cpu'

        # Create masks for each parameter
        masks = {}
        for name, param in model.named_parameters():
            if name in param_names:
                # True indicates this weight should be tentatively pruned
                if param.device.type == 'cuda' and device == 'cpu':
                    # Move to CPU to avoid CUDA OOM
                    mask = param.data.abs().cpu() <= threshold
                else:
                    mask = param.data.abs() <= threshold
                    if param.device.type == 'cuda':
                        mask = mask.cpu()  # Always return masks on CPU
                masks[name] = mask
            else:
                # Don't prune this parameter
                masks[name] = torch.zeros(param.data.shape, dtype=torch.bool, device='cpu')

        return masks

    def _compute_threshold_gpu(self, param_list, sparsity):
        """Compute threshold on GPU (fast but requires VRAM)."""
        all_weights = []
        for param in param_list:
            all_weights.append(param.data.abs().flatten())

        all_weights_flat = torch.cat(all_weights)
        num_weights_to_prune = int(sparsity * len(all_weights_flat))

        if num_weights_to_prune == 0:
            return -1.0  # No pruning

        threshold = torch.topk(
            all_weights_flat,
            num_weights_to_prune,
            largest=False
        ).values.max()

        return threshold.item()

    def _compute_threshold_cpu_streaming(self, param_list, sparsity):
        """Compute threshold on CPU using streaming to avoid huge memory usage."""
        # Count total weights
        total_weights = sum(p.numel() for p in param_list)
        num_weights_to_prune = int(sparsity * total_weights)

        if num_weights_to_prune == 0:
            return -1.0  # No pruning

        # Use a reservoir sampling approach with a fixed-size buffer
        # This avoids loading all weights into memory at once
        # Sample ~100M weights to estimate threshold (much less than 7B for Mistral)
        max_sample_size = min(100_000_000, total_weights)

        if total_weights <= max_sample_size:
            # Small enough to process all at once on CPU
            all_weights = []
            for param in param_list:
                all_weights.append(param.data.abs().cpu().flatten())
            all_weights_flat = torch.cat(all_weights)
            threshold = torch.topk(
                all_weights_flat,
                num_weights_to_prune,
                largest=False
            ).values.max()
            return threshold.item()

        # For very large models, use quantile estimation with sampling
        # Collect a representative sample
        import random
        sample_ratio = max_sample_size / total_weights
        samples = []

        for param in param_list:
            param_cpu = param.data.abs().cpu().flatten()
            # Sample from this parameter
            n_samples = int(len(param_cpu) * sample_ratio)
            if n_samples > 0:
                indices = torch.randperm(len(param_cpu))[:n_samples]
                samples.append(param_cpu[indices])

        # Estimate threshold from samples
        all_samples = torch.cat(samples)
        sample_threshold_idx = int(sparsity * len(all_samples))
        threshold = torch.topk(
            all_samples,
            min(sample_threshold_idx, len(all_samples)),
            largest=False
        ).values.max()

        return threshold.item()

    def get_name(self) -> str:
        """Return the name of this pruning strategy."""
        return "magnitude"
