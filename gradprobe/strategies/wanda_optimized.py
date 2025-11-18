"""
Optimized WANDA pruning strategy using statistical threshold estimation.

This is a drop-in replacement for wanda.py that uses mean/std + binary search
instead of topk, providing significant speedup and memory savings.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import psutil
import time
import gc

from .base import PruningStrategy


def log_memory(msg, log_file="/tmp/wanda_optimized_memory.log"):
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


class WANDAPruningOptimized(PruningStrategy):
    """
    Optimized WANDA pruning strategy.

    Uses streaming statistics (mean/std) + binary search instead of topk
    to find the pruning threshold. This avoids concatenating all importance scores
    into one large tensor, saving memory and providing speedup.
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
        sparsity: float,
        max_iterations: int = 20,
        tolerance: float = 0.0001
    ) -> Dict[str, torch.Tensor]:
        """
        Select weights to prune based on WANDA importance scores.

        Uses streaming statistics to estimate threshold without large memory allocation.

        Args:
            model: The neural network model to analyze
            sparsity: Target sparsity level (fraction of weights to prune, 0-1)
            max_iterations: Maximum binary search iterations (default: 20)
            tolerance: Target sparsity tolerance (default: 0.0001 = 0.01%)

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
            return self._magnitude_only_fallback(model, sparsity, max_iterations, tolerance)

        # Collect parameter info
        log_memory("Before collecting parameters")
        param_list = []

        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:  # Weight matrices only
                param_list.append((name, param))

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

        # Pass 1: Compute global statistics and build histogram for initial threshold
        log_memory("Computing global statistics")
        importance_cache = []

        for name, param in param_list:
            # Compute WANDA importance: |W| * ||X||
            if name in activation_norms:
                act_norm = activation_norms[name].cpu()
                if len(param.shape) == 2:
                    act_norm_expanded = act_norm.unsqueeze(0)
                    importance = param.data.abs().cpu() * act_norm_expanded
                else:
                    importance = param.data.abs().cpu()
            else:
                # No activation data, use magnitude only
                importance = param.data.abs().cpu()

            importance_cache.append((name, importance))

        # Find global min/max first
        min_val = min(imp.min().item() for _, imp in importance_cache)
        max_val = max(imp.max().item() for _, imp in importance_cache)

        # Compute total count
        total_count = sum(imp.numel() for _, imp in importance_cache)
        target_count = int(sparsity * total_count)

        log_memory(f"Value range: min={min_val:.6f}, max={max_val:.6f}, total_count={total_count}")

        # Check dtype and count zeros
        first_dtype = importance_cache[0][1].dtype if importance_cache else None
        zero_count = sum((imp == 0).sum().item() for _, imp in importance_cache)
        log_memory(f"Importance dtype: {first_dtype}, zeros: {zero_count}/{total_count} ({zero_count/total_count*100:.1f}%)")

        # Build histogram to estimate initial threshold
        # Use two-pass approach for bimodal distributions:
        # Pass 1: Coarse histogram to find which region target is in
        # Pass 2: If bin is heavily populated, zoom in with fine histogram
        num_bins_coarse = 1000
        num_bins_fine = 10000
        histogram = torch.zeros(num_bins_coarse)
        eps = (max_val - min_val) * 1e-6 if max_val > min_val else 1e-6
        hist_min = min_val - eps
        hist_max = max_val + eps

        log_memory(f"Pass 1: Coarse histogram with {num_bins_coarse} bins, range=[{hist_min:.6f}, {hist_max:.6f}]")
        for idx, (name, importance) in enumerate(importance_cache):
            importance_f32 = importance.float()
            hist = torch.histc(importance_f32, bins=num_bins_coarse, min=hist_min, max=hist_max)
            histogram += hist

        # Find which bin contains the target percentile
        cumsum = histogram.cumsum(0)
        coarse_bin_idx = (cumsum >= target_count).nonzero(as_tuple=True)[0]

        if len(coarse_bin_idx) == 0:
            raise RuntimeError(f"Coarse histogram failed: cumsum never reached target_count")

        coarse_bin_idx = coarse_bin_idx[0].item()
        bin_width_coarse = (hist_max - hist_min) / num_bins_coarse

        # Check if this bin is heavily populated (>10% of total values)
        bin_count = histogram[coarse_bin_idx].item()
        bin_density = bin_count / total_count

        log_memory(f"Pass 1: Target in bin {coarse_bin_idx}/{num_bins_coarse}, bin density={bin_density*100:.1f}%")

        # Track bounds for binary search
        search_min = min_val
        search_max = max_val

        if bin_density > 0.1:
            # Heavily populated bin - zoom in with fine histogram
            fine_min = hist_min + coarse_bin_idx * bin_width_coarse
            fine_max = hist_min + (coarse_bin_idx + 1) * bin_width_coarse
            fine_eps = (fine_max - fine_min) * 1e-6
            fine_min -= fine_eps
            fine_max += fine_eps

            log_memory(f"Pass 2: Zooming into [{fine_min:.6f}, {fine_max:.6f}] with {num_bins_fine} bins")

            histogram_fine = torch.zeros(num_bins_fine)
            for idx, (name, importance) in enumerate(importance_cache):
                importance_f32 = importance.float()
                hist = torch.histc(importance_f32, bins=num_bins_fine, min=fine_min, max=fine_max)
                histogram_fine += hist

            # Find threshold in fine histogram
            cumsum_fine = histogram_fine.cumsum(0)
            fine_bin_idx = (cumsum_fine >= target_count).nonzero(as_tuple=True)[0]

            if len(fine_bin_idx) == 0:
                raise RuntimeError(f"Fine histogram failed: cumsum never reached target_count")

            fine_bin_idx = fine_bin_idx[0].item()
            bin_width_fine = (fine_max - fine_min) / num_bins_fine
            initial_threshold = fine_min + (fine_bin_idx + 0.5) * bin_width_fine

            # Set binary search bounds based on element count, not bin count
            # Allow ±1% slack in sparsity (e.g., 9-11% for 10% target)
            # This adapts to density - tight in dense regions, wider in sparse regions
            slack_percent = 0.01
            lower_target = target_count * (1 - slack_percent)
            upper_target = target_count * (1 + slack_percent)

            # Find bins that bracket these element counts
            lower_bin_idx = (cumsum_fine >= lower_target).nonzero(as_tuple=True)[0]
            upper_bin_idx = (cumsum_fine >= upper_target).nonzero(as_tuple=True)[0]

            lower_bin = lower_bin_idx[0].item() if len(lower_bin_idx) > 0 else 0
            upper_bin = upper_bin_idx[0].item() if len(upper_bin_idx) > 0 else num_bins_fine - 1

            search_min = max(min_val, fine_min + lower_bin * bin_width_fine)
            search_max = min(max_val, fine_min + (upper_bin + 1) * bin_width_fine)

            log_memory(f"Pass 2: threshold from bin {fine_bin_idx}/{num_bins_fine} = {initial_threshold:.6f}, search range: bins [{lower_bin}, {upper_bin}] (±{slack_percent*100:.0f}% elements)")
        else:
            # Low density - coarse estimate is good enough
            initial_threshold = hist_min + (coarse_bin_idx + 0.5) * bin_width_coarse
            log_memory(f"Using coarse estimate: threshold={initial_threshold:.6f}")

        # Clamp to actual data range
        initial_threshold = max(min_val, min(max_val, initial_threshold))

        # Compute mean/std for diagnostics using torch operations (avoid overflow)
        # Accumulate as double precision scalars
        mean_accum = 0.0
        for _, imp in importance_cache:
            mean_accum += imp.double().sum().item()
        mean = mean_accum / total_count

        variance_accum = 0.0
        for _, imp in importance_cache:
            variance_accum += ((imp.double() - mean) ** 2).sum().item()
        std = (variance_accum / total_count) ** 0.5

        log_memory(f"Statistics: mean={mean:.6f}, std={std:.6f}, target_count={target_count}")

        # Binary search for optimal threshold
        log_memory(f"Binary search: initial_threshold={initial_threshold:.6f}, bounds=[{search_min:.6f}, {search_max:.6f}]")
        low = search_min
        high = search_max
        threshold = initial_threshold  # Already clamped above

        best_threshold = threshold
        best_error = float('inf')

        for iteration in range(max_iterations):
            # Count weights below threshold (streaming through layers)
            count_below = 0
            for _, importance in importance_cache:
                count_below += (importance <= threshold).sum().item()

            actual_sparsity = count_below / total_count
            error = abs(actual_sparsity - sparsity)

            # Debug: print first few iterations
            if iteration < 5:
                log_memory(f"Iteration {iteration}: threshold={threshold:.6f}, actual_sparsity={actual_sparsity:.6f}, error={error:.6f}, bounds=[{low:.6f}, {high:.6f}]")

            if error < best_error:
                best_error = error
                best_threshold = threshold

            if error < tolerance:
                log_memory(f"Converged in {iteration+1} iterations: threshold={best_threshold:.6f}, error={best_error:.6f}")
                break

            # Binary search update
            if actual_sparsity < sparsity:
                low = threshold
            else:
                high = threshold

            threshold = (low + high) / 2

            if abs(high - low) < 1e-10:
                log_memory(f"Search range collapsed in {iteration+1} iterations")
                break
        else:
            log_memory(f"Max iterations reached: threshold={best_threshold:.6f}, error={best_error:.6f}")

        # Create masks with best threshold
        log_memory("Creating masks")
        masks = {}
        for name, importance in importance_cache:
            masks[name] = importance <= best_threshold

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

    def _magnitude_only_fallback(
        self,
        model: nn.Module,
        sparsity: float,
        max_iterations: int,
        tolerance: float
    ) -> Dict[str, torch.Tensor]:
        """
        Fallback to magnitude-only pruning if activation collection fails.
        Uses the same optimized streaming approach.
        """
        log_memory("Fallback to magnitude-only (optimized)")

        # Collect parameters
        param_list = []
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                param_list.append((name, param))

        if not param_list:
            return {}

        # Compute statistics using torch operations
        importance_cache = []

        for name, param in param_list:
            importance = param.data.abs().cpu()
            importance_cache.append((name, importance))

        # Concatenate all importance tensors and compute statistics using torch
        all_importance = torch.cat([imp.flatten() for _, imp in importance_cache])

        mean = all_importance.mean().item()
        std = all_importance.std().item()
        total_count = all_importance.numel()

        # Find min/max for bounds
        min_val = all_importance.min().item()
        max_val = all_importance.max().item()

        # Normal approximation for initial threshold
        if sparsity < 0.5:
            p = sparsity
            sign = -1
        else:
            p = 1 - sparsity
            sign = 1

        if p > 0.0 and p < 1.0:
            t = (-2 * torch.log(torch.tensor(p))) ** 0.5
            c0, c1, c2 = 2.515517, 0.802853, 0.010328
            d1, d2, d3 = 1.432788, 0.189269, 0.001308
            z_approx = sign * (t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t))
        else:
            z_approx = 0.0

        initial_threshold = mean + z_approx * std
        initial_threshold = max(min_val, min(max_val, initial_threshold))

        # Binary search
        low = min_val
        high = max_val
        threshold = initial_threshold

        best_threshold = threshold
        best_error = float('inf')

        for iteration in range(max_iterations):
            count_below = sum((importance <= threshold).sum().item() for _, importance in importance_cache)
            actual_sparsity = count_below / total_count
            error = abs(actual_sparsity - sparsity)

            if error < best_error:
                best_error = error
                best_threshold = threshold

            if error < tolerance:
                break

            if actual_sparsity < sparsity:
                low = threshold
            else:
                high = threshold

            threshold = (low + high) / 2

            if abs(high - low) < 1e-10:
                break

        # Create masks
        masks = {}
        for name, importance in importance_cache:
            masks[name] = importance <= best_threshold

        # Add non-prunable parameters
        for name, param in model.named_parameters():
            if name not in masks:
                masks[name] = torch.zeros(param.shape, dtype=torch.bool, device='cpu')

        return masks

    def get_name(self) -> str:
        """Return the name of this pruning strategy."""
        return "WANDA_optimized"
