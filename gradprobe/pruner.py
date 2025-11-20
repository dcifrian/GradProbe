"""
GradProbe: Gradient-based Neural Network Pruner.

This module implements the main GradProbe algorithm, which uses gradient
information to make intelligent pruning decisions.
"""

from typing import Dict, Optional, Callable, Tuple, Union, List
import torch
import torch.nn as nn
from copy import deepcopy
import gc
import math

from .strategies.base import PruningStrategy
from .logger import get_logger


def _is_nonzero_threshold(gradient_threshold: Union[float, List[float], Dict[str, float], Tuple[str, float]]) -> bool:
    """Check if gradient_threshold is effectively non-zero (for tuning eligibility)."""
    if isinstance(gradient_threshold, tuple) and len(gradient_threshold) == 2 and gradient_threshold[0] == "adaptive":
        return gradient_threshold[1] > 0
    elif isinstance(gradient_threshold, dict):
        return any(v > 0 for v in gradient_threshold.values())
    elif isinstance(gradient_threshold, (list, tuple)):
        return any(v > 0 for v in gradient_threshold)
    else:
        return float(gradient_threshold) > 0


def compute_adaptive_gradient_thresholds(
    model: nn.Module,
    base_threshold: float,
    reference_layer_size: int = 38597376  # Largest layer in TinyStories-33M (wte.weight)
) -> Dict[str, float]:
    """
    Compute adaptive per-layer gradient thresholds.

    Formula: threshold_for_layer = base_threshold * sqrt(layer_size) / sqrt(reference_layer_size)

    Larger layers get proportionally larger thresholds (more lenient),
    smaller layers get proportionally smaller thresholds (more strict).

    Args:
        model: The neural network model
        base_threshold: Base threshold value (1.0 is good for TinyStories)
        reference_layer_size: Size of reference layer (default: TinyStories wte.weight = 38,597,376)

    Returns:
        Dictionary mapping parameter names to adaptive thresholds
    """
    thresholds = {}
    sqrt_ref = math.sqrt(reference_layer_size)

    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) >= 2:  # Weight matrices only
            layer_size = param.numel()
            # Adaptive scaling: sqrt(layer_size) / sqrt(reference_size)
            threshold = base_threshold * (math.sqrt(layer_size) / sqrt_ref)
            thresholds[name] = threshold
            get_logger().debug(f"Adaptive threshold for {name} (size={layer_size}): {threshold:.4f}")

    return thresholds


class GradProbe:
    """
    Gradient-comparison based neural network pruner.

    The pruning algorithm works as follows:
    1. Forward pass with original network and record gradients
    2. Use a pruning strategy to determine initial weights to tentatively prune
    3. Set those weights to 1/10 of their value (instead of 0)
    4. Do another forward pass and compare gradients
    5. Prune (set to 0) weights where gradients decreased or stayed the same
    6. Restore weights where gradients increased

    This approach is more intelligent than naive pruning as it considers how
    the gradient changes when a weight is reduced, indicating its importance.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: PruningStrategy,
        device: str = "auto",
        low_memory_mode: bool = False,
        use_fp16: bool = False,
        use_gradient_checkpointing: bool = False
    ):
        """
        Initialize the GradProbe pruner.

        Args:
            model: Neural network model to prune
            strategy: Pruning strategy to use for selecting initial candidates
            device: Device to use for computation.
                   "auto" (default): auto-detect CUDA if available, otherwise CPU
                   "cuda": use CUDA
                   "cpu": use CPU
            low_memory_mode: If True, minimize memory usage by:
                           - Not caching gradients during layerwise pruning
                           - Moving tensors to CPU when not needed
                           - Clearing CUDA cache aggressively
                           - Using layer-by-layer processing
                           Useful for large models like Mistral-7B
            use_fp16: If True, use float16 for gradient storage and saved states
                     This cuts memory usage in half for these components
                     Model weights remain in their original precision
            use_gradient_checkpointing: If True, enable gradient checkpointing
                                       to reduce activation memory during backprop
                                       Trade-off: increases compute time ~30-50%
        """
        # Auto-detect device if requested
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            get_logger().info(f"Auto-detected device: {device}")

        self.model = model.to(device)
        self.strategy = strategy
        self.device = device
        self.low_memory_mode = low_memory_mode
        self.use_fp16 = use_fp16
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.original_state = None
        self.pruning_mask = None

        # Set gradient dtype based on use_fp16
        self.grad_dtype = torch.float16 if use_fp16 else torch.float32

        if low_memory_mode:
            get_logger().info(f"Low memory mode enabled - will use layer-by-layer processing")
        if use_fp16:
            get_logger().info(f"FP16 mode enabled - gradients and saved states will use float16")
        if use_gradient_checkpointing:
            get_logger().info(f"Gradient checkpointing enabled - will trade compute for memory")
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for supported models."""
        # Try to enable gradient checkpointing if model supports it
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            try:
                self.model.gradient_checkpointing_enable()
                get_logger().debug("  ✓ Gradient checkpointing enabled on model")
            except Exception as e:
                get_logger().warning(f"  ⚠ Could not enable gradient checkpointing: {e}")
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
            # For transformer models, need to disable cache for gradient checkpointing
            try:
                self.model.config.use_cache = False
                if hasattr(self.model, 'gradient_checkpointing_enable'):
                    self.model.gradient_checkpointing_enable()
                    get_logger().debug("  ✓ Gradient checkpointing enabled on model")
            except Exception as e:
                get_logger().warning(f"  ⚠ Could not enable gradient checkpointing: {e}")
        else:
            get_logger().warning("  ⚠ Model does not support gradient checkpointing")

    def prune(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        sparsity: float,
        num_batches: int = 1,
        reduction_factor: float = 0.1,
        gradient_threshold: float = 0.0,
        verbose: bool = True,
        compare_baseline: bool = False,
        eval_fn: Callable = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prune the model using gradient-comparison.

        Args:
            dataloader: DataLoader providing input data for gradient computation
            loss_fn: Loss function to use for computing gradients
            sparsity: Target sparsity level (fraction of weights to prune, 0-1)
            num_batches: Number of batches to use for gradient computation
            reduction_factor: Factor to reduce tentative weights by (default: 0.1)
            gradient_threshold: Relative gradient increase threshold (default: 0.0)
                               Weights are pruned if: new_grad <= old_grad * (1 + threshold)
                               e.g., 0.0 = only prune if gradient decreased or stayed same
                                     0.1 = allow up to 10% gradient increase before restoring
            verbose: Whether to print progress information
            compare_baseline: If True, also evaluate what accuracy would be with magnitude-only pruning
            eval_fn: Evaluation function (required if compare_baseline=True), returns accuracy as percentage

        Returns:
            Dictionary mapping parameter names to pruning masks (True = pruned)
        """
        if verbose:
            get_logger().info(f"Starting GradProbe with {self.strategy.get_name()} strategy")
            get_logger().info(f"Target sparsity: {sparsity:.2%}")
            get_logger().info(f"Reduction factor: {reduction_factor}")
            get_logger().info(f"Gradient threshold: {gradient_threshold:.2%}")

        # Use layer-by-layer streaming in low_memory_mode to dramatically reduce memory usage
        if self.low_memory_mode:
            if verbose:
                get_logger().info(f"\nUsing layer-by-layer gradient streaming (low memory mode)")
            return self._prune_layer_by_layer_streaming(
                dataloader=dataloader,
                loss_fn=loss_fn,
                sparsity=sparsity,
                num_batches=num_batches,
                reduction_factor=reduction_factor,
                gradient_threshold=gradient_threshold,
                verbose=verbose,
                compare_baseline=compare_baseline,
                eval_fn=eval_fn
            )

        # Save original model state
        # In low_memory_mode, save to CPU to avoid doubling GPU memory usage
        # In fp16 mode, save in half precision to reduce memory usage
        if self.low_memory_mode:
            if self.use_fp16:
                self.original_state = {
                    name: param.data.cpu().half() for name, param in self.model.named_parameters()
                }
            else:
                self.original_state = {
                    name: param.data.cpu().clone() for name, param in self.model.named_parameters()
                }
        else:
            if self.use_fp16:
                self.original_state = {
                    name: param.data.half() for name, param in self.model.named_parameters()
                }
            else:
                self.original_state = {
                    name: param.data.clone() for name, param in self.model.named_parameters()
                }

        # Step 1: Get initial gradients with original model
        if verbose:
            get_logger().info("\nStep 1: Computing gradients with original model...")
        original_gradients = self._compute_gradients(dataloader, loss_fn, num_batches)

        # Step 2: Use strategy to select weights to tentatively prune
        if verbose:
            get_logger().info(f"\nStep 2: Selecting weights using {self.strategy.get_name()} strategy...")
        tentative_masks = self.strategy.select_weights_to_prune(self.model, sparsity)

        # Count tentative pruning candidates
        total_tentative = sum(mask.sum().item() for mask in tentative_masks.values())
        if verbose:
            get_logger().info(f"Tentative pruning candidates: {total_tentative}")

        # Step 3: Set tentative weights to reduction_factor * original value
        if verbose:
            get_logger().info(f"\nStep 3: Reducing tentative weights to {reduction_factor}x...")
        self._apply_tentative_reduction(tentative_masks, reduction_factor)

        # Step 4: Compute gradients with reduced weights
        if verbose:
            get_logger().info("\nStep 4: Computing gradients with reduced weights...")
        modified_gradients = self._compute_gradients(dataloader, loss_fn, num_batches)

        # Cache these for threshold tuning
        self._cached_original_gradients = original_gradients
        self._cached_modified_gradients = modified_gradients
        self._cached_tentative_masks = tentative_masks

        # Step 5: Compare gradients and make final pruning decisions
        if verbose:
            get_logger().info("\nStep 5: Comparing gradients and making pruning decisions...")
        final_masks = self._compare_gradients_and_decide(
            original_gradients,
            modified_gradients,
            tentative_masks,
            gradient_threshold=gradient_threshold,
            verbose=verbose
        )

        # Step 6: Apply final pruning
        if verbose:
            get_logger().info("\nStep 6: Applying final pruning...")
        self._apply_final_pruning(final_masks)

        self.pruning_mask = final_masks

        # Print statistics
        if verbose:
            self._print_pruning_statistics(final_masks)

        # Compare with baseline (magnitude-only pruning) if requested
        if compare_baseline:
            if eval_fn is None:
                raise ValueError("eval_fn must be provided when compare_baseline=True")

            if verbose:
                get_logger().info("\n" + "="*60)
                get_logger().info("BASELINE COMPARISON")
                get_logger().info("="*60)

            # Save gradient-pruned state
            if self.low_memory_mode:
                gradient_pruned_state = {
                    name: param.data.cpu().clone() for name, param in self.model.named_parameters()
                }
            else:
                gradient_pruned_state = {
                    name: param.data.clone() for name, param in self.model.named_parameters()
                }

            # Evaluate gradient-based pruning
            gradient_accuracy = eval_fn(self.model)
            gradient_sparsity = self.get_sparsity()

            # Apply magnitude-only (all tentative candidates)
            for name, param in self.model.named_parameters():
                if name in self.original_state:
                    param.data.copy_(self.original_state[name])
                if name in tentative_masks:
                    mask = self._move_mask_to_param_device(tentative_masks[name], param)
                    param.data[mask] = 0

            # Evaluate magnitude-only pruning
            magnitude_accuracy = eval_fn(self.model)
            magnitude_sparsity = self.get_sparsity()

            # Restore gradient-pruned state
            for name, param in self.model.named_parameters():
                if name in gradient_pruned_state:
                    param.data.copy_(gradient_pruned_state[name])

            if verbose:
                get_logger().info(f"Magnitude-only (all candidates):")
                get_logger().info(f"  Sparsity: {magnitude_sparsity:.2%}")
                get_logger().info(f"  Accuracy: {magnitude_accuracy:.2f}%")
                get_logger().info(f"\nGradient-filtered:")
                get_logger().info(f"  Sparsity: {gradient_sparsity:.2%}")
                get_logger().info(f"  Accuracy: {gradient_accuracy:.2f}%")
                get_logger().info(f"\nAccuracy gain from gradient filtering: {gradient_accuracy - magnitude_accuracy:+.2f}%")
                get_logger().info("="*60)

        return final_masks

    def _compute_gradients(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        num_batches: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute and return gradients for all parameters using absolute maximum across batches.
        Disables dropout to match inference conditions but keeps model in train mode.

        Args:
            dataloader: DataLoader providing input data
            loss_fn: Loss function
            num_batches: Number of batches to process

        Returns:
            Dictionary mapping parameter names to gradient tensors (abs max across batches)
        """
        # Disable dropout for deterministic gradients (inference behavior)
        # but keep model in train mode for other layers
        dropout_states = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                dropout_states[name] = module.training
                module.eval()

        # Store gradients on CPU to save GPU memory and keep all tensor ops on CPU
        # Use self.grad_dtype (fp16 if use_fp16=True) to reduce memory usage
        gradients = {name: torch.zeros(param.shape, dtype=self.grad_dtype, device='cpu')
                     for name, param in self.model.named_parameters()
                     if param.requires_grad}

        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_count >= num_batches:
                break

            # Zero gradients
            self.model.zero_grad()

            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = inputs  # For autoencoder-style tasks

            # Forward pass
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass
            loss.backward()

            # Take element-wise maximum of absolute gradients
            # Move to CPU immediately to save GPU memory and keep all tensor ops on CPU
            # Convert to self.grad_dtype (fp16 if enabled) to save memory
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_abs = param.grad.data.abs().cpu().to(self.grad_dtype)
                    if batch_count == 0:
                        gradients[name] = grad_abs
                    else:
                        gradients[name] = torch.maximum(gradients[name], grad_abs)

            batch_count += 1

        # Restore dropout states
        for name, module in self.model.named_modules():
            if name in dropout_states:
                if dropout_states[name]:
                    module.train()

        return gradients

    def _move_mask_to_param_device(self, mask: torch.Tensor, param: torch.nn.Parameter) -> torch.Tensor:
        """
        Move mask to the same device as parameter if needed.

        This is important for large models where masks may be on CPU to save GPU memory,
        but need to be temporarily moved to GPU to apply them.

        Args:
            mask: Boolean mask tensor
            param: Parameter to apply mask to

        Returns:
            Mask on the same device as param
        """
        if mask.device != param.device:
            return mask.to(param.device)
        return mask

    def _apply_tentative_reduction(
        self,
        masks: Dict[str, torch.Tensor],
        reduction_factor: float
    ):
        """
        Reduce tentatively pruned weights by the reduction factor.

        Args:
            masks: Dictionary of pruning masks
            reduction_factor: Factor to multiply tentative weights by
        """
        for name, param in self.model.named_parameters():
            if name in masks:
                mask = masks[name]
                # Move mask to same device as param if needed (e.g., CPU mask -> CUDA param)
                if mask.device != param.device:
                    mask = mask.to(param.device)
                param.data[mask] *= reduction_factor

    def _compare_gradients_and_decide(
        self,
        original_grads: Dict[str, torch.Tensor],
        modified_grads: Dict[str, torch.Tensor],
        tentative_masks: Dict[str, torch.Tensor],
        gradient_threshold: float = 0.0,
        verbose: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compare gradients and decide which weights to actually prune.

        For each tentatively pruned weight:
        - If gradient increased by more than threshold → keep it (restore original value)
        - Otherwise → prune it (set to 0)

        Args:
            original_grads: Gradients from original model
            modified_grads: Gradients from modified model
            tentative_masks: Tentative pruning masks
            gradient_threshold: Relative gradient increase threshold
                               Prune if: new_grad <= old_grad * (1 + threshold)
            verbose: Whether to print statistics

        Returns:
            Final pruning masks
        """
        final_masks = {}
        stats = {
            'pruned': 0,
            'kept': 0,
            'total_tentative': 0
        }

        for name in tentative_masks:
            if name not in original_grads or name not in modified_grads:
                # Create empty mask on CPU
                final_masks[name] = torch.zeros(tentative_masks[name].shape, dtype=torch.bool, device='cpu')
                continue

            tentative_mask = tentative_masks[name]
            orig_grad = original_grads[name]
            mod_grad = modified_grads[name]

            # For tentatively pruned weights, compare gradients
            # Prune if gradient didn't increase beyond threshold
            # mod_grad <= orig_grad * (1 + gradient_threshold)
            # All tensors are on CPU, so this operation stays on CPU
            gradient_below_threshold = mod_grad <= orig_grad * (1.0 + gradient_threshold)

            # Final decision: prune if gradient is below threshold
            # (at the tentative locations)
            # Both masks are on CPU, result stays on CPU
            final_mask = tentative_mask & gradient_below_threshold

            final_masks[name] = final_mask

            # Update statistics
            stats['total_tentative'] += tentative_mask.sum().item()
            stats['pruned'] += final_mask.sum().item()
            stats['kept'] += (tentative_mask & ~final_mask).sum().item()

        if verbose:
            get_logger().info(f"  Tentative candidates: {stats['total_tentative']}")
            get_logger().info(f"  Final pruned: {stats['pruned']}")
            get_logger().info(f"  Restored (gradient increased): {stats['kept']}")
            if stats['total_tentative'] > 0:
                prune_rate = stats['pruned'] / stats['total_tentative']
                get_logger().info(f"  Pruning rate from tentative: {prune_rate:.2%}")

        return final_masks

    def _apply_final_pruning(self, final_masks: Dict[str, torch.Tensor]):
        """
        Apply final pruning decisions to the model.

        Args:
            final_masks: Final pruning masks
        """
        # First, restore all weights to original values
        # Handle dtype conversion if original_state was saved in fp16
        for name, param in self.model.named_parameters():
            if name in self.original_state:
                saved_state = self.original_state[name]
                # Convert to param dtype if needed (e.g., fp16 -> fp32)
                if saved_state.dtype != param.dtype:
                    param.data.copy_(saved_state.to(param.dtype))
                else:
                    param.data.copy_(saved_state)

        # Then apply final pruning (set to 0)
        for name, param in self.model.named_parameters():
            if name in final_masks:
                mask = final_masks[name]
                # Move mask to same device as param if needed (e.g., CPU mask -> CUDA param)
                if mask.device != param.device:
                    mask = mask.to(param.device)
                param.data[mask] = 0

    def _print_pruning_statistics(self, masks: Dict[str, torch.Tensor]):
        """Print detailed pruning statistics."""
        get_logger().info("\n" + "="*60)
        get_logger().info("PRUNING STATISTICS")
        get_logger().info("="*60)

        total_params = 0
        total_pruned = 0

        for name, param in self.model.named_parameters():
            param_count = param.numel()
            if name in masks:
                pruned_count = masks[name].sum().item()
            else:
                pruned_count = 0

            total_params += param_count
            total_pruned += pruned_count

            if param.requires_grad:
                sparsity = pruned_count / param_count if param_count > 0 else 0
                get_logger().info(f"{name:40s}: {pruned_count:8d} / {param_count:8d} ({sparsity:6.2%})")

        overall_sparsity = total_pruned / total_params if total_params > 0 else 0
        get_logger().info("="*60)
        get_logger().info(f"{'TOTAL':40s}: {total_pruned:8d} / {total_params:8d} ({overall_sparsity:6.2%})")
        get_logger().info("="*60)

    def _prune_layer_by_layer_streaming(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        sparsity: float,
        num_batches: int = 1,
        reduction_factor: float = 0.1,
        gradient_threshold: float = 0.0,
        verbose: bool = True,
        compare_baseline: bool = False,
        eval_fn: Callable = None
    ) -> Dict[str, torch.Tensor]:
        """
        Memory-efficient layer-by-layer gradient streaming.

        Instead of storing gradients for ALL parameters simultaneously,
        process one layer at a time:
        1. Compute original gradient for layer N only
        2. Apply tentative reduction to layer N
        3. Compute modified gradient for layer N only
        4. Compare and decide for layer N
        5. Clear gradients, move to layer N+1

        This reduces memory from 4x model size to ~2x model size.
        """
        # Save original model state (still need this, but in fp16 if enabled)
        if self.use_fp16:
            self.original_state = {
                name: param.data.cpu().half() for name, param in self.model.named_parameters()
            }
        else:
            self.original_state = {
                name: param.data.cpu().clone() for name, param in self.model.named_parameters()
            }

        # Get tentative masks from strategy (this is cheap, just boolean masks)
        if verbose:
            get_logger().info(f"\nSelecting weights using {self.strategy.get_name()} strategy...")
        tentative_masks = self.strategy.select_weights_to_prune(self.model, sparsity)

        total_tentative = sum(mask.sum().item() for mask in tentative_masks.values())
        if verbose:
            get_logger().info(f"Tentative pruning candidates: {total_tentative}")

        # Get all weight layers (2D or higher dimensional parameters)
        layer_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                layer_params.append((name, param))

        if verbose:
            get_logger().info(f"\nProcessing {len(layer_params)} layers sequentially...")
            get_logger().info(f"Memory savings: storing gradients for 1 layer at a time instead of all {len(layer_params)} layers")

        # Process each layer individually
        final_masks = {}

        for layer_idx, (layer_name, layer_param) in enumerate(layer_params):
            if verbose and layer_idx % 5 == 0:  # Print progress every 5 layers
                get_logger().debug(f"  Layer {layer_idx+1}/{len(layer_params)}: {layer_name}")

            # Skip if not in tentative masks
            if layer_name not in tentative_masks:
                continue

            # Freeze all other parameters to save memory
            original_requires_grad = {}
            for name, param in self.model.named_parameters():
                original_requires_grad[name] = param.requires_grad
                if name != layer_name:
                    param.requires_grad = False

            # Compute original gradient for this layer ONLY
            original_grad = self._compute_gradients_single_layer(
                dataloader, loss_fn, num_batches, layer_name
            )

            # Apply tentative reduction to this layer
            mask = tentative_masks[layer_name]
            if mask.device != layer_param.device:
                mask = mask.to(layer_param.device)
            layer_param.data[mask] *= reduction_factor

            # Compute modified gradient for this layer ONLY
            modified_grad = self._compute_gradients_single_layer(
                dataloader, loss_fn, num_batches, layer_name
            )

            # Compare gradients and decide (only for this layer)
            gradient_below_threshold = modified_grad <= original_grad * (1.0 + gradient_threshold)
            final_mask = tentative_masks[layer_name] & gradient_below_threshold
            final_masks[layer_name] = final_mask

            # Restore original values for this layer
            if layer_name in self.original_state:
                saved_state = self.original_state[layer_name]
                if saved_state.dtype != layer_param.dtype:
                    layer_param.data.copy_(saved_state.to(layer_param.dtype))
                else:
                    layer_param.data.copy_(saved_state)

            # Apply final pruning to this layer
            final_mask_device = final_mask.to(layer_param.device) if final_mask.device != layer_param.device else final_mask
            layer_param.data[final_mask_device] = 0

            # Restore requires_grad
            for name, param in self.model.named_parameters():
                param.requires_grad = original_requires_grad[name]

            # Explicitly clear CUDA cache if on GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # Fill in masks for other parameters (biases, layer norms, etc.)
        for name, param in self.model.named_parameters():
            if name not in final_masks:
                final_masks[name] = torch.zeros(param.data.shape, dtype=torch.bool, device='cpu')

        self.pruning_mask = final_masks

        if verbose:
            get_logger().info(f"\nLayer-by-layer streaming complete!")
            self._print_pruning_statistics(final_masks)

        # Compare with baseline if requested
        if compare_baseline:
            if eval_fn is None:
                raise ValueError("eval_fn must be provided when compare_baseline=True")

            if verbose:
                get_logger().info("\n" + "="*60)
                get_logger().info("BASELINE COMPARISON")
                get_logger().info("="*60)

            # Save gradient-pruned state
            if self.use_fp16:
                gradient_pruned_state = {
                    name: param.data.cpu().half() for name, param in self.model.named_parameters()
                }
            else:
                gradient_pruned_state = {
                    name: param.data.cpu().clone() for name, param in self.model.named_parameters()
                }

            # Evaluate gradient-based pruning
            gradient_accuracy = eval_fn(self.model)
            gradient_sparsity = self.get_sparsity()

            # Apply magnitude-only (all tentative candidates)
            for name, param in self.model.named_parameters():
                if name in self.original_state:
                    saved_state = self.original_state[name]
                    if saved_state.dtype != param.dtype:
                        param.data.copy_(saved_state.to(param.dtype))
                    else:
                        param.data.copy_(saved_state)
                if name in tentative_masks:
                    mask = tentative_masks[name].to(param.device) if tentative_masks[name].device != param.device else tentative_masks[name]
                    param.data[mask] = 0

            # Evaluate magnitude-only pruning
            magnitude_accuracy = eval_fn(self.model)
            magnitude_sparsity = self.get_sparsity()

            # Restore gradient-pruned state
            for name, param in self.model.named_parameters():
                if name in gradient_pruned_state:
                    saved_state = gradient_pruned_state[name]
                    if saved_state.dtype != param.dtype:
                        param.data.copy_(saved_state.to(param.dtype))
                    else:
                        param.data.copy_(saved_state)

            if verbose:
                get_logger().info(f"Magnitude-only (all candidates):")
                get_logger().info(f"  Sparsity: {magnitude_sparsity:.2%}")
                get_logger().info(f"  Accuracy: {magnitude_accuracy:.2f}%")
                get_logger().info(f"\nGradient-filtered:")
                get_logger().info(f"  Sparsity: {gradient_sparsity:.2%}")
                get_logger().info(f"  Accuracy: {gradient_accuracy:.2f}%")
                get_logger().info(f"\nAccuracy gain from gradient filtering: {gradient_accuracy - magnitude_accuracy:+.2f}%")
                get_logger().info("="*60)

        return final_masks

    def _compute_gradients_single_layer(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        num_batches: int,
        layer_name: str
    ) -> torch.Tensor:
        """
        Compute gradients for a SINGLE layer only.

        This is the key memory optimization - we only store gradient for one layer
        instead of all layers simultaneously.

        Returns:
            Gradient tensor for the specified layer (absolute maximum across batches)
        """
        # Disable dropout for deterministic gradients
        dropout_states = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                dropout_states[name] = module.training
                module.eval()

        # Get the parameter
        layer_param = dict(self.model.named_parameters())[layer_name]

        # Initialize gradient storage for this ONE layer only (use fp16 if enabled)
        gradient = torch.zeros(layer_param.shape, dtype=self.grad_dtype, device='cpu')

        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_count >= num_batches:
                break

            # Zero gradients
            self.model.zero_grad()

            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = inputs

            # Forward pass
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass
            loss.backward()

            # Extract gradient for this layer ONLY and convert to fp16 if enabled
            if layer_param.grad is not None:
                grad_abs = layer_param.grad.data.abs().cpu().to(self.grad_dtype)
                if batch_count == 0:
                    gradient = grad_abs
                else:
                    gradient = torch.maximum(gradient, grad_abs)

            batch_count += 1

        # Restore dropout states
        for name, module in self.model.named_modules():
            if name in dropout_states:
                if dropout_states[name]:
                    module.train()

        return gradient

    def _apply_gradient_filtering_single_layer(
        self,
        layer_name: str,
        tentative_masks: Dict[str, torch.Tensor],
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        num_batches: int,
        reduction_factor: float,
        gradient_threshold: float
    ) -> Dict[str, torch.Tensor]:
        """
        Apply gradient filtering to a single layer.

        This is the core GradProbe algorithm for one layer:
        1. Compute original gradients for the layer
        2. Apply tentative reduction to the layer
        3. Compute modified gradients for the layer
        4. Compare gradients and decide final pruning mask
        5. Restore original layer state

        Args:
            layer_name: Name of the layer to process
            tentative_masks: Dictionary with tentative mask for this layer
            dataloader: DataLoader for computing gradients
            loss_fn: Loss function
            num_batches: Number of batches for gradient computation
            reduction_factor: Factor to reduce tentative weights by
            gradient_threshold: Relative gradient increase threshold

        Returns:
            Dictionary with final mask for this layer
        """
        # Get the parameter
        layer_param = dict(self.model.named_parameters())[layer_name]

        # Save original state for this layer
        original_layer_state = layer_param.data.clone()

        # Step 1: Compute original gradients for this layer
        original_grad = self._compute_gradients_single_layer(
            dataloader=dataloader,
            loss_fn=loss_fn,
            num_batches=num_batches,
            layer_name=layer_name
        )

        # Step 2: Apply tentative reduction to this layer
        tentative_mask = tentative_masks[layer_name]
        # Move mask to same device as param for reduction
        if tentative_mask.device != layer_param.device:
            tentative_mask_device = tentative_mask.to(layer_param.device)
        else:
            tentative_mask_device = tentative_mask
        layer_param.data[tentative_mask_device] *= reduction_factor

        # Step 3: Compute modified gradients for this layer
        modified_grad = self._compute_gradients_single_layer(
            dataloader=dataloader,
            loss_fn=loss_fn,
            num_batches=num_batches,
            layer_name=layer_name
        )

        # Step 4: Compare gradients and decide final mask
        original_grads = {layer_name: original_grad}
        modified_grads = {layer_name: modified_grad}

        final_masks = self._compare_gradients_and_decide(
            original_grads=original_grads,
            modified_grads=modified_grads,
            tentative_masks=tentative_masks,
            gradient_threshold=gradient_threshold,
            verbose=False  # Don't print stats for individual layers
        )

        # Step 5: Restore original layer state
        layer_param.data.copy_(original_layer_state)

        return final_masks

    def get_sparsity(self) -> float:
        """
        Get the current sparsity of the model.

        Returns:
            Fraction of weights that are zero
        """
        total_params = 0
        zero_params = 0

        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0

    def apply_mask_permanently(self):
        """
        Register pruning masks as buffers to make pruning permanent.
        This prevents pruned weights from being updated during training.
        """
        if self.pruning_mask is None:
            raise RuntimeError("No pruning has been performed yet")

        def hook_factory(mask):
            def hook(grad):
                return grad * (~mask).float()
            return hook

        for name, param in self.model.named_parameters():
            if name in self.pruning_mask:
                mask = self.pruning_mask[name]
                param.register_hook(hook_factory(mask))

    def sweep_gradient_thresholds(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        eval_fn: Callable,
        sparsity: float,
        thresholds: list = None,
        num_batches: int = 1,
        reduction_factor: float = 0.1,
        verbose: bool = True
    ) -> Dict:
        """
        Test different gradient thresholds and report results.

        This method helps understand the trade-off between pruning aggressiveness
        and model performance by testing different threshold values.

        Args:
            dataloader: DataLoader for gradient computation
            loss_fn: Loss function for computing gradients
            eval_fn: Function that takes model and returns accuracy/metric
                    Should have signature: eval_fn(model) -> float
            sparsity: Target sparsity level
            thresholds: List of threshold values to test (default: [0.0, 0.05, 0.1, 0.2, 0.5])
            num_batches: Number of batches for gradient computation
            reduction_factor: Weight reduction factor
            verbose: Whether to print progress

        Returns:
            Dictionary with threshold sweep results containing:
            - 'thresholds': list of tested thresholds
            - 'sparsities': list of actual sparsities achieved
            - 'weights_pruned': list of number of weights pruned
            - 'metrics': list of metric values (accuracy/loss)
        """
        if thresholds is None:
            thresholds = [0.0, 0.05, 0.1, 0.2, 0.5]

        results = {
            'thresholds': [],
            'sparsities': [],
            'weights_pruned': [],
            'metrics': []
        }

        # Save original model state once
        original_state_backup = {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }

        # Extract batches to use for all tests (ensures consistency)
        # This prevents different thresholds from using different random batches
        cached_batches = []
        batch_count = 0
        for batch in dataloader:
            if batch_count >= num_batches:
                break
            # Store batches on CPU to save memory if needed
            if isinstance(batch, (tuple, list)):
                cached_batches.append([b.cpu() for b in batch])
            else:
                cached_batches.append(batch.cpu())
            batch_count += 1

        if verbose:
            get_logger().info("="*70)
            get_logger().info("GRADIENT THRESHOLD SWEEP")
            get_logger().info("="*70)
            get_logger().info(f"Testing {len(thresholds)} threshold values...")
            get_logger().info(f"Target sparsity: {sparsity:.2%}")
            get_logger().info(f"Using {len(cached_batches)} cached batches for consistent comparison")
            get_logger().info("")

        for threshold in thresholds:
            if verbose:
                get_logger().debug(f"\nTesting threshold: {threshold:.2%}")
                get_logger().debug("-" * 50)

            # Restore model to original state
            for name, param in self.model.named_parameters():
                if name in original_state_backup:
                    param.data.copy_(original_state_backup[name])

            # Prune with this threshold using cached batches
            masks = self._prune_with_cached_batches(
                cached_batches=cached_batches,
                loss_fn=loss_fn,
                sparsity=sparsity,
                reduction_factor=reduction_factor,
                gradient_threshold=threshold,
                verbose=False
            )

            # Get actual sparsity
            actual_sparsity = self.get_sparsity()
            total_params = sum(p.numel() for p in self.model.parameters())
            num_pruned = int(actual_sparsity * total_params)

            # Evaluate model
            metric = eval_fn(self.model)

            # Store results
            results['thresholds'].append(threshold)
            results['sparsities'].append(actual_sparsity)
            results['weights_pruned'].append(num_pruned)
            results['metrics'].append(metric)

            if verbose:
                get_logger().debug(f"  Threshold: {threshold:.2%}")
                get_logger().debug(f"  Sparsity: {actual_sparsity:.2%}")
                get_logger().debug(f"  Weights pruned: {num_pruned:,} / {total_params:,}")
                get_logger().debug(f"  Metric: {metric:.2f}")

        # Restore model to original state
        for name, param in self.model.named_parameters():
            if name in original_state_backup:
                param.data.copy_(original_state_backup[name])

        if verbose:
            get_logger().info("\n" + "="*70)
            get_logger().info("SWEEP RESULTS SUMMARY")
            get_logger().info("="*70)
            get_logger().info(f"{'Threshold':<12} {'Sparsity':<12} {'Pruned':<15} {'Metric':<10}")
            get_logger().info("-" * 70)
            for i in range(len(thresholds)):
                get_logger().info(f"{results['thresholds'][i]:<12.2%} "
                      f"{results['sparsities'][i]:<12.2%} "
                      f"{results['weights_pruned'][i]:<15,} "
                      f"{results['metrics'][i]:<10.2f}")
            get_logger().info("="*70)

        return results

    def _prune_with_cached_batches(
        self,
        cached_batches: list,
        loss_fn: Callable,
        sparsity: float,
        reduction_factor: float = 0.1,
        gradient_threshold: float = 0.0,
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Internal pruning method that uses pre-cached batches.

        This ensures consistent batch usage across multiple pruning runs,
        which is essential for fair threshold comparisons.
        """
        # Save original model state
        self.original_state = {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }

        # Step 1: Get initial gradients with original model
        original_gradients = self._compute_gradients_from_cached(cached_batches, loss_fn)

        # Step 2: Use strategy to select weights to tentatively prune
        tentative_masks = self.strategy.select_weights_to_prune(self.model, sparsity)

        # Step 3: Set tentative weights to reduction_factor * original value
        self._apply_tentative_reduction(tentative_masks, reduction_factor)

        # Step 4: Compute gradients with reduced weights
        modified_gradients = self._compute_gradients_from_cached(cached_batches, loss_fn)

        # Step 5: Compare gradients and make final pruning decisions
        final_masks = self._compare_gradients_and_decide(
            original_gradients,
            modified_gradients,
            tentative_masks,
            gradient_threshold=gradient_threshold,
            verbose=verbose
        )

        # Step 6: Apply final pruning
        self._apply_final_pruning(final_masks)

        self.pruning_mask = final_masks

        return final_masks

    def _compute_gradients_from_cached(
        self,
        cached_batches: list,
        loss_fn: Callable
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradients using pre-cached batches with absolute maximum across batches.
        Disables dropout to match inference conditions but keeps model in train mode.
        """
        # Disable dropout for deterministic gradients (inference behavior)
        # but keep model in train mode for other layers
        dropout_states = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                dropout_states[name] = module.training
                module.eval()

        # Use self.grad_dtype (fp16 if use_fp16=True) to reduce memory usage
        gradients = {name: torch.zeros(param.shape, dtype=self.grad_dtype, device=param.device)
                     for name, param in self.model.named_parameters()
                     if param.requires_grad}

        batch_count = 0
        for batch in cached_batches:
            # Zero gradients
            self.model.zero_grad()

            # Move batch to device and handle different formats
            if isinstance(batch, list):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = inputs  # For autoencoder-style tasks

            # Forward pass
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass
            loss.backward()

            # Take element-wise maximum of absolute gradients
            # Convert to self.grad_dtype (fp16 if enabled) to save memory
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_abs = param.grad.data.abs().to(self.grad_dtype)
                    if batch_count == 0:
                        gradients[name] = grad_abs
                    else:
                        gradients[name] = torch.maximum(gradients[name], grad_abs)

            batch_count += 1

        # Restore dropout states
        for name, module in self.model.named_modules():
            if name in dropout_states:
                if dropout_states[name]:
                    module.train()

        return gradients

    def prune_layerwise(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        sparsity: float,
        num_batches: int = 1,
        reduction_factor: float = 0.1,
        gradient_threshold: Union[float, List[float], Dict[str, float], Tuple[str, float]] = 0.0,
        verbose: bool = True,
        layer_order: str = "reverse"
    ) -> Dict[str, torch.Tensor]:
        """
        Prune layer-by-layer with configurable ordering.

        This approach freezes already-pruned layers to isolate the gradient
        signal for each layer, which can work better for deep networks.

        Args:
            dataloader: DataLoader providing input data
            loss_fn: Loss function
            sparsity: Target sparsity level per layer
            num_batches: Number of batches for gradient computation
            reduction_factor: Factor to reduce tentative weights by
            gradient_threshold: Relative gradient increase threshold. Can be:
                - float: Single threshold for all layers
                - List[float]: Per-layer thresholds (must match number of layers)
                - Dict[str, float]: Mapping of layer names to thresholds
                - Tuple[str, float]: ("adaptive", base_threshold) for adaptive scaling
            verbose: Whether to print progress
            layer_order: Order to prune layers:
                        "reverse" - output to input (default)
                        "size" - largest to smallest (by number of weights)
                        "forward" - input to output

        Returns:
            Dictionary mapping parameter names to pruning masks
        """
        # Process gradient_threshold into per-layer dictionary
        if isinstance(gradient_threshold, tuple) and len(gradient_threshold) == 2 and gradient_threshold[0] == "adaptive":
            # Adaptive thresholds
            base_threshold = gradient_threshold[1]
            layer_thresholds = compute_adaptive_gradient_thresholds(self.model, base_threshold)
            if verbose:
                get_logger().info(f"Using adaptive gradient thresholds (base={base_threshold:.2f})")
        elif isinstance(gradient_threshold, dict):
            # Already a dictionary
            layer_thresholds = gradient_threshold
        elif isinstance(gradient_threshold, (list, tuple)):
            # List of thresholds - will be mapped to layers after ordering
            layer_thresholds = None
            threshold_list = list(gradient_threshold)
        else:
            # Single threshold for all layers
            layer_thresholds = None
            single_threshold = float(gradient_threshold)

        if verbose:
            get_logger().info("="*70)
            get_logger().info("LAYER-BY-LAYER PRUNING")
            get_logger().info("="*70)
            get_logger().info(f"Target sparsity per layer: {sparsity:.2%}")
            if isinstance(gradient_threshold, tuple) and len(gradient_threshold) == 2 and gradient_threshold[0] == "adaptive":
                get_logger().info(f"Gradient threshold: adaptive (base={gradient_threshold[1]:.2f})")
            elif isinstance(gradient_threshold, dict):
                get_logger().info(f"Gradient threshold: per-layer dictionary")
            elif isinstance(gradient_threshold, (list, tuple)):
                get_logger().info(f"Gradient threshold: per-layer list")
            else:
                get_logger().info(f"Gradient threshold: {gradient_threshold:.2%}")
            get_logger().info(f"Layer order: {layer_order}")
            get_logger().info("")

        # Get all weight layers
        layer_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:  # Weight matrices only
                layer_params.append((name, param))

        # Apply ordering
        if layer_order == "reverse":
            # Output to input (reverse model order)
            layer_params = list(reversed(layer_params))
            order_desc = "reverse order (output to input)"
        elif layer_order == "size":
            # Largest to smallest by number of weights
            layer_params = sorted(layer_params, key=lambda x: x[1].numel(), reverse=True)
            order_desc = "size order (largest to smallest)"
        elif layer_order == "forward":
            # Input to output (model order)
            order_desc = "forward order (input to output)"
        else:
            raise ValueError(f"Unknown layer_order: {layer_order}. Must be 'reverse', 'size', or 'forward'")

        # Map list of thresholds to layers after ordering
        if isinstance(gradient_threshold, (list, tuple)) and not (isinstance(gradient_threshold, tuple) and len(gradient_threshold) == 2 and gradient_threshold[0] == "adaptive"):
            if len(threshold_list) != len(layer_params):
                raise ValueError(f"Length of gradient_threshold list ({len(threshold_list)}) must match number of layers ({len(layer_params)})")
            layer_thresholds = {name: threshold_list[i] for i, (name, param) in enumerate(layer_params)}

        if verbose:
            get_logger().info(f"Pruning {len(layer_params)} layers in {order_desc}:")
            for i, (name, param) in enumerate(layer_params):
                if layer_thresholds:
                    thresh_str = f" (threshold={layer_thresholds[name]:.4f})"
                else:
                    thresh_str = f" (threshold={single_threshold:.4f})"
                get_logger().info(f"  {i+1}. {name} ({param.numel():,} weights){thresh_str}")
            get_logger().info("")

        # Save original model state
        # In low_memory_mode, save to CPU to avoid doubling GPU memory usage
        # In fp16 mode, save in half precision to reduce memory usage
        if self.low_memory_mode:
            if self.use_fp16:
                self.original_state = {
                    name: param.data.cpu().half() for name, param in self.model.named_parameters()
                }
            else:
                self.original_state = {
                    name: param.data.cpu().clone() for name, param in self.model.named_parameters()
                }
        else:
            if self.use_fp16:
                self.original_state = {
                    name: param.data.half() for name, param in self.model.named_parameters()
                }
            else:
                self.original_state = {
                    name: param.data.clone() for name, param in self.model.named_parameters()
                }

        all_masks = {}

        # Accumulators for caching gradients and masks across all layers
        # Only cache if not in low_memory_mode
        cached_original_gradients = {}
        cached_modified_gradients = {}
        cached_tentative_masks = {}

        # Prune each layer
        for layer_idx, (layer_name, layer_param) in enumerate(layer_params):
            if verbose:
                get_logger().debug(f"Layer {layer_idx+1}/{len(layer_params)}: {layer_name}")
                get_logger().debug("-" * 50)

            # Freeze all other layers (set requires_grad=False)
            original_requires_grad = {}
            for name, param in self.model.named_parameters():
                original_requires_grad[name] = param.requires_grad
                if name != layer_name:
                    param.requires_grad = False

            # Recompute strategy masks for current model state (with previous layers pruned)
            # This is the key difference from the "optimized" version that cached upfront
            if verbose:
                get_logger().debug(f"  Computing {self.strategy.get_name()} importance scores...")
            full_strategy_masks = self.strategy.select_weights_to_prune(self.model, sparsity)

            # Extract just this layer's mask
            if layer_name not in full_strategy_masks:
                # Shouldn't happen, but fallback to empty mask
                all_masks[layer_name] = torch.zeros(layer_param.data.shape, dtype=torch.bool, device='cpu')
                if verbose:
                    get_logger().warning(f"  Warning: No mask for {layer_name}, skipping")
                # Restore requires_grad
                for name, param in self.model.named_parameters():
                    param.requires_grad = original_requires_grad[name]
                continue

            tentative_mask_for_layer = {layer_name: full_strategy_masks[layer_name]}

            # Apply gradient filtering for this layer only
            # Use per-layer threshold if available
            if layer_thresholds:
                current_threshold = layer_thresholds[layer_name]
            else:
                current_threshold = single_threshold

            layer_masks = self._apply_gradient_filtering_single_layer(
                layer_name=layer_name,
                tentative_masks=tentative_mask_for_layer,
                dataloader=dataloader,
                loss_fn=loss_fn,
                num_batches=num_batches,
                reduction_factor=reduction_factor,
                gradient_threshold=current_threshold
            )

            # Store masks for this layer
            all_masks[layer_name] = layer_masks[layer_name]

            # CRITICAL: Actually apply pruning to this layer before moving to next
            # This allows next layer's WANDA/Magnitude to see the effect of pruning
            layer_mask = layer_masks[layer_name]
            if layer_mask.device != layer_param.device:
                layer_mask_device = layer_mask.to(layer_param.device)
            else:
                layer_mask_device = layer_mask
            layer_param.data[layer_mask_device] = 0

            # Accumulate cached gradients and masks from this layer
            # Skip caching in low_memory_mode to save GPU memory
            if not self.low_memory_mode:
                if hasattr(self, '_cached_original_gradients'):
                    cached_original_gradients.update(self._cached_original_gradients)
                if hasattr(self, '_cached_modified_gradients'):
                    cached_modified_gradients.update(self._cached_modified_gradients)
                if hasattr(self, '_cached_tentative_masks'):
                    cached_tentative_masks.update(self._cached_tentative_masks)

            # In low_memory_mode, clear cached data after each layer
            if self.low_memory_mode:
                if hasattr(self, '_cached_original_gradients'):
                    del self._cached_original_gradients
                if hasattr(self, '_cached_modified_gradients'):
                    del self._cached_modified_gradients
                if hasattr(self, '_cached_tentative_masks'):
                    del self._cached_tentative_masks
                # Clear CUDA cache if on GPU
                #if self.device == 'cuda':
                    #torch.cuda.empty_cache()

            # Restore requires_grad
            for name, param in self.model.named_parameters():
                param.requires_grad = original_requires_grad[name]

            # Print stats for this layer
            if verbose:
                layer_sparsity = layer_masks[layer_name].sum().item() / layer_masks[layer_name].numel()
                get_logger().debug(f"  Pruned: {layer_masks[layer_name].sum().item()} / {layer_masks[layer_name].numel()} "
                      f"({layer_sparsity:.2%})")
                get_logger().debug("")

        # Store accumulated caches for threshold tuning (only if not in low_memory_mode)
        if not self.low_memory_mode:
            self._cached_original_gradients = cached_original_gradients
            self._cached_modified_gradients = cached_modified_gradients
            self._cached_tentative_masks = cached_tentative_masks
        else:
            # Clear any cached data
            if hasattr(self, '_cached_original_gradients'):
                del self._cached_original_gradients
            if hasattr(self, '_cached_modified_gradients'):
                del self._cached_modified_gradients
            if hasattr(self, '_cached_tentative_masks'):
                del self._cached_tentative_masks

        # Fill in masks for other parameters (bias, etc.) - on CPU
        for name, param in self.model.named_parameters():
            if name not in all_masks:
                all_masks[name] = torch.zeros(param.data.shape, dtype=torch.bool, device='cpu')

        self.pruning_mask = all_masks

        if verbose:
            get_logger().info("="*70)
            self._print_pruning_statistics(all_masks)

        return all_masks

    def _tune_threshold_at_sparsity(
        self,
        current_sparsity: float,
        base_threshold: float,
        original_state_backup: Dict,
        cumulative_masks: Dict,
        tentative_masks: Dict,
        original_gradients: Dict,
        modified_gradients: Dict,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        eval_fn: Callable,
        initial_accuracy: float,
        max_accuracy_drop: float,
        num_batches: int,
        reduction_factor: float,
        layerwise: bool,
        verbose: bool,
        prev_step_data: Dict = None,
        layer_order: str = "reverse"
    ) -> Dict:
        """
        Fine-tune gradient threshold when accuracy target is missed.

        Single-step mode: Reuses the same gradients and tentative masks, only varying
        the threshold in the gradient comparison step.

        Two-step mode (experimental): When prev_step_data is provided, completely re-runs
        the pruning process for BOTH the previous and current steps with new thresholds.
        This explores different pruning paths since changing the threshold at step N-1
        creates a different model state going into step N.

        Tries threshold * 0.9 and * 1.1 to find which direction improves accuracy,
        then continues in that direction until results stop improving.

        Args:
            prev_step_data: EXPERIMENTAL Optional dict with previous step's data.
                           If provided, will re-prune both steps from scratch with new threshold.
                           Dict should contain: 'sparsity' (target for previous step),
                           'base_masks' (cumulative masks from before previous step)

        Returns:
            Dictionary with 'threshold', 'masks', 'sparsity', 'accuracy' if successful,
            None if no improvement found that meets accuracy requirement
        """
        # Implement two-step tuning when prev_step_data is provided
        use_two_step = prev_step_data is not None

        if use_two_step and verbose:
            get_logger().info(f"    Using two-step tuning (re-pruning both step {prev_step_data['sparsity']:.0%} and {current_sparsity:.0%})")

        # Try both directions
        lower_threshold = base_threshold * 0.9
        higher_threshold = base_threshold * 1.1

        if verbose:
            get_logger().debug(f"    Testing threshold {lower_threshold:.2f} and {higher_threshold:.2f}")

        # Test lower threshold
        if use_two_step:
            lower_result = self._apply_threshold_two_steps(
                threshold=lower_threshold,
                prev_step_data=prev_step_data,
                current_sparsity=current_sparsity,
                dataloader=dataloader,
                loss_fn=loss_fn,
                original_state_backup=original_state_backup,
                eval_fn=eval_fn,
                num_batches=num_batches,
                reduction_factor=reduction_factor,
                layerwise=layerwise,
                verbose=verbose,
                layer_order=layer_order
            )
        else:
            lower_result = self._apply_threshold_and_evaluate(
                threshold=lower_threshold,
                tentative_masks=tentative_masks,
                original_gradients=original_gradients,
                modified_gradients=modified_gradients,
                original_state_backup=original_state_backup,
                cumulative_masks=cumulative_masks,
                eval_fn=eval_fn,
                verbose=False
            )

        # Test higher threshold
        if use_two_step:
            higher_result = self._apply_threshold_two_steps(
                threshold=higher_threshold,
                prev_step_data=prev_step_data,
                current_sparsity=current_sparsity,
                dataloader=dataloader,
                loss_fn=loss_fn,
                original_state_backup=original_state_backup,
                eval_fn=eval_fn,
                num_batches=num_batches,
                reduction_factor=reduction_factor,
                layerwise=layerwise,
                verbose=verbose,
                layer_order=layer_order
            )
        else:
            higher_result = self._apply_threshold_and_evaluate(
                threshold=higher_threshold,
                tentative_masks=tentative_masks,
                original_gradients=original_gradients,
                modified_gradients=modified_gradients,
                original_state_backup=original_state_backup,
                cumulative_masks=cumulative_masks,
                eval_fn=eval_fn,
                verbose=False
            )

        # Determine which direction is better (higher accuracy)
        best_result = None
        direction_multiplier = None

        lower_drop = initial_accuracy - lower_result['accuracy']
        higher_drop = initial_accuracy - higher_result['accuracy']

        if verbose:
            get_logger().info(f"    Lower threshold ({lower_threshold:.2f}): sparsity={lower_result['sparsity']:.2%}, accuracy={lower_result['accuracy']:.2f}%, drop={lower_drop:.2f}%")
            get_logger().info(f"    Higher threshold ({higher_threshold:.2f}): sparsity={higher_result['sparsity']:.2%}, accuracy={higher_result['accuracy']:.2f}%, drop={higher_drop:.2f}%")

        # Check if either direction meets the requirement
        if lower_drop <= max_accuracy_drop:
            best_result = lower_result
            best_result['threshold'] = lower_threshold
            direction_multiplier = 0.9  # Go lower

        if higher_drop <= max_accuracy_drop and (best_result is None or higher_result['accuracy'] > best_result['accuracy']):
            best_result = higher_result
            best_result['threshold'] = higher_threshold
            direction_multiplier = 1.1  # Go higher

        # If neither direction meets requirements, still try the better direction
        # (it might improve with further iterations)
        if best_result is None:
            if verbose:
                get_logger().info(f"    Neither direction meets accuracy requirement")
                get_logger().info(f"    Trying the better direction anyway...")

            # Pick the direction with better (or equal) accuracy
            if lower_result['accuracy'] >= higher_result['accuracy']:
                best_result = lower_result
                best_result['threshold'] = lower_threshold
                direction_multiplier = 0.9
                if verbose:
                    get_logger().debug(f"    Chose lower threshold (better/equal accuracy)")
            else:
                best_result = higher_result
                best_result['threshold'] = higher_threshold
                direction_multiplier = 1.1
                if verbose:
                    get_logger().debug(f"    Chose higher threshold (better accuracy)")

        # Continue in the better direction
        if verbose:
            if direction_multiplier == 0.9:
                get_logger().info(f"    Direction: decreasing threshold (×0.9)")
            else:
                get_logger().info(f"    Direction: increasing threshold (×1.1)")

        current_threshold = best_result['threshold']
        previous_accuracy = best_result['accuracy']

        # Track best result that meets requirements (may be None initially)
        best_valid_result = best_result if (initial_accuracy - best_result['accuracy']) <= max_accuracy_drop else None

        # Keep going in that direction as long as accuracy doesn't degrade significantly
        # Allow 1% tolerance for small numerical variations
        max_iterations = 20
        for i in range(max_iterations):
            next_threshold = current_threshold * direction_multiplier

            if use_two_step:
                next_result = self._apply_threshold_two_steps(
                    threshold=next_threshold,
                    prev_step_data=prev_step_data,
                    current_sparsity=current_sparsity,
                    dataloader=dataloader,
                    loss_fn=loss_fn,
                    original_state_backup=original_state_backup,
                    eval_fn=eval_fn,
                    num_batches=num_batches,
                    reduction_factor=reduction_factor,
                    layerwise=layerwise,
                    verbose=False,
                    layer_order=layer_order
                )
            else:
                next_result = self._apply_threshold_and_evaluate(
                    threshold=next_threshold,
                    tentative_masks=tentative_masks,
                    original_gradients=original_gradients,
                    modified_gradients=modified_gradients,
                    original_state_backup=original_state_backup,
                    cumulative_masks=cumulative_masks,
                    eval_fn=eval_fn,
                    verbose=False
                )

            next_drop = initial_accuracy - next_result['accuracy']

            if verbose:
                get_logger().info(f"    Trying threshold {next_threshold:.2f}: sparsity={next_result['sparsity']:.2%}, accuracy={next_result['accuracy']:.2f}%, drop={next_drop:.2f}%")

            # Allow up to 1% degradation relative to current accuracy to handle numerical noise
            tolerance = abs(previous_accuracy) * 0.01
            accuracy_change = next_result['accuracy'] - previous_accuracy

            # Continue as long as accuracy doesn't degrade by more than tolerance
            if accuracy_change >= -tolerance:
                # Update tracking
                current_threshold = next_threshold
                previous_accuracy = next_result['accuracy']

                # Track this as best valid result if it meets requirements
                # Prefer highest sparsity among valid results
                if next_drop <= max_accuracy_drop:
                    if best_valid_result is None or next_result['sparsity'] > best_valid_result['sparsity']:
                        best_valid_result = next_result
                        best_valid_result['threshold'] = next_threshold
                        if verbose:
                            get_logger().info(f"      ✓ New best valid result (sparsity={next_result['sparsity']:.2%})")
            else:
                # Accuracy degraded beyond tolerance, stop
                if verbose:
                    get_logger().info(f"    Stopping tuning (accuracy degraded by {-accuracy_change:.2f}% > tolerance {tolerance:.2f}%)")
                break

        # Return the best result that meets requirements (or None if we never found one)
        return best_valid_result

    def _apply_threshold_two_steps(
        self,
        threshold: float,
        prev_step_data: Dict,
        current_sparsity: float,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        original_state_backup: Dict,
        eval_fn: Callable,
        num_batches: int,
        reduction_factor: float,
        layerwise: bool,
        verbose: bool,
        layer_order: str = "reverse"
    ) -> Dict:
        """
        Re-prune BOTH previous and current steps with a new threshold.

        This completely re-runs the pruning process for both steps:
        1. Reset model to state BEFORE previous step (step N-2)
        2. Re-run full pruning for previous step (N-1) with new threshold
        3. Re-run full pruning for current step (N) with new threshold
        4. Evaluate the result

        This explores a different pruning path since changing the threshold at
        step N-1 creates a different model state going into step N.

        Args:
            threshold: Gradient threshold to use for both steps
            prev_step_data: Dict with 'sparsity' (target for previous step),
                           'base_masks' (cumulative masks from before previous step)
            current_sparsity: Target sparsity for current step
            dataloader: DataLoader for gradient computation
            loss_fn: Loss function
            original_state_backup: Original unpruned model state
            eval_fn: Evaluation function
            num_batches: Number of batches for gradient computation
            reduction_factor: Weight reduction factor
            layerwise: Whether to use layer-by-layer pruning
            verbose: Print details
            layer_order: Order for layerwise pruning (only used if layerwise=True)

        Returns:
            Dictionary with 'masks', 'sparsity', 'accuracy'
        """
        # Reset model to step N-2 state (before previous step)
        base_masks = prev_step_data['base_masks']
        for name, param in self.model.named_parameters():
            if name in original_state_backup:
                param.data.copy_(original_state_backup[name])
            if name in base_masks:
                mask = self._move_mask_to_param_device(base_masks[name], param)
                param.data[mask] = 0

        if verbose:
            get_logger().debug(f"        [Two-step] Re-pruning step {prev_step_data['sparsity']:.0%} with threshold {threshold:.2f}...")

        # Re-run pruning for previous step with new threshold
        prev_sparsity = prev_step_data['sparsity']
        if layerwise:
            prev_new_masks = self.prune_layerwise(
                dataloader=dataloader,
                loss_fn=loss_fn,
                sparsity=prev_sparsity,
                num_batches=num_batches,
                reduction_factor=reduction_factor,
                gradient_threshold=threshold,
                verbose=False,
                layer_order=layer_order
            )
        else:
            prev_new_masks = self.prune(
                dataloader=dataloader,
                loss_fn=loss_fn,
                sparsity=prev_sparsity,
                num_batches=num_batches,
                reduction_factor=reduction_factor,
                gradient_threshold=threshold,
                verbose=False
            )

        # Accumulate with base masks
        prev_cumulative_masks = {name: mask.clone() for name, mask in base_masks.items()}
        for name in prev_cumulative_masks:
            if name in prev_new_masks:
                prev_cumulative_masks[name] = prev_cumulative_masks[name] | prev_new_masks[name]

        # Apply previous step's cumulative masks to create the state for current step
        for name, param in self.model.named_parameters():
            if name in prev_cumulative_masks:
                mask = self._move_mask_to_param_device(prev_cumulative_masks[name], param)
                param.data[mask] = 0

        if verbose:
            get_logger().debug(f"        [Two-step] Re-pruning step {current_sparsity:.0%} with threshold {threshold:.2f}...")

        # Re-run pruning for current step with new threshold
        if layerwise:
            curr_new_masks = self.prune_layerwise(
                dataloader=dataloader,
                loss_fn=loss_fn,
                sparsity=current_sparsity,
                num_batches=num_batches,
                reduction_factor=reduction_factor,
                gradient_threshold=threshold,
                verbose=False,
                layer_order=layer_order
            )
        else:
            curr_new_masks = self.prune(
                dataloader=dataloader,
                loss_fn=loss_fn,
                sparsity=current_sparsity,
                num_batches=num_batches,
                reduction_factor=reduction_factor,
                gradient_threshold=threshold,
                verbose=False
            )

        # Accumulate with previous step's cumulative masks
        combined_masks = {name: mask.clone() for name, mask in prev_cumulative_masks.items()}
        for name in combined_masks:
            if name in curr_new_masks:
                combined_masks[name] = combined_masks[name] | curr_new_masks[name]

        # Apply combined masks to get final state
        for name, param in self.model.named_parameters():
            if name in original_state_backup:
                param.data.copy_(original_state_backup[name])
            if name in combined_masks:
                mask = self._move_mask_to_param_device(combined_masks[name], param)
                param.data[mask] = 0

        # Evaluate
        actual_sparsity = self.get_sparsity()
        accuracy = eval_fn(self.model)

        return {
            'masks': combined_masks,
            'sparsity': actual_sparsity,
            'accuracy': accuracy
        }

    def _apply_threshold_and_evaluate(
        self,
        threshold: float,
        tentative_masks: Dict,
        original_gradients: Dict,
        modified_gradients: Dict,
        original_state_backup: Dict,
        cumulative_masks: Dict,
        eval_fn: Callable,
        verbose: bool
    ) -> Dict:
        """
        Apply gradient comparison with a specific threshold and evaluate the result.

        This method reuses pre-computed gradients and tentative masks, only
        varying the threshold in the gradient comparison step.

        Returns:
            Dictionary with 'masks', 'sparsity', 'accuracy'
        """
        # Apply gradient comparison with the specified threshold
        final_masks = self._compare_gradients_and_decide(
            original_gradients,
            modified_gradients,
            tentative_masks,
            gradient_threshold=threshold,
            verbose=verbose
        )

        # Accumulate with previous masks
        test_cumulative_masks = {name: mask.clone() for name, mask in cumulative_masks.items()}
        for name in test_cumulative_masks:
            if name in final_masks:
                test_cumulative_masks[name] = test_cumulative_masks[name] | final_masks[name]

        # Apply cumulative masks to the model
        for name, param in self.model.named_parameters():
            if name in original_state_backup:
                param.data.copy_(original_state_backup[name])
            if name in test_cumulative_masks:
                mask = self._move_mask_to_param_device(test_cumulative_masks[name], param)
                param.data[mask] = 0

        # Evaluate
        actual_sparsity = self.get_sparsity()
        accuracy = eval_fn(self.model)

        return {
            'masks': test_cumulative_masks,
            'sparsity': actual_sparsity,
            'accuracy': accuracy
        }

    def iterative_prune(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        eval_fn: Callable,
        initial_sparsity: float = 0.1,
        sparsity_step: float = 0.1,
        max_accuracy_drop: float = 1.0,
        num_batches: int = 1,
        reduction_factor: float = 0.1,
        gradient_threshold: Union[float, List[float], Dict[str, float], Tuple[str, float]] = 0.0,
        layerwise: bool = False,
        verbose: bool = True,
        compare_baseline: bool = False,
        tune_threshold_on_fail: bool = True,
        experimental_tune_both_steps: bool = False,
        layer_order: str = "reverse"
    ) -> Dict:
        """
        Iteratively increase pruning until accuracy drops beyond threshold.

        This method gradually increases sparsity without retraining,
        stopping when accuracy degrades too much.

        Args:
            dataloader: DataLoader for gradient computation
            loss_fn: Loss function
            eval_fn: Function to evaluate model, returns accuracy/metric (as percentage)
            initial_sparsity: Starting sparsity level (default: 0.1)
            sparsity_step: How much to increase sparsity each iteration (default: 0.1)
            max_accuracy_drop: Stop if accuracy drops by this much in percentage points (default: 1.0 = 1%)
            num_batches: Number of batches for gradient computation
            reduction_factor: Factor to reduce tentative weights by
            gradient_threshold: Relative gradient increase threshold (for layerwise pruning, can be adaptive). Can be:
                - float: Single threshold for all layers
                - List[float]: Per-layer thresholds (must match number of layers)
                - Dict[str, float]: Mapping of layer names to thresholds
                - Tuple[str, float]: ("adaptive", base_threshold) for adaptive scaling
            layerwise: Whether to use layer-by-layer pruning (required for adaptive thresholds)
            verbose: Whether to print progress
            compare_baseline: If True, compare with strategy-only pruning at each step
            tune_threshold_on_fail: If True, when accuracy target is missed, fine-tune
                                   gradient_threshold by trying ±10% adjustments to find
                                   a better setting that meets the accuracy requirement
            experimental_tune_both_steps: EXPERIMENTAL If True, when tuning threshold,
                                         also re-prune the previous successful step with
                                         the same adjusted threshold. This explores tuning
                                         both steps together. (default: False)
            layer_order: Order for layerwise pruning (only used if layerwise=True):
                        "reverse" - output to input (default)
                        "size" - largest to smallest
                        "forward" - input to output

        Returns:
            Dictionary with results containing:
            - 'sparsity_history': list of sparsity values tested
            - 'accuracy_history': list of accuracies achieved
            - 'final_sparsity': final sparsity achieved
            - 'final_accuracy': final accuracy
            - 'initial_accuracy': accuracy before pruning
            - 'final_masks': final pruning masks
        """
        # Warn about incompatible options in low_memory_mode
        if self.low_memory_mode:
            if tune_threshold_on_fail and layerwise:
                get_logger().warning("\n⚠ WARNING: low_memory_mode + layerwise + tune_threshold_on_fail")
                get_logger().warning("   Threshold tuning will be DISABLED for layerwise pruning in low_memory_mode")
                get_logger().warning("   (requires cached gradients which are not stored to save memory)")
                tune_threshold_on_fail = False
            if experimental_tune_both_steps:
                get_logger().warning("\n⚠ WARNING: low_memory_mode + experimental_tune_both_steps")
                get_logger().warning("   Two-step tuning will be DISABLED in low_memory_mode")
                get_logger().warning("   (requires re-pruning which is very expensive)")
                experimental_tune_both_steps = False

        if verbose:
            get_logger().info("="*70)
            get_logger().info("ITERATIVE PRUNING")
            get_logger().info("="*70)
            get_logger().info(f"Initial sparsity: {initial_sparsity:.2%}")
            get_logger().info(f"Sparsity step: {sparsity_step:.2%}")
            get_logger().info(f"Max accuracy drop: {max_accuracy_drop:.2f}%")
            get_logger().info(f"Layerwise: {layerwise}")
            if layerwise:
                get_logger().info(f"Layer order: {layer_order}")
            if self.low_memory_mode:
                get_logger().info(f"Low memory mode: ENABLED")
            get_logger().info("")

        original_state_backup = {
            name: param.data.cpu().clone() for name, param in self.model.named_parameters()
        }

        # Measure initial accuracy
        initial_accuracy = eval_fn(self.model)
        if verbose:
            get_logger().info(f"Initial accuracy: {initial_accuracy:.2f}%\n")

        results = {
            'sparsity_history': [],
            'accuracy_history': [],
            'initial_accuracy': initial_accuracy,
            'final_sparsity': 0.0,
            'final_accuracy': initial_accuracy,
            'final_masks': None
        }

        current_sparsity = initial_sparsity
        best_masks = None
        best_sparsity = 0.0
        best_accuracy = initial_accuracy

        # Track sparsity progress for stop condition
        previous_sparsity = 0.0
        consecutive_no_increase = 0

        # Track cumulative masks across iterations (on CPU to save GPU memory)
        cumulative_masks = {name: torch.zeros(param.data.shape, dtype=torch.bool, device='cpu')
                           for name, param in self.model.named_parameters()}

        # Track previous iteration's cached data for experimental two-step tuning
        prev_cached_original_gradients = None
        prev_cached_modified_gradients = None
        prev_cached_tentative_masks = None
        prev_sparsity = None
        prev_step_base_masks = None  # Masks from BEFORE the previous step

        while True:
            if verbose:
                get_logger().info(f"Trying sparsity: {current_sparsity:.2%}")
                get_logger().info("-" * 50)

            # Don't restore - continue from previous iteration's pruned state
            # (first iteration starts from original)

            # Prune with current sparsity
            if layerwise:
                new_masks = self.prune_layerwise(
                    dataloader=dataloader,
                    loss_fn=loss_fn,
                    sparsity=current_sparsity,
                    num_batches=num_batches,
                    reduction_factor=reduction_factor,
                    gradient_threshold=gradient_threshold,
                    verbose=False,
                    layer_order=layer_order
                )
            else:
                new_masks = self.prune(
                    dataloader=dataloader,
                    loss_fn=loss_fn,
                    sparsity=current_sparsity,
                    num_batches=num_batches,
                    reduction_factor=reduction_factor,
                    gradient_threshold=gradient_threshold,
                    verbose=False
                )

            # Accumulate masks (union of previous and new)
            for name in cumulative_masks:
                if name in new_masks:
                    cumulative_masks[name] = cumulative_masks[name] | new_masks[name]

            # Explicitly delete new_masks to free memory immediately
            del new_masks
            gc.collect()

            # Apply cumulative masks
            for name, param in self.model.named_parameters():
                if name in cumulative_masks:
                    mask = self._move_mask_to_param_device(cumulative_masks[name], param)
                    param.data[mask] = 0

            # Evaluate
            actual_sparsity = self.get_sparsity()
            accuracy = eval_fn(self.model)
            accuracy_drop = initial_accuracy - accuracy

            results['sparsity_history'].append(actual_sparsity)
            results['accuracy_history'].append(accuracy)

            if verbose:
                get_logger().info(f"  Target sparsity: {current_sparsity:.2%}")
                get_logger().info(f"  Gradient-filtered sparsity: {actual_sparsity:.2%}")
                get_logger().info(f"  Gradient-filtered accuracy: {accuracy:.2f}%")
                get_logger().info(f"  Accuracy drop: {accuracy_drop:.2f}%")

            # Compare with baseline (strategy-only without gradient filtering) if requested
            if compare_baseline:
                # Save gradient-filtered state
                gradient_state = {
                    name: param.data.cpu().clone() for name, param in self.model.named_parameters()
                }

                # Get what strategy-only would select at this target sparsity
                # (without gradient filtering)
                # Use strategy directly on original state
                for name, param in self.model.named_parameters():
                    if name in original_state_backup:
                        param.data.copy_(original_state_backup[name])

                strategy_masks = self.strategy.select_weights_to_prune(self.model, current_sparsity)

                # Apply strategy-only masks
                for name, param in self.model.named_parameters():
                    if name in strategy_masks:
                        mask = self._move_mask_to_param_device(strategy_masks[name], param)
                        param.data[mask] = 0

                # Evaluate strategy-only
                strategy_sparsity = self.get_sparsity()
                strategy_accuracy = eval_fn(self.model)
                strategy_drop = initial_accuracy - strategy_accuracy

                # Get strategy name for display
                strategy_name = self.strategy.get_name()

                if verbose:
                    get_logger().info(f"  {strategy_name}-only sparsity: {strategy_sparsity:.2%}")
                    get_logger().info(f"  {strategy_name}-only accuracy: {strategy_accuracy:.2f}%")
                    get_logger().info(f"  Accuracy gain from filtering: {accuracy - strategy_accuracy:+.2f}%")

                # Restore gradient-filtered state
                for name, param in self.model.named_parameters():
                    if name in gradient_state:
                        param.data.copy_(gradient_state[name])

                # Explicitly delete comparison artifacts to free memory
                del gradient_state
                del strategy_masks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Check if sparsity is not increasing
            sparsity_increase = actual_sparsity - previous_sparsity
            if sparsity_increase <= 0:
                consecutive_no_increase += 1
                if verbose:
                    get_logger().warning(f"  ⚠ Sparsity did not increase (Δ={sparsity_increase:+.2%})")
                    get_logger().warning(f"  Consecutive no-increase count: {consecutive_no_increase}")
            else:
                consecutive_no_increase = 0

            # Check if we should stop due to sparsity plateau
            if consecutive_no_increase >= 2:
                if verbose:
                    get_logger().info(f"  ✗ Sparsity plateau detected (no increase for {consecutive_no_increase} iterations)")
                    get_logger().info(f"  Stopping - gradient filtering is restoring too many weights")
                # Revert to previous best state
                for name, param in self.model.named_parameters():
                    if name in original_state_backup:
                        param.data.copy_(original_state_backup[name])
                    if name in best_masks:
                        mask = self._move_mask_to_param_device(best_masks[name], param)
                        param.data[mask] = 0
                break

            # Check if we should stop due to accuracy drop
            if accuracy_drop > max_accuracy_drop:
                if verbose:
                    get_logger().info(f"  ✗ Accuracy dropped by {accuracy_drop:.2f}% > {max_accuracy_drop:.2f}%")

                # Try to fine-tune gradient threshold to squeeze out more sparsity
                tuned_result = None
                if tune_threshold_on_fail and _is_nonzero_threshold(gradient_threshold):
                    # Check if we have cached gradients and masks
                    if (hasattr(self, '_cached_original_gradients') and
                        hasattr(self, '_cached_modified_gradients') and
                        hasattr(self, '_cached_tentative_masks')):
                        if verbose:
                            mode_str = "two-step" if (experimental_tune_both_steps and prev_cached_original_gradients is not None) else "single-step"
                            get_logger().info(f"  ⚙ Attempting threshold tuning ({mode_str})...")

                        # Pass previous step's data if experimental mode is enabled
                        prev_data = None
                        if experimental_tune_both_steps and prev_cached_original_gradients is not None:
                            prev_data = {
                                'original_gradients': prev_cached_original_gradients,
                                'modified_gradients': prev_cached_modified_gradients,
                                'tentative_masks': prev_cached_tentative_masks,
                                'sparsity': prev_sparsity,
                                'base_masks': prev_step_base_masks  # Masks from BEFORE previous step
                            }

                        tuned_result = self._tune_threshold_at_sparsity(
                            current_sparsity=current_sparsity,
                            base_threshold=gradient_threshold,
                            original_state_backup=original_state_backup,
                            cumulative_masks=best_masks,  # Use best_masks, not cumulative_masks!
                            tentative_masks=self._cached_tentative_masks,
                            original_gradients=self._cached_original_gradients,
                            modified_gradients=self._cached_modified_gradients,
                            dataloader=dataloader,
                            loss_fn=loss_fn,
                            eval_fn=eval_fn,
                            initial_accuracy=initial_accuracy,
                            max_accuracy_drop=max_accuracy_drop,
                            num_batches=num_batches,
                            reduction_factor=reduction_factor,
                            layerwise=layerwise,
                            verbose=verbose,
                            prev_step_data=prev_data,
                            layer_order=layer_order
                        )
                    elif verbose:
                        get_logger().warning(f"  ⚠ Cannot tune threshold - cached gradients not available")

                if tuned_result is not None:
                    # Found a better threshold that meets requirements
                    if verbose:
                        get_logger().info(f"  ✓ Tuning successful! Using threshold {tuned_result['threshold']:.2f}")
                        get_logger().info(f"  ✓ Achieved {tuned_result['sparsity']:.2%} sparsity at {tuned_result['accuracy']:.2f}% accuracy")

                    best_masks = tuned_result['masks']
                    best_sparsity = tuned_result['sparsity']
                    best_accuracy = tuned_result['accuracy']

                    # Apply the tuned masks
                    for name, param in self.model.named_parameters():
                        if name in original_state_backup:
                            param.data.copy_(original_state_backup[name])
                        if name in best_masks:
                            mask = self._move_mask_to_param_device(best_masks[name], param)

                            param.data[mask] = 0
                else:
                    if verbose:
                        if tune_threshold_on_fail and _is_nonzero_threshold(gradient_threshold):
                            get_logger().info(f"  ✗ Threshold tuning unsuccessful")
                        get_logger().info(f"  Stopping - reverting to previous iteration")
                    # Revert to previous best state
                    for name, param in self.model.named_parameters():
                        if name in original_state_backup:
                            param.data.copy_(original_state_backup[name])
                        if name in best_masks:
                            mask = self._move_mask_to_param_device(best_masks[name], param)

                            param.data[mask] = 0
                break
            else:
                if verbose:
                    get_logger().info(f"  ✓ Accuracy drop acceptable")

                # Save current iteration's cached data as "previous" for next iteration
                # IMPORTANT: Save BEFORE updating best_masks, so we have masks from before this step
                if (hasattr(self, '_cached_original_gradients') and
                    hasattr(self, '_cached_modified_gradients') and
                    hasattr(self, '_cached_tentative_masks')):
                    # Save masks from BEFORE this iteration (for two-step tuning base)
                    if best_masks is not None:
                        prev_step_base_masks = {k: v.clone() for k, v in best_masks.items()}
                    else:
                        # First iteration, no previous masks (create on CPU)
                        prev_step_base_masks = {name: torch.zeros(param.data.shape, dtype=torch.bool, device='cpu')
                                               for name, param in self.model.named_parameters()}

                    # Save this step's cached data
                    prev_cached_original_gradients = {k: v.clone() for k, v in self._cached_original_gradients.items()}
                    prev_cached_modified_gradients = {k: v.clone() for k, v in self._cached_modified_gradients.items()}
                    prev_cached_tentative_masks = {k: v.clone() for k, v in self._cached_tentative_masks.items()}
                    prev_sparsity = current_sparsity

                # This is good, keep it
                best_masks = {name: mask.clone() for name, mask in cumulative_masks.items()}
                best_sparsity = actual_sparsity
                best_accuracy = accuracy

            # Update previous sparsity for next iteration
            previous_sparsity = actual_sparsity

            if verbose:
                get_logger().info("")

            # Force garbage collection and free GPU memory between iterations
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Increase sparsity for next iteration
            current_sparsity += sparsity_step

            # Stop if we've reached 100% sparsity
            if current_sparsity > 1.0:
                if verbose:
                    get_logger().info("Reached maximum sparsity (1.0)")
                break

        # Model should already have best masks applied (we didn't restore)
        # Just update internal state
        if best_masks is not None:
            self.pruning_mask = best_masks
            results['final_masks'] = best_masks
            results['final_sparsity'] = best_sparsity
            results['final_accuracy'] = best_accuracy
        else:
            # No successful iteration, restore to original
            for name, param in self.model.named_parameters():
                if name in original_state_backup:
                    param.data.copy_(original_state_backup[name])

        if verbose:
            get_logger().info("\n" + "="*70)
            get_logger().info("ITERATIVE PRUNING RESULTS")
            get_logger().info("="*70)
            get_logger().info(f"Initial accuracy: {initial_accuracy:.2f}%")
            get_logger().info(f"Final accuracy: {best_accuracy:.2f}%")
            get_logger().info(f"Accuracy drop: {initial_accuracy - best_accuracy:.2f}%")
            get_logger().info(f"Final sparsity: {best_sparsity:.2%}")
            get_logger().info("="*70)
            if best_masks is not None:
                self._print_pruning_statistics(best_masks)

        return results
