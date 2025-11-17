"""
WANDA (Pruning by Weights And activations) pruning strategy.

This strategy computes importance scores as |weight| * ||activation||,
combining weight magnitude with activation norms to identify unimportant weights.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from .base import PruningStrategy


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
        if not 0 <= sparsity <= 1:
            raise ValueError(f"Sparsity must be between 0 and 1, got {sparsity}")

        # Collect activations for each layer
        activation_norms = self._collect_activation_norms(model)

        if not activation_norms:
            # Fall back to magnitude-only if no activations collected
            return self._magnitude_only_fallback(model, sparsity)

        # Compute importance scores: |weight| * ||activation||
        importance_scores = []
        param_names = []

        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:  # Weight matrices only
                if name in activation_norms:
                    # Compute importance: |W| * ||X||
                    # activation_norms[name] has shape matching input dimension
                    act_norm = activation_norms[name]

                    # For linear layers: weight shape is [out_features, in_features]
                    # activation norm has shape [in_features]
                    # We want to broadcast: |W[i,j]| * ||X[j]||
                    if len(param.shape) == 2:
                        # Expand activation norm to match weight shape
                        act_norm_expanded = act_norm.unsqueeze(0)  # [1, in_features]
                        importance = param.data.abs() * act_norm_expanded
                    else:
                        # For conv layers or other shapes, use simpler approach
                        importance = param.data.abs()

                    importance_scores.append(importance.flatten())
                    param_names.append(name)
                else:
                    # No activation data, use magnitude only
                    importance_scores.append(param.data.abs().flatten())
                    param_names.append(name)

        if not importance_scores:
            return {}

        # Concatenate all importance scores and find the threshold
        all_scores_flat = torch.cat(importance_scores)
        num_weights_to_prune = int(sparsity * len(all_scores_flat))

        if num_weights_to_prune == 0:
            # Return empty masks (no pruning)
            return {name: torch.zeros_like(model.state_dict()[name], dtype=torch.bool)
                    for name in param_names}

        # Find the threshold value (the importance at which we prune)
        threshold = torch.topk(
            all_scores_flat,
            num_weights_to_prune,
            largest=False
        ).values.max()

        # Create masks for each parameter
        # Create on CPU if model is on CUDA to save GPU memory
        masks = {}
        for name, param in model.named_parameters():
            if name in param_names and name in activation_norms:
                # Compute importance again for this parameter
                act_norm = activation_norms[name]
                if len(param.shape) == 2:
                    act_norm_expanded = act_norm.unsqueeze(0)
                    importance = param.data.abs() * act_norm_expanded
                else:
                    importance = param.data.abs()

                # True indicates this weight should be tentatively pruned
                # Create mask on CPU if param is on CUDA to save GPU memory
                mask = importance <= threshold
                if param.device.type == 'cuda':
                    masks[name] = mask.cpu()
                else:
                    masks[name] = mask
            elif name in param_names:
                # No activation data, use magnitude only
                mask = param.data.abs() <= threshold
                if param.device.type == 'cuda':
                    masks[name] = mask.cpu()
                else:
                    masks[name] = mask
            else:
                # Don't prune this parameter
                # Create on CPU if param is on CUDA
                if param.device.type == 'cuda':
                    masks[name] = torch.zeros(param.data.shape, dtype=torch.bool, device='cpu')
                else:
                    masks[name] = torch.zeros_like(param.data, dtype=torch.bool)

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
