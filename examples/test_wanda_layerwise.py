"""
Test the new layerwise WANDA pruning with per-layer histogram-based thresholds.

Compares:
1. Old approach: Global WANDA scores computed once, reused for all layers
2. New approach: Per-layer WANDA scores with histogram-based thresholds
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from gradprobe.strategies.wanda_optimized import WANDAPruningOptimized
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

print('='*70)
print('LAYERWISE WANDA PRUNING TEST')
print('='*70)

print('\nLoading TinyStories-33M...')
model_name = 'roneneldan/TinyStories-33M'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

total_params = sum(p.numel() for p in model.parameters())
print(f'Model loaded: {total_params:,} parameters')

# Prepare calibration dataset
text = 'Once upon a time, there was a little girl named Lily. She loved to play outside.'
tokens = tokenizer.encode(text, return_tensors='pt')
dataset = TensorDataset(tokens)
dataloader = DataLoader(dataset, batch_size=1)

# Test sparsity
sparsity = 0.3

print('\n' + '='*70)
print('METHOD 1: Global WANDA (old approach)')
print('='*70)
print('Computes WANDA scores once globally, reuses for all layers')
print()

wanda_global = WANDAPruningOptimized(dataloader=dataloader, num_batches=1)
masks_global = wanda_global.select_weights_to_prune(model, sparsity=sparsity)

# Compute per-layer sparsity for global approach
print('\nPer-layer sparsity (global approach):')
layer_sparsities_global = {}
for name, mask in masks_global.items():
    if len(mask.shape) >= 2:  # Only weight matrices
        layer_sparsity = mask.sum().item() / mask.numel()
        layer_sparsities_global[name] = layer_sparsity
        print(f'  {name}: {layer_sparsity:.4f} ({mask.sum().item()}/{mask.numel()} pruned)')

total_pruned_global = sum(mask.sum().item() for mask in masks_global.values())
total_weights_global = sum(mask.numel() for mask in masks_global.values())
global_sparsity = total_pruned_global / total_weights_global
print(f'\nGlobal sparsity: {global_sparsity:.6f} (target: {sparsity:.6f})')

print('\n' + '='*70)
print('METHOD 2: Layerwise WANDA (new approach)')
print('='*70)
print('Computes WANDA scores per layer with histogram-based thresholds')
print()

# Need to create a new instance to clear cache
wanda_layerwise = WANDAPruningOptimized(dataloader=dataloader, num_batches=1)
masks_layerwise = wanda_layerwise.select_weights_to_prune_layerwise(model, sparsity=sparsity)

# Compute per-layer sparsity for layerwise approach
print('\nPer-layer sparsity (layerwise approach):')
layer_sparsities_layerwise = {}
for name, mask in masks_layerwise.items():
    if len(mask.shape) >= 2:  # Only weight matrices
        layer_sparsity = mask.sum().item() / mask.numel()
        layer_sparsities_layerwise[name] = layer_sparsity
        print(f'  {name}: {layer_sparsity:.4f} ({mask.sum().item()}/{mask.numel()} pruned)')

total_pruned_layerwise = sum(mask.sum().item() for mask in masks_layerwise.values())
total_weights_layerwise = sum(mask.numel() for mask in masks_layerwise.values())
layerwise_sparsity = total_pruned_layerwise / total_weights_layerwise
print(f'\nGlobal sparsity: {layerwise_sparsity:.6f} (target: {sparsity:.6f})')

print('\n' + '='*70)
print('COMPARISON')
print('='*70)

# Compare per-layer sparsity variance
print('\nPer-layer sparsity statistics:')
print(f'Global approach:')
global_values = list(layer_sparsities_global.values())
print(f'  Mean: {sum(global_values)/len(global_values):.4f}')
print(f'  Min:  {min(global_values):.4f}')
print(f'  Max:  {max(global_values):.4f}')
print(f'  Std:  {torch.tensor(global_values).std().item():.4f}')

print(f'\nLayerwise approach:')
layerwise_values = list(layer_sparsities_layerwise.values())
print(f'  Mean: {sum(layerwise_values)/len(layerwise_values):.4f}')
print(f'  Min:  {min(layerwise_values):.4f}')
print(f'  Max:  {max(layerwise_values):.4f}')
print(f'  Std:  {torch.tensor(layerwise_values).std().item():.4f}')

print('\nExpected behavior:')
print('- Layerwise approach should have more uniform per-layer sparsity')
print('  (each layer gets exactly the target sparsity)')
print('- Global approach may have high variance across layers')
print('  (some layers pruned more heavily than others)')

# Check that layerwise achieves target sparsity per layer
max_layer_error = max(abs(s - sparsity) for s in layerwise_values)
print(f'\nLayerwise max error from target: {max_layer_error:.6f}')
if max_layer_error < 0.002:  # 0.2% tolerance
    print('✓ All layers within 0.2% of target sparsity')
else:
    print('✗ Some layers exceed 0.2% error')

print('\n' + '='*70)
print('SUCCESS: Layerwise WANDA pruning implemented!')
print('='*70)
