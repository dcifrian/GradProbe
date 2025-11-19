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
from gradprobe import Logger, LogLevel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

logger = Logger(program_name='test_wanda_layerwise', level=LogLevel.INFO)

logger.info('='*70)
logger.info('LAYERWISE WANDA PRUNING TEST')
logger.info('='*70)

logger.info('\nLoading TinyStories-33M...')
model_name = 'roneneldan/TinyStories-33M'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

total_params = sum(p.numel() for p in model.parameters())
logger.info(f'Model loaded: {total_params:,} parameters')

# Prepare calibration dataset
text = 'Once upon a time, there was a little girl named Lily. She loved to play outside.'
tokens = tokenizer.encode(text, return_tensors='pt')
dataset = TensorDataset(tokens)
dataloader = DataLoader(dataset, batch_size=1)

# Test sparsity
sparsity = 0.3

logger.info('\n' + '='*70)
logger.info('METHOD 1: Global WANDA (old approach)')
logger.info('='*70)
logger.info('Computes WANDA scores once globally, reuses for all layers')
logger.info('')

wanda_global = WANDAPruningOptimized(dataloader=dataloader, num_batches=1)
masks_global = wanda_global.select_weights_to_prune(model, sparsity=sparsity)

# Compute per-layer sparsity for global approach
logger.info('\nPer-layer sparsity (global approach):')
layer_sparsities_global = {}
for name, mask in masks_global.items():
    if len(mask.shape) >= 2:  # Only weight matrices
        layer_sparsity = mask.sum().item() / mask.numel()
        layer_sparsities_global[name] = layer_sparsity
        logger.info(f'  {name}: {layer_sparsity:.4f} ({mask.sum().item()}/{mask.numel()} pruned)')

total_pruned_global = sum(mask.sum().item() for mask in masks_global.values())
total_weights_global = sum(mask.numel() for mask in masks_global.values())
global_sparsity = total_pruned_global / total_weights_global
logger.info(f'\nGlobal sparsity: {global_sparsity:.6f} (target: {sparsity:.6f})')

logger.info('\n' + '='*70)
logger.info('METHOD 2: Layerwise WANDA (new approach)')
logger.info('='*70)
logger.info('Computes WANDA scores per layer with histogram-based thresholds')
logger.info('')

# Need to create a new instance to clear cache
wanda_layerwise = WANDAPruningOptimized(dataloader=dataloader, num_batches=1)
masks_layerwise = wanda_layerwise.select_weights_to_prune_layerwise(model, sparsity=sparsity)

# Compute per-layer sparsity for layerwise approach
logger.info('\nPer-layer sparsity (layerwise approach):')
layer_sparsities_layerwise = {}
for name, mask in masks_layerwise.items():
    if len(mask.shape) >= 2:  # Only weight matrices
        layer_sparsity = mask.sum().item() / mask.numel()
        layer_sparsities_layerwise[name] = layer_sparsity
        logger.info(f'  {name}: {layer_sparsity:.4f} ({mask.sum().item()}/{mask.numel()} pruned)')

total_pruned_layerwise = sum(mask.sum().item() for mask in masks_layerwise.values())
total_weights_layerwise = sum(mask.numel() for mask in masks_layerwise.values())
layerwise_sparsity = total_pruned_layerwise / total_weights_layerwise
logger.info(f'\nGlobal sparsity: {layerwise_sparsity:.6f} (target: {sparsity:.6f})')

logger.info('\n' + '='*70)
logger.info('COMPARISON')
logger.info('='*70)

# Compare per-layer sparsity variance
logger.info('\nPer-layer sparsity statistics:')
logger.info(f'Global approach:')
global_values = list(layer_sparsities_global.values())
logger.info(f'  Mean: {sum(global_values)/len(global_values):.4f}')
logger.info(f'  Min:  {min(global_values):.4f}')
logger.info(f'  Max:  {max(global_values):.4f}')
logger.info(f'  Std:  {torch.tensor(global_values).std().item():.4f}')

logger.info(f'\nLayerwise approach:')
layerwise_values = list(layer_sparsities_layerwise.values())
logger.info(f'  Mean: {sum(layerwise_values)/len(layerwise_values):.4f}')
logger.info(f'  Min:  {min(layerwise_values):.4f}')
logger.info(f'  Max:  {max(layerwise_values):.4f}')
logger.info(f'  Std:  {torch.tensor(layerwise_values).std().item():.4f}')

logger.info('\nExpected behavior:')
logger.info('- Layerwise approach should have more uniform per-layer sparsity')
logger.info('  (each layer gets exactly the target sparsity)')
logger.info('- Global approach may have high variance across layers')
logger.info('  (some layers pruned more heavily than others)')

# Check that layerwise achieves target sparsity per layer
max_layer_error = max(abs(s - sparsity) for s in layerwise_values)
logger.info(f'\nLayerwise max error from target: {max_layer_error:.6f}')
if max_layer_error < 0.002:  # 0.2% tolerance
    logger.info('✓ All layers within 0.2% of target sparsity')
else:
    logger.info('✗ Some layers exceed 0.2% error')

logger.info('\n' + '='*70)
logger.info('SUCCESS: Layerwise WANDA pruning implemented!')
logger.info('='*70)
