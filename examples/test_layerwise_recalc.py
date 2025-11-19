"""
Test the restored layerwise approach that recalculates WANDA after each layer.

This should replicate the old successful approach (f320eb98) but with fast histogram-based WANDA.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from gradprobe import GradProbe, WANDAPruning, Logger, LogLevel

logger = Logger(program_name='test_layerwise_recalc', level=LogLevel.INFO)

logger.info('='*70)
logger.info('LAYERWISE WANDA WITH RECALCULATION TEST')
logger.info('='*70)

# Load model
logger.info('\nLoading TinyStories-33M...')
model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
tokenizer = AutoTokenizer.from_pretrained('roneneldan/TinyStories-33M')

# Simple calibration data
text = 'Once upon a time, there was a little girl named Lily.'
tokens = tokenizer(text, return_tensors='pt')
input_ids = tokens['input_ids']
dataset = [(input_ids, input_ids)]  # (input, target) tuples
dataloader = DataLoader(dataset, batch_size=1)

# Loss function (must match pruner's expected signature: loss_fn(outputs, targets))
def loss_fn(outputs, targets):
    # outputs is the model output, targets is the shifted input
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    # Shift targets for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

# Create pruner with WANDA strategy
wanda = WANDAPruning(dataloader=dataloader, num_batches=1)
pruner = GradProbe(model=model, strategy=wanda)

# Test layerwise pruning at 30% sparsity
logger.info('\nPruning layerwise at 30% sparsity...')
logger.info('This will recalculate WANDA after each layer is pruned.')
logger.info('')

masks = pruner.prune_layerwise(
    dataloader=dataloader,
    loss_fn=loss_fn,
    sparsity=0.3,
    num_batches=1,
    reduction_factor=0.1,
    gradient_threshold=0.0,
    verbose=True,
    layer_order="reverse"
)

# Check results
total_pruned = sum(mask.sum().item() for mask in masks.values())
total_weights = sum(mask.numel() for mask in masks.values())
actual_sparsity = total_pruned / total_weights

logger.info(f'\n' + '='*70)
logger.info(f'RESULTS')
logger.info(f'='*70)
logger.info(f'Total pruned: {total_pruned:,} / {total_weights:,}')
logger.info(f'Actual sparsity: {actual_sparsity:.4f} (target: 0.3000)')

# Check per-layer variance
layer_sparsities = []
for name, mask in masks.items():
    if len(mask.shape) >= 2:
        layer_sparsity = mask.sum().item() / mask.numel()
        layer_sparsities.append(layer_sparsity)

if layer_sparsities:
    import statistics
    logger.info(f'\nPer-layer sparsity stats:')
    logger.info(f'  Mean: {statistics.mean(layer_sparsities):.4f}')
    logger.info(f'  Std:  {statistics.stdev(layer_sparsities):.4f}')
    logger.info(f'  Min:  {min(layer_sparsities):.4f}')
    logger.info(f'  Max:  {max(layer_sparsities):.4f}')

logger.info(f'\n' + '='*70)
logger.info('SUCCESS: Layerwise pruning with recalculation complete!')
logger.info('='*70)
