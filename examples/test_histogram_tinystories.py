"""
Quick test of histogram-based threshold on TinyStories.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import MagnitudePruning, WANDAPruning, Logger, LogLevel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

# Initialize logger
logger = Logger(program_name='test_histogram_tinystories', level=LogLevel.INFO)

logger.info("="*70)
logger.info("HISTOGRAM THRESHOLD TEST ON TINYSTORIES-33M")
logger.info("="*70)

logger.info("\nLoading TinyStories-33M...")
model_name = "roneneldan/TinyStories-33M"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Model loaded: {total_params:,} parameters")

# Prepare tiny calibration dataset
text = "Once upon a time, there was a little girl named Lily."
tokens = tokenizer.encode(text, return_tensors='pt')
dataset = TensorDataset(tokens)
dataloader = DataLoader(dataset, batch_size=1)

logger.info("\nTesting Magnitude strategy...")
magnitude = MagnitudePruning()
masks_mag = magnitude.select_weights_to_prune(model, sparsity=0.1)

# Count actual sparsity
total_pruned = sum(mask.sum().item() for mask in masks_mag.values())
total_in_masks = sum(mask.numel() for mask in masks_mag.values())
actual_sparsity = total_pruned / total_in_masks

logger.info(f"Magnitude: target=0.1, actual={actual_sparsity:.6f}, error={abs(actual_sparsity - 0.1):.6f}")
assert abs(actual_sparsity - 0.1) < 0.01, f"Magnitude sparsity error too high: {abs(actual_sparsity - 0.1)}"

logger.info("\nTesting WANDA strategy...")
wanda = WANDAPruning(dataloader=dataloader, num_batches=1)
masks_wanda = wanda.select_weights_to_prune(model, sparsity=0.1)

# Count actual sparsity
total_pruned = sum(mask.sum().item() for mask in masks_wanda.values())
total_in_masks = sum(mask.numel() for mask in masks_wanda.values())
actual_sparsity = total_pruned / total_in_masks

logger.info(f"WANDA: target=0.1, actual={actual_sparsity:.6f}, error={abs(actual_sparsity - 0.1):.6f}")
assert abs(actual_sparsity - 0.1) < 0.01, f"WANDA sparsity error too high: {abs(actual_sparsity - 0.1)}"

logger.info("\n" + "="*70)
logger.info("SUCCESS: Both strategies working correctly on TinyStories!")
logger.info("="*70)
