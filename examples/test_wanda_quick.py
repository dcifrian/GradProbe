"""
Quick test to verify WANDA works on TinyStories.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, WANDAPruning
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading TinyStories-33M model...")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Prepare minimal data
test_text = "Once upon a time, there was a little girl. " * 50
tokens = tokenizer.encode(test_text, return_tensors='pt')
print(f"Tokens: {tokens.shape[1]}")

# Create sequences
seq_length = 64
sequences = []
for i in range(0, min(tokens.shape[1] - seq_length - 1, 100), 32):
    input_seq = tokens[:, i:i+seq_length]
    target_seq = tokens[:, i+1:i+seq_length+1]
    if input_seq.shape[1] == seq_length and target_seq.shape[1] == seq_length:
        sequences.append((input_seq, target_seq))

print(f"Created {len(sequences)} sequences")

# Create dataloader
all_inputs = torch.cat([s[0] for s in sequences], dim=0)
all_targets = torch.cat([s[1] for s in sequences], dim=0)
dataset = TensorDataset(all_inputs, all_targets)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Test WANDA strategy
print("\nTesting WANDA activation collection...")
wanda_strategy = WANDAPruning(dataloader=dataloader, num_batches=2)

# Try to select weights to prune
print("Testing select_weights_to_prune...")
try:
    masks = wanda_strategy.select_weights_to_prune(model, sparsity=0.1)
    print(f"✓ Success! Generated {len(masks)} masks")

    # Count total weights to prune
    total_pruned = sum(mask.sum().item() for mask in masks.values())
    total_weights = sum(mask.numel() for mask in masks.values())
    actual_sparsity = total_pruned / total_weights if total_weights > 0 else 0
    print(f"✓ Actual sparsity: {actual_sparsity:.2%}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
