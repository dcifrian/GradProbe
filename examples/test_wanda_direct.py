"""
Direct test of WANDA strategy on Mistral-7B to debug threshold computation.
This tests ONLY the WANDA strategy, not the full iterative pruner.
"""

import torch
import sys
import os
import psutil
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import WANDAPruning

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("transformers library not found. Please install it:")
    print("pip install transformers")
    sys.exit(1)

def get_memory_gb():
    """Get current memory usage in GB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024**3), mem_info.vms / (1024**3)

def print_memory(label):
    """Print current memory usage."""
    rss, vms = get_memory_gb()
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024**3)
        vram_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"[{label}] RAM: {rss:.2f}GB (VMS: {vms:.2f}GB) | VRAM: {vram_allocated:.2f}GB (Reserved: {vram_reserved:.2f}GB)")
    else:
        print(f"[{label}] RAM: {rss:.2f}GB (VMS: {vms:.2f}GB)")

print("="*70)
print("WANDA STRATEGY DIRECT TEST ON MISTRAL-7B")
print("="*70)

print("\nLoading Mistral-7B-Instruct-v0.2...")
print_memory("Before loading")

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"\nModel loaded successfully")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print_memory("After loading model")

# Prepare small calibration dataset for WANDA
print("\nPreparing calibration data for WANDA...")
calibration_text = """The future of artificial intelligence is a topic of great interest and debate.
Machine learning models continue to grow in capability and scale.
Language models can now generate coherent and contextually relevant text.
Computer vision systems can identify objects with remarkable accuracy.
Robotics and automation are transforming manufacturing and logistics."""

tokens = tokenizer.encode(calibration_text, return_tensors='pt')
print(f"Calibration tokens: {tokens.shape[1]}")

# Create simple dataset
from torch.utils.data import DataLoader, TensorDataset
seq_length = 64
input_sequences = []
for i in range(0, tokens.shape[1] - seq_length, 32):
    seq = tokens[:, i:i+seq_length]
    if seq.shape[1] == seq_length:
        input_sequences.append(seq)

print(f"Created {len(input_sequences)} sequences")
dataset = TensorDataset(torch.cat(input_sequences, dim=0))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print_memory("After preparing data")

# Test different sparsity levels
test_sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]

for sparsity in test_sparsities:
    print("\n" + "="*70)
    print(f"TESTING SPARSITY: {sparsity:.1%}")
    print("="*70)

    gc.collect()
    torch.cuda.empty_cache()
    print_memory("Before WANDA strategy")

    # Create WANDA strategy
    wanda = WANDAPruning(dataloader=dataloader, num_batches=3)

    print(f"\nCalling compute_mask with sparsity={sparsity}...")
    print_memory("Before compute_mask")

    # Call compute_mask and capture what happens
    masks = wanda.compute_mask(model, sparsity=sparsity)

    print_memory("After compute_mask")

    # Analyze results
    print(f"\nResults for sparsity={sparsity:.1%}:")
    print(f"  Number of masks created: {len(masks)}")

    # Count actual sparsity
    total_pruned = 0
    total_params_in_masks = 0
    for name, mask in masks.items():
        pruned = mask.sum().item()
        total_pruned += pruned
        total_params_in_masks += mask.numel()

    actual_sparsity = total_pruned / total_params_in_masks if total_params_in_masks > 0 else 0
    error = abs(actual_sparsity - sparsity)

    print(f"  Target sparsity: {sparsity:.4f}")
    print(f"  Actual sparsity: {actual_sparsity:.4f}")
    print(f"  Error: {error:.4f} ({error/sparsity*100:.1f}% relative error)")
    print(f"  Total parameters masked: {total_params_in_masks:,}")
    print(f"  Total parameters pruned: {total_pruned:,}")

    # Clean up
    del masks
    gc.collect()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print_memory("Final")
