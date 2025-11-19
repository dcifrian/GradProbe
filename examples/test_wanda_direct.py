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

from gradprobe import WANDAPruning, Logger, LogLevel

logger = Logger(program_name='test_wanda_direct', level=LogLevel.INFO)

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    logger.error("transformers library not found. Please install it:")
    logger.error("pip install transformers")
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
        logger.memory(f"[{label}] RAM: {rss:.2f}GB (VMS: {vms:.2f}GB) | VRAM: {vram_allocated:.2f}GB (Reserved: {vram_reserved:.2f}GB)")
    else:
        logger.memory(f"[{label}] RAM: {rss:.2f}GB (VMS: {vms:.2f}GB)")

logger.info("="*70)
logger.info("WANDA STRATEGY DIRECT TEST ON MISTRAL-7B")
logger.info("="*70)

logger.info("\nLoading Mistral-7B...")
print_memory("Before loading")

model_name = "mistralai/Mistral-7B-v0.3"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

logger.info(f"\nModel loaded successfully")
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Total parameters: {total_params:,}")
print_memory("After loading model")

# Prepare small calibration dataset for WANDA
logger.info("\nPreparing calibration data for WANDA...")
calibration_text = """The future of artificial intelligence is a topic of great interest and debate.
Machine learning models continue to grow in capability and scale.
Language models can now generate coherent and contextually relevant text.
Computer vision systems can identify objects with remarkable accuracy.
Robotics and automation are transforming manufacturing and logistics."""

tokens = tokenizer.encode(calibration_text, return_tensors='pt')
logger.info(f"Calibration tokens: {tokens.shape[1]}")

# Create simple dataset
from torch.utils.data import DataLoader, TensorDataset
seq_length = 64
input_sequences = []
for i in range(0, tokens.shape[1] - seq_length, 32):
    seq = tokens[:, i:i+seq_length]
    if seq.shape[1] == seq_length:
        input_sequences.append(seq)

logger.info(f"Created {len(input_sequences)} sequences")
dataset = TensorDataset(torch.cat(input_sequences, dim=0))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print_memory("After preparing data")

# Test different sparsity levels
test_sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]

for sparsity in test_sparsities:
    logger.info("\n" + "="*70)
    logger.info(f"TESTING SPARSITY: {sparsity:.1%}")
    logger.info("="*70)

    gc.collect()
    torch.cuda.empty_cache()
    print_memory("Before WANDA strategy")

    # Create WANDA strategy
    wanda = WANDAPruning(dataloader=dataloader, num_batches=3)

    logger.info(f"\nCalling select_weights_to_prune with sparsity={sparsity}...")
    print_memory("Before select_weights_to_prune")

    # Call select_weights_to_prune and capture what happens
    masks = wanda.select_weights_to_prune(model, sparsity=sparsity)

    print_memory("After select_weights_to_prune")

    # Analyze results
    logger.info(f"\nResults for sparsity={sparsity:.1%}:")
    logger.info(f"  Number of masks created: {len(masks)}")

    # Count actual sparsity
    total_pruned = 0
    total_params_in_masks = 0
    for name, mask in masks.items():
        pruned = mask.sum().item()
        total_pruned += pruned
        total_params_in_masks += mask.numel()

    actual_sparsity = total_pruned / total_params_in_masks if total_params_in_masks > 0 else 0
    error = abs(actual_sparsity - sparsity)

    logger.info(f"  Target sparsity: {sparsity:.4f}")
    logger.info(f"  Actual sparsity: {actual_sparsity:.4f}")
    logger.info(f"  Error: {error:.4f} ({error/sparsity*100:.1f}% relative error)")
    logger.info(f"  Total parameters masked: {total_params_in_masks:,}")
    logger.info(f"  Total parameters pruned: {total_pruned:,}")

    # Clean up
    del masks
    gc.collect()

logger.info("\n" + "="*70)
logger.info("TEST COMPLETE")
logger.info("="*70)
print_memory("Final")
