"""
Memory-profiled version of test_tinystories.py

This script runs the GradProbe algorithm on TinyStories-33M with detailed
memory profiling to identify where memory is being consumed.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, MagnitudePruning, WANDAPruning
from profile_memory import MemoryProfiler, analyze_model_memory, analyze_gradient_memory, estimate_activation_memory

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("transformers library not found. Please install it:")
    print("pip install transformers")
    sys.exit(1)

# Initialize memory profiler
profiler = MemoryProfiler()

print("="*70)
print("MEMORY PROFILING: GradProbe on TinyStories-33M")
print("="*70)
print()

profiler.snapshot("00_initial")

print("Loading TinyStories-33M model...")
model_name = "roneneldan/TinyStories-33M"
try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying to download...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

profiler.snapshot("01_model_loaded")

print(f"Model loaded successfully")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Analyze model parameter memory
num_params, param_bytes = analyze_model_memory(model)

# Prepare test data - use a smaller dataset for profiling
test_text = """Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the park. She ran to get it and kicked it very hard. The ball flew high into the sky and landed in a tree. Lily was sad because she couldn't reach it. Then, a kind boy came and helped her get the ball down. Lily was happy again and they played together all day.

Once there was a brave knight named Tom. He had a shiny sword and a big horse. One day, Tom went on an adventure to find a dragon. He rode his horse through the dark forest and over the tall mountains. When he found the dragon, it was sleeping in a cave. Tom was very quiet and sneaked past the dragon. He found a treasure chest full of gold coins. Tom took the treasure and rode back home to share it with his family.

There was a happy bunny who lived in the woods. The bunny liked to hop and play with his friends. One sunny morning, the bunny found a big carrot in the garden. He was so excited! He called all his friends to come and see. Together, they shared the carrot and had a wonderful picnic. The bunny and his friends played games until the sun went down. Then they all went home to sleep.

A little boy named Max loved to build things. He had many blocks of different colors. One day, Max decided to build a tall tower. He stacked the blocks very carefully, one on top of another. The tower grew taller and taller until it was as high as Max's head. Max was very proud of his tower. He showed it to his mom and dad, and they clapped their hands. That night, Max dreamed about building even bigger towers."""

# Tokenize
print("\nPreparing data...")
tokens = tokenizer.encode(test_text, return_tensors='pt')
print(f"Total tokens: {tokens.shape[1]}")

# Create dataset with sliding windows - smaller for profiling
seq_length = 128
stride = 64
input_sequences = []
target_sequences = []

for i in range(0, tokens.shape[1] - seq_length - 1, stride):
    input_seq = tokens[:, i:i+seq_length]
    target_seq = tokens[:, i+1:i+seq_length+1]
    if input_seq.shape[1] == seq_length and target_seq.shape[1] == seq_length:
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

print(f"Created {len(input_sequences)} sequences of length {seq_length}")

# Create dataloaders
all_inputs = torch.cat(input_sequences, dim=0)
all_targets = torch.cat(target_sequences, dim=0)
dataset = TensorDataset(all_inputs, all_targets)
dataloader_pruning = DataLoader(dataset, batch_size=1, shuffle=False)

profiler.snapshot("02_data_prepared")

# Estimate activation memory
estimate_activation_memory(model, batch_size=1, seq_length=seq_length)

# Custom loss function for pruning
def loss_fn(outputs, targets):
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs

    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    return loss

print("\n" + "="*70)
print("STARTING GRADPROBE PRUNING - MEMORY PROFILING")
print("="*70)

# We'll instrument the pruning process by manually stepping through it
# This allows us to profile each stage

# Initialize pruner
print("\nInitializing GradProbe with Magnitude Pruning strategy...")
pruner = GradProbe(model, MagnitudePruning(), device='cpu')
profiler.snapshot("03_pruner_initialized")

sparsity = 0.1  # Small sparsity for profiling
num_batches = 5  # Small number of batches for profiling
reduction_factor = 0.1
gradient_threshold = 2.0

print(f"\nPruning configuration:")
print(f"  Target sparsity: {sparsity:.2%}")
print(f"  Num batches: {num_batches}")
print(f"  Reduction factor: {reduction_factor}")
print(f"  Gradient threshold: {gradient_threshold}")

# Save original model state
print("\nSaving original model state...")
original_state = {
    name: param.data.clone() for name, param in model.named_parameters()
}
profiler.snapshot("04_saved_original_state")

# Step 1: Compute original gradients
print("\nStep 1: Computing gradients with original model...")
print("  (This stores gradients for all parameters)")

# Manually implement gradient computation with profiling
dropout_states = {}
for name, module in model.named_modules():
    if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
        dropout_states[name] = module.training
        module.eval()

gradients = {name: torch.zeros_like(param)
             for name, param in model.named_parameters()
             if param.requires_grad}

profiler.snapshot("05_allocated_gradient_storage")

# Analyze gradient storage memory
print("\nAnalyzing gradient storage:")
grad_bytes = analyze_gradient_memory(gradients)
print(f"\nNote: We're storing TWO sets of gradients (original + modified)")
print(f"Expected gradient memory: {2 * grad_bytes / 1024 / 1024:.2f} MB")

batch_count = 0
for batch_idx, batch in enumerate(dataloader_pruning):
    if batch_count >= num_batches:
        break

    # Zero gradients
    model.zero_grad()

    # Handle batch
    inputs, targets = batch[0], batch[1]

    # Forward pass
    print(f"  Processing batch {batch_count + 1}/{num_batches}...")
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    profiler.snapshot(f"06_forward_pass_batch_{batch_count}")

    # Backward pass
    loss.backward()

    profiler.snapshot(f"07_backward_pass_batch_{batch_count}")

    # Accumulate gradients
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if batch_count == 0:
                gradients[name] = param.grad.data.abs()
            else:
                gradients[name] = torch.maximum(gradients[name], param.grad.data.abs())

    batch_count += 1

    # Only profile first 2 batches in detail
    if batch_count >= 2:
        break

profiler.snapshot("08_original_gradients_computed")

# Restore dropout states
for name, module in model.named_modules():
    if name in dropout_states:
        if dropout_states[name]:
            module.train()

print("\nOriginal gradients computed and stored")
print(f"Gradient storage: {len(gradients)} tensors")

# Step 2: Select weights to prune
print("\nStep 2: Selecting weights to prune using Magnitude strategy...")
tentative_masks = pruner.strategy.select_weights_to_prune(model, sparsity)
profiler.snapshot("09_tentative_masks_created")

total_tentative = sum(mask.sum().item() for mask in tentative_masks.values())
print(f"Tentative pruning candidates: {total_tentative}")

# Analyze mask memory
print("\nAnalyzing pruning mask memory:")
mask_bytes = 0
for name, mask in tentative_masks.items():
    mask_bytes += mask.element_size() * mask.numel()
print(f"Mask storage: {mask_bytes / 1024 / 1024:.2f} MB")
print(f"Note: We store masks for all parameters")

# Step 3: Reduce tentative weights
print(f"\nStep 3: Reducing tentative weights to {reduction_factor}x...")
for name, param in model.named_parameters():
    if name in tentative_masks:
        mask = tentative_masks[name]
        param.data[mask] *= reduction_factor

profiler.snapshot("10_weights_reduced")

# Step 4: Compute modified gradients
print("\nStep 4: Computing gradients with reduced weights...")
print("  (This stores a SECOND set of gradients)")

# Initialize storage for modified gradients
modified_gradients = {name: torch.zeros_like(param)
                      for name, param in model.named_parameters()
                      if param.requires_grad}

profiler.snapshot("11_allocated_modified_gradient_storage")

print("\nNOTE: At this point we have:")
print(f"  - Original model weights: {param_bytes / 1024 / 1024:.2f} MB")
print(f"  - Original gradients: {grad_bytes / 1024 / 1024:.2f} MB")
print(f"  - Modified gradients: {grad_bytes / 1024 / 1024:.2f} MB")
print(f"  - Pruning masks: {mask_bytes / 1024 / 1024:.2f} MB")
print(f"  - Saved original state: {param_bytes / 1024 / 1024:.2f} MB")
print(f"  TOTAL (weights + gradients): {(2 * param_bytes + 2 * grad_bytes + mask_bytes) / 1024 / 1024:.2f} MB")

# Compute modified gradients (just first 2 batches for profiling)
batch_count = 0
for batch_idx, batch in enumerate(dataloader_pruning):
    if batch_count >= 2:
        break

    model.zero_grad()
    inputs, targets = batch[0], batch[1]

    print(f"  Processing batch {batch_count + 1}/2...")
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    profiler.snapshot(f"12_modified_forward_batch_{batch_count}")

    loss.backward()

    profiler.snapshot(f"13_modified_backward_batch_{batch_count}")

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if batch_count == 0:
                modified_gradients[name] = param.grad.data.abs()
            else:
                modified_gradients[name] = torch.maximum(modified_gradients[name], param.grad.data.abs())

    batch_count += 1

profiler.snapshot("14_modified_gradients_computed")

# Step 5: Compare gradients and decide
print("\nStep 5: Comparing gradients and making pruning decisions...")

final_masks = {}
for name in tentative_masks:
    if name in gradients and name in modified_gradients:
        tentative_mask = tentative_masks[name]
        orig_grad = gradients[name]
        mod_grad = modified_gradients[name]

        gradient_below_threshold = mod_grad <= orig_grad * (1.0 + gradient_threshold)
        final_mask = tentative_mask & gradient_below_threshold
        final_masks[name] = final_mask

profiler.snapshot("15_final_masks_computed")

# Step 6: Apply final pruning
print("\nStep 6: Applying final pruning...")

# Restore weights
for name, param in model.named_parameters():
    if name in original_state:
        param.data.copy_(original_state[name])

# Apply masks
for name, param in model.named_parameters():
    if name in final_masks:
        mask = final_masks[name]
        param.data[mask] = 0

profiler.snapshot("16_final_pruning_applied")

# Print final statistics
total_params = sum(p.numel() for p in model.parameters())
zero_params = sum((p.data == 0).sum().item() for p in model.parameters())
final_sparsity = zero_params / total_params

print(f"\nFinal sparsity: {final_sparsity:.2%}")
print(f"Pruned {zero_params:,} out of {total_params:,} parameters")

# Print full profiling results
print("\n" + "="*70)
print("MEMORY PROFILING RESULTS")
print("="*70)

profiler.print_summary()

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)
print(f"\n1. Model Parameters: {param_bytes / 1024 / 1024:.2f} MB")
print(f"   - Stored once in model")
print(f"   - Copied once in 'original_state' dict")
print(f"   - TOTAL: {2 * param_bytes / 1024 / 1024:.2f} MB")

print(f"\n2. Gradients: {grad_bytes / 1024 / 1024:.2f} MB per set")
print(f"   - Original gradients: {grad_bytes / 1024 / 1024:.2f} MB")
print(f"   - Modified gradients: {grad_bytes / 1024 / 1024:.2f} MB")
print(f"   - TOTAL: {2 * grad_bytes / 1024 / 1024:.2f} MB")

print(f"\n3. Pruning Masks: {mask_bytes / 1024 / 1024:.2f} MB")
print(f"   - Boolean masks for all parameters")

print(f"\n4. Activations (per forward pass):")
print(f"   - Estimated from model config")
print(f"   - Not stored between passes, but used during forward/backward")

expected_total = (2 * param_bytes + 2 * grad_bytes + mask_bytes) / 1024 / 1024
print(f"\nEXPECTED MINIMUM MEMORY (without activations): {expected_total:.2f} MB")

# Get actual peak
peak_snapshot = max(profiler.snapshots, key=lambda s: s.process_rss_mb)
print(f"ACTUAL PEAK MEMORY: {peak_snapshot.process_rss_mb:.2f} MB")
print(f"  at stage: {peak_snapshot.stage}")

overhead = peak_snapshot.process_rss_mb - expected_total
print(f"\nOVERHEAD (activations + Python + OS): {overhead:.2f} MB")

print("\n" + "="*70)
print("DETAILED SNAPSHOTS")
print("="*70)
profiler.print_all_snapshots()

print("\n" + "="*70)
print("PROFILING COMPLETE")
print("="*70)
