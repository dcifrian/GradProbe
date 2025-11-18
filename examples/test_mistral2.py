"""
Test Mistral-7B with iterative WANDA pruning.

Based on profile_mistral_wanda.py but using iterative_prune() instead of prune()
to test multiple sparsity levels while keeping the same efficient configuration.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import time
import psutil
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, WANDAPruning

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("transformers library not found. Please install it:")
    print("pip install transformers")
    sys.exit(1)

# Configuration - match profile_mistral_wanda.py
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTH = 128
NUM_BATCHES_GRADIENT = 5  # Same as profiler
NUM_SEQUENCES = 3  # Same as profiler

# Iterative pruning configuration
INITIAL_SPARSITY = 0.1
SPARSITY_STEP = 0.1
MAX_ACCURACY_DROP = 1.0  # 1% perplexity increase allowed

print("="*70)
print("GRADPROBE - MISTRAL-7B ITERATIVE WANDA PRUNING TEST")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Sequence length: {SEQ_LENGTH}")
print(f"Num sequences: {NUM_SEQUENCES}")
print(f"Num batches: {NUM_BATCHES_GRADIENT}")
print(f"Initial sparsity: {INITIAL_SPARSITY:.0%}")
print(f"Sparsity step: {SPARSITY_STEP:.0%}")
print(f"Max accuracy drop: {MAX_ACCURACY_DROP:.1f}%")
print("="*70)

# Memory monitoring helper
def print_memory(label):
    """Print current memory usage."""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024 / 1024
    vms_mb = process.memory_info().vms / 1024 / 1024

    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
        vram_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"[{label}] RAM: {ram_mb:.0f}MB, VMS: {vms_mb:.0f}MB, "
              f"VRAM: {vram_mb:.0f}MB, VRAM Reserved: {vram_reserved_mb:.0f}MB")
    else:
        print(f"[{label}] RAM: {ram_mb:.0f}MB, VMS: {vms_mb:.0f}MB")

# Load model
print(f"\nLoading Mistral-7B...")
print_memory("Before model load")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Model in FP16
    low_cpu_mem_usage=True,
    device_map="auto" if DEVICE == "cuda" else None
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print_memory("After model load")

# Prepare calibration data - same as profiler
CALIBRATION_TEXT = """The quick brown fox jumps over the lazy dog. Machine learning has revolutionized how we interact with computers. These models learn patterns from vast amounts of text data. The transformer architecture marked a significant breakthrough in natural language processing. Climate change represents one of the most pressing challenges facing humanity.

The history of computing dates back to ancient times when humans first created tools for calculation. The abacus was invented thousands of years ago. In the 19th century, Charles Babbage designed the Analytical Engine, a mechanical general-purpose computer. Although never completed in his lifetime, Babbage's designs laid the groundwork for modern computers.

Throughout history, explorers have ventured into the unknown, driven by curiosity and the desire to discover new lands and cultures. From ancient Polynesian navigators crossing vast oceans to modern astronauts venturing into space, the spirit of exploration has been a defining characteristic of human civilization.

Mathematics is often called the language of the universe. From the elegant simplicity of Euclidean geometry to the abstract complexities of modern algebra and topology, mathematics provides tools for understanding patterns, structures, and relationships in both the natural and abstract worlds."""

print(f"\nPreparing calibration data...")
print_memory("Before data prep")

tokens = tokenizer.encode(CALIBRATION_TEXT, return_tensors='pt')
print(f"Total tokens: {tokens.shape[1]}")

# Create sequences - same logic as profiler
input_sequences = []
target_sequences = []
stride = SEQ_LENGTH // 2

for i in range(0, min(tokens.shape[1] - SEQ_LENGTH - 1, NUM_SEQUENCES * stride), stride):
    input_seq = tokens[:, i:i+SEQ_LENGTH]
    target_seq = tokens[:, i+1:i+SEQ_LENGTH+1]
    if input_seq.shape[1] == SEQ_LENGTH and target_seq.shape[1] == SEQ_LENGTH:
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
        if len(input_sequences) >= NUM_SEQUENCES:
            break

if len(input_sequences) == 0:
    print(f"\nWarning: Text too short for {NUM_SEQUENCES} sequences of length {SEQ_LENGTH}")
    print(f"Creating single sequence from available text...")
    # Pad if necessary
    if tokens.shape[1] < SEQ_LENGTH + 1:
        padding_needed = SEQ_LENGTH + 1 - tokens.shape[1]
        tokens = torch.cat([tokens, tokens[:, :padding_needed]], dim=1)
    input_sequences = [tokens[:, :SEQ_LENGTH]]
    target_sequences = [tokens[:, 1:SEQ_LENGTH+1]]

print(f"Created {len(input_sequences)} sequences")

all_inputs = torch.cat(input_sequences, dim=0)
all_targets = torch.cat(target_sequences, dim=0)
dataset = TensorDataset(all_inputs, all_targets)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print_memory("After data prep")

# Store for evaluation (use slices to avoid duplicate memory)
eval_inputs = all_inputs[:min(5, len(all_inputs))]
eval_targets = all_targets[:min(5, len(all_inputs))]

# Loss function
def loss_fn(outputs, targets):
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

# Evaluation function - compute perplexity
def eval_perplexity(m):
    """Evaluate model perplexity on eval sequences."""
    m.eval()
    total_loss = 0
    total_tokens = 0
    device = next(m.parameters()).device

    with torch.no_grad():
        # Use eval sequences
        input_batch = eval_inputs.to(device)
        target_batch = eval_targets.to(device)

        outputs = m(input_batch)
        logits = outputs.logits

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_batch.reshape(-1),
            reduction='sum'
        )

        total_loss += loss.item()
        total_tokens += target_batch.numel()

    perplexity = math.exp(total_loss / total_tokens)
    return perplexity

# Wrapper for iterative_prune - return negative perplexity
# This makes the stopping logic work: increases in perplexity = decreases in score
def eval_fn_adjusted(m):
    perplexity = eval_perplexity(m)
    return -perplexity

# Initialize pruner with all optimizations (same as profiler)
print(f"\nInitializing GradProbe with WANDA strategy...")
print_memory("Before pruner init")

pruner = GradProbe(
    model,
    WANDAPruning(dataloader, num_batches=NUM_BATCHES_GRADIENT),
    device=DEVICE,
    low_memory_mode=True,
    use_fp16=True,
    use_gradient_checkpointing=True
)

print(f"Pruner initialized")
print_memory("After pruner init")

# Measure baseline perplexity
print("\n" + "="*70)
print("Measuring baseline perplexity...")
baseline_perplexity = eval_perplexity(model)
print(f"Baseline perplexity: {baseline_perplexity:.2f}")
print("="*70)

# Run iterative pruning
print(f"\nStarting iterative pruning...")
print(f"Initial sparsity: {INITIAL_SPARSITY:.0%}, step: {SPARSITY_STEP:.0%}")
print(f"Max allowed perplexity increase: {MAX_ACCURACY_DROP:.1f}%")
print()

start_time = time.time()
print_memory("Before pruning")

# Note: layerwise=False by default, same as profiler
results = pruner.iterative_prune(
    dataloader=dataloader,
    loss_fn=loss_fn,
    eval_fn=eval_fn_adjusted,  # Use adjusted function that returns -perplexity
    initial_sparsity=INITIAL_SPARSITY,
    sparsity_step=SPARSITY_STEP,
    max_accuracy_drop=MAX_ACCURACY_DROP,
    num_batches=NUM_BATCHES_GRADIENT,
    reduction_factor=0.1,
    gradient_threshold=0.5,  # Changed from 0.0 (too restrictive)
    layerwise=False,  # Explicitly set to False like profiler
    verbose=True,
    compare_baseline=True
)

elapsed_time = time.time() - start_time
print_memory("After pruning")

print(f"\nIterative pruning completed in {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")

# Print results
print("\n" + "="*70)
print("PRUNING RESULTS")
print("="*70)

# Note: results contain negative perplexity, so negate back for display
print(f"\nBaseline perplexity: {-results['initial_accuracy']:.2f}")
print(f"Final perplexity: {-results['final_accuracy']:.2f}")
print(f"Perplexity change: {-results['final_accuracy'] - (-results['initial_accuracy']):.2f}")

total_params = sum(p.numel() for p in model.parameters())
zero_params = sum((p.data == 0).sum().item() for p in model.parameters())
final_sparsity = zero_params / total_params

print(f"\nFinal sparsity: {final_sparsity:.2%}")
print(f"Pruned {zero_params:,} out of {total_params:,} parameters")

if 'sparsity_history' in results:
    print("\nSparsity progression:")
    for i, (sparsity, acc) in enumerate(zip(results['sparsity_history'], results['accuracy_history'])):
        # acc is negative perplexity, so negate it back for display
        print(f"  Step {i+1}: {sparsity:.1%} sparsity -> perplexity {-acc:.2f}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print_memory("Final")
