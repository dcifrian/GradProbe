"""
Memory-profiled version of test_tinystories.py with optimizations ENABLED

This script tests the memory improvements from:
1. FP16 gradients and saved state (2x reduction)
2. Layer-by-layer gradient streaming (eliminates dual gradient storage)
3. Gradient checkpointing (reduces activation memory)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, MagnitudePruning
from gradprobe.logger import Logger, LogLevel
from profile_memory import MemoryProfiler, analyze_model_memory, estimate_activation_memory

# Initialize logger
logger = Logger(program_name='profile_tinystories_optimized', level=LogLevel.INFO)

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    logger.error("transformers library not found. Please install it:")
    logger.error("pip install transformers")
    sys.exit(1)

# Initialize memory profiler
profiler = MemoryProfiler()

logger.info("="*70)
logger.info("MEMORY PROFILING: GradProbe with OPTIMIZATIONS ENABLED")
logger.info("="*70)
logger.info("Optimizations:")
logger.info("  ✓ FP16 gradients and saved state (2x memory reduction)")
logger.info("  ✓ Layer-by-layer gradient streaming (eliminates dual storage)")
logger.info("  ✓ Gradient checkpointing (reduces activation memory)")
logger.info("="*70)
logger.info()

profiler.snapshot("00_initial")

logger.info("Loading TinyStories-33M model...")
model_name = "roneneldan/TinyStories-33M"
try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.info("Trying to download...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

profiler.snapshot("01_model_loaded")

logger.info(f"Model loaded successfully")
logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Analyze model parameter memory
num_params, param_bytes = analyze_model_memory(model)

# Prepare test data
test_text = """Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the park. She ran to get it and kicked it very hard. The ball flew high into the sky and landed in a tree. Lily was sad because she couldn't reach it. Then, a kind boy came and helped her get the ball down. Lily was happy again and they played together all day.

Once there was a brave knight named Tom. He had a shiny sword and a big horse. One day, Tom went on an adventure to find a dragon. He rode his horse through the dark forest and over the tall mountains. When he found the dragon, it was sleeping in a cave. Tom was very quiet and sneaked past the dragon. He found a treasure chest full of gold coins. Tom took the treasure and rode back home to share it with his family.

There was a happy bunny who lived in the woods. The bunny liked to hop and play with his friends. One sunny morning, the bunny found a big carrot in the garden. He was so excited! He called all his friends to come and see. Together, they shared the carrot and had a wonderful picnic. The bunny and his friends played games until the sun went down. Then they all went home to sleep.

A little boy named Max loved to build things. He had many blocks of different colors. One day, Max decided to build a tall tower. He stacked the blocks very carefully, one on top of another. The tower grew taller and taller until it was as high as Max's head. Max was very proud of his tower. He showed it to his mom and dad, and they clapped their hands. That night, Max dreamed about building even bigger towers."""

# Tokenize
logger.info("\nPreparing data...")
tokens = tokenizer.encode(test_text, return_tensors='pt')
logger.info(f"Total tokens: {tokens.shape[1]}")

# Create dataset with sliding windows
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

logger.info(f"Created {len(input_sequences)} sequences of length {seq_length}")

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

logger.info("\n" + "="*70)
logger.info("STARTING GRADPROBE PRUNING WITH OPTIMIZATIONS")
logger.info("="*70)

# Initialize pruner with ALL optimizations enabled
logger.info("\nInitializing GradProbe with optimizations...")
pruner = GradProbe(
    model,
    MagnitudePruning(),
    device='cpu',
    low_memory_mode=True,  # Enables layer-by-layer streaming
    use_fp16=True,  # FP16 gradients and saved state
    use_gradient_checkpointing=True  # Gradient checkpointing
)
profiler.snapshot("03_pruner_initialized")

sparsity = 0.1
num_batches = 5
reduction_factor = 0.1
gradient_threshold = 2.0

logger.info(f"\nPruning configuration:")
logger.info(f"  Target sparsity: {sparsity:.2%}")
logger.info(f"  Num batches: {num_batches}")
logger.info(f"  Reduction factor: {reduction_factor}")
logger.info(f"  Gradient threshold: {gradient_threshold}")

profiler.snapshot("04_before_pruning")

# Run pruning with optimizations
logger.info("\nRunning pruning...")
final_masks = pruner.prune(
    dataloader=dataloader_pruning,
    loss_fn=loss_fn,
    sparsity=sparsity,
    num_batches=num_batches,
    reduction_factor=reduction_factor,
    gradient_threshold=gradient_threshold,
    verbose=True
)

profiler.snapshot("05_after_pruning")

# Print final statistics
total_params = sum(p.numel() for p in model.parameters())
zero_params = sum((p.data == 0).sum().item() for p in model.parameters())
final_sparsity = zero_params / total_params

logger.info(f"\nFinal sparsity: {final_sparsity:.2%}")
logger.info(f"Pruned {zero_params:,} out of {total_params:,} parameters")

# Print profiling results
logger.info("\n" + "="*70)
logger.info("MEMORY PROFILING RESULTS - WITH OPTIMIZATIONS")
logger.info("="*70)

profiler.print_summary()

# Calculate expected memory savings
logger.info("\n" + "="*70)
logger.info("MEMORY COMPARISON")
logger.info("="*70)

# Original implementation
original_weights = 2 * param_bytes  # model + saved_state
original_gradients = 2 * param_bytes  # original + modified gradients (both full model size)
original_masks = param_bytes / 4  # boolean masks
original_total = original_weights + original_gradients + original_masks

# Optimized implementation
opt_weights = param_bytes  # model
opt_saved_state = param_bytes / 2 if pruner.use_fp16 else param_bytes  # fp16 saved state
opt_gradients = (param_bytes / len([p for p in model.parameters() if len(p.shape) >= 2])) / 2  # one layer at a time, in fp16
opt_masks = param_bytes / 4  # boolean masks
opt_total = opt_weights + opt_saved_state + opt_gradients + opt_masks

logger.info(f"\nORIGINAL IMPLEMENTATION (without optimizations):")
logger.memory(f"  Model weights: {param_bytes / 1024 / 1024:.2f} MB")
logger.memory(f"  Saved state (fp32): {param_bytes / 1024 / 1024:.2f} MB")
logger.memory(f"  Original gradients (fp32, all layers): {param_bytes / 1024 / 1024:.2f} MB")
logger.memory(f"  Modified gradients (fp32, all layers): {param_bytes / 1024 / 1024:.2f} MB")
logger.memory(f"  Masks: {(param_bytes / 4) / 1024 / 1024:.2f} MB")
logger.memory(f"  TOTAL (before activations): {original_total / 1024 / 1024:.2f} MB")

logger.info(f"\nOPTIMIZED IMPLEMENTATION (with all optimizations):")
logger.memory(f"  Model weights: {param_bytes / 1024 / 1024:.2f} MB")
logger.memory(f"  Saved state (fp16): {opt_saved_state / 1024 / 1024:.2f} MB")
logger.memory(f"  Gradients (fp16, ONE layer at a time): {opt_gradients / 1024 / 1024:.2f} MB")
logger.memory(f"  Masks: {(param_bytes / 4) / 1024 / 1024:.2f} MB")
logger.memory(f"  TOTAL (before activations): {opt_total / 1024 / 1024:.2f} MB")

logger.info(f"\nMEMORY SAVINGS:")
logger.memory(f"  Reduction: {original_total / 1024 / 1024:.2f} MB → {opt_total / 1024 / 1024:.2f} MB")
logger.memory(f"  Savings: {(original_total - opt_total) / 1024 / 1024:.2f} MB ({(1 - opt_total/original_total)*100:.1f}%)")

# Scaling projection
mistral_params = 7_000_000_000
mistral_param_bytes = mistral_params * 4  # fp32
mistral_original_total = (mistral_param_bytes * 4.25) / 1024 / 1024 / 1024  # 4.25x before activations
mistral_opt_total = (mistral_param_bytes * 1.65) / 1024 / 1024 / 1024  # ~1.65x with optimizations

logger.info(f"\nSCALING TO MISTRAL-7B:")
logger.memory(f"  Original implementation: ~{mistral_original_total:.1f} GB (before activations)")
logger.memory(f"  Optimized implementation: ~{mistral_opt_total:.1f} GB (before activations)")
logger.memory(f"  Savings: ~{mistral_original_total - mistral_opt_total:.1f} GB ({(1 - mistral_opt_total/mistral_original_total)*100:.1f}%)")

logger.info("\n" + "="*70)
logger.info("PROFILING COMPLETE")
logger.info("="*70)
