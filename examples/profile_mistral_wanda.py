"""
Performance and memory profiling for Mistral-7B with GradProbe WANDA strategy.

This script profiles:
- Time per layer and overall runtime
- RAM usage (system memory)
- VRAM usage (GPU memory)
- GPU utilization
- Bottleneck identification
- WANDA-specific activation collection overhead
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import time
import psutil
from collections import defaultdict
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, WANDAPruning
from gradprobe.logger import Logger, LogLevel

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    logger = Logger(program_name='profile_mistral_wanda', level=LogLevel.INFO)
    logger.error("transformers library not found. Please install it:")
    logger.error("pip install transformers")
    sys.exit(1)

# Initialize logger
logger = Logger(program_name='profile_mistral_wanda', level=LogLevel.INFO)

@dataclass
class PerformanceMetrics:
    """Store performance metrics for a single operation."""
    operation: str
    layer_name: str
    duration_s: float
    ram_mb: float
    vram_mb: float
    gpu_util_percent: float = 0.0

class PerformanceProfiler:
    """Profile performance and memory usage during pruning."""

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()

    def get_ram_usage_mb(self):
        """Get current RAM usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_vram_usage_mb(self):
        """Get current VRAM usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def get_gpu_utilization(self):
        """Get GPU utilization percentage (requires pynvml)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except:
            return 0.0

    def record(self, operation: str, layer_name: str, duration_s: float):
        """Record a performance metric."""
        metric = PerformanceMetrics(
            operation=operation,
            layer_name=layer_name,
            duration_s=duration_s,
            ram_mb=self.get_ram_usage_mb(),
            vram_mb=self.get_vram_usage_mb(),
            gpu_util_percent=self.get_gpu_utilization()
        )
        self.metrics.append(metric)
        return metric

    def print_summary(self):
        """Print performance summary."""
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE PROFILING SUMMARY")
        logger.info("="*80)

        # Group by operation
        by_operation = defaultdict(list)
        for m in self.metrics:
            by_operation[m.operation].append(m)

        logger.info(f"\n{'Operation':<30} {'Count':<8} {'Total (s)':<12} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
        logger.info("-"*80)

        for op, metrics in sorted(by_operation.items()):
            count = len(metrics)
            total_time = sum(m.duration_s for m in metrics)
            avg_time = total_time / count
            min_time = min(m.duration_s for m in metrics)
            max_time = max(m.duration_s for m in metrics)

            logger.info(f"{op:<30} {count:<8} {total_time:<12.2f} {avg_time:<12.2f} {min_time:<12.2f} {max_time:<12.2f}")

        # Overall stats
        total_time = time.time() - self.start_time
        peak_ram = max(m.ram_mb for m in self.metrics)
        peak_vram = max(m.vram_mb for m in self.metrics)
        avg_gpu_util = sum(m.gpu_util_percent for m in self.metrics) / len(self.metrics) if self.metrics else 0

        logger.info("\n" + "="*80)
        logger.info("OVERALL STATISTICS")
        logger.info("="*80)
        logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.memory(f"Peak RAM: {peak_ram:.2f} MB ({peak_ram/1024:.2f} GB)")
        logger.memory(f"Peak VRAM: {peak_vram:.2f} MB ({peak_vram/1024:.2f} GB)")
        logger.info(f"Average GPU utilization: {avg_gpu_util:.1f}%")

        # Bottleneck analysis
        logger.info("\n" + "="*80)
        logger.info("BOTTLENECK ANALYSIS")
        logger.info("="*80)

        # Find slowest operations
        slowest = sorted(self.metrics, key=lambda m: m.duration_s, reverse=True)[:10]
        logger.info("\nTop 10 slowest operations:")
        logger.info(f"{'Operation':<30} {'Layer':<40} {'Time (s)':<12} {'GPU %':<8}")
        logger.info("-"*80)
        for m in slowest:
            logger.info(f"{m.operation:<30} {m.layer_name:<40} {m.duration_s:<12.2f} {m.gpu_util_percent:<8.1f}")

        # Time breakdown by operation type
        logger.info("\nTime breakdown by operation:")
        total_op_time = sum(m.duration_s for m in self.metrics)
        for op, metrics in sorted(by_operation.items(), key=lambda x: sum(m.duration_s for m in x[1]), reverse=True):
            op_time = sum(m.duration_s for m in metrics)
            pct = (op_time / total_op_time * 100) if total_op_time > 0 else 0
            logger.info(f"  {op:<30} {op_time:>10.2f}s ({pct:>5.1f}%)")


# Configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTH = 128
NUM_BATCHES_GRADIENT = 5  # Keep small for profiling
SPARSITY = 0.1  # 10% sparsity
NUM_SEQUENCES = 3  # Small number of sequences for profiling

logger.info("="*80)
logger.info("MISTRAL-7B PERFORMANCE PROFILING WITH WANDA STRATEGY")
logger.info("="*80)
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Device: {DEVICE}")
logger.info(f"Strategy: WANDA (Weight AND Activation pruning)")
logger.info(f"Optimizations enabled:")
logger.info(f"  ✓ low_memory_mode=True (layer-by-layer streaming)")
logger.info(f"  ✓ use_fp16=True (FP16 gradients and saved state)")
logger.info(f"  ✓ use_gradient_checkpointing=True (reduce activation memory)")
logger.info("="*80)

# Initialize profiler
profiler = PerformanceProfiler()

# Load model
logger.info(f"\nLoading Mistral-7B...")
start = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Model in FP16
    low_cpu_mem_usage=True,
    device_map="auto" if DEVICE == "cuda" else None
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

load_time = time.time() - start
profiler.record("model_load", "N/A", load_time)

logger.info(f"Model loaded in {load_time:.2f}s")
logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
logger.memory(f"RAM: {profiler.get_ram_usage_mb():.2f} MB")
logger.memory(f"VRAM: {profiler.get_vram_usage_mb():.2f} MB")

# Prepare calibration data - need long enough text for multiple sequences
CALIBRATION_TEXT = """The quick brown fox jumps over the lazy dog. Machine learning has revolutionized how we interact with computers. These models learn patterns from vast amounts of text data. The transformer architecture marked a significant breakthrough in natural language processing. Climate change represents one of the most pressing challenges facing humanity.

The history of computing dates back to ancient times when humans first created tools for calculation. The abacus was invented thousands of years ago. In the 19th century, Charles Babbage designed the Analytical Engine, a mechanical general-purpose computer. Although never completed in his lifetime, Babbage's designs laid the groundwork for modern computers.

Throughout history, explorers have ventured into the unknown, driven by curiosity and the desire to discover new lands and cultures. From ancient Polynesian navigators crossing vast oceans to modern astronauts venturing into space, the spirit of exploration has been a defining characteristic of human civilization.

Mathematics is often called the language of the universe. From the elegant simplicity of Euclidean geometry to the abstract complexities of modern algebra and topology, mathematics provides tools for understanding patterns, structures, and relationships in both the natural and abstract worlds."""

logger.info(f"\nPreparing calibration data...")
start = time.time()

tokens = tokenizer.encode(CALIBRATION_TEXT, return_tensors='pt')
logger.info(f"Total tokens: {tokens.shape[1]}")

# Create sequences
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
    logger.info(f"\n⚠ Warning: Text too short for {NUM_SEQUENCES} sequences of length {SEQ_LENGTH}")
    logger.info(f"Creating single sequence from available text...")
    # Pad if necessary
    if tokens.shape[1] < SEQ_LENGTH + 1:
        padding_needed = SEQ_LENGTH + 1 - tokens.shape[1]
        tokens = torch.cat([tokens, tokens[:, :padding_needed]], dim=1)
    input_sequences = [tokens[:, :SEQ_LENGTH]]
    target_sequences = [tokens[:, 1:SEQ_LENGTH+1]]

logger.info(f"Created {len(input_sequences)} sequences")

all_inputs = torch.cat(input_sequences, dim=0)
all_targets = torch.cat(target_sequences, dim=0)
dataset = TensorDataset(all_inputs, all_targets)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

data_time = time.time() - start
profiler.record("data_preparation", "N/A", data_time)
logger.info(f"Data prepared in {data_time:.2f}s")

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

# Initialize pruner with all optimizations
logger.info(f"\nInitializing GradProbe with all optimizations...")
start = time.time()

# Monkey-patch to add timing to layer processing
original_compute_single = GradProbe._compute_gradients_single_layer

def timed_compute_single_layer(self, dataloader, loss_fn, num_batches, layer_name):
    start = time.time()
    result = original_compute_single(self, dataloader, loss_fn, num_batches, layer_name)
    duration = time.time() - start

    # Determine if this is original or modified gradient computation
    # We'll track both separately by checking if weights are reduced
    layer_param = dict(self.model.named_parameters())[layer_name]
    operation = "gradient_computation"

    profiler.record(operation, layer_name, duration)

    # Print progress
    logger.memory(f"  [{operation}] {layer_name}: {duration:.2f}s "
          f"(RAM: {profiler.get_ram_usage_mb():.0f}MB, "
          f"VRAM: {profiler.get_vram_usage_mb():.0f}MB, "
          f"GPU: {profiler.get_gpu_utilization():.0f}%)")

    return result

GradProbe._compute_gradients_single_layer = timed_compute_single_layer

pruner = GradProbe(
    model,
    WANDAPruning(dataloader, num_batches=NUM_BATCHES_GRADIENT),
    device=DEVICE,
    low_memory_mode=True,
    use_fp16=True,
    use_gradient_checkpointing=True
)

init_time = time.time() - start
profiler.record("pruner_init", "N/A", init_time)
logger.info(f"Pruner initialized in {init_time:.2f}s")

# Run pruning
logger.info(f"\nStarting pruning (sparsity={SPARSITY:.1%})...")
logger.info(f"This will process {len([p for p in model.parameters() if len(p.shape) >= 2])} layers sequentially")
logger.info()

start = time.time()

final_masks = pruner.prune(
    dataloader=dataloader,
    loss_fn=loss_fn,
    sparsity=SPARSITY,
    num_batches=NUM_BATCHES_GRADIENT,
    reduction_factor=0.1,
    gradient_threshold=0.0,
    verbose=True
)

prune_time = time.time() - start
profiler.record("total_pruning", "N/A", prune_time)

logger.info(f"\nPruning completed in {prune_time:.2f}s ({prune_time/60:.2f} minutes)")

# Final statistics
total_params = sum(p.numel() for p in model.parameters())
zero_params = sum((p.data == 0).sum().item() for p in model.parameters())
final_sparsity = zero_params / total_params

logger.info(f"\nFinal sparsity: {final_sparsity:.2%}")
logger.info(f"Pruned {zero_params:,} out of {total_params:,} parameters")

# Print full profiling summary
profiler.print_summary()

# Additional insights
logger.info("\n" + "="*80)
logger.info("OPTIMIZATION RECOMMENDATIONS")
logger.info("="*80)

avg_gpu_util = sum(m.gpu_util_percent for m in profiler.metrics) / len(profiler.metrics) if profiler.metrics else 0

if avg_gpu_util < 20:
    logger.info("\n⚠ LOW GPU UTILIZATION DETECTED")
    logger.info(f"  Average GPU utilization: {avg_gpu_util:.1f}%")
    logger.info("\nPossible causes:")
    logger.info("  1. Layer-by-layer processing is inherently sequential")
    logger.info("  2. Gradient checkpointing causes recomputation overhead")
    logger.info("  3. CPU-GPU data transfer bottleneck")
    logger.info("  4. Small batch size (batch_size=1)")
    logger.info("\nRecommendations:")
    logger.info("  - If VRAM allows, disable gradient checkpointing: use_gradient_checkpointing=False")
    logger.info("  - Consider processing multiple layers in parallel if memory permits")
    logger.info("  - Increase batch size if memory allows")
    logger.info("  - Profile with torch.profiler to identify exact bottlenecks")

peak_vram = max(m.vram_mb for m in profiler.metrics)
if DEVICE == "cuda":
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    vram_usage_pct = (peak_vram / total_vram) * 100

    logger.memory(f"\n📊 VRAM USAGE: {peak_vram:.0f} MB / {total_vram:.0f} MB ({vram_usage_pct:.1f}%)")

    if vram_usage_pct < 50:
        logger.info("\n✓ LOW VRAM USAGE - You can disable gradient checkpointing!")
        logger.info("  This will:")
        logger.info("    - Increase VRAM usage by ~50-100%")
        logger.info("    - Reduce compute time by ~30-50%")
        logger.info("    - Improve GPU utilization")
        logger.info("\n  Set: use_gradient_checkpointing=False")

logger.info("\n" + "="*80)
logger.info("PROFILING COMPLETE")
logger.info("="*80)
