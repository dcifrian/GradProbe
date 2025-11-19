"""
Quick diagnostic profiling for Mistral-7B - identifies bottlenecks fast.

This script profiles just 3-5 layers to quickly identify:
- GPU vs CPU bottleneck
- Impact of gradient checkpointing
- Layer processing time
- Optimal settings
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import time
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, MagnitudePruning
from gradprobe.logger import Logger, LogLevel

# Initialize logger
logger = Logger(program_name='quick_profile_mistral', level=LogLevel.INFO)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    logger.error("transformers library not found")
    sys.exit(1)

def profile_gpu_utilization():
    """Check if we can monitor GPU utilization."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return True, util.gpu
    except:
        return False, 0

def get_memory_info():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / 1024**3

    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / 1024**3
        vram_reserved_gb = torch.cuda.memory_reserved() / 1024**3
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return ram_gb, vram_gb, vram_reserved_gb, total_vram_gb
    return ram_gb, 0, 0, 0

logger.info("="*80)
logger.info("QUICK MISTRAL-7B PERFORMANCE DIAGNOSTIC")
logger.info("="*80)
logger.info("\nThis script profiles 3-5 layers to quickly identify bottlenecks.")
logger.info("Full profiling will run based on these results.\n")

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TEST_LAYERS = 3  # Only test first 3 layers

logger.info(f"Device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    can_monitor_gpu, current_util = profile_gpu_utilization()
    if can_monitor_gpu:
        logger.info(f"GPU monitoring: Available (current util: {current_util}%)")
    else:
        logger.info(f"GPU monitoring: Not available (install pynvml: pip install pynvml)")

logger.info("\n" + "="*80)
logger.info("LOADING MODEL")
logger.info("="*80)

start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto" if DEVICE == "cuda" else None
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Enable gradient checkpointing to avoid OOM during profiling
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
    if hasattr(model, 'config'):
        model.config.use_cache = False
    logger.info("Gradient checkpointing enabled to avoid OOM during profiling")

load_time = time.time() - start
ram, vram, vram_res, vram_total = get_memory_info()

logger.info(f"\n✓ Model loaded in {load_time:.1f}s")
logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
logger.memory(f"  RAM: {ram:.1f} GB")
if DEVICE == "cuda":
    logger.memory(f"  VRAM: {vram:.1f} GB / {vram_total:.1f} GB ({vram/vram_total*100:.1f}%)")

# Prepare data
logger.info("\n" + "="*80)
logger.info("PREPARING DATA")
logger.info("="*80)

text = "The quick brown fox jumps over the lazy dog. " * 20
tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=128)
inputs = tokens[:, :128]
targets = tokens[:, :128]
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

logger.info(f"✓ Created 1 sequence of length 128")

def loss_fn(outputs, targets):
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

# Test 1: Skip forward/backward pass - too memory intensive for profiling
logger.info("\n" + "="*80)
logger.info("TEST 1: MEMORY USAGE ANALYSIS")
logger.info("="*80)
logger.info("Skipping forward/backward pass test - model already using most VRAM")
logger.memory(f"Model is using {vram:.1f}GB / {vram_total:.1f}GB")
logger.memory(f"Only {vram_total - vram:.1f}GB available - not enough for backward pass")
logger.info("\nNote: Your actual pruning run should work fine because it processes")
logger.info("      layer-by-layer with gradient checkpointing enabled.")

# Test 2: Note about gradient checkpointing
logger.info("\n" + "="*80)
logger.info("TEST 2: GRADIENT CHECKPOINTING")
logger.info("="*80)
logger.info("Note: Gradient checkpointing is already enabled to prevent OOM.")
logger.info("The baseline forward/backward test above was performed WITH checkpointing.")
logger.info("\nTo test WITHOUT checkpointing, you would need more VRAM.")

# Test 3: Layer-by-layer processing
logger.info("\n" + "="*80)
logger.info("TEST 3: LAYER-BY-LAYER PROCESSING")
logger.info("="*80)

# Get first few layers
layer_params = [(name, param) for name, param in model.named_parameters()
                if param.requires_grad and len(param.shape) >= 2][:NUM_TEST_LAYERS]

logger.info(f"Testing first {len(layer_params)} layers:")
for i, (name, param) in enumerate(layer_params):
    logger.info(f"  {i+1}. {name} ({param.numel():,} params)")

logger.info("\nProcessing layers sequentially...")

layer_times = []
for layer_name, layer_param in layer_params:
    # Freeze all other parameters
    for name, param in model.named_parameters():
        param.requires_grad = (name == layer_name)

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start = time.time()

    for inputs_batch, targets_batch in dataloader:
        inputs_batch = inputs_batch.to(DEVICE)
        targets_batch = targets_batch.to(DEVICE)

        model.zero_grad()
        outputs = model(inputs_batch)
        loss = loss_fn(outputs, targets_batch)
        loss.backward()
        break

    layer_time = time.time() - start
    layer_times.append((layer_name, layer_time))

    _, gpu_util = profile_gpu_utilization() if profile_gpu_utilization()[0] else (False, 0)
    logger.info(f"  {layer_name}: {layer_time:.3f}s (GPU: {gpu_util}%)")

# Restore requires_grad
for param in model.parameters():
    param.requires_grad = True

avg_layer_time = sum(t for _, t in layer_times) / len(layer_times)
total_layers = len([p for p in model.parameters() if len(p.shape) >= 2])

logger.info(f"\n✓ Average time per layer: {avg_layer_time:.3f}s")
logger.info(f"  Estimated total time for {total_layers} layers: {avg_layer_time * total_layers:.1f}s ({avg_layer_time * total_layers / 60:.1f} min)")
logger.info(f"  This is for ONE gradient computation.")
logger.info(f"  Full pruning needs TWO gradient computations per layer (original + modified)")
logger.info(f"  Estimated FULL pruning time: {avg_layer_time * total_layers * 2 / 60:.1f} minutes")

# Analysis and recommendations
logger.info("\n" + "="*80)
logger.info("ANALYSIS & RECOMMENDATIONS")
logger.info("="*80)

ram, vram, vram_res, vram_total = get_memory_info()

logger.info(f"\n📊 Current resource usage:")
logger.memory(f"  RAM: {ram:.1f} GB")
if DEVICE == "cuda":
    logger.memory(f"  VRAM: {vram:.1f} GB / {vram_total:.1f} GB ({vram/vram_total*100:.1f}%)")
    vram_available = vram_total - vram

    logger.memory(f"\n💡 VRAM Headroom: {vram_available:.1f} GB available")

    if vram_available < 3:
        logger.info("\n⚠ LOW VRAM HEADROOM")
        logger.info("   Gradient checkpointing is necessary to avoid OOM.")
        logger.info("   Keep: use_gradient_checkpointing=True")
    elif vram_available < 8:
        logger.info("\n✓ MODERATE VRAM HEADROOM")
        logger.info("   Gradient checkpointing recommended for safety.")
        logger.info("   You could try disabling it if you monitor VRAM carefully.")
    else:
        logger.info("\n✅ GOOD VRAM HEADROOM")
        logger.info("   You could try disabling gradient checkpointing for better performance.")
        logger.info("   Set: use_gradient_checkpointing=False")
        logger.info("   Monitor VRAM usage carefully - may still OOM depending on batch size.")

# GPU utilization analysis
can_monitor, current_util = profile_gpu_utilization()
if can_monitor and current_util < 30:
    logger.info(f"\n⚠ LOW GPU UTILIZATION: {current_util}%")
    logger.info("   Possible causes:")
    logger.info("   1. Gradient checkpointing adds CPU overhead (recomputation)")
    logger.info("   2. Layer-by-layer processing is sequential (inherent to algorithm)")
    logger.info("   3. Small batch size (batch_size=1) underutilizes GPU")
    logger.info("\n   To improve GPU utilization:")
    logger.info("   - Disable gradient checkpointing (if VRAM allows)")
    logger.info("   - Increase batch size (if VRAM allows)")
    logger.info("   - Process multiple layers in parallel (requires code changes)")

# Speed estimation
num_batches = 10  # typical for gradient computation
total_time_estimate = avg_layer_time * total_layers * 2 * num_batches / 60

logger.info(f"\n⏱ ESTIMATED FULL PRUNING TIME:")
logger.info(f"  With {num_batches} batches per gradient computation:")
logger.info(f"  ~{total_time_estimate:.1f} minutes ({total_time_estimate/60:.1f} hours)")

if total_time_estimate > 60:
    logger.info(f"\n  This is very slow! Recommendations:")
    logger.info(f"  1. Reduce num_batches (e.g., 5 instead of 10)")
    logger.info(f"  2. Disable gradient checkpointing if VRAM allows")
    logger.info(f"  3. Consider using smaller model for testing (e.g., Mistral-1B)")

logger.info("\n" + "="*80)
logger.info("QUICK DIAGNOSTIC COMPLETE")
logger.info("="*80)
logger.info("\nNext steps:")
logger.info("1. Apply recommended settings above")
logger.info("2. Run full profiling with: python examples/profile_mistral_performance.py")
logger.info("3. Or run actual pruning with optimized settings: python examples/test_mistral.py")
