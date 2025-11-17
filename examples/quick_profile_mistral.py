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

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("transformers library not found")
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

print("="*80)
print("QUICK MISTRAL-7B PERFORMANCE DIAGNOSTIC")
print("="*80)
print("\nThis script profiles 3-5 layers to quickly identify bottlenecks.")
print("Full profiling will run based on these results.\n")

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TEST_LAYERS = 3  # Only test first 3 layers

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    can_monitor_gpu, current_util = profile_gpu_utilization()
    if can_monitor_gpu:
        print(f"GPU monitoring: Available (current util: {current_util}%)")
    else:
        print(f"GPU monitoring: Not available (install pynvml: pip install pynvml)")

print("\n" + "="*80)
print("LOADING MODEL")
print("="*80)

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

load_time = time.time() - start
ram, vram, vram_res, vram_total = get_memory_info()

print(f"\n✓ Model loaded in {load_time:.1f}s")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  RAM: {ram:.1f} GB")
if DEVICE == "cuda":
    print(f"  VRAM: {vram:.1f} GB / {vram_total:.1f} GB ({vram/vram_total*100:.1f}%)")

# Prepare data
print("\n" + "="*80)
print("PREPARING DATA")
print("="*80)

text = "The quick brown fox jumps over the lazy dog. " * 20
tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=128)
inputs = tokens[:, :128]
targets = tokens[:, :128]
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print(f"✓ Created 1 sequence of length 128")

def loss_fn(outputs, targets):
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

# Test 1: Single forward/backward pass
print("\n" + "="*80)
print("TEST 1: BASELINE FORWARD/BACKWARD PASS")
print("="*80)

model.eval()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

start = time.time()
ram_before, vram_before, _, _ = get_memory_info()

# Forward pass
for inputs_batch, targets_batch in dataloader:
    inputs_batch = inputs_batch.to(DEVICE)
    targets_batch = targets_batch.to(DEVICE)

    model.zero_grad()
    outputs = model(inputs_batch)
    loss = loss_fn(outputs, targets_batch)

    # Backward pass
    loss.backward()
    break

forward_backward_time = time.time() - start
ram_after, vram_after, _, _ = get_memory_info()

print(f"\n✓ Forward + Backward: {forward_backward_time:.3f}s")
print(f"  RAM delta: {ram_after - ram_before:.2f} GB")
if DEVICE == "cuda":
    print(f"  VRAM delta: {vram_after - vram_before:.2f} GB")
    _, gpu_util = profile_gpu_utilization() if profile_gpu_utilization()[0] else (False, 0)
    print(f"  GPU utilization: {gpu_util}%")

# Test 2: With gradient checkpointing
print("\n" + "="*80)
print("TEST 2: WITH GRADIENT CHECKPOINTING")
print("="*80)

if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
    if hasattr(model, 'config'):
        model.config.use_cache = False
    print("✓ Gradient checkpointing enabled")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    start = time.time()
    ram_before, vram_before, _, _ = get_memory_info()

    for inputs_batch, targets_batch in dataloader:
        inputs_batch = inputs_batch.to(DEVICE)
        targets_batch = targets_batch.to(DEVICE)

        model.zero_grad()
        outputs = model(inputs_batch)
        loss = loss_fn(outputs, targets_batch)
        loss.backward()
        break

    checkpointing_time = time.time() - start
    ram_after, vram_after, _, _ = get_memory_info()

    print(f"\n✓ Forward + Backward (checkpointed): {checkpointing_time:.3f}s")
    print(f"  RAM delta: {ram_after - ram_before:.2f} GB")
    if DEVICE == "cuda":
        print(f"  VRAM delta: {vram_after - vram_before:.2f} GB")

    slowdown = (checkpointing_time / forward_backward_time - 1) * 100
    print(f"\n  Checkpointing slowdown: {slowdown:+.1f}%")
else:
    print("✗ Model does not support gradient checkpointing")

# Test 3: Layer-by-layer processing
print("\n" + "="*80)
print("TEST 3: LAYER-BY-LAYER PROCESSING")
print("="*80)

# Get first few layers
layer_params = [(name, param) for name, param in model.named_parameters()
                if param.requires_grad and len(param.shape) >= 2][:NUM_TEST_LAYERS]

print(f"Testing first {len(layer_params)} layers:")
for i, (name, param) in enumerate(layer_params):
    print(f"  {i+1}. {name} ({param.numel():,} params)")

print("\nProcessing layers sequentially...")

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
    print(f"  {layer_name}: {layer_time:.3f}s (GPU: {gpu_util}%)")

# Restore requires_grad
for param in model.parameters():
    param.requires_grad = True

avg_layer_time = sum(t for _, t in layer_times) / len(layer_times)
total_layers = len([p for p in model.parameters() if len(p.shape) >= 2])

print(f"\n✓ Average time per layer: {avg_layer_time:.3f}s")
print(f"  Estimated total time for {total_layers} layers: {avg_layer_time * total_layers:.1f}s ({avg_layer_time * total_layers / 60:.1f} min)")
print(f"  This is for ONE gradient computation.")
print(f"  Full pruning needs TWO gradient computations per layer (original + modified)")
print(f"  Estimated FULL pruning time: {avg_layer_time * total_layers * 2 / 60:.1f} minutes")

# Analysis and recommendations
print("\n" + "="*80)
print("ANALYSIS & RECOMMENDATIONS")
print("="*80)

ram, vram, vram_res, vram_total = get_memory_info()

print(f"\n📊 Current resource usage:")
print(f"  RAM: {ram:.1f} GB")
if DEVICE == "cuda":
    print(f"  VRAM: {vram:.1f} GB / {vram_total:.1f} GB ({vram/vram_total*100:.1f}%)")
    vram_available = vram_total - vram

    print(f"\n💡 VRAM Headroom: {vram_available:.1f} GB available")

    if vram_available > 10:
        print("\n✅ RECOMMENDATION: Disable gradient checkpointing")
        print("   You have plenty of VRAM available. Gradient checkpointing is slowing you down.")
        print("   Change: use_gradient_checkpointing=False")
        print(f"   Expected speedup: ~{slowdown:.0f}% faster")

    if vram_available > 15:
        print("\n✅ RECOMMENDATION: Consider processing multiple layers in parallel")
        print("   You have significant VRAM headroom. Could process 2-4 layers simultaneously.")
        print("   This would require code changes to batch layer processing.")

    if vram/vram_total < 0.5:
        print("\n✅ RECOMMENDATION: You could use full fp32 instead of fp16")
        print("   Your VRAM usage is low enough for full precision.")

# GPU utilization analysis
can_monitor, current_util = profile_gpu_utilization()
if can_monitor and current_util < 30:
    print(f"\n⚠ LOW GPU UTILIZATION: {current_util}%")
    print("   Possible causes:")
    print("   1. Gradient checkpointing adds CPU overhead (recomputation)")
    print("   2. Layer-by-layer processing is sequential (inherent to algorithm)")
    print("   3. Small batch size (batch_size=1) underutilizes GPU")
    print("\n   To improve GPU utilization:")
    print("   - Disable gradient checkpointing (if VRAM allows)")
    print("   - Increase batch size (if VRAM allows)")
    print("   - Process multiple layers in parallel (requires code changes)")

# Speed estimation
num_batches = 10  # typical for gradient computation
total_time_estimate = avg_layer_time * total_layers * 2 * num_batches / 60

print(f"\n⏱ ESTIMATED FULL PRUNING TIME:")
print(f"  With {num_batches} batches per gradient computation:")
print(f"  ~{total_time_estimate:.1f} minutes ({total_time_estimate/60:.1f} hours)")

if total_time_estimate > 60:
    print(f"\n  This is very slow! Recommendations:")
    print(f"  1. Reduce num_batches (e.g., 5 instead of 10)")
    print(f"  2. Disable gradient checkpointing if VRAM allows")
    print(f"  3. Consider using smaller model for testing (e.g., Mistral-1B)")

print("\n" + "="*80)
print("QUICK DIAGNOSTIC COMPLETE")
print("="*80)
print("\nNext steps:")
print("1. Apply recommended settings above")
print("2. Run full profiling with: python examples/profile_mistral_performance.py")
print("3. Or run actual pruning with optimized settings: python examples/test_mistral.py")
