# GradProbe Memory Analysis Report
## TinyStories-33M Model Profiling

**Date**: 2025-11-17
**Model**: TinyStories-33M (68.5M parameters)
**Algorithm**: GradProbe gradient-based pruning

---

## Executive Summary

The GradProbe algorithm is experiencing severe memory issues when scaling to larger models. For a 33M parameter model (261 MB of weights), the algorithm uses **2.6 GB** of RAM during pruning. This scales to approximately **300GB+ for Mistral-7B** (7B parameters).

### Root Cause

The algorithm stores **multiple complete copies** of model weights and gradients simultaneously:

| Component | Size (33M model) | Mistral-7B Estimate | Multiplier |
|-----------|------------------|---------------------|------------|
| Model weights (in model) | 261 MB | ~28 GB | 1x |
| Saved original state | 261 MB | ~28 GB | 1x |
| Original gradients | 261 MB | ~28 GB | 1x |
| Modified gradients | 261 MB | ~28 GB | 1x |
| Pruning masks | 65 MB | ~7 GB | 0.25x |
| **TOTAL (before activations)** | **1,110 MB** | **~119 GB** | **4.25x** |
| Activations during forward/backward | ~1,500 MB | ~180 GB | Variable |
| **PEAK TOTAL** | **2,605 MB** | **~299 GB** | **~10x** |

---

## Detailed Analysis

### 1. Memory Breakdown by Component

#### A. Model Weights (2x storage)
- **Original model weights**: 261.36 MB
  - Lives in the model's parameters
  - Modified during the algorithm (reduced to 0.1x for tentative pruning)

- **Saved original state**: 261.36 MB
  - Complete copy created at line 109-117 in `pruner.py`
  - Purpose: Restore weights after gradient comparison
  - **Issue**: This doubles the weight memory footprint

**Code location**: `gradprobe/pruner.py:109-117`
```python
self.original_state = {
    name: param.data.clone() for name, param in self.model.named_parameters()
}
```

#### B. Gradients (2x storage)
- **Original gradients**: 261.36 MB
  - Computed via forward/backward pass on unmodified model
  - Stored in dictionary `original_gradients`
  - Size equals model weights (same shape, same dtype)

- **Modified gradients**: 261.36 MB
  - Computed via forward/backward pass on modified model (tentative weights reduced)
  - Stored in dictionary `modified_gradients`
  - **Issue**: Both gradient sets stored simultaneously for comparison

**Code locations**:
- Original gradients: `gradprobe/pruner.py:122` → `_compute_gradients()` → `pruner.py:224-295`
- Modified gradients: `gradprobe/pruner.py:142` → `_compute_gradients()` → `pruner.py:224-295`

**Storage implementation** (`pruner.py:250-253`):
```python
gradients = {name: torch.zeros_like(param)
             for name, param in self.model.named_parameters()
             if param.requires_grad}
```

#### C. Pruning Masks
- **Tentative masks**: ~65 MB (boolean tensors)
  - Created by pruning strategy (magnitude, WANDA, etc.)
  - Boolean tensor (1 bit per weight, but stored as uint8 = 1 byte per weight)
  - Size = model parameters / 4 (due to boolean vs float32)

- **Final masks**: ~65 MB
  - Result of gradient comparison
  - Additional copy of mask data

**Code location**: `gradprobe/pruner.py:127`

#### D. Activations (largest overhead during forward/backward)
- **Estimated**: 11.5 MB per layer × 4 layers = ~46 MB theoretical
- **Actual**: ~1,500 MB overhead during backward passes

The discrepancy is because:
1. PyTorch stores intermediate activations for all layers during forward pass
2. These are kept for backpropagation
3. For transformers: attention matrices (batch × heads × seq × seq) are large
4. FFN intermediate activations (4× hidden size)

**Backward pass memory spikes**:
- First backward pass: 2,025 MB (delta +290 MB from forward)
- Second backward pass: 2,480 MB (delta +196 MB from forward)

---

### 2. Memory Usage Timeline

| Stage | RSS (MB) | Delta | What Happens |
|-------|----------|-------|--------------|
| Initial | 660 | - | Python + transformers loaded |
| Model loaded | 759 | +99 | TinyStories-33M loaded |
| **Saved original state** | **1,298** | **+523** | **Full copy of weights** |
| **Allocated gradients** | **1,560** | **+261** | **Gradient storage allocated** |
| Forward pass batch 0 | 1,735 | +175 | Activations allocated |
| **Backward pass batch 0** | **2,025** | **+290** | **Gradients computed + stored** |
| Original grads done | 2,135 | +45 | Accumulated gradients |
| Tentative masks | 2,259 | +125 | Masks created |
| **Allocated modified grads** | **2,411** | **+147** | **Second gradient storage** |
| Modified backward 0 | 2,480 | +196 | Second set gradients computed |
| **PEAK** | **2,605** | **+1,945** | **All data in memory** |

---

### 3. Scaling to Larger Models

#### Mistral-7B Projections

Mistral-7B has approximately 7 billion parameters:
- Weights: 7B × 4 bytes (float32) = 28 GB
- Gradients (2 sets): 2 × 28 GB = 56 GB
- Saved state: 28 GB
- Masks: 7 GB
- **Minimum (no activations)**: ~119 GB
- **With activations overhead (10x multiplier)**: ~**300 GB**

This explains why the user observed 300GB+ memory usage with Mistral-7B.

---

### 4. Key Issues Identified

#### Issue #1: Duplicate Weight Storage (522 MB → 56 GB for Mistral)
**Problem**: The algorithm stores a complete copy of all model weights in `self.original_state`.

**Code**: `gradprobe/pruner.py:109-117`

**Purpose**: Restore weights after testing tentative pruning

**Impact**: Doubles weight memory footprint (2× model size)

#### Issue #2: Dual Gradient Storage (522 MB → 56 GB for Mistral)
**Problem**: Both original and modified gradients stored simultaneously for comparison.

**Code**:
- `gradprobe/pruner.py:122` (original)
- `gradprobe/pruner.py:142` (modified)

**Purpose**: Compare gradient magnitudes to decide which weights to prune

**Impact**: Doubles gradient memory (2× model size)

#### Issue #3: Gradients Stored on CPU
**Problem**: Gradients are moved to CPU for storage (`pruner.py:250-253`), but:
1. This actually increases total memory usage (weights on GPU + gradients on CPU)
2. For CPU-only execution, doesn't help
3. Adds CPU ↔ GPU transfer overhead

**Code**: `gradprobe/pruner.py:281`
```python
grad_abs = param.grad.data.abs().cpu()
```

#### Issue #4: Activation Memory During Backward Pass
**Problem**: Each forward/backward pass requires storing activations for all layers.

**Impact**:
- Single batch: ~290 MB overhead
- For larger models/sequences: scales with batch_size × seq_length × num_layers

**Cannot be avoided** (required for backpropagation), but can be minimized with smaller batches.

---

### 5. Memory Efficiency Observations

#### What Works Well
1. **Boolean masks** use 4x less memory than float32 (1 byte vs 4 bytes per element)
2. **Gradient accumulation** uses `torch.maximum()` instead of storing all batch gradients
3. **Low memory mode** exists but has limited impact

#### What Doesn't Work
1. **Storing complete copies** of weights and gradients
2. **No streaming/chunking** - all parameters processed simultaneously
3. **No gradient checkpointing** during forward passes
4. **No parameter-wise processing** - could process layer-by-layer with gradient clearing

---

## Recommended Solutions (For Future Discussion)

> **Note**: As requested, I have NOT implemented any changes. These are observations only.

### High-Impact Optimizations

1. **Eliminate `original_state` copy**
   - Store only the tentatively pruned indices and their original values
   - Memory reduction: 261 MB → <10 MB (only pruned weights)
   - Savings: ~28 GB for Mistral-7B

2. **Stream gradient comparison layer-by-layer**
   - Process one layer at a time:
     1. Compute original gradients for layer N
     2. Apply tentative reduction
     3. Compute modified gradients for layer N
     4. Compare and decide
     5. Clear gradients, move to layer N+1
   - Memory reduction: From 2× gradients to ~0.1× gradients (one layer)
   - Savings: ~50 GB for Mistral-7B

3. **Use gradient checkpointing during forward passes**
   - Trade computation for memory (recompute activations during backward)
   - Can reduce activation memory by 50-90%
   - Savings: ~90-162 GB for Mistral-7B

4. **Quantize gradients to int8 or float16**
   - Use lower precision for gradient comparison
   - Memory reduction: 4× or 2×
   - Savings: 14-28 GB for Mistral-7B

### Layerwise Pruning Observations

The code already has a `prune_layerwise()` method, but:
- It still stores gradients for ALL parameters (see `pruner.py:810-814`)
- Could be modified to only store gradients for current layer
- Would reduce memory proportional to (1 / num_layers)

---

## Profiling Methodology

### Tools Used
1. **Python `psutil`**: Process memory (RSS, VMS)
2. **Python `tracemalloc`**: Python heap memory
3. **PyTorch memory APIs**: CUDA allocated/reserved (when applicable)
4. **Custom MemoryProfiler**: Snapshots at key algorithm stages

### Test Configuration
- Model: TinyStories-33M (68.5M params)
- Sequence length: 128 tokens
- Batch size: 1
- Number of batches: 2 (for profiling speed)
- Device: CPU (as requested)
- Sparsity target: 10%

### Accuracy
The profiling shows **actual memory usage** from the OS perspective (RSS = Resident Set Size). This is more reliable than tracking individual tensor sizes because it includes:
- Python interpreter overhead
- PyTorch framework overhead
- Memory fragmentation
- OS page alignment

---

## Comparison to Theoretical Minimum

### Theoretical Minimum (If Algorithm Could Be Optimized)
- Model weights (1×): 261 MB
- Gradients (streaming, 1 layer at a time): ~13 MB
- Masks (sparse storage): ~10 MB
- Activations (gradient checkpointing): ~50 MB
- **Total**: ~334 MB (~1.3× model size)

### Current Implementation
- **Total**: 2,605 MB (~10× model size)

### Gap: 7.8× MORE memory than theoretical minimum

This shows there is significant room for optimization.

---

## Conclusion

The GradProbe algorithm in its current form has a memory footprint of approximately **10× the model size**. This makes it unsuitable for large language models (Mistral-7B, Llama, etc.) without significant modifications.

The primary memory consumers are:
1. **Duplicate weight storage** (2× model size)
2. **Dual gradient storage** (2× model size)
3. **Activation memory** during backpropagation (variable, ~2-4× model size)

For a 7B parameter model, this results in ~300 GB memory usage, which exceeds typical system RAM.

The good news: Most of this memory usage can be eliminated through algorithmic changes without fundamentally altering the gradient-comparison approach. The algorithm's core insight (compare gradients before/after weight reduction) can be preserved while drastically reducing memory footprint.

---

## Appendix: Detailed Memory Snapshots

See `profile_output.txt` for complete snapshot-by-snapshot breakdown.

### Key Snapshots

**Before first gradient computation** (Stage 05):
- RSS: 1,560 MB
- Components: Model (261 MB) + Saved state (261 MB) + Gradient storage allocated (261 MB)

**After first backward pass** (Stage 07):
- RSS: 2,025 MB (+465 MB)
- New: Gradients computed and stored + activations

**After second gradient storage allocated** (Stage 11):
- RSS: 2,411 MB (+851 MB from stage 05)
- New: Second gradient storage (261 MB) + masks (65 MB)

**Peak** (Stage 15):
- RSS: 2,605 MB
- All components in memory simultaneously

---

## Files Generated

1. `profile_memory.py` - Memory profiling utilities
2. `examples/profile_tinystories.py` - Instrumented test script
3. `profile_output.txt` - Raw profiling output
4. `MEMORY_ANALYSIS_REPORT.md` - This report
