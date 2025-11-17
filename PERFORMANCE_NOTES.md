# GradProbe Performance Analysis & Optimization

## Current Performance Issue with Mistral-7B

### Problem
- **GPU Utilization**: 4% (should be 70-90%)
- **VRAM Usage**: 15GB / 24GB (62.5% unused)
- **Speed**: Very slow, GPU mostly idle

### Root Causes

#### 1. Gradient Checkpointing Overhead ⚠️
**Impact**: ~30-50% slowdown

When `use_gradient_checkpointing=True` is enabled:
- Forward pass: Normal speed
- Backward pass: Recomputes activations on-the-fly (trades memory for compute)
- Result: Adds 30-50% overhead to every backward pass

**Your situation**:
- Using 15GB / 24GB VRAM
- **9GB of headroom available**
- Gradient checkpointing is HURTING performance unnecessarily!

**Fix**:
```python
pruner = GradProbe(
    model,
    strategy,
    low_memory_mode=True,  # Keep this
    use_fp16=True,  # Keep this
    use_gradient_checkpointing=False  # DISABLE THIS
)
```

#### 2. Layer-by-Layer Sequential Processing
**Impact**: Inherent to the algorithm, but can be optimized

Current implementation:
```python
for each layer in model:
    freeze_all_except(layer)  # Make only this layer trainable
    compute_gradient(layer)   # Full forward + backward pass
```

**Problem**: Setting `requires_grad=False` on other layers doesn't prevent computation, it just prevents gradient storage. We're still:
- Computing forward pass through ALL layers
- Backpropagating through ALL layers
- Only storing gradient for ONE layer

**Better approach** (doesn't freeze):
```python
for each layer in model:
    # Don't freeze anything
    compute_gradient_all_layers()
    # Only extract and store the gradient for current layer
    current_grad = grads[layer_name]
    # Discard other gradients
```

This allows PyTorch to better optimize the computation graph.

#### 3. Small Batch Size
**Impact**: GPU underutilization

- Current: `batch_size=1`
- GPU has parallel processing power for much larger batches
- With 9GB VRAM headroom, could likely use `batch_size=4` or more

#### 4. Multiple Gradient Computations per Layer
**Impact**: Doubles the time

For each layer, we compute gradients TWICE:
1. Original gradient (with full weights)
2. Modified gradient (with reduced weights)

This is necessary for the algorithm but doubles the computation.

### Quick Wins

#### Fix 1: Disable Gradient Checkpointing
**Expected improvement**: 30-50% faster

```python
pruner = GradProbe(
    model,
    strategy,
    low_memory_mode=True,
    use_fp16=True,
    use_gradient_checkpointing=False  # Change this
)
```

**VRAM impact**: Will use ~18-20GB instead of 15GB (still safe for 24GB GPU)

#### Fix 2: Increase Batch Size (if memory allows)
**Expected improvement**: 2-4x faster gradient computation

Test progressively:
```python
# Try batch_size=2 first
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# If that works, try batch_size=4
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
```

Monitor VRAM usage and stop before hitting limits.

#### Fix 3: Optimize Layer Processing (code change needed)

Current code freezes layers unnecessarily. Better approach:

```python
def _compute_gradients_single_layer_optimized(self, dataloader, loss_fn, num_batches, layer_name):
    """Optimized: Don't freeze other layers, just extract the one we need."""

    # Disable dropout
    dropout_states = {}
    for name, module in self.model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            dropout_states[name] = module.training
            module.eval()

    # Get target parameter
    layer_param = dict(self.model.named_parameters())[layer_name]
    gradient = torch.zeros(layer_param.shape, dtype=self.grad_dtype, device='cpu')

    batch_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_count >= num_batches:
            break

        self.model.zero_grad()

        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward - computes ALL gradients
        loss.backward()

        # Only extract the gradient we care about
        if layer_param.grad is not None:
            grad_abs = layer_param.grad.data.abs().cpu().to(self.grad_dtype)
            if batch_count == 0:
                gradient = grad_abs
            else:
                gradient = torch.maximum(gradient, grad_abs)

        batch_count += 1

    # Restore dropout
    for name, module in self.model.named_modules():
        if name in dropout_states and dropout_states[name]:
            module.train()

    return gradient
```

**Why this is better**:
- PyTorch can optimize the computation graph better
- No need to modify requires_grad flags (overhead)
- Cleaner code

### Expected Performance After Fixes

| Optimization | Current | After Fix | Improvement |
|--------------|---------|-----------|-------------|
| Gradient checkpointing OFF | 15 min | 10 min | 33% faster |
| Batch size 2 | 10 min | 5 min | 50% faster |
| Batch size 4 | 5 min | 3-4 min | 20-30% faster |
| **Total** | **15 min** | **3-4 min** | **~75% faster** |

### Memory vs Speed Trade-offs

| Configuration | VRAM Usage | Speed | GPU Util | Recommendation |
|---------------|------------|-------|----------|----------------|
| Current (all optimizations) | 15 GB | Slow | 4% | ❌ Over-optimized for memory |
| Disable checkpointing | 18-20 GB | Medium | 30-50% | ✅ **Best for 24GB GPU** |
| + Batch size 2 | 20-22 GB | Fast | 50-70% | ✅ Recommended |
| + Batch size 4 | 23-24 GB | Very Fast | 70-90% | ⚠️ Close to limit |

### Implementation Priority

1. **Immediate** (no code changes):
   - Disable gradient checkpointing: `use_gradient_checkpointing=False`
   - Increase batch size to 2: `batch_size=2`
   - Expected: 3-4x speedup

2. **Short-term** (minor code changes):
   - Remove layer freezing in `_compute_gradients_single_layer`
   - Expected: Additional 10-20% speedup

3. **Long-term** (significant changes):
   - Parallel layer processing (process 2-4 layers simultaneously)
   - Mixed precision training (fp16 forward, fp32 backward)
   - Gradient accumulation across batches

### Diagnostic Commands

Run these to diagnose your system:

```bash
# Quick diagnostic (3-5 minutes)
python examples/quick_profile_mistral.py

# Full performance profiling (will take current runtime)
python examples/profile_mistral_performance.py
```

### Monitoring During Pruning

Watch these metrics:
```bash
# GPU utilization
watch -n 1 nvidia-smi

# RAM usage
watch -n 1 free -h

# VRAM details
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv'
```

### Why TinyStories Was Fast

TinyStories-33M:
- 4 layers (vs 32 for Mistral)
- 68M params (vs 7B for Mistral)
- Each layer processes in ~0.5s (vs ~5-10s for Mistral)
- Total: ~30 seconds (vs ~15 minutes for Mistral)

The algorithm scales with:
- **O(num_layers × num_batches × 2)** - each layer processed twice
- Mistral has 8x more layers
- Mistral's layers are 100x larger
- Result: ~800x slower

### Conclusion

Your current settings are optimized for a system with limited VRAM (e.g., 16GB GPU). With 24GB VRAM and 9GB headroom:

**Recommended configuration**:
```python
pruner = GradProbe(
    model,
    MagnitudePruning(),
    device='cuda',
    low_memory_mode=True,  # Keep for layer-by-layer processing
    use_fp16=True,  # Keep for 2x memory savings
    use_gradient_checkpointing=False  # DISABLE - you have plenty of VRAM
)

# Use larger batch size
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
```

**Expected result**:
- VRAM: 20-22 GB (safe margin)
- GPU utilization: 50-70% (much better!)
- Speed: 3-5x faster than current
