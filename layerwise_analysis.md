# Layerwise WANDA Pruning - Performance Analysis

## Test Details
- Model: TinyStories-33M (68M parameters, 26 weight layers)
- Strategy: WANDA with iterative pruning + layerwise=True
- Configuration: 100 batches/layer, gradient_threshold=1.0

## Performance Results

### TEST 1: Magnitude (layerwise=False)
- Completed in: ~3 minutes
- Final sparsity: 47.16%
- Final perplexity: 6.80 (from 3.89 baseline)
- **Fast and efficient!**

### TEST 2: WANDA (layerwise=True)  
- Runtime: **97+ minutes** (killed - still at 60% sparsity)
- Expected: ~10 minutes
- **Actual: 10x+ slower than expected!**

## Root Cause Analysis

The extreme slowness is due to layerwise processing in iterative pruning:

**Computation per iteration:**
- With `layerwise=True`, each call to `iterative_prune()` at a given sparsity level:
  1. Calls `prune_layerwise()` which processes **26 layers sequentially**
  2. For each layer:
     - Freezes all other layers  
     - Calls `self.prune()` for that single layer
     - `prune()` calls `strategy.select_weights_to_prune()`
     - Computes gradients **twice** (original + modified) × 100 batches
  3. Then evaluates perplexity
  4. Compares with WANDA-only baseline (another call to select_weights_to_prune for all 26 params)

**Total operations for 6 sparsity levels (10%-60%):**
- Layerwise gradient computations: 6 levels × 26 layers × 2 passes × 100 batches = **31,200 forward/backward passes**
- Plus baseline comparisons: 6 levels × 2 × 100 batches = **1,200 forward/backward passes**  
- **Grand total: 32,400+ forward/backward passes**

Compare to non-layerwise:
- 6 levels × 2 passes × 100 batches = **1,200 forward/backward passes**
- **27x fewer passes!**

## Hotspot Identification

### Primary Bottleneck
The gradient computation in layerwise mode dominates runtime:
- Each layer requires full model forward/backward passes
- Even with activation caching in WANDA, gradients must be recomputed
- 26 layers × multiple sparsity levels = massive overhead

### Secondary Issues  
1. **Per-layer strategy calls**: WANDA's `select_weights_to_prune()` is called 26 times per sparsity level
   - Activation norms ARE cached (working correctly!)
   - But importance computation and threshold calculation repeated 26 times
   
2. **Baseline comparison overhead**: With `compare_baseline=True`, adds another full strategy call per iteration

3. **Evaluation overhead**: Perplexity evaluation after each sparsity level

## Recommendations

### Option 1: Batch Layer Processing
Instead of 26 sequential `prune()` calls, compute gradients for multiple layers in parallel:
- Group compatible layers (same dimensions)
- Compute gradients for multiple layers in single pass
- Could reduce passes from 26× to ~5-10×

### Option 2: Gradient Caching (if memory allows)
- Cache gradient computations across layers within same sparsity level
- Trade memory for speed
- Would reduce redundant gradient calculations

### Option 3: Skip Baseline Comparisons in Layerwise
- `compare_baseline=True` doubles the strategy calls
- Could make it optional for layerwise mode

### Option 4: Optimize Threshold Computation  
- WANDA computes global threshold by sampling all parameters
- In layerwise mode, could compute per-layer threshold more efficiently
- Avoid collecting importance for all 26 params when pruning just 1

## Conclusion

Layerwise pruning with iterative WANDA is **extremely slow** due to:
1. 26 sequential layer processings per sparsity level
2. Each requiring full gradient computations (2 × 100 batches)
3. Multiplied across 6+ sparsity levels = 31,200+ passes

**Immediate fix needed**: Optimize layerwise gradient computation to avoid redundant forward/backward passes.
