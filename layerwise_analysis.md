# Layerwise WANDA Timing Analysis - CORRECTED

## Test Configuration
- Model: TinyStories-33M (68M parameters, 26 weight layers)
- Strategy: WANDA with iterative pruning + layerwise=True
- Batches: 100 per layer, gradient_threshold=1.0
- Experimental: experimental_tune_both_steps=True

## Timing Results (Wall-Clock Time)

### TEST 1: Magnitude (layerwise=False)
- Start: ~09:49
- Sparsity levels: 10-50%
- End: ~09:51 
- **Total time: ~2 minutes**
- Final sparsity: 47.16%
- Final perplexity: 6.80

### TEST 2: WANDA (layerwise=True)  
- Start: 09:51
- Breakdown by sparsity level:
  - 10%: 09:51:12 → 09:51:41 = **29 seconds**
  - 20%: 09:51:41 → 09:52:12 = **31 seconds**  
  - 30%: 09:52:12 → 09:52:46 = **34 seconds**
  - 40%: 09:52:46 → 09:53:19 = **33 seconds**
  - 50%: 09:53:19 → 09:53:53 = **34 seconds**
  - 60%: 09:53:53 → ~10:15 = **~21 minutes** (threshold tuning!)
- End: ~10:15
- **Total time: ~24 minutes**
- Final sparsity: 57.72%
- Final perplexity: 6.44

## Time Breakdown Analysis

### Per-Sparsity Level (10-50%)
Average: ~32 seconds per level

**Components:**
1. **WANDA calls**: 26 layers × <1 second = ~26 seconds
   - Each call: "Using cached activation norms" → very fast
   - Computes importance + threshold for 1 parameter
   
2. **Gradient computation**: ~5-10 seconds
   - 26 layers processed sequentially
   - Each layer unfreezes, computes gradients, refreezes
   
3. **Evaluation**: <1 second

### 60% Sparsity Level
**21 minutes** due to experimental two-step threshold tuning:
- Accuracy dropped 14.50% > 3.00% threshold
- Triggered `experimental_tune_both_steps=True`
- Re-prunes BOTH steps 50% and 60% multiple times
- Searches for optimal threshold adjustment
- Each iteration: 26 WANDA calls × 2 steps × multiple trials

## Key Findings

### 1. WANDA Calls Are FAST
- Per-layer WANDA call: <1 second  
- 26 calls per sparsity level = ~26 seconds total
- Activations correctly cached across all calls
- **BUT**: Importance computation + threshold repeated 26 times

### 2. Gradient Computation Is MODERATE
- ~5-10 seconds per sparsity level for all 26 layers
- NOT the primary bottleneck (contrary to initial analysis)

### 3. Threshold Tuning Is EXPENSIVE
- When enabled and triggered, dominates runtime
- Two-step tuning: 21 minutes for single sparsity level
- Regular tuning would be much faster (only tunes current step)

## Optimization Opportunity

**Cache WANDA Results Across Layers:**

Currently in `prune_layerwise()`:
```
For each of 26 layers:
    Call strategy.select_weights_to_prune(model, sparsity)
        → Computes importance for ALL 26 parameters
        → Computes global threshold
        → Returns masks for all parameters (but only uses 1)
```

The weights haven't changed between layers! We could:
```
At start of prune_layerwise():
    Call strategy.select_weights_to_prune(model, sparsity) ONCE
    Cache all 26 masks

For each of 26 layers:
    Extract cached mask for this layer
    Skip strategy call entirely
```

**Expected speedup:**
- Current: 26 WANDA calls × 1 sec = 26 seconds
- Optimized: 1 WANDA call × 1 sec = 1 second  
- **Savings: 25 seconds per sparsity level**

For 6 sparsity levels: **2.5 minutes saved**

This preserves the core insight (gradient recomputation per layer) while eliminating redundant WANDA computations.
