# WANDA Optimization Results

**Date**: 2025-11-18
**Optimization**: Statistical threshold estimation for WANDA pruning strategy

## Summary

Applied the same mean/std + binary search optimization to WANDA that was successful for Magnitude.

### Performance Results (TinyStories-33M)

| Sparsity | Current Time | Optimized Time | Speedup | Agreement |
|----------|-------------|----------------|---------|-----------|
| 10% | 3.31s | 3.28s | **1.01x** | 99.96% |
| 30% | 4.89s | 3.24s | **1.51x** | 99.97% |
| 50% | 6.59s | 2.30s | **2.86x** | 99.99% |

**Average: 1.79x speedup with 99.97% mask agreement**

## Key Findings

### 1. Speedup Increases with Sparsity

The optimization becomes **more effective at higher sparsity levels**:

- **Low sparsity (10%)**: Minimal speedup (1.01x)
  - Both approaches are fast
  - topk overhead is low when k is small

- **Medium sparsity (30%)**: Moderate speedup (1.51x)
  - topk starts to slow down
  - Binary search remains fast

- **High sparsity (50%)**: Significant speedup (2.86x)
  - topk is expensive (sorting 17M values)
  - Binary search converges quickly (threshold near median)

### 2. Why This Pattern?

**Current approach (topk):**
- Time complexity: O(n log k) where k = sparsity × n
- At 50% sparsity: k = 17M, very expensive
- Gets slower as sparsity increases

**Optimized approach (binary search):**
- Time complexity: O(iterations × n) where iterations ≈ 10-20
- Constant iterations regardless of sparsity
- Actually **faster** at higher sparsity (better initial guess)

### 3. WANDA-Specific Considerations

WANDA is more complex than Magnitude because importance = |weight| × ||activation||:

1. **Activation collection**: Same for both (not optimized)
2. **Importance computation**: Same for both (|W| × ||X|| per layer)
3. **Threshold selection**: **This is what we optimized**
4. **Mask creation**: Same for both

The optimization only affects step 3, which is why speedup is less dramatic than Magnitude (which had 4.5x average speedup).

### 4. Memory Comparison

Optimized version uses slightly **more** memory than current:
- Current: Concatenates importance scores temporarily, then discards
- Optimized: Keeps importance cache for binary search iterations

This is acceptable because:
- Memory difference is small (~10-60MB)
- Trade-off for 1.8x speedup
- Still far better than the massive concat tensor in old approach

## Comparison with Magnitude Optimization

| Strategy | Avg Speedup | Agreement | Notes |
|----------|------------|-----------|-------|
| **Magnitude** | 4.48x | 100% | Simpler importance (just \|W\|) |
| **WANDA** | 1.79x | 99.97% | Complex importance (\|W\| × \|\|X\|\|) |

WANDA has lower speedup because:
1. Activation collection dominates at low sparsity
2. Importance computation more complex
3. Threshold selection is smaller fraction of total time

But **1.8x is still significant**, especially for large models!

## Convergence Analysis

### Binary Search Iterations

- **10% sparsity**: 20 iterations (max iterations reached)
  - Initial guess far from target (z ≈ -1.28)
  - Harder to estimate for skewed threshold

- **30% sparsity**: 20 iterations (max iterations reached)
  - Better initial guess (z ≈ -0.52)
  - But still needs refinement

- **50% sparsity**: 13 iterations (converged!)
  - Excellent initial guess (z = 0, median)
  - Converges quickly

**Observation**: The normal distribution approximation works best at 50% sparsity (median), as expected.

## WANDA Importance Distribution

The WANDA importance scores (|W| × ||X||) have interesting properties:

- **Mean**: 0.116
- **Std**: 0.231
- **Range**: [0, 33.6]
- **Distribution**: Highly skewed (not normal!)

The skewed distribution explains why:
1. Agreement is 99.97% instead of 100%
2. Convergence takes more iterations
3. Normal approximation is less accurate

But 99.97% agreement is still excellent!

## Scalability for Large Models

### Mistral-7B Projection

For Mistral-7B at 50% sparsity:

**Current approach:**
- topk on 3.5B values = very slow
- Estimated: ~60-90 seconds for threshold selection

**Optimized approach:**
- 13 iterations × (3.5B comparisons)
- Estimated: ~15-20 seconds for threshold selection
- **3-4x faster**

The speedup will be even more pronounced on larger models!

## Recommendations

### When to Use Optimized WANDA

✅ **Use optimized version when:**
- Medium to high sparsity (30%+)
- Large models (Mistral-7B, Llama-7B+)
- Memory-constrained environments
- Iterative pruning (many threshold selections)

⚠️ **May keep current version when:**
- Very low sparsity (10%)
- Small models (< 100M params)
- One-shot pruning only

### Integration Strategy

**Conservative approach:**
1. Keep both implementations
2. Auto-select based on model size and sparsity
3. Use optimized for sparsity >= 30% or model >= 1B params

**Aggressive approach:**
1. Replace current WANDA with optimized version
2. 99.97% agreement is close enough for practical purposes
3. 1.8x speedup worth the tiny difference

## Next Steps

1. ✅ Magnitude optimized: 4.5x speedup
2. ✅ WANDA optimized: 1.8x speedup
3. 🔄 Test on Mistral-7B to validate large model benefits
4. 📊 Consider hybrid: use optimized for sparsity >= 30%
5. 📝 Update user documentation

## Conclusion

The WANDA optimization is **successful** with:
- 1.79x average speedup
- Up to 2.86x at 50% sparsity
- 99.97% mask agreement
- Especially beneficial for large models

Combined with the Magnitude optimization (4.5x), we've significantly improved the performance of both core pruning strategies!

---

## Appendix: Implementation Details

### Algorithm

```python
# Pass 1: Collect activations (same as current)
activation_norms = collect_activation_norms(model)

# Pass 2: Compute WANDA importance (streaming)
for layer in layers:
    importance = |W| × ||X||
    update_statistics(importance)  # No concatenation!

# Pass 3: Binary search for threshold
mean, std = compute_global_stats()
initial_threshold = mean + inverse_normal_cdf(sparsity) * std

for iteration in range(max_iterations):
    count_below = count_weights_below(threshold)  # Streaming
    if close_enough(count_below, target):
        break
    update_threshold()  # Binary search

# Pass 4: Create masks
for layer in layers:
    masks[layer] = importance <= threshold
```

### Key Differences from Magnitude

1. **Importance computation**: |W| × ||X|| instead of just |W|
2. **Distribution**: More skewed, harder to approximate
3. **Activation overhead**: Must collect activations first
4. **Cache reuse**: Activations cached across sparsity levels

The optimization still provides significant benefits despite the added complexity!
