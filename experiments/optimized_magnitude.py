"""
Experimental optimized magnitude pruning using statistical estimation.

Instead of computing exact threshold with kthvalue, we:
1. Compute mean and std of importance scores (streaming, no concat)
2. Use statistical estimation to find threshold for target sparsity
3. Binary search to fine-tune if needed

This should reduce memory usage and potentially be faster.
"""

import torch
import time
import gc
import psutil
from typing import Dict, Tuple
import sys
sys.path.insert(0, '/home/user/GradProbe')

from gradprobe.logger import Logger, LogLevel

# Initialize logger
logger = Logger(program_name='optimized_magnitude', level=LogLevel.INFO)


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def current_magnitude_single_layer(
    param: torch.nn.Parameter,
    sparsity: float,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, float, float]:
    """
    Current approach: compute all importance, concat, use kthvalue.

    Returns:
        mask, time_taken, peak_memory_gb
    """
    start_mem = get_memory_usage()
    start_time = time.time()

    # Compute importance
    importance = param.data.abs().to(device)

    # Flatten for threshold computation
    importance_flat = importance.flatten()

    # Compute threshold using kthvalue
    num_weights_to_prune = int(sparsity * importance_flat.numel())
    if num_weights_to_prune == 0:
        threshold = float('inf')
    else:
        # This is the expensive operation
        threshold = torch.kthvalue(importance_flat, num_weights_to_prune + 1)[0].item()

    # Create mask
    mask = importance < threshold

    end_time = time.time()
    peak_mem = get_memory_usage()

    return mask, end_time - start_time, peak_mem - start_mem


def optimized_magnitude_single_layer(
    param: torch.nn.Parameter,
    sparsity: float,
    device: str = 'cpu',
    max_iterations: int = 20,
    tolerance: float = 0.001
) -> Tuple[torch.Tensor, float, float]:
    """
    Optimized approach: use mean/std to estimate threshold, binary search.

    Returns:
        mask, time_taken, peak_memory_gb
    """
    start_mem = get_memory_usage()
    start_time = time.time()

    # Compute importance
    importance = param.data.abs().to(device)

    # Compute statistics (fast operations)
    mean = importance.mean()
    std = importance.std()
    total_count = importance.numel()
    target_count = int(sparsity * total_count)

    # Initial threshold guess using normal distribution approximation
    # For sparsity s, we want the s-th quantile
    # Using inverse CDF approximation: threshold ≈ mean + z*std
    # where z is computed from sparsity

    # Simplified z-score approximation (faster than scipy)
    # For common sparsity values:
    # 10% -> z ≈ -1.28
    # 20% -> z ≈ -0.84
    # 30% -> z ≈ -0.52
    # 40% -> z ≈ -0.25
    # 50% -> z ≈ 0.00

    # Better approximation using polynomial (Abramowitz & Stegun)
    if sparsity < 0.5:
        p = sparsity
        sign = -1
    else:
        p = 1 - sparsity
        sign = 1

    # Rational approximation for normal quantile
    if p > 0.0 and p < 1.0:
        t = torch.sqrt(-2 * torch.log(torch.tensor(p)))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z_approx = sign * (t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t))
    else:
        z_approx = 0.0

    initial_threshold = mean + z_approx * std

    # Binary search to fine-tune threshold
    # Set bounds based on data range
    low = importance.min().item()
    high = importance.max().item()
    threshold = initial_threshold.item()

    # Clamp initial guess to valid range
    threshold = max(low, min(high, threshold))

    best_threshold = threshold
    best_error = float('inf')

    for iteration in range(max_iterations):
        # Count weights below threshold (fast operation)
        count_below = (importance < threshold).sum().item()
        actual_sparsity = count_below / total_count

        error = abs(actual_sparsity - sparsity)
        if error < best_error:
            best_error = error
            best_threshold = threshold

        # Check if we're close enough
        if error < tolerance:
            break

        # Binary search update
        if actual_sparsity < sparsity:
            # Need higher threshold to prune more
            low = threshold
        else:
            # Need lower threshold to prune less
            high = threshold

        # Update threshold
        threshold = (low + high) / 2

        # Prevent infinite loops
        if abs(high - low) < 1e-10:
            break

    # Create mask with best threshold found
    mask = importance < best_threshold

    end_time = time.time()
    peak_mem = get_memory_usage()

    return mask, end_time - start_time, peak_mem - start_mem


def compare_approaches(
    param: torch.nn.Parameter,
    sparsity: float,
    device: str = 'cpu',
    num_trials: int = 5
):
    """
    Compare current vs optimized approach.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Comparing Magnitude Pruning Approaches")
    logger.info(f"{'='*70}")
    logger.info(f"Parameter shape: {param.shape}")
    logger.info(f"Total weights: {param.numel():,}")
    logger.info(f"Target sparsity: {sparsity:.1%}")
    logger.info(f"Device: {device}")
    logger.info(f"Trials: {num_trials}")
    logger.info()

    # Warm up
    _ = current_magnitude_single_layer(param, sparsity, device)
    _ = optimized_magnitude_single_layer(param, sparsity, device)
    gc.collect()

    # Test current approach
    logger.info("Testing CURRENT approach (kthvalue)...")
    current_times = []
    current_mems = []
    current_mask = None

    for i in range(num_trials):
        gc.collect()
        mask, time_taken, mem_used = current_magnitude_single_layer(param, sparsity, device)
        current_times.append(time_taken)
        current_mems.append(mem_used)
        if i == 0:
            current_mask = mask
        logger.memory(f"  Trial {i+1}: {time_taken*1000:.2f}ms, {mem_used*1000:.2f}MB")

    current_sparsity = current_mask.sum().item() / current_mask.numel()

    logger.info()

    # Test optimized approach
    logger.info("Testing OPTIMIZED approach (mean/std + binary search)...")
    opt_times = []
    opt_mems = []
    opt_mask = None

    for i in range(num_trials):
        gc.collect()
        mask, time_taken, mem_used = optimized_magnitude_single_layer(param, sparsity, device)
        opt_times.append(time_taken)
        opt_mems.append(mem_used)
        if i == 0:
            opt_mask = mask
        logger.memory(f"  Trial {i+1}: {time_taken*1000:.2f}ms, {mem_used*1000:.2f}MB")

    opt_sparsity = opt_mask.sum().item() / opt_mask.numel()

    # Check agreement between masks
    agreement = (current_mask == opt_mask).sum().item() / current_mask.numel()

    logger.info()
    logger.info(f"{'='*70}")
    logger.info("RESULTS")
    logger.info(f"{'='*70}")

    logger.info(f"\nSparsity achieved:")
    logger.info(f"  Target:     {sparsity:.4%}")
    logger.info(f"  Current:    {current_sparsity:.4%} (error: {abs(current_sparsity - sparsity):.4%})")
    logger.info(f"  Optimized:  {opt_sparsity:.4%} (error: {abs(opt_sparsity - sparsity):.4%})")

    logger.info(f"\nMask agreement: {agreement:.4%}")

    logger.info(f"\nTime (average over {num_trials} trials):")
    avg_current_time = sum(current_times) / len(current_times)
    avg_opt_time = sum(opt_times) / len(opt_times)
    logger.info(f"  Current:    {avg_current_time*1000:.2f}ms")
    logger.info(f"  Optimized:  {avg_opt_time*1000:.2f}ms")
    if avg_current_time > 0:
        speedup = avg_current_time / avg_opt_time
        logger.info(f"  Speedup:    {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'}")

    logger.info(f"\nMemory (average over {num_trials} trials):")
    avg_current_mem = sum(current_mems) / len(current_mems)
    avg_opt_mem = sum(opt_mems) / len(opt_mems)
    logger.memory(f"  Current:    {avg_current_mem*1000:.2f}MB")
    logger.memory(f"  Optimized:  {avg_opt_mem*1000:.2f}MB")
    if avg_current_mem > 0:
        mem_savings = (avg_current_mem - avg_opt_mem) / avg_current_mem * 100
        logger.info(f"  Savings:    {mem_savings:.1f}%")

    logger.info(f"\n{'='*70}\n")

    return {
        'current_time': avg_current_time,
        'opt_time': avg_opt_time,
        'speedup': avg_current_time / avg_opt_time if avg_opt_time > 0 else 0,
        'current_mem': avg_current_mem,
        'opt_mem': avg_opt_mem,
        'mem_savings': mem_savings if avg_current_mem > 0 else 0,
        'agreement': agreement,
        'current_sparsity': current_sparsity,
        'opt_sparsity': opt_sparsity
    }


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM

    logger.info("Loading TinyStories-33M model...")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    logger.info(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test on different layer sizes
    test_layers = [
        ("Small layer", model.transformer.h[0].attn.attention.k_proj.weight),  # ~589K params
        ("Medium layer", model.transformer.h[0].mlp.c_fc.weight),              # ~2.3M params
        ("Large layer", model.transformer.wte.weight),                          # ~38M params
    ]

    sparsities = [0.1, 0.3, 0.5]

    all_results = []

    for layer_name, param in test_layers:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Testing: {layer_name}")
        logger.info(f"{'#'*70}")

        for sparsity in sparsities:
            results = compare_approaches(param, sparsity, device='cpu', num_trials=3)
            results['layer'] = layer_name
            results['sparsity'] = sparsity
            results['num_params'] = param.numel()
            all_results.append(results)

    # Summary
    logger.info(f"\n{'#'*70}")
    logger.info("# SUMMARY")
    logger.info(f"{'#'*70}\n")

    logger.info(f"{'Layer':<20} {'Sparsity':<10} {'Speedup':<10} {'Mem Savings':<12} {'Agreement':<12}")
    logger.info("-" * 70)
    for r in all_results:
        logger.info(f"{r['layer']:<20} {r['sparsity']:<10.1%} {r['speedup']:<10.2f}x "
              f"{r['mem_savings']:<12.1f}% {r['agreement']:<12.2%}")

    # Overall statistics
    avg_speedup = sum(r['speedup'] for r in all_results) / len(all_results)
    avg_mem_savings = sum(r['mem_savings'] for r in all_results) / len(all_results)
    avg_agreement = sum(r['agreement'] for r in all_results) / len(all_results)

    logger.info("\n" + "="*70)
    logger.info("OVERALL AVERAGES:")
    logger.info(f"  Speedup:        {avg_speedup:.2f}x")
    logger.info(f"  Memory savings: {avg_mem_savings:.1f}%")
    logger.info(f"  Mask agreement: {avg_agreement:.2%}")
    logger.info("="*70)

    if avg_speedup > 1.2 and avg_agreement > 0.95:
        logger.info("\n✅ OPTIMIZATION SUCCESSFUL!")
        logger.info("   The optimized approach is significantly faster with high agreement.")
    elif avg_speedup > 1.0 and avg_agreement > 0.98:
        logger.info("\n✅ OPTIMIZATION PROMISING!")
        logger.info("   The optimized approach is faster with very high agreement.")
    else:
        logger.info("\n⚠️  OPTIMIZATION NEEDS WORK")
        logger.info("   Either not fast enough or agreement is too low.")
