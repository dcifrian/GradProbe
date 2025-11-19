"""
Multi-layer test: This is the real use case for WANDA/Magnitude strategies.

Current approach:
1. Collect importance for all layers
2. Concatenate into one big tensor
3. kthvalue on concatenated tensor
4. Create masks

Optimized approach:
1. Stream through layers computing mean/std (no concatenation!)
2. Use stats to estimate threshold
3. Binary search with streaming (no big tensor)
4. Create masks

The benefit: avoid the memory spike from concatenating all layers.
"""

import torch
import time
import gc
import psutil
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, '/home/user/GradProbe')

from gradprobe.logger import Logger, LogLevel

# Initialize logger
logger = Logger(program_name='optimized_magnitude_multilayer', level=LogLevel.INFO)


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def current_magnitude_all_layers(
    layers: List[Tuple[str, torch.nn.Parameter]],
    sparsity: float,
    device: str = 'cpu'
) -> Tuple[Dict[str, torch.Tensor], float, float, float]:
    """
    Current approach: collect all importance, concat, kthvalue.

    Returns:
        masks, time_taken, peak_memory_gb, concat_tensor_size_gb
    """
    start_mem = get_memory_usage()
    start_time = time.time()

    # Collect importance for all layers
    importances = []
    layer_shapes = {}

    for name, param in layers:
        importance = param.data.abs().to(device)
        importance_flat = importance.flatten()
        importances.append(importance_flat)
        layer_shapes[name] = importance.shape

    # Concatenate all importance scores - THIS IS THE MEMORY SPIKE
    concat_start = time.time()
    all_importance = torch.cat(importances)
    concat_time = time.time() - concat_start
    concat_size_gb = all_importance.element_size() * all_importance.numel() / 1024**3

    # Compute threshold using kthvalue
    num_weights_to_prune = int(sparsity * all_importance.numel())
    if num_weights_to_prune == 0:
        threshold = float('inf')
    else:
        threshold = torch.kthvalue(all_importance, num_weights_to_prune + 1)[0].item()

    # Create masks for each layer
    masks = {}
    for name, param in layers:
        importance = param.data.abs().to(device)
        masks[name] = importance < threshold

    end_time = time.time()
    peak_mem = get_memory_usage()

    return masks, end_time - start_time, peak_mem - start_mem, concat_size_gb


def optimized_magnitude_all_layers(
    layers: List[Tuple[str, torch.nn.Parameter]],
    sparsity: float,
    device: str = 'cpu',
    max_iterations: int = 20,
    tolerance: float = 0.0001
) -> Tuple[Dict[str, torch.Tensor], float, float]:
    """
    Optimized approach: streaming statistics, no concatenation.

    Returns:
        masks, time_taken, peak_memory_gb
    """
    start_mem = get_memory_usage()
    start_time = time.time()

    # Pass 1: Compute global statistics (streaming, no concatenation!)
    total_count = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    importance_cache = []

    for name, param in layers:
        importance = param.data.abs().to(device)
        importance_cache.append((name, importance))

        # Update running statistics
        total_count += importance.numel()
        total_sum += importance.sum().item()
        total_sum_sq += (importance ** 2).sum().item()

    # Compute global mean and std
    mean = total_sum / total_count
    variance = (total_sum_sq / total_count) - (mean ** 2)
    std = (variance ** 0.5)

    target_count = int(sparsity * total_count)

    # Initial threshold estimate using normal approximation
    if sparsity < 0.5:
        p = sparsity
        sign = -1
    else:
        p = 1 - sparsity
        sign = 1

    if p > 0.0 and p < 1.0:
        t = (-2 * torch.log(torch.tensor(p))) ** 0.5
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z_approx = sign * (t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t))
    else:
        z_approx = 0.0

    initial_threshold = mean + z_approx * std

    # Find min/max across all layers for bounds
    min_val = min(importance.min().item() for _, importance in importance_cache)
    max_val = max(importance.max().item() for _, importance in importance_cache)

    # Binary search
    low = min_val
    high = max_val
    threshold = max(low, min(high, initial_threshold))

    best_threshold = threshold
    best_error = float('inf')

    for iteration in range(max_iterations):
        # Count weights below threshold (streaming through layers)
        count_below = 0
        for _, importance in importance_cache:
            count_below += (importance < threshold).sum().item()

        actual_sparsity = count_below / total_count
        error = abs(actual_sparsity - sparsity)

        if error < best_error:
            best_error = error
            best_threshold = threshold

        if error < tolerance:
            break

        # Binary search update
        if actual_sparsity < sparsity:
            low = threshold
        else:
            high = threshold

        threshold = (low + high) / 2

        if abs(high - low) < 1e-10:
            break

    # Create masks with best threshold
    masks = {}
    for name, importance in importance_cache:
        masks[name] = importance < best_threshold

    end_time = time.time()
    peak_mem = get_memory_usage()

    return masks, end_time - start_time, peak_mem - start_mem


def compare_multilayer(
    layers: List[Tuple[str, torch.nn.Parameter]],
    sparsity: float,
    device: str = 'cpu',
    num_trials: int = 3
):
    """Compare approaches on multiple layers."""

    total_params = sum(p.numel() for _, p in layers)

    logger.info(f"\n{'='*70}")
    logger.info(f"Comparing Multi-Layer Magnitude Pruning")
    logger.info(f"{'='*70}")
    logger.info(f"Number of layers: {len(layers)}")
    logger.info(f"Total weights: {total_params:,}")
    logger.info(f"Target sparsity: {sparsity:.1%}")
    logger.info(f"Device: {device}")
    logger.info(f"Trials: {num_trials}")
    logger.info()

    # Warm up
    _ = current_magnitude_all_layers(layers, sparsity, device)
    _ = optimized_magnitude_all_layers(layers, sparsity, device)
    gc.collect()

    # Test current approach
    logger.info("Testing CURRENT approach (concat + kthvalue)...")
    current_times = []
    current_mems = []
    current_masks = None
    concat_size = 0

    for i in range(num_trials):
        gc.collect()
        masks, time_taken, mem_used, concat_gb = current_magnitude_all_layers(layers, sparsity, device)
        current_times.append(time_taken)
        current_mems.append(mem_used)
        if i == 0:
            current_masks = masks
            concat_size = concat_gb
        logger.memory(f"  Trial {i+1}: {time_taken*1000:.2f}ms, {mem_used*1000:.2f}MB")

    logger.info()

    # Test optimized approach
    logger.info("Testing OPTIMIZED approach (streaming + binary search)...")
    opt_times = []
    opt_mems = []
    opt_masks = None

    for i in range(num_trials):
        gc.collect()
        masks, time_taken, mem_used = optimized_magnitude_all_layers(layers, sparsity, device)
        opt_times.append(time_taken)
        opt_mems.append(mem_used)
        if i == 0:
            opt_masks = masks
        logger.memory(f"  Trial {i+1}: {time_taken*1000:.2f}ms, {mem_used*1000:.2f}MB")

    # Check sparsity and agreement
    current_sparsity = sum(m.sum().item() for m in current_masks.values()) / total_params
    opt_sparsity = sum(m.sum().item() for m in opt_masks.values()) / total_params

    total_agree = sum((current_masks[name] == opt_masks[name]).sum().item() for name in current_masks)
    agreement = total_agree / total_params

    logger.info()
    logger.info(f"{'='*70}")
    logger.info("RESULTS")
    logger.info(f"{'='*70}")

    logger.memory(f"\nConcatenated tensor size: {concat_size*1024:.1f}MB")

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
    else:
        mem_savings = 0

    logger.info(f"\n{'='*70}\n")

    return {
        'num_layers': len(layers),
        'total_params': total_params,
        'current_time': avg_current_time,
        'opt_time': avg_opt_time,
        'speedup': avg_current_time / avg_opt_time if avg_opt_time > 0 else 0,
        'current_mem': avg_current_mem,
        'opt_mem': avg_opt_mem,
        'mem_savings': mem_savings,
        'agreement': agreement,
        'concat_size_mb': concat_size * 1024
    }


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM

    logger.info("Loading TinyStories-33M model...")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    logger.info(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Test scenarios
    scenarios = [
        ("Single transformer block (4 layers)", [
            ("k_proj", model.transformer.h[0].attn.attention.k_proj.weight),
            ("v_proj", model.transformer.h[0].attn.attention.v_proj.weight),
            ("q_proj", model.transformer.h[0].attn.attention.q_proj.weight),
            ("out_proj", model.transformer.h[0].attn.attention.out_proj.weight),
        ]),
        ("Full transformer block (6 layers)", [
            ("k_proj", model.transformer.h[0].attn.attention.k_proj.weight),
            ("v_proj", model.transformer.h[0].attn.attention.v_proj.weight),
            ("q_proj", model.transformer.h[0].attn.attention.q_proj.weight),
            ("out_proj", model.transformer.h[0].attn.attention.out_proj.weight),
            ("mlp.c_fc", model.transformer.h[0].mlp.c_fc.weight),
            ("mlp.c_proj", model.transformer.h[0].mlp.c_proj.weight),
        ]),
        ("All weight layers (26 layers)", [
            (name, param) for name, param in model.named_parameters()
            if param.requires_grad and len(param.shape) >= 2 and 'wte' not in name and 'wpe' not in name
        ])
    ]

    sparsities = [0.3, 0.5]

    all_results = []

    for scenario_name, layers in scenarios:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Scenario: {scenario_name}")
        logger.info(f"{'#'*70}")

        for sparsity in sparsities:
            results = compare_multilayer(layers, sparsity, device='cpu', num_trials=3)
            results['scenario'] = scenario_name
            results['sparsity'] = sparsity
            all_results.append(results)

    # Summary
    logger.info(f"\n{'#'*70}")
    logger.info("# SUMMARY")
    logger.info(f"{'#'*70}\n")

    logger.info(f"{'Scenario':<30} {'Params':<12} {'Sparsity':<10} {'Speedup':<10} {'Concat Size':<12}")
    logger.info("-" * 80)
    for r in all_results:
        logger.info(f"{r['scenario']:<30} {r['total_params']:<12,} {r['sparsity']:<10.1%} "
              f"{r['speedup']:<10.2f}x {r['concat_size_mb']:<12.1f}MB")

    avg_speedup = sum(r['speedup'] for r in all_results) / len(all_results)
    avg_agreement = sum(r['agreement'] for r in all_results) / len(all_results)

    logger.info("\n" + "="*80)
    logger.info("OVERALL RESULTS:")
    logger.info(f"  Average speedup:    {avg_speedup:.2f}x")
    logger.info(f"  Average agreement:  {avg_agreement:.4%}")
    logger.info("="*80)

    if avg_speedup > 1.2 and avg_agreement > 0.995:
        logger.info("\n✅ OPTIMIZATION SUCCESSFUL FOR MULTI-LAYER!")
        logger.info("   Significantly faster with high mask agreement.")
        logger.info("   This optimization should be applied to real strategies.")
    elif avg_speedup > 1.0:
        logger.info("\n⚠️  OPTIMIZATION SHOWS PROMISE")
        logger.info("   Modest speedup. May need more tuning.")
    else:
        logger.info("\n❌ OPTIMIZATION NOT BENEFICIAL")
        logger.info("   Current approach is faster.")
