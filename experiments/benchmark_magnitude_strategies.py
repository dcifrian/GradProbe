"""
Benchmark current magnitude strategy vs optimized version on full model.
"""

import sys
sys.path.insert(0, '/home/user/GradProbe')

import torch
import time
import gc
import psutil
from transformers import AutoModelForCausalLM

from gradprobe.strategies.magnitude import MagnitudePruning
from gradprobe.strategies.magnitude_optimized import MagnitudePruningOptimized


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def compare_strategies(model, sparsity, num_trials=3):
    """Compare current vs optimized magnitude strategy."""

    print(f"\n{'='*70}")
    print(f"Benchmarking Magnitude Strategies")
    print(f"{'='*70}")
    print(f"Model: TinyStories-33M")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Target sparsity: {sparsity:.1%}")
    print(f"Trials: {num_trials}\n")

    # Test current strategy
    print("Testing CURRENT MagnitudePruning...")
    current_strategy = MagnitudePruning()
    current_times = []
    current_mems = []
    current_masks = None

    for i in range(num_trials):
        gc.collect()
        start_mem = get_memory_usage()
        start_time = time.time()

        masks = current_strategy.select_weights_to_prune(model, sparsity)

        end_time = time.time()
        peak_mem = get_memory_usage()

        time_taken = end_time - start_time
        mem_used = peak_mem - start_mem

        current_times.append(time_taken)
        current_mems.append(mem_used)
        if i == 0:
            current_masks = masks

        print(f"  Trial {i+1}: {time_taken:.3f}s, {mem_used*1024:.1f}MB")

    print()

    # Test optimized strategy
    print("Testing OPTIMIZED MagnitudePruningOptimized...")
    opt_strategy = MagnitudePruningOptimized()
    opt_times = []
    opt_mems = []
    opt_masks = None

    for i in range(num_trials):
        gc.collect()
        start_mem = get_memory_usage()
        start_time = time.time()

        masks = opt_strategy.select_weights_to_prune(model, sparsity)

        end_time = time.time()
        peak_mem = get_memory_usage()

        time_taken = end_time - start_time
        mem_used = peak_mem - start_mem

        opt_times.append(time_taken)
        opt_mems.append(mem_used)
        if i == 0:
            opt_masks = masks

        print(f"  Trial {i+1}: {time_taken:.3f}s, {mem_used*1024:.1f}MB")

    # Calculate sparsity and agreement
    total_params = sum(p.numel() for p in model.parameters())
    current_sparsity = sum(m.sum().item() for m in current_masks.values()) / total_params
    opt_sparsity = sum(m.sum().item() for m in opt_masks.values()) / total_params

    # Check mask agreement (only for weight matrices)
    total_weights = 0
    total_agree = 0
    for name in current_masks:
        if name in opt_masks:
            total_weights += current_masks[name].numel()
            total_agree += (current_masks[name] == opt_masks[name]).sum().item()

    agreement = total_agree / total_weights if total_weights > 0 else 1.0

    print()
    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\nSparsity achieved:")
    print(f"  Target:     {sparsity:.4%}")
    print(f"  Current:    {current_sparsity:.4%} (error: {abs(current_sparsity - sparsity):.4%})")
    print(f"  Optimized:  {opt_sparsity:.4%} (error: {abs(opt_sparsity - sparsity):.4%})")

    print(f"\nMask agreement: {agreement:.4%}")

    print(f"\nTime (average over {num_trials} trials):")
    avg_current_time = sum(current_times) / len(current_times)
    avg_opt_time = sum(opt_times) / len(opt_times)
    print(f"  Current:    {avg_current_time:.3f}s")
    print(f"  Optimized:  {avg_opt_time:.3f}s")
    speedup = avg_current_time / avg_opt_time if avg_opt_time > 0 else 0
    print(f"  Speedup:    {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'}")

    print(f"\nMemory (average over {num_trials} trials):")
    avg_current_mem = sum(current_mems) / len(current_mems)
    avg_opt_mem = sum(opt_mems) / len(opt_mems)
    print(f"  Current:    {avg_current_mem*1024:.1f}MB")
    print(f"  Optimized:  {avg_opt_mem*1024:.1f}MB")
    if avg_current_mem > 0:
        mem_savings = (avg_current_mem - avg_opt_mem) / avg_current_mem * 100
        print(f"  Savings:    {mem_savings:.1f}%")
    else:
        mem_savings = 0

    print(f"\n{'='*70}\n")

    return {
        'sparsity': sparsity,
        'speedup': speedup,
        'mem_savings': mem_savings,
        'agreement': agreement,
        'current_time': avg_current_time,
        'opt_time': avg_opt_time
    }


if __name__ == "__main__":
    print("Loading TinyStories-33M model...")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    print(f"Model loaded successfully\n")

    sparsities = [0.1, 0.3, 0.5]
    results = []

    for sparsity in sparsities:
        result = compare_strategies(model, sparsity, num_trials=3)
        results.append(result)

    # Summary
    print(f"\n{'#'*70}")
    print("# SUMMARY")
    print(f"{'#'*70}\n")

    print(f"{'Sparsity':<12} {'Current Time':<15} {'Opt Time':<15} {'Speedup':<12} {'Mem Savings':<15} {'Agreement':<12}")
    print("-" * 90)
    for r in results:
        print(f"{r['sparsity']:<12.1%} {r['current_time']:<15.3f}s {r['opt_time']:<15.3f}s "
              f"{r['speedup']:<12.2f}x {r['mem_savings']:<15.1f}% {r['agreement']:<12.4%}")

    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    avg_mem_savings = sum(r['mem_savings'] for r in results) / len(results)
    avg_agreement = sum(r['agreement'] for r in results) / len(results)

    print("\n" + "="*90)
    print("OVERALL AVERAGES:")
    print(f"  Speedup:        {avg_speedup:.2f}x")
    print(f"  Memory savings: {avg_mem_savings:.1f}%")
    print(f"  Mask agreement: {avg_agreement:.4%}")
    print("="*90)

    if avg_speedup > 3.0 and avg_agreement > 0.995:
        print("\n✅ OPTIMIZATION HIGHLY SUCCESSFUL!")
        print("   Ready to replace current magnitude implementation.")
    elif avg_speedup > 1.5 and avg_agreement > 0.99:
        print("\n✅ OPTIMIZATION SUCCESSFUL!")
        print("   Significant improvement, consider replacing current implementation.")
    else:
        print("\n⚠️  OPTIMIZATION NEEDS MORE WORK")
        print("   May not be worth replacing current implementation.")
