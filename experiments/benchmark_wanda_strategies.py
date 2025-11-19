"""
Benchmark current WANDA strategy vs optimized version on full model.
"""

import sys
sys.path.insert(0, '/home/user/GradProbe')

import torch
import time
import gc
import psutil
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader, TensorDataset

from gradprobe.strategies.wanda import WANDAPruning
from gradprobe.strategies.wanda_optimized import WANDAPruningOptimized
from gradprobe.logger import Logger, LogLevel

# Initialize logger
logger = Logger(program_name='benchmark_wanda_strategies', level=LogLevel.INFO)


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def create_dummy_dataloader(model, num_samples=10, seq_length=128):
    """Create a dummy dataloader for activation collection."""
    vocab_size = model.config.vocab_size

    # Create random token sequences
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    dataset = TensorDataset(input_ids)

    return DataLoader(dataset, batch_size=2, shuffle=False)


def compare_strategies(model, dataloader, sparsity, num_trials=3):
    """Compare current vs optimized WANDA strategy."""

    logger.info(f"\n{'='*70}")
    logger.info(f"Benchmarking WANDA Strategies")
    logger.info(f"{'='*70}")
    logger.info(f"Model: TinyStories-33M")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Target sparsity: {sparsity:.1%}")
    logger.info(f"Dataloader: {len(dataloader)} batches for activation collection")
    logger.info(f"Trials: {num_trials}\n")

    # Test current strategy
    logger.info("Testing CURRENT WANDAPruning...")
    current_times = []
    current_mems = []
    current_masks = None

    for i in range(num_trials):
        gc.collect()

        # Create fresh strategy instance (to avoid activation caching across trials)
        current_strategy = WANDAPruning(dataloader, num_batches=5)

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

        logger.memory(f"  Trial {i+1}: {time_taken:.3f}s, {mem_used*1024:.1f}MB")

    logger.info()

    # Test optimized strategy
    logger.info("Testing OPTIMIZED WANDAPruningOptimized...")
    opt_times = []
    opt_mems = []
    opt_masks = None

    for i in range(num_trials):
        gc.collect()

        # Create fresh strategy instance
        opt_strategy = WANDAPruningOptimized(dataloader, num_batches=5)

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

        logger.memory(f"  Trial {i+1}: {time_taken:.3f}s, {mem_used*1024:.1f}MB")

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
    logger.info(f"  Current:    {avg_current_time:.3f}s")
    logger.info(f"  Optimized:  {avg_opt_time:.3f}s")
    speedup = avg_current_time / avg_opt_time if avg_opt_time > 0 else 0
    logger.info(f"  Speedup:    {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'}")

    logger.info(f"\nMemory (average over {num_trials} trials):")
    avg_current_mem = sum(current_mems) / len(current_mems)
    avg_opt_mem = sum(opt_mems) / len(opt_mems)
    logger.memory(f"  Current:    {avg_current_mem*1024:.1f}MB")
    logger.memory(f"  Optimized:  {avg_opt_mem*1024:.1f}MB")
    if avg_current_mem > 0:
        mem_savings = (avg_current_mem - avg_opt_mem) / avg_current_mem * 100
        logger.info(f"  Savings:    {mem_savings:.1f}%")
    else:
        mem_savings = 0

    logger.info(f"\n{'='*70}\n")

    return {
        'sparsity': sparsity,
        'speedup': speedup,
        'mem_savings': mem_savings,
        'agreement': agreement,
        'current_time': avg_current_time,
        'opt_time': avg_opt_time
    }


if __name__ == "__main__":
    logger.info("Loading TinyStories-33M model...")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    logger.info(f"Model loaded successfully\n")

    logger.info("Creating dummy dataloader for activation collection...")
    dataloader = create_dummy_dataloader(model, num_samples=10, seq_length=128)
    logger.info(f"Dataloader created: {len(dataloader)} batches\n")

    sparsities = [0.1, 0.3, 0.5]
    results = []

    for sparsity in sparsities:
        result = compare_strategies(model, dataloader, sparsity, num_trials=3)
        results.append(result)

    # Summary
    logger.info(f"\n{'#'*70}")
    logger.info("# SUMMARY")
    logger.info(f"{'#'*70}\n")

    logger.info(f"{'Sparsity':<12} {'Current Time':<15} {'Opt Time':<15} {'Speedup':<12} {'Mem Savings':<15} {'Agreement':<12}")
    logger.info("-" * 90)
    for r in results:
        logger.info(f"{r['sparsity']:<12.1%} {r['current_time']:<15.3f}s {r['opt_time']:<15.3f}s "
              f"{r['speedup']:<12.2f}x {r['mem_savings']:<15.1f}% {r['agreement']:<12.4%}")

    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    avg_mem_savings = sum(r['mem_savings'] for r in results) / len(results)
    avg_agreement = sum(r['agreement'] for r in results) / len(results)

    logger.info("\n" + "="*90)
    logger.info("OVERALL AVERAGES:")
    logger.info(f"  Speedup:        {avg_speedup:.2f}x")
    logger.info(f"  Memory savings: {avg_mem_savings:.1f}%")
    logger.info(f"  Mask agreement: {avg_agreement:.4%}")
    logger.info("="*90)

    if avg_speedup > 3.0 and avg_agreement > 0.99:
        logger.info("\n✅ OPTIMIZATION HIGHLY SUCCESSFUL!")
        logger.info("   Ready to replace current WANDA implementation.")
    elif avg_speedup > 1.5 and avg_agreement > 0.98:
        logger.info("\n✅ OPTIMIZATION SUCCESSFUL!")
        logger.info("   Significant improvement, consider replacing current implementation.")
    else:
        logger.info("\n⚠️  OPTIMIZATION NEEDS MORE WORK")
        logger.info("   May not be worth replacing current implementation.")
