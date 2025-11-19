"""
Memory profiler for GradProbe algorithm.

This script profiles memory usage during the pruning process to identify
where memory is being consumed (weights, gradients, activations, etc.)
"""

import torch
import torch.nn as nn
import sys
import os
import gc
import tracemalloc
from dataclasses import dataclass
from typing import Dict, List
import psutil
from gradprobe import get_logger

@dataclass
class MemorySnapshot:
    """Stores memory usage information at a specific point in time."""
    stage: str
    process_rss_mb: float  # Process resident set size in MB
    process_vms_mb: float  # Process virtual memory size in MB
    python_allocated_mb: float  # Python tracemalloc allocated memory in MB
    torch_allocated_mb: float  # PyTorch allocated memory in MB
    torch_reserved_mb: float  # PyTorch reserved memory in MB
    torch_cached_mb: float  # PyTorch cached memory in MB
    tensors: Dict[str, int]  # Name -> tensor size in bytes


class MemoryProfiler:
    """Profiles memory usage throughout the pruning process."""

    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.process = psutil.Process(os.getpid())
        tracemalloc.start()

    def get_process_memory(self):
        """Get current process memory usage."""
        mem_info = self.process.memory_info()
        return mem_info.rss / 1024 / 1024, mem_info.vms / 1024 / 1024

    def get_python_memory(self):
        """Get Python allocated memory from tracemalloc."""
        current, peak = tracemalloc.get_traced_memory()
        return current / 1024 / 1024

    def get_torch_memory(self):
        """Get PyTorch memory usage."""
        # These functions work regardless of device
        allocated = 0
        reserved = 0
        cached = 0

        # Check if CUDA is available and has allocated memory
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0:
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            # Cached is reserved - allocated
            cached = reserved - allocated

        return allocated, reserved, cached

    def get_tensor_memory(self, name_prefix="", obj=None) -> Dict[str, int]:
        """Get memory used by all tensor objects.

        Args:
            name_prefix: Prefix to add to tensor names
            obj: Object to scan for tensors (defaults to all Python objects)

        Returns:
            Dictionary mapping tensor names to sizes in bytes
        """
        tensors = {}

        if obj is None:
            # Scan all Python objects
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj):
                        size = obj.element_size() * obj.nelement()
                        shape = tuple(obj.shape)
                        dtype = str(obj.dtype)
                        device = str(obj.device)
                        name = f"{name_prefix}tensor_{shape}_{dtype}_{device}"

                        if name in tensors:
                            tensors[name] += size
                        else:
                            tensors[name] = size
                except:
                    pass
        else:
            # Scan specific object
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if torch.is_tensor(value):
                        size = value.element_size() * value.nelement()
                        tensors[f"{name_prefix}{key}"] = size
                    elif isinstance(value, dict):
                        tensors.update(self.get_tensor_memory(f"{name_prefix}{key}.", value))

        return tensors

    def snapshot(self, stage: str, tensors_dict: Dict = None):
        """Take a memory snapshot at current point.

        Args:
            stage: Description of current stage
            tensors_dict: Optional dictionary of named tensors to track
        """
        gc.collect()

        rss, vms = self.get_process_memory()
        python_mem = self.get_python_memory()
        torch_allocated, torch_reserved, torch_cached = self.get_torch_memory()

        if tensors_dict:
            tensors = self.get_tensor_memory(obj=tensors_dict)
        else:
            tensors = {}

        snapshot = MemorySnapshot(
            stage=stage,
            process_rss_mb=rss,
            process_vms_mb=vms,
            python_allocated_mb=python_mem,
            torch_allocated_mb=torch_allocated,
            torch_reserved_mb=torch_reserved,
            torch_cached_mb=torch_cached,
            tensors=tensors
        )

        self.snapshots.append(snapshot)
        return snapshot

    def print_snapshot(self, snapshot: MemorySnapshot):
        """Print a single snapshot."""
        get_logger().memory(f"\n{'='*70}")
        get_logger().memory(f"Stage: {snapshot.stage}")
        get_logger().memory(f"{'='*70}")
        get_logger().memory(f"Process Memory:")
        get_logger().memory(f"  RSS: {snapshot.process_rss_mb:.2f} MB")
        get_logger().memory(f"  VMS: {snapshot.process_vms_mb:.2f} MB")
        get_logger().memory(f"Python Allocated: {snapshot.python_allocated_mb:.2f} MB")
        get_logger().memory(f"PyTorch Memory:")
        get_logger().memory(f"  Allocated: {snapshot.torch_allocated_mb:.2f} MB")
        get_logger().memory(f"  Reserved: {snapshot.torch_reserved_mb:.2f} MB")
        get_logger().memory(f"  Cached: {snapshot.torch_cached_mb:.2f} MB")

        if snapshot.tensors:
            get_logger().memory(f"\nTracked Tensors:")
            total_tensor_mb = 0
            for name, size_bytes in sorted(snapshot.tensors.items(), key=lambda x: x[1], reverse=True)[:10]:
                size_mb = size_bytes / 1024 / 1024
                total_tensor_mb += size_mb
                get_logger().memory(f"  {name}: {size_mb:.2f} MB")
            get_logger().memory(f"  Total (top 10): {total_tensor_mb:.2f} MB")

    def print_all_snapshots(self):
        """Print all snapshots."""
        for snapshot in self.snapshots:
            self.print_snapshot(snapshot)

    def print_summary(self):
        """Print summary comparing all snapshots."""
        if not self.snapshots:
            get_logger().memory("No snapshots taken")
            return

        get_logger().memory(f"\n{'='*70}")
        get_logger().memory("MEMORY PROFILE SUMMARY")
        get_logger().memory(f"{'='*70}")
        get_logger().memory(f"{'Stage':<40} {'Process RSS (MB)':<20} {'Delta (MB)':<15}")
        get_logger().memory(f"{'-'*70}")

        baseline_rss = self.snapshots[0].process_rss_mb
        for snapshot in self.snapshots:
            delta = snapshot.process_rss_mb - baseline_rss
            get_logger().memory(f"{snapshot.stage:<40} {snapshot.process_rss_mb:>15.2f} {delta:>15.2f}")

        get_logger().memory(f"\n{'='*70}")
        get_logger().memory("PEAK MEMORY USAGE")
        get_logger().memory(f"{'='*70}")

        max_snapshot = max(self.snapshots, key=lambda s: s.process_rss_mb)
        get_logger().memory(f"Stage: {max_snapshot.stage}")
        get_logger().memory(f"Process RSS: {max_snapshot.process_rss_mb:.2f} MB")
        get_logger().memory(f"Process VMS: {max_snapshot.process_vms_mb:.2f} MB")
        get_logger().memory(f"Python Allocated: {max_snapshot.python_allocated_mb:.2f} MB")
        get_logger().memory(f"PyTorch Allocated: {max_snapshot.torch_allocated_mb:.2f} MB")
        get_logger().memory(f"PyTorch Reserved: {max_snapshot.torch_reserved_mb:.2f} MB")

        get_logger().memory(f"\n{'='*70}")
        get_logger().memory("MEMORY INCREASE BY STAGE")
        get_logger().memory(f"{'='*70}")

        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]
            delta_rss = curr.process_rss_mb - prev.process_rss_mb
            delta_torch = curr.torch_allocated_mb - prev.torch_allocated_mb

            get_logger().memory(f"\n{prev.stage} -> {curr.stage}")
            get_logger().memory(f"  Process RSS delta: {delta_rss:+.2f} MB")
            get_logger().memory(f"  PyTorch allocated delta: {delta_torch:+.2f} MB")


def analyze_model_memory(model):
    """Analyze memory used by model parameters."""
    total_params = 0
    total_bytes = 0

    get_logger().memory(f"\n{'='*70}")
    get_logger().memory("MODEL PARAMETER MEMORY ANALYSIS")
    get_logger().memory(f"{'='*70}")
    get_logger().memory(f"{'Parameter Name':<50} {'Shape':<25} {'Size (MB)':<15}")
    get_logger().memory(f"{'-'*70}")

    param_info = []
    for name, param in model.named_parameters():
        num_params = param.numel()
        num_bytes = num_params * param.element_size()
        total_params += num_params
        total_bytes += num_bytes

        param_info.append((name, str(tuple(param.shape)), num_bytes))

    # Sort by size
    param_info.sort(key=lambda x: x[2], reverse=True)

    for name, shape, num_bytes in param_info:
        size_mb = num_bytes / 1024 / 1024
        get_logger().memory(f"{name:<50} {shape:<25} {size_mb:>12.2f}")

    get_logger().memory(f"{'-'*70}")
    get_logger().memory(f"{'TOTAL':<50} {total_params:>25,} {total_bytes/1024/1024:>12.2f}")
    get_logger().memory(f"{'='*70}")

    return total_params, total_bytes


def analyze_gradient_memory(gradients_dict):
    """Analyze memory used by gradients."""
    total_bytes = 0

    get_logger().memory(f"\n{'='*70}")
    get_logger().memory("GRADIENT MEMORY ANALYSIS")
    get_logger().memory(f"{'='*70}")
    get_logger().memory(f"{'Parameter Name':<50} {'Shape':<25} {'Size (MB)':<15}")
    get_logger().memory(f"{'-'*70}")

    grad_info = []
    for name, grad in gradients_dict.items():
        if grad is not None:
            num_bytes = grad.element_size() * grad.nelement()
            total_bytes += num_bytes
            grad_info.append((name, str(tuple(grad.shape)), num_bytes))

    # Sort by size
    grad_info.sort(key=lambda x: x[2], reverse=True)

    for name, shape, num_bytes in grad_info:
        size_mb = num_bytes / 1024 / 1024
        get_logger().memory(f"{name:<50} {shape:<25} {size_mb:>12.2f}")

    get_logger().memory(f"{'-'*70}")
    get_logger().memory(f"{'TOTAL':<50} {'':>25} {total_bytes/1024/1024:>12.2f}")
    get_logger().memory(f"{'='*70}")

    return total_bytes


def estimate_activation_memory(model, batch_size, seq_length):
    """Estimate activation memory for transformer models."""
    get_logger().memory(f"\n{'='*70}")
    get_logger().memory("ACTIVATION MEMORY ESTIMATION")
    get_logger().memory(f"{'='*70}")

    # Get model config if available
    if hasattr(model, 'config'):
        config = model.config
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads

        get_logger().memory(f"Model: {config.model_type}")
        get_logger().memory(f"Hidden size: {hidden_size}")
        get_logger().memory(f"Number of layers: {num_layers}")
        get_logger().memory(f"Number of attention heads: {num_heads}")
        get_logger().memory(f"Batch size: {batch_size}")
        get_logger().memory(f"Sequence length: {seq_length}")
        get_logger().memory("")

        # Estimate activation memory per layer
        # Each layer has:
        # - Hidden states: batch * seq * hidden
        # - Attention scores: batch * heads * seq * seq
        # - Attention output: batch * seq * hidden
        # - FFN intermediate: batch * seq * (4 * hidden) typically

        hidden_states_mb = (batch_size * seq_length * hidden_size * 4) / 1024 / 1024  # 4 bytes for float32
        attention_scores_mb = (batch_size * num_heads * seq_length * seq_length * 4) / 1024 / 1024
        ffn_intermediate_mb = (batch_size * seq_length * 4 * hidden_size * 4) / 1024 / 1024

        per_layer_mb = hidden_states_mb + attention_scores_mb + ffn_intermediate_mb
        total_activation_mb = per_layer_mb * num_layers

        get_logger().memory(f"Estimated memory per layer:")
        get_logger().memory(f"  Hidden states: {hidden_states_mb:.2f} MB")
        get_logger().memory(f"  Attention scores: {attention_scores_mb:.2f} MB")
        get_logger().memory(f"  FFN intermediate: {ffn_intermediate_mb:.2f} MB")
        get_logger().memory(f"  Total per layer: {per_layer_mb:.2f} MB")
        get_logger().memory(f"\nTotal activation memory (all layers): {total_activation_mb:.2f} MB")
        get_logger().memory(f"{'='*70}")

        return total_activation_mb
    else:
        get_logger().memory("Model config not available for activation estimation")
        get_logger().memory(f"{'='*70}")
        return 0


if __name__ == "__main__":
    from gradprobe import Logger, LogLevel
    logger = Logger(program_name='profile_memory', level=LogLevel.INFO)

    logger.info("Memory profiler utility for GradProbe")
    logger.info("Usage: import this module and use MemoryProfiler class")
    logger.info("\nExample:")
    logger.info("  profiler = MemoryProfiler()")
    logger.info("  profiler.snapshot('initial')")
    logger.info("  # ... do some work ...")
    logger.info("  profiler.snapshot('after_work')")
    logger.info("  profiler.print_summary()")
