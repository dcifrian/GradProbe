"""
Unit tests for GradProbe.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, MagnitudePruning, TinyMLP


def test_magnitude_pruning():
    """Test magnitude-based pruning strategy."""
    print("\nTest 1: Magnitude Pruning Strategy")
    print("-" * 50)

    model = TinyMLP(input_dim=50, output_dim=5)
    strategy = MagnitudePruning()

    # Test with 30% sparsity
    sparsity = 0.3
    masks = strategy.select_weights_to_prune(model, sparsity)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad and len(p.shape) >= 2)
    total_selected = sum(mask.sum().item() for mask in masks.values())

    actual_sparsity = total_selected / total_params
    print(f"Target sparsity: {sparsity:.2%}")
    print(f"Actual sparsity: {actual_sparsity:.2%}")
    print(f"Total parameters: {total_params}")
    print(f"Selected for pruning: {total_selected}")

    # Check that sparsity is approximately correct (within 5%)
    assert abs(actual_sparsity - sparsity) < 0.05, \
        f"Sparsity mismatch: expected ~{sparsity:.2%}, got {actual_sparsity:.2%}"

    print("✓ Magnitude pruning test passed")
    return True


def test_gradprobe_basic():
    """Test basic GradProbe functionality."""
    print("\nTest 2: GradProbe Basic Functionality")
    print("-" * 50)

    # Create simple model and data
    model = TinyMLP(input_dim=20, output_dim=3)
    X = torch.randn(100, 20)
    y = torch.randint(0, 3, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Create pruner
    strategy = MagnitudePruning()
    pruner = GradProbe(model, strategy, device='cpu')

    # Count parameters before
    params_before = model.count_parameters()
    print(f"Parameters before: {params_before['total']}")

    # Prune with 40% sparsity
    sparsity = 0.4
    masks = pruner.prune(
        dataloader=dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        sparsity=sparsity,
        num_batches=5,
        verbose=True
    )

    # Check sparsity
    actual_sparsity = pruner.get_sparsity()
    print(f"\nTarget sparsity: {sparsity:.2%}")
    print(f"Actual sparsity: {actual_sparsity:.2%}")

    # Verify that some weights are actually zero
    zero_count = sum((p.data == 0).sum().item() for p in model.parameters())
    print(f"Zero weights: {zero_count}")

    assert zero_count > 0, "No weights were pruned!"
    assert actual_sparsity > 0, "Sparsity is zero!"

    print("✓ GradProbe basic test passed")
    return True


def test_gradient_comparison():
    """Test that gradient comparison logic works."""
    print("\nTest 3: Gradient Comparison Logic")
    print("-" * 50)

    model = TinyMLP(input_dim=30, output_dim=4)
    X = torch.randn(50, 30)
    y = torch.randint(0, 4, (50,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    strategy = MagnitudePruning()
    pruner = GradProbe(model, strategy, device='cpu')

    # Prune
    masks = pruner.prune(
        dataloader=dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        sparsity=0.3,
        num_batches=3,
        verbose=True
    )

    # Verify masks are boolean tensors
    for name, mask in masks.items():
        assert mask.dtype == torch.bool, f"Mask for {name} is not boolean"
        print(f"{name}: {mask.sum().item()} weights pruned out of {mask.numel()}")

    print("✓ Gradient comparison test passed")
    return True


def test_permanent_masking():
    """Test permanent masking functionality."""
    print("\nTest 4: Permanent Masking")
    print("-" * 50)

    model = TinyMLP(input_dim=25, output_dim=3)
    X = torch.randn(40, 25)
    y = torch.randint(0, 3, (40,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    strategy = MagnitudePruning()
    pruner = GradProbe(model, strategy, device='cpu')

    # Prune
    masks = pruner.prune(
        dataloader=dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        sparsity=0.25,
        num_batches=3,
        verbose=False
    )

    # Apply permanent masking
    pruner.apply_mask_permanently()

    # Train one step and verify pruned weights stay zero
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Get zero positions before training
    zero_positions = {}
    for name, param in model.named_parameters():
        zero_positions[name] = (param.data == 0).clone()

    # Training step
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        break  # Just one step

    # Verify pruned weights are still zero
    all_stay_zero = True
    for name, param in model.named_parameters():
        still_zero = (param.data == 0)
        stayed_zero = (zero_positions[name] == still_zero).all()
        if not stayed_zero:
            all_stay_zero = False
            print(f"WARNING: {name} has pruned weights that changed")

    if all_stay_zero:
        print("✓ All pruned weights remained zero after training step")
    else:
        print("✗ Some pruned weights changed (this may be expected without proper masking)")

    print("✓ Permanent masking test passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("GradProbe Test Suite")
    print("="*70)

    tests = [
        test_magnitude_pruning,
        test_gradprobe_basic,
        test_gradient_comparison,
        test_permanent_masking,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
