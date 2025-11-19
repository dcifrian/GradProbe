"""
Example script demonstrating GradProbe on a simple MLP.

This script trains a small MLP on synthetic data and then prunes it
using the gradient-comparison algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, MagnitudePruning, SimpleMLP, Logger, LogLevel, get_logger


def generate_synthetic_data(num_samples=1000, input_dim=100, output_dim=10, pattern_weights=None):
    """
    Generate synthetic classification data with a learnable pattern.

    Args:
        num_samples: Number of samples to generate
        input_dim: Input dimension
        output_dim: Number of classes
        pattern_weights: Shared weight matrix to define the pattern (if None, create new)

    Returns:
        DataLoader with synthetic data, pattern_weights used
    """
    # Generate random input features
    X = torch.randn(num_samples, input_dim)

    # Create or use existing pattern
    if pattern_weights is None:
        pattern_weights = torch.randn(input_dim, output_dim)

    # Create labels based on the pattern
    scores = X @ pattern_weights
    y = scores.argmax(dim=1)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader, pattern_weights


def train_model(model, dataloader, epochs=5, lr=0.001):
    """
    Train the model on the given data.

    Args:
        model: Model to train
        dataloader: Training data
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use

    Returns:
        Trained model
    """
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    get_logger().info(f"\nTraining model for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        get_logger().info(f"Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model


def evaluate_model(model, dataloader, device=None):
    """
    Evaluate model accuracy.

    Args:
        model: Model to evaluate
        dataloader: Test data
        device: Device to use (optional, auto-detected from model if not provided)

    Returns:
        Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0

    # Auto-detect device from model if not provided
    if device is None:
        device = next(model.parameters()).device

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def main():
    """Main function."""
    logger = Logger(program_name='prune_mlp', level=LogLevel.INFO)

    logger.info("="*70)
    logger.info("GradProbe: Gradient-based Neural Network Pruning")
    logger.info("="*70)

    # Configuration
    device = 'cpu'
    input_dim = 100
    output_dim = 10
    hidden_dims = [128, 64]
    sparsity = 0.5  # Prune 50% of weights

    # Generate synthetic data
    # Note: For synthetic data, we evaluate on the training set to see the
    # impact of pruning. With real models/data, use a separate test set.
    logger.info("\n1. Generating synthetic data...")
    train_loader, pattern_weights = generate_synthetic_data(
        num_samples=1000,
        input_dim=input_dim,
        output_dim=output_dim
    )

    # Create model
    logger.info("\n2. Creating model...")
    model = SimpleMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation='relu',
        dropout=0.1
    )

    param_counts = model.count_parameters()
    logger.info(f"Model created with {param_counts['total']} total parameters")
    logger.info(f"Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")

    # Train model
    logger.info("\n3. Training model...")
    model = train_model(model, train_loader, epochs=100, lr=0.001, device=device)
    model = train_model(model, train_loader, epochs=100, lr=0.0001, device=device)

    # Evaluate before pruning (on training set for synthetic data)
    logger.info("\n4. Evaluating model before pruning...")
    accuracy_before = evaluate_model(model, train_loader, device=device)
    logger.info(f"Training accuracy before pruning: {accuracy_before:.2f}%")

    # Create pruner
    logger.info("\n5. Creating GradProbe pruner...")
    strategy = MagnitudePruning()
    pruner = GradProbe(model, strategy, device=device)

    # Prune the model
    logger.info(f"\n6. Pruning model with target sparsity: {sparsity:.1%}...")
    logger.info("-" * 70)
    pruning_masks = pruner.prune(
        dataloader=train_loader,
        loss_fn=nn.CrossEntropyLoss(),
        sparsity=sparsity,
        num_batches=10,  # Use 10 batches for gradient computation
        reduction_factor=0.1,  # Reduce weights to 1/10
        verbose=True
    )
    logger.info("-" * 70)

    # Evaluate after pruning
    logger.info("\n7. Evaluating model after pruning...")
    accuracy_after = evaluate_model(model, train_loader, device=device)
    logger.info(f"Training accuracy after pruning: {accuracy_after:.2f}%")

    # Print comparison
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)
    actual_sparsity = pruner.get_sparsity()
    logger.info(f"Target sparsity:        {sparsity:.2%}")
    logger.info(f"Actual sparsity:        {actual_sparsity:.2%}")
    logger.info(f"Accuracy before:        {accuracy_before:.2f}%")
    logger.info(f"Accuracy after:         {accuracy_after:.2f}%")
    logger.info(f"Accuracy change:        {accuracy_after - accuracy_before:+.2f}%")
    logger.info(f"Parameters remaining:   {(1-actual_sparsity)*param_counts['total']:.0f} / {param_counts['total']}")
    logger.info("="*70)

    # Optional: Fine-tune the pruned model
    logger.info("\n8. Fine-tuning pruned model...")
    pruner.apply_mask_permanently()  # Make pruning permanent
    model = train_model(model, train_loader, epochs=5, lr=0.0001, device=device)

    accuracy_finetuned = evaluate_model(model, train_loader, device=device)
    logger.info(f"\nTraining accuracy after fine-tuning: {accuracy_finetuned:.2f}%")
    logger.info(f"Final accuracy change:               {accuracy_finetuned - accuracy_before:+.2f}%")

    # Optional: Test different gradient thresholds
    logger.info("\n" + "="*70)
    logger.info("9. BONUS: Testing different gradient thresholds")
    logger.info("="*70)
    logger.info("This shows how threshold affects pruning aggressiveness vs accuracy")

    # Restore model to pre-pruning state for fair comparison
    # Re-train from scratch
    model = SimpleMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation='relu',
        dropout=0.1
    )
    model = train_model(model, train_loader, epochs=100, lr=0.001, device=device)
    model = train_model(model, train_loader, epochs=100, lr=0.0001, device=device)

    # Create fresh pruner
    strategy = MagnitudePruning()
    pruner = GradProbe(model, strategy, device=device)

    # Define evaluation function
    def eval_accuracy(model):
        return evaluate_model(model, train_loader, device=device)

    # Sweep thresholds
    sweep_results = pruner.sweep_gradient_thresholds(
        dataloader=train_loader,
        loss_fn=nn.CrossEntropyLoss(),
        eval_fn=eval_accuracy,
        sparsity=sparsity,
        thresholds=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
        num_batches=10,
        reduction_factor=0.1,
        verbose=True
    )

    logger.info("\n" + "="*70)
    logger.info("Done!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
