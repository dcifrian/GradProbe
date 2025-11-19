"""
Test WANDA pruning strategy on SimpleMLP.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, WANDAPruning, SimpleMLP, Logger, LogLevel

# Initialize logger
logger = Logger(program_name='test_wanda', level=LogLevel.INFO)

# Generate synthetic data
torch.manual_seed(42)
X = torch.randn(1000, 100)
pattern = torch.randn(100, 10)
y = (X @ pattern).argmax(dim=1)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
dataloader_pruning = DataLoader(dataset, batch_size=1, shuffle=False)  # For pruning

# Train model
model = SimpleMLP(input_dim=100, hidden_dims=[128, 64], output_dim=10, dropout=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

logger.info("Training model...")
for epoch in range(100):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

for epoch in range(100):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluation function
def eval_fn(m):
    m.eval()
    correct = 0
    total = 0
    # Detect model device
    device = next(m.parameters()).device
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = m(inputs)
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    return 100.0 * correct / total

logger.info(f"Trained accuracy: {eval_fn(model):.2f}%\n")

# Save model state
import copy
saved_state = copy.deepcopy(model.state_dict())

# Test WANDA pruning with iterative approach
logger.info("="*70)
logger.info("TEST: WANDA Iterative Pruning")
logger.info("="*70)
model.load_state_dict(saved_state)

# WANDA needs the dataloader to collect activations
wanda_strategy = WANDAPruning(dataloader=dataloader_pruning, num_batches=100)
pruner = GradProbe(model, wanda_strategy)

results = pruner.iterative_prune(
    dataloader=dataloader_pruning,
    loss_fn=nn.CrossEntropyLoss(),
    eval_fn=eval_fn,
    initial_sparsity=0.1,
    sparsity_step=0.1,
    max_accuracy_drop=1.0,  # Stop at 1 percentage point drop
    num_batches=320,  # batch_size=1, so 320 batches = 320 samples
    gradient_threshold=5.0,  # WANDA works well with layerwise and threshold 5.0
    layerwise=True,
    verbose=True,
    compare_baseline=True
)

logger.info("\n\n" + "="*70)
logger.info("COMPARISON SUMMARY")
logger.info("="*70)
logger.info(f"WANDA pruning:")
logger.info(f"  Final sparsity: {results['final_sparsity']:.2%}")
logger.info(f"  Final accuracy: {results['final_accuracy']:.2f}%")
logger.info("="*70)
logger.info("\nWANDA combines weight magnitude with activation norms to identify")
logger.info("unimportant weights. It should perform better than pure magnitude pruning")
logger.info("by considering how much each weight is actually used during inference.")
