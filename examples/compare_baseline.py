"""
Example demonstrating baseline comparison feature.

This shows the accuracy gain from gradient-based filtering vs magnitude-only pruning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, MagnitudePruning, SimpleMLP, Logger, LogLevel

# Initialize logger for this example
logger = Logger(program_name='compare_baseline', level=LogLevel.INFO)


# Generate synthetic data
torch.manual_seed(42)
X = torch.randn(1000, 100)
pattern = torch.randn(100, 10)
y = (X @ pattern).argmax(dim=1)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

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

# Test pruning with baseline comparison
logger.info("="*70)
logger.info("PRUNING WITH BASELINE COMPARISON")
logger.info("="*70)
model.load_state_dict(saved_state)
pruner = GradProbe(model, MagnitudePruning())

pruner.prune(
    dataloader=dataloader,
    loss_fn=nn.CrossEntropyLoss(),
    sparsity=0.5,
    num_batches=10,
    gradient_threshold=10.0,
    verbose=True,
    compare_baseline=True,
    eval_fn=eval_fn
)
