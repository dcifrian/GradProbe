# GradProbe: Gradient-based Neural Network Pruner

GradProbe is a sophisticated neural network pruning library that uses gradient information to make intelligent pruning decisions. Unlike traditional pruning methods that simply remove weights based on magnitude or other static criteria, GradProbe uses a novel gradient-comparison algorithm to determine which weights are truly redundant.

## Algorithm Overview

The GradProbe algorithm works as follows:

1. **Initial Gradient Recording**: Perform a forward pass with the original network and record gradients for all parameters
2. **Tentative Selection**: Use a pluggable pruning strategy (e.g., magnitude-based) to identify candidate weights for pruning
3. **Weight Reduction**: Instead of immediately setting selected weights to zero, reduce them to 1/10 of their original value
4. **Gradient Comparison**: Perform another forward pass and compare the new gradients with the original gradients
5. **Smart Pruning Decision**:
   - If a weight's gradient **decreased or stayed the same** → the weight is redundant, set it to **zero**
   - If a weight's gradient **increased** → the weight is important, **restore** it to its original value

This approach is more intelligent than naive pruning because it considers how the gradient changes when a weight is reduced, providing insight into the weight's actual importance to the model's learning dynamics.

## Features

- **Gradient-aware pruning**: Uses gradient information to make smart pruning decisions
- **Pluggable strategies**: Easy to add new weight selection strategies (currently supports magnitude-based, WANDA coming soon)
- **Flexible architecture**: Works with any PyTorch model
- **GPU/CPU support**: Works efficiently on both GPU and CPU
- **Comprehensive testing**: Includes test suite and example scripts

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GradProbe.git
cd GradProbe

# Install PyTorch and dependencies
pip install torch
pip install -r requirements.txt
```

## Quick Start

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gradprobe import GradProbe, MagnitudePruning, SimpleMLP

# Create your model
model = SimpleMLP(input_dim=784, hidden_dims=[512, 256], output_dim=10)

# Create a pruning strategy
strategy = MagnitudePruning()

# Initialize GradProbe
pruner = GradProbe(model, strategy, device='cpu')

# Prepare your data
# dataloader = DataLoader(your_dataset, batch_size=32, shuffle=True)

# Prune the model
pruning_masks = pruner.prune(
    dataloader=dataloader,
    loss_fn=nn.CrossEntropyLoss(),
    sparsity=0.5,  # Remove 50% of weights
    num_batches=10,  # Use 10 batches for gradient computation
    reduction_factor=0.1,  # Reduce tentative weights to 1/10
    gradient_threshold=0.0,  # Only prune if gradient didn't increase
    verbose=True
)

# Check the results
print(f"Actual sparsity: {pruner.get_sparsity():.2%}")

# Make pruning permanent (optional)
pruner.apply_mask_permanently()
```

### Testing Different Gradient Thresholds

The `gradient_threshold` parameter controls how aggressive the pruning is:
- `0.0` (default): Only prune weights where gradient decreased or stayed same
- `0.1`: Allow up to 10% gradient increase before restoring a weight
- Higher values: More aggressive pruning, may impact accuracy more

You can sweep different thresholds to understand the trade-off:

```python
# Define evaluation function
def eval_accuracy(model):
    return evaluate_model(model, test_loader)

# Sweep different thresholds
results = pruner.sweep_gradient_thresholds(
    dataloader=train_loader,
    loss_fn=nn.CrossEntropyLoss(),
    eval_fn=eval_accuracy,
    sparsity=0.5,
    thresholds=[0.0, 0.05, 0.1, 0.2, 0.5],
    num_batches=10,
    verbose=True
)

# Results contains:
# - 'thresholds': tested threshold values
# - 'sparsities': actual sparsity achieved for each
# - 'weights_pruned': number of weights pruned
# - 'metrics': accuracy/metric for each threshold
```

## Running the Examples

### Basic MLP Example

```bash
python examples/prune_mlp.py
```

This script demonstrates the complete workflow:
1. Creates a simple MLP
2. Trains it on synthetic data
3. Prunes the model using GradProbe
4. Evaluates performance before and after pruning
5. Fine-tunes the pruned model
6. Sweeps different gradient thresholds to analyze trade-offs

### Running Tests

```bash
python tests/test_pruner.py
```

The test suite includes:
- Magnitude pruning strategy tests
- GradProbe basic functionality tests
- Gradient comparison logic tests
- Permanent masking tests

## Project Structure

```
GradProbe/
├── gradprobe/
│   ├── __init__.py
│   ├── pruner.py              # Main GradProbe implementation
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py            # Base pruning strategy interface
│   │   └── magnitude.py       # Magnitude-based pruning
│   └── models/
│       ├── __init__.py
│       └── mlp.py             # Simple MLP models for testing
├── examples/
│   └── prune_mlp.py           # Complete example script
├── tests/
│   └── test_pruner.py         # Test suite
├── requirements.txt
└── README.md
```

## Adding New Pruning Strategies

GradProbe is designed to be extensible. You can easily add new pruning strategies by inheriting from `PruningStrategy`:

```python
from gradprobe.strategies.base import PruningStrategy
import torch
import torch.nn as nn

class MyCustomStrategy(PruningStrategy):
    def select_weights_to_prune(self, model: nn.Module, sparsity: float):
        # Your custom logic here
        masks = {}
        for name, param in model.named_parameters():
            # Create boolean mask where True = prune this weight
            masks[name] = your_selection_logic(param, sparsity)
        return masks

    def get_name(self) -> str:
        return "my_custom_strategy"

# Use it
strategy = MyCustomStrategy()
pruner = GradProbe(model, strategy)
```

## Future Work

- [ ] Implement WANDA pruning strategy
- [ ] Add support for structured pruning
- [ ] Integration with HuggingFace TinyStories-33M model
- [ ] Support for iterative pruning
- [ ] Pruning schedule support
- [ ] More sophisticated gradient comparison metrics
- [ ] Visualization tools for pruning analysis

## Understanding the Algorithm

### Why Gradient Comparison?

Traditional magnitude-based pruning assumes that small weights are unimportant. However, this is not always true. A small weight might be:
- Important for gradient flow during training
- Critical for specific input patterns
- Part of a redundant pathway that's already well-represented

By observing how gradients change when we reduce a weight, we get insight into:
- **Gradient decreases**: The optimization is "giving up" on this weight → it's truly redundant
- **Gradient increases**: The optimization is "fighting" to recover this weight → it's important

### Parameters

- **sparsity**: Target fraction of weights to prune (0.0 to 1.0)
- **num_batches**: More batches = more accurate gradient estimation but slower
- **reduction_factor**: How much to reduce tentative weights (default: 0.1 = reduce to 10%)
  - Smaller values make the test more conservative
  - Larger values make the test more aggressive
- **gradient_threshold**: Relative gradient increase tolerance (default: 0.0)
  - 0.0 = only prune if gradient decreased/stayed same
  - 0.1 = allow up to 10% gradient increase
  - Higher values = more aggressive pruning but may impact accuracy
  - Use `sweep_gradient_thresholds()` to find optimal value

## Citation

If you use GradProbe in your research, please cite:

```bibtex
@software{gradprobe2024,
  title={GradProbe: Gradient-based Neural Network Pruning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/GradProbe}
}
```

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas where we'd especially appreciate contributions:

- New pruning strategies (WANDA, activation-based, etc.)
- Support for more model architectures
- Performance optimizations
- Better visualization tools
- Documentation improvements

## Acknowledgments

This project implements a novel gradient-based pruning approach. The pluggable architecture was inspired by modern pruning libraries, and the implementation uses PyTorch for efficient computation.
