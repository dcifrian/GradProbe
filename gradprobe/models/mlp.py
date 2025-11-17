"""
Simple MLP models for testing GradProbe.
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron for testing pruning.

    This is a basic feedforward network that can be used for
    classification or regression tasks.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list = [512, 256, 128],
        output_dim: int = 10,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Initialize the MLP.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function ("relu", "tanh", "gelu")
            dropout: Dropout probability (0 = no dropout)
        """
        super(SimpleMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


class TinyMLP(nn.Module):
    """
    A very small MLP for quick testing.
    """

    def __init__(self, input_dim: int = 100, output_dim: int = 10):
        """
        Initialize tiny MLP.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super(TinyMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        """Forward pass."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
