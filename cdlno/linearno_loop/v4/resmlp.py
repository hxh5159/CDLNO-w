"""FLARE-style same-width ResidualMLP adapted to LinearNO outer blocks."""
import torch
from torch import nn

class ResidualMLP(nn.Module):
    def __init__(self, width, ratio, num_layers, *, input_residual=True, output_residual=True):
        super().__init__()
        if type(width) is not int or width < 1 or type(ratio) is not int or ratio < 1:
            raise ValueError("width and ratio must be positive integers")
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.width, self.ratio, self.num_layers = width, ratio, num_layers
        hidden = width * ratio
        self.fc1 = nn.Linear(width, hidden)
        self.hidden = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])
        self.fc2 = nn.Linear(hidden, width)
        self.input_residual = bool(input_residual and width == hidden)
        self.output_residual = bool(output_residual and hidden == width)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x):
        h = self.activation(self.fc1(x))
        if self.input_residual:
            h = h + x
        for layer in self.hidden:
            h = h + self.activation(layer(h))
        y = self.fc2(h)
        return y + h if self.output_residual else y
