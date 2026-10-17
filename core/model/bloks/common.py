"""
Common blocks for the model.
"""
import torch
from torch import nn


class SkipConnection(nn.Module):
    """
    A simple skip connection module.
    """

    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.sub = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the skip connection.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        return self.sub(x) + x


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) module.
    """

    def __init__(self, in_features: int, out_features: int, hidden_features: int | None = None, depth: int = 1,
                 dropout: float = 0.0):
        """
        Initializes the MLP with the specified parameters.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            hidden_features (int | None): Number of hidden features. If None, defaults to in_features.
            depth (int): Number of layers in the MLP.
            dropout (float): Dropout rate applied after the first linear layer (if 0 no dropout is applied).
        """
        super().__init__()
        if hidden_features is None:
            hidden_features = in_features

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.depth = depth
        self.dropout = dropout

        self.layers = []
        for _ in range(self.depth):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.in_features, self.hidden_features),
                    # (B, ..., in_features) -> (B, ..., hidden_features)
                    ##nn.SiLU(),
                    nn.GELU(),
                    nn.LayerNorm(self.hidden_features),
                    nn.Linear(self.hidden_features, self.out_features),
                    # (B, ..., hidden_features) -> (B, ..., out_features)
                    nn.Dropout(self.dropout)
                )
            )
        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the MLP to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, ..., in_features)

        Returns:
            torch.Tensor: Output tensor of shape (B, ..., out_features)
        """
        return self.mlp(x)
