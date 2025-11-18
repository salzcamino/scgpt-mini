"""
Decoder modules for scGPT-mini.

This module contains:
- ExpressionDecoder: Predicts gene expression values
- ClassificationDecoder: Predicts cell types
"""

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class ExpressionDecoder(nn.Module):
    """
    Decoder for predicting gene expression values.

    Uses a 3-layer MLP to decode transformer outputs to expression values.
    Optionally supports explicit zero probability modeling.

    Args:
        d_model: Input dimension (from transformer)
        explicit_zero_prob: If True, separately model probability of zero expression
        dropout: Dropout rate

    Example:
        >>> decoder = ExpressionDecoder(d_model=32)
        >>> hidden_states = torch.randn(4, 100, 32)  # (batch, seq_len, d_model)
        >>> output = decoder(hidden_states)
        >>> print(output["pred"].shape)
        torch.Size([4, 100])
    """

    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Main prediction head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.explicit_zero_prob = explicit_zero_prob

        # Optional zero probability head
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x: Transformer output of shape (batch_size, seq_len, d_model)

        Returns:
            Dictionary with keys:
            - "pred": Predicted expression values of shape (batch_size, seq_len)
            - "zero_probs" (optional): Probability of zero expression, shape (batch_size, seq_len)
        """
        # Predict expression value
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return {"pred": pred_value}

        # Predict zero probability
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)

        return {"pred": pred_value, "zero_probs": zero_probs}


class ClassificationDecoder(nn.Module):
    """
    Decoder for cell type classification.

    Multi-layer MLP that takes the CLS token embedding and predicts cell type.

    Args:
        d_model: Input dimension (from transformer)
        n_classes: Number of cell types
        n_layers: Number of hidden layers (default: 3)
        dropout: Dropout rate
        activation: Activation function class

    Example:
        >>> decoder = ClassificationDecoder(d_model=32, n_classes=10)
        >>> cls_embedding = torch.randn(4, 32)  # (batch, d_model)
        >>> logits = decoder(cls_embedding)
        >>> print(logits.shape)
        torch.Size([4, 10])
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation: type = nn.ReLU,
    ):
        super().__init__()

        # Build hidden layers
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(activation())
            layers.append(nn.LayerNorm(d_model))
            layers.append(nn.Dropout(dropout))

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.out_layer = nn.Linear(d_model, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: CLS token embedding of shape (batch_size, d_model)

        Returns:
            Class logits of shape (batch_size, n_classes)
        """
        x = self.hidden_layers(x)
        x = self.out_layer(x)
        return x


class BinnedExpressionDecoder(nn.Module):
    """
    Decoder for predicting binned expression values.

    Predicts expression as a classification problem over bins.

    Args:
        d_model: Input dimension (from transformer)
        n_bins: Number of expression bins
        dropout: Dropout rate

    Example:
        >>> decoder = BinnedExpressionDecoder(d_model=32, n_bins=51)
        >>> hidden_states = torch.randn(4, 100, 32)  # (batch, seq_len, d_model)
        >>> logits = decoder(hidden_states)
        >>> print(logits.shape)
        torch.Size([4, 100, 51])
    """

    def __init__(
        self,
        d_model: int,
        n_bins: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_bins),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Transformer output of shape (batch_size, seq_len, d_model)

        Returns:
            Bin logits of shape (batch_size, seq_len, n_bins)
        """
        return self.fc(x)
