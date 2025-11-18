"""
Encoder modules for scGPT-mini.

This module contains:
- GeneEncoder: Embeds gene IDs
- ContinuousValueEncoder: Encodes continuous expression values
- CategoryValueEncoder: Encodes binned expression values
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class GeneEncoder(nn.Module):
    """
    Encoder for gene tokens.

    Embeds gene IDs into dense vectors and applies layer normalization.

    Args:
        num_embeddings: Size of gene vocabulary
        embedding_dim: Dimension of embeddings (d_model)
        padding_idx: Index to use for padding (typically 0)

    Example:
        >>> encoder = GeneEncoder(num_embeddings=10000, embedding_dim=32, padding_idx=0)
        >>> gene_ids = torch.randint(0, 10000, (4, 100))  # batch_size=4, seq_len=100
        >>> embeddings = encoder(gene_ids)
        >>> print(embeddings.shape)
        torch.Size([4, 100, 32])
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Gene ID tensor of shape (batch_size, seq_len)

        Returns:
            Gene embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    """
    Encoder for continuous expression values.

    Uses a 2-layer MLP to project continuous values to embedding space.

    Args:
        d_model: Embedding dimension
        dropout: Dropout rate
        max_value: Maximum value to clip inputs to

    Example:
        >>> encoder = ContinuousValueEncoder(d_model=32, dropout=0.1)
        >>> values = torch.randn(4, 100)  # batch_size=4, seq_len=100
        >>> embeddings = encoder(values)
        >>> print(embeddings.shape)
        torch.Size([4, 100, 32])
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: float = 30.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Expression value tensor of shape (batch_size, seq_len)

        Returns:
            Value embeddings of shape (batch_size, seq_len, d_model)
        """
        # Expand last dimension: (batch, seq_len) -> (batch, seq_len, 1)
        x = x.unsqueeze(-1)

        # Clip to max value (prevents very large values from dominating)
        x = torch.clamp(x, max=self.max_value)

        # Two-layer MLP with activation and normalization
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)

        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    """
    Encoder for categorical (binned) expression values.

    Uses an embedding layer to encode discrete expression bins.

    Args:
        num_embeddings: Number of expression bins
        embedding_dim: Dimension of embeddings (d_model)
        padding_idx: Index to use for padding

    Example:
        >>> encoder = CategoryValueEncoder(num_embeddings=51, embedding_dim=32, padding_idx=0)
        >>> binned_values = torch.randint(0, 51, (4, 100))  # batch_size=4, seq_len=100
        >>> embeddings = encoder(binned_values)
        >>> print(embeddings.shape)
        torch.Size([4, 100, 32])
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Binned expression value tensor of shape (batch_size, seq_len)

        Returns:
            Value embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        x = x.long()  # Ensure integer type
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        x = self.enc_norm(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sinusoidal functions.

    This adds position information to token embeddings using sine and cosine functions
    of different frequencies.

    Args:
        d_model: Embedding dimension
        dropout: Dropout rate
        max_len: Maximum sequence length

    Example:
        >>> pos_encoder = PositionalEncoding(d_model=32, dropout=0.1, max_len=1000)
        >>> embeddings = torch.randn(100, 4, 32)  # (seq_len, batch, d_model)
        >>> pos_embeddings = pos_encoder(embeddings)
        >>> print(pos_embeddings.shape)
        torch.Size([100, 4, 32])
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
               or (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        # Handle both (seq_len, batch, d_model) and (batch, seq_len, d_model) formats
        if x.dim() == 3:
            # Assume (batch, seq_len, d_model) format (batch_first=True)
            seq_len = x.size(1)
            # Add positional encoding
            x = x + self.pe[:seq_len, 0, :].unsqueeze(0)
        else:
            raise ValueError(f"Expected 3D input tensor, got shape {x.shape}")

        return self.dropout(x)
