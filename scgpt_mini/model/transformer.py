"""
Main transformer model for scGPT-mini.

This module contains the TransformerModel class that combines all components.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from scgpt_mini.tokenizer.vocabulary import GeneVocab
from scgpt_mini.model.encoder import (
    GeneEncoder,
    ContinuousValueEncoder,
    CategoryValueEncoder,
    PositionalEncoding,
)
from scgpt_mini.model.decoder import (
    ExpressionDecoder,
    ClassificationDecoder,
    BinnedExpressionDecoder,
)


class TransformerModel(nn.Module):
    """
    Transformer model for single-cell gene expression.

    This model implements a transformer architecture for learning from
    single-cell RNA-seq data using masked language modeling and optional
    cell type classification.

    Architecture:
    1. Gene Encoder: Embeds gene IDs
    2. Value Encoder: Embeds expression values
    3. Embedding Combination: Add gene + value embeddings
    4. Positional Encoding (optional)
    5. Transformer Encoder: Stack of self-attention layers
    6. Expression Decoder: Predicts masked expression values
    7. Classification Decoder (optional): Predicts cell types from CLS token

    Args:
        vocab_size: Size of gene vocabulary
        d_model: Embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        d_hid: Hidden dimension in feedforward network
        n_classes: Number of cell types for classification (optional)
        n_bins: Number of expression bins (if using binned mode)
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        value_mode: "continuous" or "binned" for expression encoding
        use_positional_encoding: Whether to add positional encoding
        activation: Activation function for transformer
        norm_first: Use pre-LN (True) or post-LN (False)
        explicit_zero_prob: Model zero expression separately
        vocab: GeneVocab instance (optional, for getting padding index)

    Example:
        >>> from scgpt_mini.tokenizer import GeneVocab
        >>> vocab = GeneVocab(["GENE1", "GENE2", "GENE3"])
        >>> model = TransformerModel(
        ...     vocab_size=len(vocab),
        ...     d_model=32,
        ...     nhead=2,
        ...     num_layers=2,
        ...     d_hid=64,
        ...     n_classes=10,
        ... )
        >>>
        >>> # Forward pass
        >>> batch_size, seq_len = 4, 100
        >>> genes = torch.randint(0, len(vocab), (batch_size, seq_len))
        >>> values = torch.randn(batch_size, seq_len)
        >>> attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        >>>
        >>> output = model(genes, values, attention_mask)
        >>> print(output["expr_pred"].shape)
        torch.Size([4, 100])
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 2,
        d_hid: int = 64,
        n_classes: Optional[int] = None,
        n_bins: int = 51,
        dropout: float = 0.1,
        max_seq_len: int = 1001,
        value_mode: str = "continuous",
        use_positional_encoding: bool = False,
        activation: str = "gelu",
        norm_first: bool = True,
        explicit_zero_prob: bool = False,
        vocab: Optional[GeneVocab] = None,
    ):
        super().__init__()

        # Store configuration
        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "d_hid": d_hid,
            "n_classes": n_classes,
            "n_bins": n_bins,
            "dropout": dropout,
            "max_seq_len": max_seq_len,
            "value_mode": value_mode,
            "use_positional_encoding": use_positional_encoding,
            "activation": activation,
            "norm_first": norm_first,
            "explicit_zero_prob": explicit_zero_prob,
        }

        self.d_model = d_model
        self.value_mode = value_mode
        self.use_positional_encoding = use_positional_encoding

        # Get padding index from vocab if available
        pad_idx = vocab.pad_id if vocab is not None else 0

        # Gene encoder
        self.gene_encoder = GeneEncoder(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx,
        )

        # Value encoder
        if value_mode == "continuous":
            self.value_encoder = ContinuousValueEncoder(
                d_model=d_model,
                dropout=dropout,
            )
        elif value_mode == "binned":
            self.value_encoder = CategoryValueEncoder(
                num_embeddings=n_bins,
                embedding_dim=d_model,
                padding_idx=0,  # Bin 0 is for zero values
            )
        else:
            raise ValueError(f"Unknown value_mode: {value_mode}")

        # Positional encoding (optional)
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                d_model=d_model,
                dropout=dropout,
                max_len=max_seq_len,
            )

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # Important: (batch, seq, feature) format
            norm_first=norm_first,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Expression decoder
        if value_mode == "continuous":
            self.expr_decoder = ExpressionDecoder(
                d_model=d_model,
                explicit_zero_prob=explicit_zero_prob,
                dropout=dropout,
            )
        elif value_mode == "binned":
            self.expr_decoder = BinnedExpressionDecoder(
                d_model=d_model,
                n_bins=n_bins,
                dropout=dropout,
            )

        # Classification decoder (optional)
        self.cls_decoder = None
        if n_classes is not None and n_classes > 0:
            self.cls_decoder = ClassificationDecoder(
                d_model=d_model,
                n_classes=n_classes,
                dropout=dropout,
            )

    def _encode(
        self,
        genes: Tensor,
        values: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        Encode genes and values into combined embeddings.

        Args:
            genes: Gene IDs of shape (batch_size, seq_len)
            values: Expression values of shape (batch_size, seq_len)
            attention_mask: Boolean mask of shape (batch_size, seq_len)

        Returns:
            Combined embeddings of shape (batch_size, seq_len, d_model)
        """
        # Encode genes
        gene_emb = self.gene_encoder(genes)  # (batch, seq_len, d_model)

        # Encode values
        value_emb = self.value_encoder(values)  # (batch, seq_len, d_model)

        # Combine embeddings (addition)
        combined_emb = gene_emb + value_emb  # (batch, seq_len, d_model)

        # Optional positional encoding
        if self.use_positional_encoding:
            combined_emb = self.pos_encoder(combined_emb)

        return combined_emb

    def forward(
        self,
        genes: Tensor,
        values: Tensor,
        attention_mask: Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the model.

        Args:
            genes: Gene IDs of shape (batch_size, seq_len)
            values: Expression values of shape (batch_size, seq_len)
            attention_mask: Boolean mask of shape (batch_size, seq_len)
                True for valid positions, False for padding
            return_embeddings: If True, return transformer embeddings

        Returns:
            Dictionary containing:
            - "expr_pred": Expression predictions (shape depends on value_mode)
                - continuous: (batch_size, seq_len)
                - binned: (batch_size, seq_len, n_bins)
            - "cls_pred" (optional): Cell type logits of shape (batch_size, n_classes)
            - "embeddings" (optional): Transformer embeddings of shape (batch_size, seq_len, d_model)
            - "cls_embedding" (optional): CLS token embedding of shape (batch_size, d_model)
        """
        # Encode input
        embeddings = self._encode(genes, values, attention_mask)  # (batch, seq_len, d_model)

        # Create attention mask for transformer
        # PyTorch transformer expects: True = mask out, False = attend
        # We have: True = valid, False = padding
        # So we need to invert
        src_key_padding_mask = ~attention_mask  # (batch, seq_len)

        # Pass through transformer
        transformer_output = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask,
        )  # (batch, seq_len, d_model)

        # Decode expression values
        expr_output = self.expr_decoder(transformer_output)

        # Prepare output dictionary
        output = {}

        # Handle expression predictions based on value_mode
        if self.value_mode == "continuous":
            output["expr_pred"] = expr_output["pred"]
            if "zero_probs" in expr_output:
                output["zero_probs"] = expr_output["zero_probs"]
        elif self.value_mode == "binned":
            output["expr_pred"] = expr_output  # Logits over bins

        # Classification prediction (from CLS token)
        if self.cls_decoder is not None:
            cls_embedding = transformer_output[:, 0, :]  # (batch, d_model)
            cls_pred = self.cls_decoder(cls_embedding)
            output["cls_pred"] = cls_pred
            if return_embeddings:
                output["cls_embedding"] = cls_embedding

        # Optional: return full embeddings
        if return_embeddings:
            output["embeddings"] = transformer_output

        return output

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        if isinstance(path, str):
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.config,
            "state_dict": self.state_dict(),
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        vocab: Optional[GeneVocab] = None,
    ) -> "TransformerModel":
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            vocab: GeneVocab instance (optional)

        Returns:
            Loaded TransformerModel
        """
        checkpoint = torch.load(path, map_location="cpu")

        # Create model from config
        model = cls(vocab=vocab, **checkpoint["config"])

        # Load state dict
        model.load_state_dict(checkpoint["state_dict"])

        print(f"Checkpoint loaded from {path}")
        return model

    def get_num_parameters(self) -> int:
        """
        Count the number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_model_info(self) -> None:
        """Print model architecture and parameter information."""
        print("=" * 70)
        print("scGPT-mini Model Information")
        print("=" * 70)
        print("\nConfiguration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

        print(f"\nTotal trainable parameters: {self.get_num_parameters():,}")

        # Estimate model size
        param_size_mb = self.get_num_parameters() * 4 / (1024 ** 2)  # 4 bytes per float32
        print(f"Estimated model size: {param_size_mb:.2f} MB")

        print("=" * 70)
