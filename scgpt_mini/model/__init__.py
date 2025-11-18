"""
Model architecture module for scGPT-mini.

This module contains the transformer model components:
- Gene and value encoders
- Expression and classification decoders
- Main transformer model
"""

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
from scgpt_mini.model.transformer import TransformerModel

__all__ = [
    # Encoders
    "GeneEncoder",
    "ContinuousValueEncoder",
    "CategoryValueEncoder",
    "PositionalEncoding",
    # Decoders
    "ExpressionDecoder",
    "ClassificationDecoder",
    "BinnedExpressionDecoder",
    # Main model
    "TransformerModel",
]
