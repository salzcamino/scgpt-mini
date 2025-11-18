"""
scGPT-mini: A miniature, educational version of scGPT for single-cell RNA-seq analysis.

This package provides a simplified transformer-based model for learning and understanding
how foundation models work with single-cell genomics data.

Main modules:
- tokenizer: Gene vocabulary and tokenization
- model: Transformer architecture
- data: Data preprocessing and collation
- training: Training infrastructure
- tasks: Downstream tasks (embedding, annotation)
- utils: Utility functions
"""

__version__ = "0.1.0"
__author__ = "scGPT-mini contributors"
__license__ = "MIT"

# Import main components for convenience
from scgpt_mini import tokenizer, model, data, training, tasks, utils

__all__ = [
    "tokenizer",
    "model",
    "data",
    "training",
    "tasks",
    "utils",
    "__version__",
]
