"""
Training infrastructure module for scGPT-mini.

This module contains:
- Loss functions
- Training metrics
- Training loop and checkpointing
"""

from scgpt_mini.training.losses import (
    masked_mse_loss,
    masked_mae_loss,
    masked_cross_entropy_loss,
    classification_loss,
    zero_probability_loss,
    MLMLoss,
    CombinedLoss,
)
from scgpt_mini.training.metrics import (
    masked_mse,
    masked_mae,
    masked_pearson,
    compute_mlm_metrics,
    compute_classification_metrics,
    MetricsTracker,
    evaluate_model,
    print_metrics,
    format_metrics_str,
)
from scgpt_mini.training.trainer import Trainer

__all__ = [
    # Loss functions
    "masked_mse_loss",
    "masked_mae_loss",
    "masked_cross_entropy_loss",
    "classification_loss",
    "zero_probability_loss",
    "MLMLoss",
    "CombinedLoss",
    # Metrics
    "masked_mse",
    "masked_mae",
    "masked_pearson",
    "compute_mlm_metrics",
    "compute_classification_metrics",
    "MetricsTracker",
    "evaluate_model",
    "print_metrics",
    "format_metrics_str",
    # Trainer
    "Trainer",
]
