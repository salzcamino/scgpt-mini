"""
Metrics for evaluating scGPT-mini model performance.

This module provides metrics for:
- Masked language modeling (MLM) evaluation
- Cell type classification evaluation
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def masked_mse(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
) -> float:
    """
    Compute masked MSE metric.

    Args:
        pred: Predictions of shape (batch_size, seq_len)
        target: Targets of shape (batch_size, seq_len)
        mask: Boolean mask of shape (batch_size, seq_len)

    Returns:
        MSE value
    """
    mask_float = mask.float()
    if mask_float.sum() == 0:
        return 0.0

    mse = ((pred - target) ** 2) * mask_float
    return (mse.sum() / mask_float.sum()).item()


def masked_mae(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
) -> float:
    """
    Compute masked MAE metric.

    Args:
        pred: Predictions of shape (batch_size, seq_len)
        target: Targets of shape (batch_size, seq_len)
        mask: Boolean mask of shape (batch_size, seq_len)

    Returns:
        MAE value
    """
    mask_float = mask.float()
    if mask_float.sum() == 0:
        return 0.0

    mae = torch.abs(pred - target) * mask_float
    return (mae.sum() / mask_float.sum()).item()


def masked_pearson(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
) -> float:
    """
    Compute Pearson correlation on masked positions.

    Args:
        pred: Predictions of shape (batch_size, seq_len)
        target: Targets of shape (batch_size, seq_len)
        mask: Boolean mask of shape (batch_size, seq_len)

    Returns:
        Pearson correlation coefficient
    """
    # Get masked values
    pred_masked = pred[mask]
    target_masked = target[mask]

    if len(pred_masked) < 2:
        return 0.0

    # Compute Pearson correlation
    pred_mean = pred_masked.mean()
    target_mean = target_masked.mean()

    pred_centered = pred_masked - pred_mean
    target_centered = target_masked - target_mean

    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())

    if denominator == 0:
        return 0.0

    return (numerator / denominator).item()


def compute_mlm_metrics(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Dict[str, float]:
    """
    Compute all MLM metrics.

    Args:
        pred: Predictions of shape (batch_size, seq_len)
        target: Targets of shape (batch_size, seq_len)
        mask: Boolean mask of shape (batch_size, seq_len)

    Returns:
        Dictionary of metrics
    """
    return {
        "mse": masked_mse(pred, target, mask),
        "mae": masked_mae(pred, target, mask),
        "pearson": masked_pearson(pred, target, mask),
    }


def compute_classification_metrics(
    pred: Tensor,
    target: Tensor,
    return_per_class: bool = False,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        pred: Predicted logits of shape (batch_size, n_classes)
        target: Target class indices of shape (batch_size,)
        return_per_class: If True, return per-class metrics

    Returns:
        Dictionary of metrics
    """
    # Get predicted classes
    pred_classes = pred.argmax(dim=-1)

    # Convert to numpy for sklearn
    pred_np = pred_classes.cpu().numpy()
    target_np = target.cpu().numpy()

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(target_np, pred_np),
        "balanced_accuracy": balanced_accuracy_score(target_np, pred_np),
        "macro_f1": f1_score(target_np, pred_np, average="macro", zero_division=0),
        "weighted_f1": f1_score(target_np, pred_np, average="weighted", zero_division=0),
    }

    # Per-class metrics if requested
    if return_per_class:
        precision, recall, f1, support = precision_recall_fscore_support(
            target_np, pred_np, zero_division=0
        )
        metrics["per_class"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        }

    return metrics


class MetricsTracker:
    """
    Tracks metrics across training epochs.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.update({"loss": 0.5, "accuracy": 0.8})
        >>> tracker.update({"loss": 0.4, "accuracy": 0.85})
        >>> print(tracker.get_average())
        {'loss': 0.45, 'accuracy': 0.825}
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        """Add a new set of metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def get_average(self) -> Dict[str, float]:
        """Get average of all tracked metrics."""
        return {key: np.mean(values) for key, values in self.metrics.items()}

    def get_last(self) -> Dict[str, float]:
        """Get last value of all tracked metrics."""
        return {key: values[-1] for key, values in self.metrics.items()}

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.metrics = {}

    def __repr__(self) -> str:
        avg_metrics = self.get_average()
        return f"MetricsTracker({avg_metrics})"


def evaluate_model(
    model,
    dataloader,
    device: str = "cpu",
    value_mode: str = "continuous",
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: TransformerModel instance
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        value_mode: "continuous" or "binned"

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    tracker = MetricsTracker()

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            genes = batch["genes"].to(device)
            values = batch["values"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get masked values and targets
            if "masked_values" in batch:
                masked_values = batch["masked_values"].to(device)
                mask_positions = batch["mask_positions"].to(device)
                target_values = batch["target_values"].to(device)
            else:
                masked_values = values
                mask_positions = torch.zeros_like(values, dtype=torch.bool)
                target_values = values

            # Forward pass
            output = model(genes, masked_values, attention_mask)

            # Compute MLM metrics
            if value_mode == "continuous":
                mlm_metrics = compute_mlm_metrics(
                    output["expr_pred"],
                    target_values,
                    mask_positions,
                )
                tracker.update(mlm_metrics)

            # Compute classification metrics if available
            if "cls_pred" in output and "labels" in batch:
                labels = batch["labels"].to(device)
                cls_metrics = compute_classification_metrics(
                    output["cls_pred"],
                    labels,
                )
                tracker.update(cls_metrics)

    return tracker.get_average()


def print_metrics(
    metrics: Dict[str, float],
    prefix: str = "",
    width: int = 60,
) -> None:
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for metric names (e.g., "Train", "Val")
        width: Width of output
    """
    if prefix:
        print("=" * width)
        print(f"{prefix} Metrics")
        print("=" * width)

    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key:20s}: {value:.4f}")

    if prefix:
        print("=" * width)


def format_metrics_str(metrics: Dict[str, float]) -> str:
    """
    Format metrics as a compact string for logging.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted string
    """
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)
