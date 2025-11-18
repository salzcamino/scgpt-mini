"""
Loss functions for scGPT-mini training.

This module provides loss functions for:
- Masked language modeling (MLM)
- Cell type classification
- Combined training objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def masked_mse_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Compute masked MSE loss for expression value prediction.

    Only computes loss on masked positions (where mask=True).

    Args:
        pred: Predicted expression values of shape (batch_size, seq_len)
        target: Target expression values of shape (batch_size, seq_len)
        mask: Boolean mask of shape (batch_size, seq_len), True = compute loss

    Returns:
        Scalar loss value

    Example:
        >>> pred = torch.randn(4, 100)
        >>> target = torch.randn(4, 100)
        >>> mask = torch.rand(4, 100) > 0.5
        >>> loss = masked_mse_loss(pred, target, mask)
    """
    # Convert mask to float for computation
    mask_float = mask.float()

    # Compute MSE only on masked positions
    mse = (pred - target) ** 2
    masked_mse = mse * mask_float

    # Average over masked positions
    if mask_float.sum() > 0:
        return masked_mse.sum() / mask_float.sum()
    else:
        return torch.tensor(0.0, device=pred.device)


def masked_mae_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Compute masked MAE (L1) loss for expression value prediction.

    Args:
        pred: Predicted expression values of shape (batch_size, seq_len)
        target: Target expression values of shape (batch_size, seq_len)
        mask: Boolean mask of shape (batch_size, seq_len), True = compute loss

    Returns:
        Scalar loss value
    """
    mask_float = mask.float()
    mae = torch.abs(pred - target)
    masked_mae = mae * mask_float

    if mask_float.sum() > 0:
        return masked_mae.sum() / mask_float.sum()
    else:
        return torch.tensor(0.0, device=pred.device)


def masked_cross_entropy_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Compute masked cross-entropy loss for binned expression prediction.

    Args:
        pred: Predicted logits of shape (batch_size, seq_len, n_bins)
        target: Target bin indices of shape (batch_size, seq_len)
        mask: Boolean mask of shape (batch_size, seq_len), True = compute loss

    Returns:
        Scalar loss value

    Example:
        >>> pred = torch.randn(4, 100, 51)  # 51 bins
        >>> target = torch.randint(0, 51, (4, 100))
        >>> mask = torch.rand(4, 100) > 0.5
        >>> loss = masked_cross_entropy_loss(pred, target, mask)
    """
    # Reshape for cross entropy
    batch_size, seq_len, n_bins = pred.shape
    pred_flat = pred.reshape(-1, n_bins)  # (batch*seq, n_bins)
    target_flat = target.reshape(-1)  # (batch*seq,)
    mask_flat = mask.reshape(-1)  # (batch*seq,)

    # Compute cross entropy
    ce = F.cross_entropy(pred_flat, target_flat, reduction='none')  # (batch*seq,)

    # Apply mask
    masked_ce = ce * mask_flat.float()

    if mask_flat.sum() > 0:
        return masked_ce.sum() / mask_flat.sum()
    else:
        return torch.tensor(0.0, device=pred.device)


def classification_loss(
    pred: Tensor,
    target: Tensor,
    class_weights: Tensor = None,
) -> Tensor:
    """
    Compute cross-entropy loss for cell type classification.

    Args:
        pred: Predicted class logits of shape (batch_size, n_classes)
        target: Target class indices of shape (batch_size,)
        class_weights: Optional class weights of shape (n_classes,)

    Returns:
        Scalar loss value

    Example:
        >>> pred = torch.randn(4, 10)  # 10 cell types
        >>> target = torch.randint(0, 10, (4,))
        >>> loss = classification_loss(pred, target)
    """
    return F.cross_entropy(pred, target, weight=class_weights)


def zero_probability_loss(
    zero_probs: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Compute binary cross-entropy loss for zero probability prediction.

    This loss encourages the model to predict when expression should be zero.

    Args:
        zero_probs: Predicted probability of zero expression, shape (batch_size, seq_len)
        target: Target expression values of shape (batch_size, seq_len)
        mask: Boolean mask of shape (batch_size, seq_len), True = compute loss

    Returns:
        Scalar loss value
    """
    # Create binary target: 1 if target is zero, 0 otherwise
    is_zero = (target == 0).float()

    # Binary cross-entropy
    bce = F.binary_cross_entropy(zero_probs, is_zero, reduction='none')

    # Apply mask
    mask_float = mask.float()
    masked_bce = bce * mask_float

    if mask_float.sum() > 0:
        return masked_bce.sum() / mask_float.sum()
    else:
        return torch.tensor(0.0, device=zero_probs.device)


class MLMLoss(nn.Module):
    """
    Combined loss for masked language modeling.

    Combines expression prediction loss with optional zero probability loss.

    Args:
        value_mode: "continuous" or "binned"
        use_zero_prob: Whether to use explicit zero probability modeling
        zero_prob_weight: Weight for zero probability loss

    Example:
        >>> criterion = MLMLoss(value_mode="continuous")
        >>> pred = {"expr_pred": torch.randn(4, 100)}
        >>> target = torch.randn(4, 100)
        >>> mask = torch.rand(4, 100) > 0.5
        >>> loss = criterion(pred, target, mask)
    """

    def __init__(
        self,
        value_mode: str = "continuous",
        use_zero_prob: bool = False,
        zero_prob_weight: float = 0.1,
    ):
        super().__init__()
        self.value_mode = value_mode
        self.use_zero_prob = use_zero_prob
        self.zero_prob_weight = zero_prob_weight

    def forward(
        self,
        pred_dict: dict,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Compute MLM loss.

        Args:
            pred_dict: Dictionary with keys:
                - "expr_pred": Expression predictions
                - "zero_probs" (optional): Zero probabilities
            target: Target expression values
            mask: Mask indicating positions to compute loss on

        Returns:
            Total loss value
        """
        # Expression prediction loss
        if self.value_mode == "continuous":
            expr_loss = masked_mse_loss(pred_dict["expr_pred"], target, mask)
        elif self.value_mode == "binned":
            expr_loss = masked_cross_entropy_loss(pred_dict["expr_pred"], target.long(), mask)
        else:
            raise ValueError(f"Unknown value_mode: {self.value_mode}")

        # Add zero probability loss if available
        if self.use_zero_prob and "zero_probs" in pred_dict:
            zero_loss = zero_probability_loss(pred_dict["zero_probs"], target, mask)
            total_loss = expr_loss + self.zero_prob_weight * zero_loss
        else:
            total_loss = expr_loss

        return total_loss


class CombinedLoss(nn.Module):
    """
    Combined loss for MLM and classification.

    Args:
        mlm_weight: Weight for MLM loss
        cls_weight: Weight for classification loss
        value_mode: "continuous" or "binned"
        use_zero_prob: Whether to use explicit zero probability
        class_weights: Optional class weights for classification

    Example:
        >>> criterion = CombinedLoss(mlm_weight=1.0, cls_weight=0.5)
        >>> pred = {
        ...     "expr_pred": torch.randn(4, 100),
        ...     "cls_pred": torch.randn(4, 10),
        ... }
        >>> target_expr = torch.randn(4, 100)
        >>> target_cls = torch.randint(0, 10, (4,))
        >>> mask = torch.rand(4, 100) > 0.5
        >>> loss = criterion(pred, target_expr, target_cls, mask)
    """

    def __init__(
        self,
        mlm_weight: float = 1.0,
        cls_weight: float = 0.5,
        value_mode: str = "continuous",
        use_zero_prob: bool = False,
        class_weights: Tensor = None,
    ):
        super().__init__()
        self.mlm_weight = mlm_weight
        self.cls_weight = cls_weight

        self.mlm_loss_fn = MLMLoss(value_mode=value_mode, use_zero_prob=use_zero_prob)
        self.class_weights = class_weights

    def forward(
        self,
        pred_dict: dict,
        target_expr: Tensor,
        target_cls: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Compute combined loss.

        Args:
            pred_dict: Dictionary with model predictions
            target_expr: Target expression values
            target_cls: Target class labels
            mask: Mask for MLM loss

        Returns:
            Total weighted loss
        """
        # MLM loss
        mlm_loss = self.mlm_loss_fn(pred_dict, target_expr, mask)

        # Classification loss
        cls_loss = classification_loss(
            pred_dict["cls_pred"],
            target_cls,
            class_weights=self.class_weights,
        )

        # Combine
        total_loss = self.mlm_weight * mlm_loss + self.cls_weight * cls_loss

        return total_loss, {"mlm_loss": mlm_loss.item(), "cls_loss": cls_loss.item()}
