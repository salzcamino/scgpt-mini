"""
Cell type annotation for scGPT-mini.

This module provides functions to:
- Fine-tune models for cell type classification
- Predict cell types on new data
- Evaluate annotation performance
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from scgpt_mini.training.losses import classification_loss
from scgpt_mini.training.metrics import compute_classification_metrics


def create_classification_dataset(
    adata,
    label_key: str = "cell_type",
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_state: int = 42,
) -> Tuple[Dict, Dict]:
    """
    Create train/val/test splits for classification.

    Args:
        adata: AnnData object with labels
        label_key: Key in adata.obs containing cell type labels
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        random_state: Random seed

    Returns:
        (split_indices, label_encoder)
        - split_indices: Dict with "train", "val", "test" indices
        - label_encoder: Dict mapping label names to indices
    """
    from sklearn.model_selection import train_test_split

    # Get labels
    labels = adata.obs[label_key].values
    unique_labels = sorted(set(labels))

    # Create label encoder
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    label_ids = np.array([label_to_id[label] for label in labels])

    # Split data
    indices = np.arange(len(labels))

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_split + test_split),
        random_state=random_state,
        stratify=label_ids,
    )

    # Second split: val vs test
    val_size_adjusted = val_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size_adjusted),
        random_state=random_state,
        stratify=label_ids[temp_idx],
    )

    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    label_encoder = {
        "label_to_id": label_to_id,
        "id_to_label": {idx: label for label, idx in label_to_id.items()},
        "n_classes": len(unique_labels),
    }

    print(f"Dataset split:")
    print(f"  Train: {len(train_idx)} cells")
    print(f"  Val: {len(val_idx)} cells")
    print(f"  Test: {len(test_idx)} cells")
    print(f"  Classes: {len(unique_labels)}")

    return split_indices, label_encoder


def finetune_for_annotation(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_classes: int,
    num_epochs: int = 10,
    learning_rate: float = 5e-5,
    freeze_encoder: bool = False,
    device: str = "cpu",
    checkpoint_dir: Union[str, Path] = "checkpoints",
    class_weights: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Fine-tune a pre-trained model for cell type annotation.

    Args:
        model: Pre-trained TransformerModel
        train_loader: Training DataLoader (with labels)
        val_loader: Validation DataLoader (with labels)
        n_classes: Number of cell types
        num_epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        freeze_encoder: If True, only train classification head
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        class_weights: Optional class weights for imbalanced data

    Returns:
        Training history dictionary

    Example:
        >>> from scgpt_mini.model import TransformerModel
        >>> from scgpt_mini.tasks import finetune_for_annotation
        >>>
        >>> model = TransformerModel.load_checkpoint("pretrained.pt")
        >>> history = finetune_for_annotation(
        ...     model, train_loader, val_loader, n_classes=10
        ... )
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)

    # Add classification head if not present or wrong size
    if model.cls_decoder is None or model.cls_decoder.out_layer.out_features != n_classes:
        from scgpt_mini.model.decoder import ClassificationDecoder
        model.cls_decoder = ClassificationDecoder(
            d_model=model.d_model,
            n_classes=n_classes,
        ).to(device)
        print(f"Added new classification head with {n_classes} classes")

    # Optionally freeze encoder
    if freeze_encoder:
        for param in model.gene_encoder.parameters():
            param.requires_grad = False
        for param in model.value_encoder.parameters():
            param.requires_grad = False
        for param in model.transformer_encoder.parameters():
            param.requires_grad = False
        print("Encoder layers frozen - training classification head only")

    # Optimizer (only for trainable parameters)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    # Loss criterion
    criterion = classification_loss

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    print("=" * 70)
    print("Fine-tuning for Cell Type Annotation")
    print("=" * 70)
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Device: {device}")
    print("=" * 70)

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            genes = batch["genes"].to(device)
            values = batch["values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            output = model(genes, values, attention_mask)
            loss = criterion(output["cls_pred"], labels, class_weights=class_weights)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            pred_labels = output["cls_pred"].argmax(dim=1)
            train_correct += (pred_labels == labels).sum().item()
            train_total += len(labels)

        # Training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                genes = batch["genes"].to(device)
                values = batch["values"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                output = model(genes, values, attention_mask)
                loss = criterion(output["cls_pred"], labels, class_weights=class_weights)

                val_loss += loss.item()
                all_preds.append(output["cls_pred"].cpu())
                all_labels.append(labels.cpu())

        # Validation metrics
        avg_val_loss = val_loss / len(val_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        val_metrics = compute_classification_metrics(all_preds, all_labels)
        val_acc = val_metrics["accuracy"]

        # Update history
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / "best_annotation_model.pt"
            model.save_checkpoint(checkpoint_path)
            print(f"  ✓ Best model saved (val_acc: {val_acc:.4f})")

    print("\n" + "=" * 70)
    print("Fine-tuning Complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 70)

    return history


def predict_cell_types(
    model,
    dataloader: DataLoader,
    label_encoder: Dict,
    device: str = "cpu",
    return_probabilities: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict cell types for new data.

    Args:
        model: Fine-tuned TransformerModel
        dataloader: DataLoader for cells to annotate
        label_encoder: Label encoder dict from create_classification_dataset
        device: Device to run on
        return_probabilities: If True, also return class probabilities

    Returns:
        Predicted labels (and probabilities if requested)

    Example:
        >>> predictions = predict_cell_types(model, test_loader, label_encoder)
        >>> cell_types = [label_encoder["id_to_label"][p] for p in predictions]
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting cell types"):
            genes = batch["genes"].to(device)
            values = batch["values"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output = model(genes, values, attention_mask)

            # Get predictions
            probs = torch.softmax(output["cls_pred"], dim=1)
            preds = probs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            if return_probabilities:
                all_probs.append(probs.cpu().numpy())

    # Concatenate results
    predictions = np.concatenate(all_preds, axis=0)

    if return_probabilities:
        probabilities = np.concatenate(all_probs, axis=0)
        return predictions, probabilities

    return predictions


def annotation_eval_report(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_encoder: Dict,
    save_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Generate comprehensive annotation evaluation report.

    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        label_encoder: Label encoder dict
        save_path: Optional path to save report

    Returns:
        Dictionary of metrics

    Example:
        >>> metrics = annotation_eval_report(
        ...     true_labels, predictions, label_encoder
        ... )
    """
    from sklearn.metrics import confusion_matrix, classification_report

    # Convert to torch for metrics computation
    true_tensor = torch.tensor(true_labels)
    # Create dummy logits from predictions
    n_classes = label_encoder["n_classes"]
    pred_logits = torch.zeros(len(pred_labels), n_classes)
    pred_logits[range(len(pred_labels)), pred_labels] = 1.0

    # Compute metrics
    metrics = compute_classification_metrics(pred_logits, true_tensor, return_per_class=True)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Print report
    print("=" * 70)
    print("Cell Type Annotation Evaluation Report")
    print("=" * 70)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nPer-class metrics:")

    id_to_label = label_encoder["id_to_label"]
    for i in range(n_classes):
        if i < len(metrics["per_class"]["f1"]):
            print(f"  {id_to_label[i]:20s}: "
                  f"F1={metrics['per_class']['f1'][i]:.3f}, "
                  f"Support={int(metrics['per_class']['support'][i])}")

    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 70)

    # Optionally save report
    if save_path is not None:
        save_path = Path(save_path)
        with open(save_path, "w") as f:
            f.write("Cell Type Annotation Evaluation Report\n")
            f.write("=" * 70 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
            f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {metrics['weighted_f1']:.4f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))

        print(f"Report saved to {save_path}")

    return {"metrics": metrics, "confusion_matrix": cm}
