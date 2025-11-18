"""
Training infrastructure for scGPT-mini.

This module provides the Trainer class for training and evaluating models.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from scgpt_mini.training.losses import MLMLoss, CombinedLoss
from scgpt_mini.training.metrics import (
    MetricsTracker,
    evaluate_model,
    print_metrics,
    format_metrics_str,
)


class Trainer:
    """
    Trainer for scGPT-mini models.

    Handles:
    - Training loop with progress bars
    - Validation and metrics tracking
    - Checkpointing (best model + periodic saves)
    - Learning rate scheduling
    - Early stopping

    Args:
        model: TransformerModel instance
        optimizer: PyTorch optimizer
        criterion: Loss function (MLMLoss or CombinedLoss)
        device: Device to train on ("cpu" or "cuda")
        scheduler: Optional learning rate scheduler
        gradient_clip: Gradient clipping value (None = no clipping)
        log_interval: How often to log within an epoch
        checkpoint_dir: Directory to save checkpoints
        value_mode: "continuous" or "binned"

    Example:
        >>> from scgpt_mini.model import TransformerModel
        >>> from scgpt_mini.training import MLMLoss, Trainer
        >>> import torch.optim as optim
        >>>
        >>> model = TransformerModel(vocab_size=1000, d_model=32)
        >>> optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        >>> criterion = MLMLoss(value_mode="continuous")
        >>> trainer = Trainer(model, optimizer, criterion)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Union[MLMLoss, CombinedLoss],
        device: str = "cpu",
        scheduler: Optional[_LRScheduler] = None,
        gradient_clip: Optional[float] = 1.0,
        log_interval: int = 10,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        value_mode: str = "continuous",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.value_mode = value_mode

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def train_epoch(
        self,
        train_loader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        tracker = MetricsTracker()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            genes = batch["genes"].to(self.device)
            values = batch["values"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            masked_values = batch["masked_values"].to(self.device)
            mask_positions = batch["mask_positions"].to(self.device)
            target_values = batch["target_values"].to(self.device)

            # Forward pass
            output = self.model(genes, masked_values, attention_mask)

            # Compute loss
            if isinstance(self.criterion, CombinedLoss):
                labels = batch["labels"].to(self.device)
                loss, loss_dict = self.criterion(
                    output, target_values, labels, mask_positions
                )
                tracker.update(loss_dict)
            else:
                loss = self.criterion(output, target_values, mask_positions)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            # Optimizer step
            self.optimizer.step()

            # Track metrics
            tracker.update({"loss": loss.item()})

            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0:
                avg_metrics = tracker.get_average()
                pbar.set_postfix({"loss": f"{avg_metrics['loss']:.4f}"})

        # Get epoch metrics
        epoch_metrics = tracker.get_average()

        return epoch_metrics

    def validate(
        self,
        val_loader,
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Dictionary of validation metrics
        """
        return evaluate_model(
            self.model,
            val_loader,
            device=self.device,
            value_mode=self.value_mode,
        )

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        save_every: int = 5,
        eval_every: int = 1,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            eval_every: Evaluate every N epochs
            early_stopping_patience: Stop if no improvement for N epochs (None = disabled)

        Returns:
            Training history dictionary
        """
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 70)

        epochs_without_improvement = 0
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.training_history["train_loss"].append(train_metrics["loss"])

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.training_history["learning_rate"].append(current_lr)

            # Validate
            if epoch % eval_every == 0:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics.get("mse", val_metrics.get("loss", 0.0))
                self.training_history["val_loss"].append(val_loss)

                # Print epoch summary
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s):")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Learning Rate: {current_lr:.6f}")

                if val_metrics:
                    print(f"  Val Metrics: {format_metrics_str(val_metrics)}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt", is_best=True)
                    print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # Early stopping check
                if early_stopping_patience is not None:
                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"\nEarly stopping after {epoch} epochs")
                        print(f"No improvement for {early_stopping_patience} epochs")
                        break

            # Periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print("=" * 70)

        return self.training_history

    def save_checkpoint(
        self,
        filename: str,
        is_best: bool = False,
    ) -> None:
        """
        Save a checkpoint.

        Args:
            filename: Filename for checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)

        if not is_best:
            print(f"  Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ) -> None:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if load_scheduler and self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.training_history = checkpoint.get("training_history", {})

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")

    def save_training_history(
        self,
        filename: str = "training_history.json",
    ) -> None:
        """
        Save training history to JSON.

        Args:
            filename: Filename for history file
        """
        history_path = self.checkpoint_dir / filename

        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        print(f"Training history saved to {history_path}")
