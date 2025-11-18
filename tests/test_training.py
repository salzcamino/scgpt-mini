"""
Unit tests for the training module.
"""

import pytest
import torch
import torch.optim as optim
from pathlib import Path

from scgpt_mini.tokenizer import GeneVocab
from scgpt_mini.model import TransformerModel
from scgpt_mini.data.collator import create_dataloader
from scgpt_mini.training import (
    masked_mse_loss,
    masked_mae_loss,
    masked_cross_entropy_loss,
    classification_loss,
    MLMLoss,
    CombinedLoss,
    masked_mse,
    masked_mae,
    masked_pearson,
    compute_mlm_metrics,
    compute_classification_metrics,
    MetricsTracker,
    Trainer,
)


class TestLossFunctions:
    """Tests for loss functions."""

    def test_masked_mse_loss(self):
        """Test masked MSE loss."""
        pred = torch.randn(4, 100)
        target = torch.randn(4, 100)
        mask = torch.rand(4, 100) > 0.5

        loss = masked_mse_loss(pred, target, mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_masked_mae_loss(self):
        """Test masked MAE loss."""
        pred = torch.randn(4, 100)
        target = torch.randn(4, 100)
        mask = torch.rand(4, 100) > 0.5

        loss = masked_mae_loss(pred, target, mask)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0

    def test_masked_cross_entropy_loss(self):
        """Test masked cross-entropy loss."""
        pred = torch.randn(4, 100, 51)  # 51 bins
        target = torch.randint(0, 51, (4, 100))
        mask = torch.rand(4, 100) > 0.5

        loss = masked_cross_entropy_loss(pred, target, mask)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0

    def test_classification_loss(self):
        """Test classification loss."""
        pred = torch.randn(4, 10)  # 10 classes
        target = torch.randint(0, 10, (4,))

        loss = classification_loss(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0

    def test_mlm_loss_continuous(self):
        """Test MLMLoss with continuous values."""
        criterion = MLMLoss(value_mode="continuous")

        pred_dict = {"expr_pred": torch.randn(4, 100)}
        target = torch.randn(4, 100)
        mask = torch.rand(4, 100) > 0.5

        loss = criterion(pred_dict, target, mask)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0

    def test_mlm_loss_binned(self):
        """Test MLMLoss with binned values."""
        criterion = MLMLoss(value_mode="binned")

        pred_dict = {"expr_pred": torch.randn(4, 100, 51)}
        target = torch.randint(0, 51, (4, 100))
        mask = torch.rand(4, 100) > 0.5

        loss = criterion(pred_dict, target, mask)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0

    def test_combined_loss(self):
        """Test CombinedLoss."""
        criterion = CombinedLoss(mlm_weight=1.0, cls_weight=0.5)

        pred_dict = {
            "expr_pred": torch.randn(4, 100),
            "cls_pred": torch.randn(4, 10),
        }
        target_expr = torch.randn(4, 100)
        target_cls = torch.randint(0, 10, (4,))
        mask = torch.rand(4, 100) > 0.5

        loss, loss_dict = criterion(pred_dict, target_expr, target_cls, mask)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0
        assert "mlm_loss" in loss_dict
        assert "cls_loss" in loss_dict


class TestMetrics:
    """Tests for metrics."""

    def test_masked_mse(self):
        """Test masked MSE metric."""
        pred = torch.randn(4, 100)
        target = torch.randn(4, 100)
        mask = torch.rand(4, 100) > 0.5

        metric = masked_mse(pred, target, mask)

        assert isinstance(metric, float)
        assert metric >= 0

    def test_masked_mae(self):
        """Test masked MAE metric."""
        pred = torch.randn(4, 100)
        target = torch.randn(4, 100)
        mask = torch.rand(4, 100) > 0.5

        metric = masked_mae(pred, target, mask)

        assert isinstance(metric, float)
        assert metric >= 0

    def test_masked_pearson(self):
        """Test masked Pearson correlation."""
        pred = torch.randn(4, 100)
        target = torch.randn(4, 100)
        mask = torch.rand(4, 100) > 0.5

        metric = masked_pearson(pred, target, mask)

        assert isinstance(metric, float)
        assert -1 <= metric <= 1

    def test_compute_mlm_metrics(self):
        """Test compute_mlm_metrics."""
        pred = torch.randn(4, 100)
        target = torch.randn(4, 100)
        mask = torch.rand(4, 100) > 0.5

        metrics = compute_mlm_metrics(pred, target, mask)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "pearson" in metrics

    def test_compute_classification_metrics(self):
        """Test compute_classification_metrics."""
        pred = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))

        metrics = compute_classification_metrics(pred, target)

        assert "accuracy" in metrics
        assert "balanced_accuracy" in metrics
        assert "macro_f1" in metrics
        assert "weighted_f1" in metrics

    def test_metrics_tracker(self):
        """Test MetricsTracker."""
        tracker = MetricsTracker()

        tracker.update({"loss": 0.5, "accuracy": 0.8})
        tracker.update({"loss": 0.4, "accuracy": 0.85})

        avg = tracker.get_average()
        assert avg["loss"] == pytest.approx(0.45)
        assert avg["accuracy"] == pytest.approx(0.825)

        last = tracker.get_last()
        assert last["loss"] == 0.4
        assert last["accuracy"] == 0.85


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def setup_training(self):
        """Set up model and data for training tests."""
        # Create simple model
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        model = TransformerModel(
            vocab_size=len(vocab),
            d_model=16,  # Small for testing
            nhead=2,
            num_layers=1,
            d_hid=32,
            vocab=vocab,
        )

        # Create dummy data
        batch_size = 4
        seq_len = 10
        n_batches = 3

        tokenized_data = []
        for _ in range(batch_size * n_batches):
            gene_ids = torch.randint(3, len(vocab), (seq_len,))
            gene_ids[0] = vocab.cls_id
            values = torch.randn(seq_len).abs()
            tokenized_data.append((gene_ids, values))

        # Create dataloaders
        train_loader = create_dataloader(
            tokenized_data[:8],
            vocab,
            batch_size=4,
            max_len=15,
            mask_ratio=0.15,
            shuffle=False,
        )

        val_loader = create_dataloader(
            tokenized_data[8:],
            vocab,
            batch_size=4,
            max_len=15,
            mask_ratio=0.15,
            shuffle=False,
        )

        return model, train_loader, val_loader

    def test_trainer_creation(self, setup_training):
        """Test Trainer creation."""
        model, train_loader, val_loader = setup_training

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = MLMLoss(value_mode="continuous")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_train_epoch(self, setup_training, tmp_path):
        """Test training for one epoch."""
        model, train_loader, val_loader = setup_training

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = MLMLoss(value_mode="continuous")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            checkpoint_dir=tmp_path,
        )

        metrics = trainer.train_epoch(train_loader, epoch=1)

        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_validate(self, setup_training):
        """Test validation."""
        model, train_loader, val_loader = setup_training

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = MLMLoss(value_mode="continuous")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
        )

        metrics = trainer.validate(val_loader)

        assert "mse" in metrics or "loss" in metrics

    def test_train(self, setup_training, tmp_path):
        """Test full training loop."""
        model, train_loader, val_loader = setup_training

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = MLMLoss(value_mode="continuous")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            checkpoint_dir=tmp_path,
        )

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            save_every=1,
            eval_every=1,
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2

    def test_checkpoint_save_load(self, setup_training, tmp_path):
        """Test checkpoint saving and loading."""
        model, train_loader, val_loader = setup_training

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = MLMLoss(value_mode="continuous")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            checkpoint_dir=tmp_path,
        )

        # Train for one epoch
        trainer.train_epoch(train_loader, epoch=1)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint("test_checkpoint.pt")

        assert checkpoint_path.exists()

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)

        assert trainer.current_epoch == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
