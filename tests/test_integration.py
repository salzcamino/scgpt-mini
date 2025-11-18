"""
Integration tests for scGPT-mini.

These tests verify that all components work together correctly
in realistic end-to-end workflows.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import all modules
from scgpt_mini.tokenizer import GeneVocab, tokenize_batch
from scgpt_mini.data import preprocess_adata, create_dataloader
from scgpt_mini.model import TransformerModel
from scgpt_mini.training import Trainer, MLMLoss
from scgpt_mini.tasks import (
    extract_cell_embeddings,
    save_embeddings_to_adata,
    embedding_eval_report,
    create_classification_dataset,
    finetune_for_annotation,
    predict_cell_types,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="module")
def mock_dataset():
    """Create a small mock dataset for testing."""
    try:
        import anndata as ad
        import pandas as pd

        n_cells = 100
        n_genes = 50

        # Create mock expression data
        np.random.seed(42)
        X = np.random.poisson(lam=2.0, size=(n_cells, n_genes)).astype(float)

        # Add some structure (cell types)
        cell_types = np.array(["TypeA"] * 40 + ["TypeB"] * 35 + ["TypeC"] * 25)

        obs = pd.DataFrame({
            "cell_type": cell_types,
            "n_counts": X.sum(axis=1),
        })

        var = pd.DataFrame(
            index=[f"Gene{i}" for i in range(n_genes)]
        )

        adata = ad.AnnData(X=X, obs=obs, var=var)
        return adata

    except ImportError:
        pytest.skip("anndata not available")


@pytest.fixture(scope="module")
def vocab():
    """Create a small vocabulary."""
    genes = [f"Gene{i}" for i in range(100)]
    return GeneVocab(genes)


# =============================================================================
# Integration Test 1: Preprocessing → Tokenization → DataLoader
# =============================================================================


def test_preprocessing_to_dataloader_pipeline(mock_dataset, vocab):
    """Test complete preprocessing and data loading pipeline."""
    # Preprocess
    adata = preprocess_adata(
        mock_dataset,
        filter_gene_by_counts=1,
        filter_cell_by_genes=5,
        normalize_total_target=1e4,
        log1p=True,
        subset_hvg=30,
        binning=False,
        inplace=False,
    )

    assert adata.n_obs > 0
    assert adata.n_vars <= 30

    # Tokenize
    data_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    gene_names = adata.var_names.values

    tokenized_data = tokenize_batch(
        data=data_matrix,
        gene_names=gene_names,
        vocab=vocab,
        append_cls=True,
        include_zero_genes=False,
        return_pt=True,
    )

    assert len(tokenized_data) == adata.n_obs

    # Create DataLoader
    dataloader = create_dataloader(
        tokenized_data=tokenized_data,
        vocab=vocab,
        batch_size=16,
        max_len=50,
        shuffle=False,
        apply_masking=True,
        mask_ratio=0.15,
    )

    # Test batch
    batch = next(iter(dataloader))
    assert "genes" in batch
    assert "masked_values" in batch
    assert "mask_positions" in batch
    assert batch["genes"].shape[0] <= 16  # batch size


# =============================================================================
# Integration Test 2: Model Creation → Training
# =============================================================================


def test_model_creation_and_training(mock_dataset, vocab, temp_dir):
    """Test model creation and short training run."""
    # Preprocess
    adata = preprocess_adata(
        mock_dataset,
        filter_gene_by_counts=1,
        normalize_total_target=1e4,
        log1p=True,
        subset_hvg=30,
        binning=False,
        inplace=False,
    )

    # Tokenize
    data_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    tokenized_data = tokenize_batch(
        data=data_matrix,
        gene_names=adata.var_names.values,
        vocab=vocab,
        append_cls=True,
        return_pt=True,
    )

    # Create DataLoaders
    n_train = int(0.8 * len(tokenized_data))
    train_data = tokenized_data[:n_train]
    val_data = tokenized_data[n_train:]

    train_loader = create_dataloader(
        train_data, vocab, batch_size=16, max_len=50, apply_masking=True
    )
    val_loader = create_dataloader(
        val_data, vocab, batch_size=16, max_len=50, apply_masking=True
    )

    # Create model
    model = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        d_model=16,
        nhead=2,
        num_layers=1,
        d_hid=32,
        dropout=0.1,
    )

    # Setup training
    criterion = MLMLoss(value_mode="continuous")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
    )

    # Train for 2 epochs
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        checkpoint_dir=temp_dir / "checkpoints",
    )

    # Check training happened
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert all(loss > 0 for loss in history["train_loss"])

    # Check checkpoint exists
    assert (temp_dir / "checkpoints" / "best_model.pt").exists()


# =============================================================================
# Integration Test 3: Training → Embedding → Evaluation
# =============================================================================


def test_training_to_embedding_pipeline(mock_dataset, vocab, temp_dir):
    """Test complete pipeline from training to embedding extraction."""
    # Preprocess
    adata = preprocess_adata(
        mock_dataset,
        filter_gene_by_counts=1,
        normalize_total_target=1e4,
        log1p=True,
        subset_hvg=30,
        binning=False,
        inplace=False,
    )

    # Tokenize
    data_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    tokenized_data = tokenize_batch(
        data=data_matrix,
        gene_names=adata.var_names.values,
        vocab=vocab,
        append_cls=True,
        return_pt=True,
    )

    # Create model and train (minimal)
    model = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        d_model=16,
        nhead=2,
        num_layers=1,
    )

    train_loader = create_dataloader(
        tokenized_data, vocab, batch_size=16, max_len=50, apply_masking=True
    )

    criterion = MLMLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, criterion)

    # Quick training
    trainer.train_epoch(train_loader, epoch=0)

    # Extract embeddings
    eval_loader = create_dataloader(
        tokenized_data, vocab, batch_size=16, max_len=50, apply_masking=False
    )

    embeddings = extract_cell_embeddings(
        model=model,
        dataloader=eval_loader,
        device="cpu",
        embedding_mode="cls",
    )

    assert embeddings.shape == (len(tokenized_data), model.d_model)

    # Save to AnnData
    save_embeddings_to_adata(adata, embeddings, key="X_scgpt")
    assert "X_scgpt" in adata.obsm

    # Evaluate
    labels = adata.obs["cell_type"].values
    metrics = embedding_eval_report(
        embeddings=embeddings,
        labels=labels,
        verbose=False,
    )

    assert "silhouette_score" in metrics
    assert "knn_accuracy" in metrics


# =============================================================================
# Integration Test 4: Complete Annotation Workflow
# =============================================================================


def test_complete_annotation_workflow(mock_dataset, vocab, temp_dir):
    """Test end-to-end annotation workflow."""
    # Preprocess
    adata = preprocess_adata(
        mock_dataset,
        filter_gene_by_counts=1,
        normalize_total_target=1e4,
        log1p=True,
        subset_hvg=30,
        binning=False,
        inplace=False,
    )

    # Create splits
    split_indices, label_encoder = create_classification_dataset(
        adata=adata,
        label_key="cell_type",
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_state=42,
    )

    # Tokenize
    data_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    tokenized_data = tokenize_batch(
        data=data_matrix,
        gene_names=adata.var_names.values,
        vocab=vocab,
        append_cls=True,
        return_pt=True,
    )

    # Add labels
    label_ids = [
        label_encoder["label_to_id"][label]
        for label in adata.obs["cell_type"].values
    ]

    labeled_data = [
        (genes, values, label_ids[i])
        for i, (genes, values) in enumerate(tokenized_data)
    ]

    # Split data
    train_data = [labeled_data[i] for i in split_indices["train"]]
    val_data = [labeled_data[i] for i in split_indices["val"]]
    test_data = [labeled_data[i] for i in split_indices["test"]]

    # Create dataloaders
    train_loader = create_dataloader(
        train_data, vocab, batch_size=16, max_len=50,
        apply_masking=False, include_labels=True
    )
    val_loader = create_dataloader(
        val_data, vocab, batch_size=16, max_len=50,
        apply_masking=False, include_labels=True
    )
    test_loader = create_dataloader(
        test_data, vocab, batch_size=16, max_len=50,
        apply_masking=False, include_labels=True
    )

    # Create and pretrain model (minimal)
    model = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        d_model=16,
        nhead=2,
        num_layers=1,
    )

    # Fine-tune for annotation (1 epoch)
    history = finetune_for_annotation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=label_encoder["n_classes"],
        num_epochs=1,
        learning_rate=1e-3,
        freeze_encoder=True,
        device="cpu",
        checkpoint_dir=temp_dir / "annotation",
    )

    assert "train_acc" in history
    assert "val_acc" in history

    # Predict
    predictions = predict_cell_types(
        model=model,
        dataloader=test_loader,
        label_encoder=label_encoder,
        device="cpu",
    )

    assert len(predictions) == len(test_data)
    assert all(0 <= p < label_encoder["n_classes"] for p in predictions)


# =============================================================================
# Integration Test 5: Save/Load Checkpoint
# =============================================================================


def test_save_load_checkpoint_integration(vocab, temp_dir):
    """Test saving and loading model checkpoints."""
    # Create model
    model1 = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        d_model=16,
        nhead=2,
        num_layers=1,
    )

    # Save checkpoint
    checkpoint_path = temp_dir / "test_model.pt"
    model1.save_checkpoint(checkpoint_path)

    assert checkpoint_path.exists()

    # Load checkpoint
    model2 = TransformerModel.load_checkpoint(checkpoint_path, vocab=vocab)

    # Check parameters match
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


# =============================================================================
# Integration Test 6: Memory Usage
# =============================================================================


def test_memory_efficiency(mock_dataset, vocab):
    """Test that memory usage stays reasonable."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Run full pipeline
    adata = preprocess_adata(
        mock_dataset,
        filter_gene_by_counts=1,
        normalize_total_target=1e4,
        log1p=True,
        subset_hvg=30,
        binning=False,
        inplace=False,
    )

    data_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    tokenized_data = tokenize_batch(
        data=data_matrix,
        gene_names=adata.var_names.values,
        vocab=vocab,
        append_cls=True,
        return_pt=True,
    )

    dataloader = create_dataloader(
        tokenized_data, vocab, batch_size=16, max_len=50
    )

    model = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        d_model=16,
        nhead=2,
        num_layers=1,
    )

    # Forward pass
    batch = next(iter(dataloader))
    with torch.no_grad():
        _ = model(
            batch["genes"],
            batch["masked_values"],
            batch["attention_mask"],
        )

    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_increase = mem_after - mem_before

    print(f"\nMemory increase: {mem_increase:.2f} MB")

    # Should use less than 500MB for this small test
    assert mem_increase < 500


# =============================================================================
# Integration Test 7: Reproducibility
# =============================================================================


def test_reproducibility(mock_dataset, vocab):
    """Test that results are reproducible with fixed seed."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Run pipeline twice
    results = []

    for _ in range(2):
        adata = preprocess_adata(
            mock_dataset,
            filter_gene_by_counts=1,
            normalize_total_target=1e4,
            log1p=True,
            subset_hvg=30,
            binning=False,
            inplace=False,
        )

        data_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        tokenized_data = tokenize_batch(
            data=data_matrix,
            gene_names=adata.var_names.values,
            vocab=vocab,
            append_cls=True,
            return_pt=True,
        )

        model = TransformerModel(
            vocab_size=len(vocab),
            vocab=vocab,
            d_model=16,
            nhead=2,
            num_layers=1,
        )

        torch.manual_seed(42)  # Reset for model initialization
        dataloader = create_dataloader(
            tokenized_data, vocab, batch_size=16, max_len=50, shuffle=False
        )

        embeddings = extract_cell_embeddings(
            model, dataloader, device="cpu", embedding_mode="cls"
        )

        results.append(embeddings)

    # Results should be identical
    assert np.allclose(results[0], results[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
