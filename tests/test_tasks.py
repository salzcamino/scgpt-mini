"""
Unit tests for downstream tasks (Phases 4 & 5).

Tests for:
- Cell embedding extraction
- Embedding quality metrics
- Cell type annotation
- Prediction and evaluation
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scgpt_mini.tokenizer import GeneVocab
from scgpt_mini.model import TransformerModel
from scgpt_mini.data import CellDataset
from scgpt_mini.tasks import (
    extract_cell_embeddings,
    save_embeddings_to_adata,
    compute_silhouette_score,
    compute_ari,
    compute_knn_accuracy,
    embedding_eval_report,
    create_classification_dataset,
    finetune_for_annotation,
    predict_cell_types,
    annotation_eval_report,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def vocab():
    """Create a small vocabulary for testing."""
    genes = [f"Gene{i}" for i in range(100)]
    return GeneVocab(genes)


@pytest.fixture
def model(vocab):
    """Create a small model for testing."""
    model = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        d_model=16,
        nhead=2,
        num_layers=1,
        d_hid=32,
        dropout=0.1,
    )
    return model


@pytest.fixture
def tokenized_data():
    """Create dummy tokenized data."""
    n_cells = 50
    data = []
    for i in range(n_cells):
        n_genes = np.random.randint(10, 30)
        genes = torch.randint(3, 100, (n_genes,))  # Avoid special tokens
        values = torch.rand(n_genes)
        data.append((genes, values))
    return data


@pytest.fixture
def labeled_tokenized_data():
    """Create dummy tokenized data with labels."""
    n_cells = 50
    data = []
    for i in range(n_cells):
        n_genes = np.random.randint(10, 30)
        genes = torch.randint(3, 100, (n_genes,))
        values = torch.rand(n_genes)
        label = i % 3  # 3 classes
        data.append((genes, values, label))
    return data


@pytest.fixture
def dataloader(tokenized_data, vocab):
    """Create a DataLoader for testing."""
    dataset = CellDataset(
        tokenized_data=tokenized_data,
        vocab=vocab,
        max_len=50,
        apply_masking=False,
    )
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def labeled_dataloader(labeled_tokenized_data, vocab):
    """Create a DataLoader with labels for testing."""
    dataset = CellDataset(
        tokenized_data=labeled_tokenized_data,
        vocab=vocab,
        max_len=50,
        apply_masking=False,
        include_labels=True,
    )
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object."""
    try:
        import anndata as ad
        import pandas as pd

        n_obs = 100
        n_vars = 50

        X = np.random.rand(n_obs, n_vars)
        obs = pd.DataFrame(
            {
                "cell_type": np.random.choice(["TypeA", "TypeB", "TypeC"], n_obs)
            }
        )
        var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_vars)])

        return ad.AnnData(X=X, obs=obs, var=var)
    except ImportError:
        pytest.skip("anndata not available")


# =============================================================================
# Phase 4: Cell Embedding Extraction Tests
# =============================================================================


def test_extract_cell_embeddings_cls(model, dataloader):
    """Test CLS token embedding extraction."""
    model.eval()

    embeddings = extract_cell_embeddings(
        model=model,
        dataloader=dataloader,
        device="cpu",
        embedding_mode="cls",
    )

    # Check shape
    assert embeddings.shape[0] == 50  # n_cells
    assert embeddings.shape[1] == model.d_model
    assert isinstance(embeddings, np.ndarray)


def test_extract_cell_embeddings_mean(model, dataloader):
    """Test mean pooling embedding extraction."""
    model.eval()

    embeddings = extract_cell_embeddings(
        model=model,
        dataloader=dataloader,
        device="cpu",
        embedding_mode="mean",
    )

    assert embeddings.shape == (50, model.d_model)


def test_extract_cell_embeddings_max(model, dataloader):
    """Test max pooling embedding extraction."""
    model.eval()

    embeddings = extract_cell_embeddings(
        model=model,
        dataloader=dataloader,
        device="cpu",
        embedding_mode="max",
    )

    assert embeddings.shape == (50, model.d_model)


def test_extract_cell_embeddings_invalid_mode(model, dataloader):
    """Test that invalid embedding mode raises error."""
    with pytest.raises(ValueError, match="embedding_mode must be one of"):
        extract_cell_embeddings(
            model=model,
            dataloader=dataloader,
            device="cpu",
            embedding_mode="invalid",
        )


def test_extract_cell_embeddings_with_labels(model, labeled_dataloader):
    """Test embedding extraction with labels."""
    model.eval()

    embeddings, labels = extract_cell_embeddings(
        model=model,
        dataloader=labeled_dataloader,
        device="cpu",
        embedding_mode="cls",
        return_labels=True,
    )

    assert embeddings.shape == (50, model.d_model)
    assert len(labels) == 50
    assert all(label in [0, 1, 2] for label in labels)


def test_save_embeddings_to_adata(mock_adata):
    """Test saving embeddings to AnnData."""
    embeddings = np.random.rand(100, 16)

    save_embeddings_to_adata(
        adata=mock_adata,
        embeddings=embeddings,
        key="X_test",
    )

    assert "X_test" in mock_adata.obsm
    assert mock_adata.obsm["X_test"].shape == (100, 16)


def test_save_embeddings_to_adata_wrong_size(mock_adata):
    """Test that wrong embedding size raises error."""
    embeddings = np.random.rand(50, 16)  # Wrong n_obs

    with pytest.raises(ValueError, match="Number of embeddings"):
        save_embeddings_to_adata(
            adata=mock_adata,
            embeddings=embeddings,
            key="X_test",
        )


# =============================================================================
# Phase 4: Embedding Quality Metrics Tests
# =============================================================================


def test_compute_silhouette_score():
    """Test silhouette score computation."""
    embeddings = np.random.rand(100, 16)
    labels = np.random.choice([0, 1, 2], 100)

    score = compute_silhouette_score(embeddings, labels)

    assert isinstance(score, float)
    assert -1 <= score <= 1


def test_compute_silhouette_score_single_cluster():
    """Test silhouette score with single cluster (should raise warning)."""
    embeddings = np.random.rand(100, 16)
    labels = np.zeros(100)  # All same cluster

    # Should handle gracefully
    score = compute_silhouette_score(embeddings, labels)
    assert score is not None or np.isnan(score)


def test_compute_ari():
    """Test Adjusted Rand Index computation."""
    embeddings = np.random.rand(100, 16)
    true_labels = np.random.choice([0, 1, 2], 100)

    ari, pred_labels = compute_ari(
        embeddings, true_labels, n_clusters=3, method="kmeans"
    )

    assert isinstance(ari, float)
    assert -1 <= ari <= 1
    assert len(pred_labels) == 100


def test_compute_ari_leiden(mock_adata):
    """Test ARI computation with Leiden clustering."""
    embeddings = np.random.rand(100, 16)
    true_labels = np.random.choice([0, 1, 2], 100)

    ari, pred_labels = compute_ari(
        embeddings, true_labels, n_clusters=3, method="leiden"
    )

    assert isinstance(ari, float)
    assert len(pred_labels) == 100


def test_compute_knn_accuracy():
    """Test k-NN accuracy computation."""
    embeddings = np.random.rand(100, 16)
    labels = np.random.choice([0, 1, 2], 100)

    accuracy = compute_knn_accuracy(embeddings, labels, k=5, test_size=0.3)

    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1


def test_embedding_eval_report():
    """Test comprehensive embedding evaluation report."""
    embeddings = np.random.rand(100, 16)
    labels = np.random.choice([0, 1, 2], 100)

    metrics = embedding_eval_report(
        embeddings=embeddings,
        labels=labels,
        n_neighbors=5,
        verbose=False,
    )

    assert "silhouette_score" in metrics
    assert "ari" in metrics
    assert "knn_accuracy" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


# =============================================================================
# Phase 5: Cell Type Annotation Tests
# =============================================================================


def test_create_classification_dataset(mock_adata):
    """Test dataset creation with train/val/test splits."""
    split_indices, label_encoder = create_classification_dataset(
        adata=mock_adata,
        label_key="cell_type",
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_state=42,
    )

    # Check splits
    assert "train" in split_indices
    assert "val" in split_indices
    assert "test" in split_indices

    n_train = len(split_indices["train"])
    n_val = len(split_indices["val"])
    n_test = len(split_indices["test"])

    assert n_train + n_val + n_test == 100
    assert n_train == 70
    assert n_val == 15
    assert n_test == 15

    # Check label encoder
    assert "label_to_id" in label_encoder
    assert "id_to_label" in label_encoder
    assert "n_classes" in label_encoder
    assert label_encoder["n_classes"] == 3


def test_finetune_for_annotation(model, labeled_dataloader):
    """Test fine-tuning for annotation (1 epoch)."""
    model.eval()

    # Split into train and val
    all_data = list(labeled_dataloader.dataset)
    train_data = all_data[:40]
    val_data = all_data[40:]

    from scgpt_mini.data import CellDataset
    train_dataset = CellDataset(
        tokenized_data=train_data,
        vocab=model.vocab,
        max_len=50,
        apply_masking=False,
        include_labels=True,
    )
    val_dataset = CellDataset(
        tokenized_data=val_data,
        vocab=model.vocab,
        max_len=50,
        apply_masking=False,
        include_labels=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Fine-tune for 1 epoch
    history = finetune_for_annotation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=3,
        num_epochs=1,
        learning_rate=1e-3,
        freeze_encoder=True,
        device="cpu",
        checkpoint_dir="test_checkpoints",
    )

    # Check history
    assert "train_loss" in history
    assert "train_acc" in history
    assert "val_loss" in history
    assert "val_acc" in history
    assert len(history["train_loss"]) == 1


def test_predict_cell_types(model, labeled_dataloader):
    """Test cell type prediction."""
    # Add classification head
    from scgpt_mini.model.decoder import ClassificationDecoder
    model.cls_decoder = ClassificationDecoder(
        d_model=model.d_model,
        n_classes=3,
    )
    model.eval()

    # Create label encoder
    label_encoder = {
        "label_to_id": {"TypeA": 0, "TypeB": 1, "TypeC": 2},
        "id_to_label": {0: "TypeA", 1: "TypeB", 2: "TypeC"},
        "n_classes": 3,
    }

    # Predict
    predictions = predict_cell_types(
        model=model,
        dataloader=labeled_dataloader,
        label_encoder=label_encoder,
        device="cpu",
        return_probabilities=False,
    )

    assert len(predictions) == 50
    assert all(p in [0, 1, 2] for p in predictions)


def test_predict_cell_types_with_probabilities(model, labeled_dataloader):
    """Test prediction with probabilities."""
    from scgpt_mini.model.decoder import ClassificationDecoder
    model.cls_decoder = ClassificationDecoder(
        d_model=model.d_model,
        n_classes=3,
    )
    model.eval()

    label_encoder = {
        "label_to_id": {"TypeA": 0, "TypeB": 1, "TypeC": 2},
        "id_to_label": {0: "TypeA", 1: "TypeB", 2: "TypeC"},
        "n_classes": 3,
    }

    predictions, probabilities = predict_cell_types(
        model=model,
        dataloader=labeled_dataloader,
        label_encoder=label_encoder,
        device="cpu",
        return_probabilities=True,
    )

    assert len(predictions) == 50
    assert probabilities.shape == (50, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1


def test_annotation_eval_report():
    """Test annotation evaluation report."""
    n_samples = 100
    true_labels = np.random.choice([0, 1, 2], n_samples)
    pred_labels = np.random.choice([0, 1, 2], n_samples)

    label_encoder = {
        "label_to_id": {"TypeA": 0, "TypeB": 1, "TypeC": 2},
        "id_to_label": {0: "TypeA", 1: "TypeB", 2: "TypeC"},
        "n_classes": 3,
    }

    result = annotation_eval_report(
        true_labels=true_labels,
        pred_labels=pred_labels,
        label_encoder=label_encoder,
        save_path=None,
    )

    assert "metrics" in result
    assert "confusion_matrix" in result
    assert "accuracy" in result["metrics"]
    assert "balanced_accuracy" in result["metrics"]
    assert result["confusion_matrix"].shape == (3, 3)


# =============================================================================
# Integration Tests
# =============================================================================


def test_end_to_end_embedding_workflow(model, dataloader):
    """Test complete embedding extraction workflow."""
    model.eval()

    # Extract embeddings
    embeddings = extract_cell_embeddings(
        model=model,
        dataloader=dataloader,
        device="cpu",
        embedding_mode="cls",
    )

    # Create dummy labels
    labels = np.random.choice([0, 1, 2], 50)

    # Compute metrics
    metrics = embedding_eval_report(
        embeddings=embeddings,
        labels=labels,
        verbose=False,
    )

    assert all(k in metrics for k in ["silhouette_score", "ari", "knn_accuracy"])


def test_end_to_end_annotation_workflow(model, labeled_tokenized_data, vocab):
    """Test complete annotation workflow (minimal)."""
    from scgpt_mini.data import CellDataset

    # Create datasets
    train_data = labeled_tokenized_data[:40]
    val_data = labeled_tokenized_data[40:]

    train_dataset = CellDataset(
        tokenized_data=train_data,
        vocab=vocab,
        max_len=50,
        apply_masking=False,
        include_labels=True,
    )
    val_dataset = CellDataset(
        tokenized_data=val_data,
        vocab=vocab,
        max_len=50,
        apply_masking=False,
        include_labels=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Fine-tune (1 epoch)
    history = finetune_for_annotation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=3,
        num_epochs=1,
        learning_rate=1e-3,
        freeze_encoder=True,
        device="cpu",
        checkpoint_dir="test_checkpoints",
    )

    # Predict
    label_encoder = {
        "label_to_id": {0: 0, 1: 1, 2: 2},
        "id_to_label": {0: 0, 1: 1, 2: 2},
        "n_classes": 3,
    }

    predictions = predict_cell_types(
        model=model,
        dataloader=val_loader,
        label_encoder=label_encoder,
        device="cpu",
    )

    assert len(predictions) == len(val_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
