"""
Cell embedding generation for scGPT-mini.

This module provides functions to:
- Extract cell embeddings from trained models
- Compute embedding quality metrics
- Save embeddings for downstream analysis
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

try:
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some metrics will not work.")


def extract_cell_embeddings(
    model,
    dataloader,
    device: str = "cpu",
    embedding_mode: str = "cls",
    return_labels: bool = False,
) -> Union[np.ndarray, tuple]:
    """
    Extract cell embeddings from a trained model.

    Args:
        model: Trained TransformerModel
        dataloader: DataLoader for cells to embed
        device: Device to run on
        embedding_mode: How to extract embeddings:
            - "cls": Use CLS token embedding (default)
            - "mean": Mean pooling over all tokens
            - "max": Max pooling over all tokens
        return_labels: If True, also return labels (if available)

    Returns:
        Cell embeddings of shape (n_cells, d_model)
        If return_labels=True: (embeddings, labels)

    Example:
        >>> from scgpt_mini.model import TransformerModel
        >>> from scgpt_mini.tasks import extract_cell_embeddings
        >>>
        >>> model = TransformerModel.load_checkpoint("model.pt")
        >>> embeddings = extract_cell_embeddings(model, dataloader, device="cpu")
        >>> print(embeddings.shape)
        (2700, 32)
    """
    model.eval()
    model.to(device)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Move to device
            genes = batch["genes"].to(device)
            values = batch["values"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass with embedding extraction
            output = model(genes, values, attention_mask, return_embeddings=True)

            # Extract embeddings based on mode
            if embedding_mode == "cls":
                # Use CLS token embedding (first position)
                embeddings = output["embeddings"][:, 0, :]  # (batch, d_model)
            elif embedding_mode == "mean":
                # Mean pooling over all tokens (excluding padding)
                embeddings = output["embeddings"]  # (batch, seq_len, d_model)
                # Mask out padding
                mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
                masked_embeddings = embeddings * mask
                # Sum and divide by number of non-padding tokens
                sum_embeddings = masked_embeddings.sum(dim=1)  # (batch, d_model)
                num_tokens = mask.sum(dim=1)  # (batch, 1)
                embeddings = sum_embeddings / num_tokens
            elif embedding_mode == "max":
                # Max pooling over all tokens
                embeddings = output["embeddings"]  # (batch, seq_len, d_model)
                # Set padding positions to very negative value
                mask = attention_mask.unsqueeze(-1).float()
                masked_embeddings = embeddings.masked_fill(mask == 0, -1e9)
                embeddings = masked_embeddings.max(dim=1)[0]  # (batch, d_model)
            else:
                raise ValueError(f"Unknown embedding_mode: {embedding_mode}")

            all_embeddings.append(embeddings.cpu().numpy())

            # Collect labels if available
            if return_labels and "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())

    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)

    if return_labels and all_labels:
        labels = np.concatenate(all_labels, axis=0)
        return embeddings, labels

    return embeddings


def save_embeddings_to_adata(
    adata,
    embeddings: np.ndarray,
    key: str = "X_scgpt",
) -> None:
    """
    Save embeddings to AnnData object.

    Args:
        adata: AnnData object
        embeddings: Cell embeddings of shape (n_cells, d_model)
        key: Key to store embeddings in adata.obsm

    Example:
        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc3k()
        >>> save_embeddings_to_adata(adata, embeddings, key="X_scgpt")
        >>> print(adata.obsm["X_scgpt"].shape)
    """
    if embeddings.shape[0] != adata.n_obs:
        raise ValueError(
            f"Number of embeddings ({embeddings.shape[0]}) does not match "
            f"number of cells ({adata.n_obs})"
        )

    adata.obsm[key] = embeddings
    print(f"Embeddings saved to adata.obsm['{key}']")


def compute_silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute silhouette score for embeddings.

    Measures how well-separated clusters are. Higher is better.
    Range: [-1, 1], where 1 means perfect clustering.

    Args:
        embeddings: Cell embeddings of shape (n_cells, d_model)
        labels: Cell type labels of shape (n_cells,)

    Returns:
        Silhouette score
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for silhouette score")

    return silhouette_score(embeddings, labels)


def compute_ari(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: Optional[int] = None,
) -> float:
    """
    Compute Adjusted Rand Index (ARI) for embeddings.

    Clusters the embeddings with k-means and compares to ground truth.
    Range: [-1, 1], where 1 means perfect agreement.

    Args:
        embeddings: Cell embeddings of shape (n_cells, d_model)
        true_labels: Ground truth labels of shape (n_cells,)
        n_clusters: Number of clusters (if None, uses number of unique labels)

    Returns:
        ARI score
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for ARI")

    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))

    # Cluster embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(embeddings)

    # Compute ARI
    return adjusted_rand_score(true_labels, pred_labels)


def compute_knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> float:
    """
    Compute k-NN classification accuracy on embeddings.

    Splits data into train/test and evaluates k-NN classifier.

    Args:
        embeddings: Cell embeddings of shape (n_cells, d_model)
        labels: Cell type labels of shape (n_cells,)
        k: Number of neighbors for k-NN
        test_size: Fraction of data to use for testing
        random_state: Random seed

    Returns:
        Classification accuracy
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for k-NN accuracy")

    from sklearn.model_selection import train_test_split

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Evaluate
    accuracy = knn.score(X_test, y_test)

    return accuracy


def embedding_eval_report(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive embedding evaluation metrics.

    Args:
        embeddings: Cell embeddings of shape (n_cells, d_model)
        labels: Cell type labels of shape (n_cells,)
        label_names: Optional list of label names

    Returns:
        Dictionary of metrics

    Example:
        >>> metrics = embedding_eval_report(embeddings, labels)
        >>> print(f"Silhouette: {metrics['silhouette']:.3f}")
        >>> print(f"ARI: {metrics['ari']:.3f}")
        >>> print(f"k-NN Accuracy: {metrics['knn_accuracy']:.3f}")
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for embedding evaluation")

    metrics = {}

    # Silhouette score
    try:
        metrics["silhouette"] = compute_silhouette_score(embeddings, labels)
    except Exception as e:
        print(f"Warning: Could not compute silhouette score: {e}")
        metrics["silhouette"] = None

    # ARI
    try:
        metrics["ari"] = compute_ari(embeddings, labels)
    except Exception as e:
        print(f"Warning: Could not compute ARI: {e}")
        metrics["ari"] = None

    # k-NN accuracy
    try:
        metrics["knn_accuracy"] = compute_knn_accuracy(embeddings, labels)
    except Exception as e:
        print(f"Warning: Could not compute k-NN accuracy: {e}")
        metrics["knn_accuracy"] = None

    # Basic stats
    metrics["n_cells"] = len(embeddings)
    metrics["n_cell_types"] = len(np.unique(labels))
    metrics["embedding_dim"] = embeddings.shape[1]

    # Print report
    print("=" * 60)
    print("Embedding Evaluation Report")
    print("=" * 60)
    print(f"Number of cells: {metrics['n_cells']}")
    print(f"Number of cell types: {metrics['n_cell_types']}")
    print(f"Embedding dimension: {metrics['embedding_dim']}")
    print("-" * 60)
    if metrics["silhouette"] is not None:
        print(f"Silhouette Score: {metrics['silhouette']:.3f}")
    if metrics["ari"] is not None:
        print(f"Adjusted Rand Index: {metrics['ari']:.3f}")
    if metrics["knn_accuracy"] is not None:
        print(f"k-NN Accuracy: {metrics['knn_accuracy']:.3f}")
    print("=" * 60)

    return metrics
