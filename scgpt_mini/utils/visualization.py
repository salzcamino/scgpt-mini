"""
Visualization utilities for scGPT-mini.

This module provides functions for:
- UMAP and t-SNE visualization
- Plotting cell embeddings
- Visualization of training curves
"""

from typing import Optional, Union, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False


def compute_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute UMAP projection of embeddings.

    Args:
        embeddings: Embeddings of shape (n_cells, d_model)
        n_components: Number of UMAP components (typically 2 for visualization)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random seed

    Returns:
        UMAP coordinates of shape (n_cells, n_components)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )

    umap_coords = reducer.fit_transform(embeddings)
    return umap_coords


def compute_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute t-SNE projection of embeddings.

    Args:
        embeddings: Embeddings of shape (n_cells, d_model)
        n_components: Number of t-SNE components (typically 2 for visualization)
        perplexity: Perplexity parameter for t-SNE
        random_state: Random seed

    Returns:
        t-SNE coordinates of shape (n_cells, n_components)
    """
    if not TSNE_AVAILABLE:
        raise ImportError("scikit-learn is required for t-SNE")

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
    )

    tsne_coords = tsne.fit_transform(embeddings)
    return tsne_coords


def plot_umap(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    title: str = "UMAP Projection",
    figsize: tuple = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    **umap_kwargs,
) -> tuple:
    """
    Plot UMAP projection of embeddings.

    Args:
        embeddings: Embeddings of shape (n_cells, d_model)
        labels: Optional labels of shape (n_cells,) for coloring
        label_names: Optional list of label names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (if provided)
        **umap_kwargs: Additional kwargs for UMAP

    Returns:
        (figure, axes, umap_coords)

    Example:
        >>> fig, ax, coords = plot_umap(embeddings, labels=cell_types)
        >>> plt.show()
    """
    # Compute UMAP
    umap_coords = compute_umap(embeddings, **umap_kwargs)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        # Plot with colors
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)

        # Use seaborn color palette if available
        if SEABORN_AVAILABLE:
            colors = sns.color_palette("husl", n_labels)
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_labels))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names[label] if label_names is not None else str(label)

            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=[colors[i]],
                label=label_name,
                alpha=0.6,
                s=20,
            )

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    else:
        # Plot without colors
        ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig, ax, umap_coords


def plot_tsne(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    title: str = "t-SNE Projection",
    figsize: tuple = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    **tsne_kwargs,
) -> tuple:
    """
    Plot t-SNE projection of embeddings.

    Args:
        embeddings: Embeddings of shape (n_cells, d_model)
        labels: Optional labels of shape (n_cells,) for coloring
        label_names: Optional list of label names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (if provided)
        **tsne_kwargs: Additional kwargs for t-SNE

    Returns:
        (figure, axes, tsne_coords)
    """
    # Compute t-SNE
    tsne_coords = compute_tsne(embeddings, **tsne_kwargs)

    # Create plot (reuse UMAP plotting logic)
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)

        if SEABORN_AVAILABLE:
            colors = sns.color_palette("husl", n_labels)
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_labels))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names[label] if label_names is not None else str(label)

            ax.scatter(
                tsne_coords[mask, 0],
                tsne_coords[mask, 1],
                c=[colors[i]],
                label=label_name,
                alpha=0.6,
                s=20,
            )

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    else:
        ax.scatter(
            tsne_coords[:, 0],
            tsne_coords[:, 1],
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig, ax, tsne_coords


def plot_training_curves(
    history: dict,
    figsize: tuple = (12, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """
    Plot training curves from training history.

    Args:
        history: Training history dictionary with keys like:
            - "train_loss", "val_loss", "learning_rate"
        figsize: Figure size
        save_path: Path to save figure (if provided)

    Returns:
        (figure, axes)

    Example:
        >>> history = trainer.training_history
        >>> fig, axes = plot_training_curves(history)
        >>> plt.show()
    """
    n_plots = 0
    has_train_loss = "train_loss" in history and len(history["train_loss"]) > 0
    has_val_loss = "val_loss" in history and len(history["val_loss"]) > 0
    has_lr = "learning_rate" in history and len(history["learning_rate"]) > 0

    if has_train_loss or has_val_loss:
        n_plots += 1
    if has_lr:
        n_plots += 1

    if n_plots == 0:
        print("No data to plot in history")
        return None, None

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot loss
    if has_train_loss or has_val_loss:
        ax = axes[plot_idx]

        if has_train_loss:
            epochs = range(1, len(history["train_loss"]) + 1)
            ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)

        if has_val_loss:
            # Val loss might be evaluated less frequently
            val_epochs = range(1, len(history["val_loss"]) + 1)
            ax.plot(val_epochs, history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot learning rate
    if has_lr:
        ax = axes[plot_idx]
        epochs = range(1, len(history["learning_rate"]) + 1)
        ax.plot(epochs, history["learning_rate"], linewidth=2, color='green')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig, axes
