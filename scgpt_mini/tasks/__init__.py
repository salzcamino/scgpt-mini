"""
Downstream tasks module for scGPT-mini.

This module contains implementations for:
- Cell embedding generation
- Cell type annotation
"""

from scgpt_mini.tasks.embedding import (
    extract_cell_embeddings,
    save_embeddings_to_adata,
    compute_silhouette_score,
    compute_ari,
    compute_knn_accuracy,
    embedding_eval_report,
)
from scgpt_mini.tasks.annotation import (
    create_classification_dataset,
    finetune_for_annotation,
    predict_cell_types,
    annotation_eval_report,
)

__all__ = [
    # Embedding functions
    "extract_cell_embeddings",
    "save_embeddings_to_adata",
    "compute_silhouette_score",
    "compute_ari",
    "compute_knn_accuracy",
    "embedding_eval_report",
    # Annotation functions
    "create_classification_dataset",
    "finetune_for_annotation",
    "predict_cell_types",
    "annotation_eval_report",
]
