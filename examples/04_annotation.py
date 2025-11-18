"""
Example: Cell Type Annotation

This script demonstrates how to:
1. Load a pretrained model
2. Create a labeled dataset with train/val/test splits
3. Fine-tune the model for cell type classification
4. Evaluate annotation performance
5. Predict cell types on unlabeled data

This covers Phase 5 functionality.
"""

import torch
import scanpy as sc
from pathlib import Path

# Import scGPT-mini modules
from scgpt_mini.tokenizer import GeneVocab, tokenize_batch
from scgpt_mini.data import preprocess_adata, create_dataloader
from scgpt_mini.model import TransformerModel
from scgpt_mini.tasks import (
    create_classification_dataset,
    finetune_for_annotation,
    predict_cell_types,
    annotation_eval_report,
)
from scgpt_mini.utils import plot_umap


def main():
    print("=" * 70)
    print("scGPT-mini Phase 5: Cell Type Annotation Example")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load and preprocess data
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading and preprocessing PBMC 3k dataset...")
    adata = sc.datasets.pbmc3k()
    print(f"  Original: {adata.n_obs} cells x {adata.n_vars} genes")

    # Annotate cell types (if not already done)
    if "cell_type" not in adata.obs.columns:
        print("  Running cell type annotation with Louvain clusters...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.louvain(adata)
        sc.tl.rank_genes_groups(adata, "louvain", method="wilcoxon")

        # Simple cell type assignment based on marker genes
        # (In practice, use sc.tl.leiden and manual annotation)
        cell_type_map = {
            "0": "CD4 T cells",
            "1": "CD14 Monocytes",
            "2": "B cells",
            "3": "CD8 T cells",
            "4": "NK cells",
            "5": "FCGR3A Monocytes",
            "6": "Dendritic cells",
            "7": "Megakaryocytes",
        }
        adata.obs["cell_type"] = adata.obs["louvain"].map(
            lambda x: cell_type_map.get(str(x), "Unknown")
        )

    print(f"  Cell types: {adata.obs['cell_type'].unique()}")

    # Preprocess for scGPT-mini
    adata = preprocess_adata(
        adata,
        filter_gene_by_counts=10,
        filter_cell_by_genes=200,
        normalize_total_target=1e4,
        log1p=True,
        subset_hvg=500,
        binning=False,
        inplace=False,
    )

    print(f"  Preprocessed: {adata.n_obs} cells x {adata.n_vars} genes")

    # -------------------------------------------------------------------------
    # Step 2: Create train/val/test splits
    # -------------------------------------------------------------------------
    print("\n[Step 2] Creating train/val/test splits...")

    split_indices, label_encoder = create_classification_dataset(
        adata=adata,
        label_key="cell_type",
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_state=42,
    )

    print(f"\n  Label mapping:")
    for label, idx in label_encoder["label_to_id"].items():
        print(f"    {idx}: {label}")

    # -------------------------------------------------------------------------
    # Step 3: Tokenize and create dataloaders
    # -------------------------------------------------------------------------
    print("\n[Step 3] Tokenizing data...")

    # Load vocabulary
    vocab = GeneVocab.from_json("scgpt_mini/tokenizer/default_vocab.json")

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

    # Add labels to tokenized data
    labels = adata.obs["cell_type"].values
    label_ids = [label_encoder["label_to_id"][label] for label in labels]

    for i, (genes, values) in enumerate(tokenized_data):
        tokenized_data[i] = (genes, values, label_ids[i])

    # Split data
    train_data = [tokenized_data[i] for i in split_indices["train"]]
    val_data = [tokenized_data[i] for i in split_indices["val"]]
    test_data = [tokenized_data[i] for i in split_indices["test"]]

    print(f"  Train: {len(train_data)} cells")
    print(f"  Val: {len(val_data)} cells")
    print(f"  Test: {len(test_data)} cells")

    # Create dataloaders
    print("\n[Step 4] Creating dataloaders...")

    train_loader = create_dataloader(
        tokenized_data=train_data,
        vocab=vocab,
        batch_size=32,
        max_len=1001,
        shuffle=True,
        apply_masking=False,  # No masking for classification
        include_labels=True,
    )

    val_loader = create_dataloader(
        tokenized_data=val_data,
        vocab=vocab,
        batch_size=32,
        max_len=1001,
        shuffle=False,
        apply_masking=False,
        include_labels=True,
    )

    test_loader = create_dataloader(
        tokenized_data=test_data,
        vocab=vocab,
        batch_size=32,
        max_len=1001,
        shuffle=False,
        apply_masking=False,
        include_labels=True,
    )

    # -------------------------------------------------------------------------
    # Step 5: Load pretrained model
    # -------------------------------------------------------------------------
    print("\n[Step 5] Loading pretrained model...")

    model_path = Path("checkpoints/pretrained_model.pt")
    if not model_path.exists():
        print(f"  ERROR: Pretrained model not found at {model_path}")
        print("  Please run 02_training.py first to create a pretrained model.")
        return

    model = TransformerModel.load_checkpoint(model_path, vocab=vocab)

    print(f"  Model loaded successfully")

    # -------------------------------------------------------------------------
    # Step 6: Fine-tune for cell type annotation
    # -------------------------------------------------------------------------
    print("\n[Step 6] Fine-tuning for cell type annotation...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Option 1: Fine-tune only classification head (faster, less overfitting)
    print("\n  Fine-tuning classification head only...")

    history_frozen = finetune_for_annotation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=label_encoder["n_classes"],
        num_epochs=10,
        learning_rate=5e-4,
        freeze_encoder=True,
        device=device,
        checkpoint_dir="checkpoints/annotation",
    )

    # Option 2: Full fine-tuning (optional, uncomment if desired)
    # print("\n  Full model fine-tuning...")
    # history_full = finetune_for_annotation(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     n_classes=label_encoder["n_classes"],
    #     num_epochs=5,
    #     learning_rate=1e-5,
    #     freeze_encoder=False,
    #     device=device,
    #     checkpoint_dir="checkpoints/annotation_full",
    # )

    # -------------------------------------------------------------------------
    # Step 7: Load best model and evaluate
    # -------------------------------------------------------------------------
    print("\n[Step 7] Loading best model and evaluating...")

    best_model_path = Path("checkpoints/annotation/best_annotation_model.pt")
    model = TransformerModel.load_checkpoint(best_model_path, vocab=vocab)

    # Predict on test set
    predictions, probabilities = predict_cell_types(
        model=model,
        dataloader=test_loader,
        label_encoder=label_encoder,
        device=device,
        return_probabilities=True,
    )

    # Get true labels
    true_labels = [label_ids[i] for i in split_indices["test"]]

    # Evaluate
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    metrics = annotation_eval_report(
        true_labels=true_labels,
        pred_labels=predictions,
        label_encoder=label_encoder,
        save_path=output_dir / "annotation_report.txt",
    )

    # -------------------------------------------------------------------------
    # Step 8: Visualize predictions
    # -------------------------------------------------------------------------
    print("\n[Step 8] Visualizing predictions...")

    # Extract embeddings for visualization
    from scgpt_mini.tasks import extract_cell_embeddings

    test_embeddings = extract_cell_embeddings(
        model=model,
        dataloader=test_loader,
        device=device,
        embedding_mode="cls",
    )

    # Plot UMAP with true labels
    fig, ax, _ = plot_umap(
        embeddings=test_embeddings,
        labels=[label_encoder["id_to_label"][l] for l in true_labels],
        title="True Cell Types (Test Set)",
        save_path=output_dir / "annotation_true_labels.png",
    )

    # Plot UMAP with predictions
    fig, ax, _ = plot_umap(
        embeddings=test_embeddings,
        labels=[label_encoder["id_to_label"][l] for l in predictions],
        title="Predicted Cell Types (Test Set)",
        save_path=output_dir / "annotation_predictions.png",
    )

    print(f"  Plots saved to {output_dir}")

    # -------------------------------------------------------------------------
    # Step 9: Analyze prediction confidence
    # -------------------------------------------------------------------------
    print("\n[Step 9] Analyzing prediction confidence...")

    # Get max probability for each prediction
    confidences = probabilities.max(axis=1)

    print(f"\n  Confidence statistics:")
    print(f"    Mean: {confidences.mean():.4f}")
    print(f"    Median: {confidences.median():.4f}")
    print(f"    Min: {confidences.min():.4f}")
    print(f"    Max: {confidences.max():.4f}")

    # Find low-confidence predictions
    low_confidence_threshold = 0.5
    low_conf_mask = confidences < low_confidence_threshold
    n_low_conf = low_conf_mask.sum()

    print(f"\n  Low confidence predictions (<{low_confidence_threshold}): {n_low_conf} / {len(predictions)} "
          f"({100 * n_low_conf / len(predictions):.1f}%)")

    if n_low_conf > 0:
        print(f"\n  Examples of low-confidence predictions:")
        low_conf_indices = low_conf_mask.nonzero()[0][:5]
        for idx in low_conf_indices:
            true_label = label_encoder["id_to_label"][true_labels[idx]]
            pred_label = label_encoder["id_to_label"][predictions[idx]]
            conf = confidences[idx]
            print(f"    True: {true_label:20s} | Pred: {pred_label:20s} | Conf: {conf:.3f}")

    # -------------------------------------------------------------------------
    # Step 10: Predict on unlabeled data (simulation)
    # -------------------------------------------------------------------------
    print("\n[Step 10] Predicting on unlabeled data...")

    # Simulate unlabeled data (use a subset of test set)
    unlabeled_data = [(genes, values) for genes, values, _ in test_data[:100]]

    unlabeled_loader = create_dataloader(
        tokenized_data=unlabeled_data,
        vocab=vocab,
        batch_size=32,
        max_len=1001,
        shuffle=False,
        apply_masking=False,
        include_labels=False,
    )

    # Predict
    unlabeled_predictions, unlabeled_probs = predict_cell_types(
        model=model,
        dataloader=unlabeled_loader,
        label_encoder=label_encoder,
        device=device,
        return_probabilities=True,
    )

    # Convert predictions to cell type names
    predicted_cell_types = [
        label_encoder["id_to_label"][pred] for pred in unlabeled_predictions
    ]

    print(f"  Predicted cell types for {len(predicted_cell_types)} unlabeled cells:")
    from collections import Counter
    type_counts = Counter(predicted_cell_types)
    for cell_type, count in type_counts.most_common():
        print(f"    {cell_type:20s}: {count:4d} cells")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 5 cell type annotation complete!")
    print("=" * 70)

    print("\nGenerated files:")
    print(f"  - {best_model_path}")
    print(f"  - {output_dir / 'annotation_report.txt'}")
    print(f"  - {output_dir / 'annotation_true_labels.png'}")
    print(f"  - {output_dir / 'annotation_predictions.png'}")

    print("\nAnnotation performance:")
    print(f"  - Accuracy: {metrics['metrics']['accuracy']:.4f}")
    print(f"  - Balanced Accuracy: {metrics['metrics']['balanced_accuracy']:.4f}")
    print(f"  - Macro F1: {metrics['metrics']['macro_f1']:.4f}")
    print(f"  - Weighted F1: {metrics['metrics']['weighted_f1']:.4f}")

    print("\nTraining history:")
    print(f"  - Final train accuracy: {history_frozen['train_acc'][-1]:.4f}")
    print(f"  - Final val accuracy: {history_frozen['val_acc'][-1]:.4f}")
    print(f"  - Best val accuracy: {max(history_frozen['val_acc']):.4f}")

    print("\nNext steps:")
    print("  - Phase 6: Integration & comprehensive testing")
    print("  - Experiment with different fine-tuning strategies")
    print("  - Try full model fine-tuning for better performance")
    print("  - Apply to your own datasets")
    print("=" * 70)


if __name__ == "__main__":
    main()
