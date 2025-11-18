"""
End-to-End Complete Workflow

This script demonstrates the complete scGPT-mini pipeline:
1. Data preprocessing
2. Model creation
3. Pretraining with MLM
4. Cell embedding extraction
5. Cell type annotation

This is an integration example showing all phases working together.
"""

import torch
import scanpy as sc
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import scGPT-mini modules
from scgpt_mini.tokenizer import GeneVocab, tokenize_batch
from scgpt_mini.data import preprocess_adata, create_dataloader
from scgpt_mini.model import TransformerModel
from scgpt_mini.training import Trainer, MLMLoss
from scgpt_mini.tasks import (
    extract_cell_embeddings,
    create_classification_dataset,
    finetune_for_annotation,
    predict_cell_types,
    embedding_eval_report,
    annotation_eval_report,
)
from scgpt_mini.utils import plot_umap, plot_training_curves


def main():
    print("=" * 80)
    print("scGPT-mini: Complete End-to-End Workflow")
    print("=" * 80)
    print("\nThis script demonstrates the full pipeline:")
    print("  Phase 1: Data Preprocessing")
    print("  Phase 2: Model Creation")
    print("  Phase 3: MLM Pretraining")
    print("  Phase 4: Cell Embedding Generation")
    print("  Phase 5: Cell Type Annotation")
    print("=" * 80)

    # Create output directories
    output_dir = Path("outputs/end_to_end")
    checkpoint_dir = Path("checkpoints/end_to_end")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # =========================================================================
    # PHASE 1: Data Preprocessing
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Data Preprocessing")
    print("=" * 80)

    print("\nLoading PBMC 3k dataset...")
    adata = sc.datasets.pbmc3k()
    print(f"  Original: {adata.n_obs} cells x {adata.n_vars} genes")

    # Preprocess
    print("\nPreprocessing data...")
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

    # Annotate cell types (simplified)
    print("\nAnnotating cell types...")
    if "cell_type" not in adata.obs.columns:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.louvain(adata)

        # Simplified cell type assignment
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

    print(f"  Cell types: {list(adata.obs['cell_type'].unique())}")

    # Load vocabulary
    print("\nLoading gene vocabulary...")
    vocab = GeneVocab.from_json("scgpt_mini/tokenizer/default_vocab.json")
    print(f"  Vocabulary size: {len(vocab)}")

    # Tokenize data
    print("\nTokenizing data...")
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
    print(f"  Tokenized {len(tokenized_data)} cells")

    # =========================================================================
    # PHASE 2: Model Creation
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Model Creation")
    print("=" * 80)

    print("\nCreating transformer model...")
    model = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        d_model=32,
        nhead=2,
        num_layers=2,
        d_hid=64,
        dropout=0.1,
        activation="gelu",
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model created with {n_params:,} parameters")
    print(f"  Embedding dimension: {model.d_model}")

    # =========================================================================
    # PHASE 3: MLM Pretraining
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: MLM Pretraining")
    print("=" * 80)

    # Split data for pretraining
    print("\nSplitting data for pretraining...")
    n_cells = len(tokenized_data)
    indices = np.random.permutation(n_cells)
    n_train = int(0.8 * n_cells)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_data = [tokenized_data[i] for i in train_indices]
    val_data = [tokenized_data[i] for i in val_indices]

    print(f"  Train: {len(train_data)} cells")
    print(f"  Val: {len(val_data)} cells")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(
        tokenized_data=train_data,
        vocab=vocab,
        batch_size=32,
        max_len=1001,
        shuffle=True,
        apply_masking=True,
        mask_ratio=0.15,
    )

    val_loader = create_dataloader(
        tokenized_data=val_data,
        vocab=vocab,
        batch_size=32,
        max_len=1001,
        shuffle=False,
        apply_masking=True,
        mask_ratio=0.15,
    )

    # Setup training
    print("\nSetting up training...")
    criterion = MLMLoss(value_mode="continuous")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # Train
    print("\nTraining model (10 epochs)...")
    print("  This may take 5-15 minutes depending on your hardware...")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        checkpoint_dir=checkpoint_dir,
        save_every=5,
    )

    # Plot training curves
    print("\nPlotting training curves...")
    fig = plot_training_curves(history, save_path=output_dir / "training_curves.png")
    plt.close(fig)

    print(f"  Best validation loss: {min(history['val_loss']):.4f}")
    print(f"  Final training loss: {history['train_loss'][-1]:.4f}")

    # =========================================================================
    # PHASE 4: Cell Embedding Generation
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: Cell Embedding Generation")
    print("=" * 80)

    # Create dataloader for all cells (no masking)
    print("\nPreparing data for embedding extraction...")
    full_dataloader = create_dataloader(
        tokenized_data=tokenized_data,
        vocab=vocab,
        batch_size=64,
        max_len=1001,
        shuffle=False,
        apply_masking=False,
    )

    # Extract embeddings
    print("\nExtracting cell embeddings...")
    embeddings = extract_cell_embeddings(
        model=model,
        dataloader=full_dataloader,
        device=device,
        embedding_mode="cls",
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # Evaluate embedding quality
    print("\nEvaluating embedding quality...")
    labels = adata.obs["cell_type"].values
    metrics = embedding_eval_report(
        embeddings=embeddings,
        labels=labels,
        verbose=False,
    )

    print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  Adjusted Rand Index: {metrics['ari']:.4f}")
    print(f"  k-NN Accuracy: {metrics['knn_accuracy']:.4f}")

    # Visualize embeddings
    print("\nVisualizing embeddings with UMAP...")
    fig, ax, umap_coords = plot_umap(
        embeddings=embeddings,
        labels=labels,
        title="scGPT-mini Cell Embeddings (After Pretraining)",
        save_path=output_dir / "embeddings_pretrained.png",
    )
    plt.close(fig)

    # =========================================================================
    # PHASE 5: Cell Type Annotation
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: Cell Type Annotation")
    print("=" * 80)

    # Create classification dataset
    print("\nCreating classification dataset...")
    split_indices, label_encoder = create_classification_dataset(
        adata=adata,
        label_key="cell_type",
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_state=42,
    )

    print(f"  Train: {len(split_indices['train'])} cells")
    print(f"  Val: {len(split_indices['val'])} cells")
    print(f"  Test: {len(split_indices['test'])} cells")
    print(f"  Number of classes: {label_encoder['n_classes']}")

    # Prepare labeled data
    print("\nPreparing labeled dataloaders...")
    label_ids = [label_encoder["label_to_id"][label] for label in labels]

    # Add labels to tokenized data
    labeled_tokenized_data = []
    for i, (genes, values) in enumerate(tokenized_data):
        labeled_tokenized_data.append((genes, values, label_ids[i]))

    # Split data
    train_data_labeled = [labeled_tokenized_data[i] for i in split_indices["train"]]
    val_data_labeled = [labeled_tokenized_data[i] for i in split_indices["val"]]
    test_data_labeled = [labeled_tokenized_data[i] for i in split_indices["test"]]

    # Create dataloaders
    train_loader_cls = create_dataloader(
        tokenized_data=train_data_labeled,
        vocab=vocab,
        batch_size=32,
        max_len=1001,
        shuffle=True,
        apply_masking=False,
        include_labels=True,
    )

    val_loader_cls = create_dataloader(
        tokenized_data=val_data_labeled,
        vocab=vocab,
        batch_size=32,
        max_len=1001,
        shuffle=False,
        apply_masking=False,
        include_labels=True,
    )

    test_loader_cls = create_dataloader(
        tokenized_data=test_data_labeled,
        vocab=vocab,
        batch_size=32,
        max_len=1001,
        shuffle=False,
        apply_masking=False,
        include_labels=True,
    )

    # Fine-tune for annotation
    print("\nFine-tuning for cell type annotation...")
    print("  Training classification head (encoder frozen)...")

    annotation_history = finetune_for_annotation(
        model=model,
        train_loader=train_loader_cls,
        val_loader=val_loader_cls,
        n_classes=label_encoder["n_classes"],
        num_epochs=10,
        learning_rate=5e-4,
        freeze_encoder=True,
        device=device,
        checkpoint_dir=checkpoint_dir / "annotation",
    )

    print(f"  Best validation accuracy: {max(annotation_history['val_acc']):.4f}")

    # Load best model
    print("\nLoading best annotation model...")
    best_model_path = checkpoint_dir / "annotation" / "best_annotation_model.pt"
    model = TransformerModel.load_checkpoint(best_model_path, vocab=vocab)

    # Predict on test set
    print("\nPredicting on test set...")
    predictions, probabilities = predict_cell_types(
        model=model,
        dataloader=test_loader_cls,
        label_encoder=label_encoder,
        device=device,
        return_probabilities=True,
    )

    # Evaluate
    true_labels = [label_ids[i] for i in split_indices["test"]]

    result = annotation_eval_report(
        true_labels=true_labels,
        pred_labels=predictions,
        label_encoder=label_encoder,
        save_path=output_dir / "annotation_report.txt",
    )

    print(f"\nAnnotation Performance:")
    print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {result['metrics']['balanced_accuracy']:.4f}")
    print(f"  Macro F1: {result['metrics']['macro_f1']:.4f}")

    # Visualize final embeddings
    print("\nExtracting embeddings from fine-tuned model...")
    final_embeddings = extract_cell_embeddings(
        model=model,
        dataloader=full_dataloader,
        device=device,
        embedding_mode="cls",
    )

    print("\nVisualizing final embeddings...")
    fig, ax, umap_coords = plot_umap(
        embeddings=final_embeddings,
        labels=labels,
        title="scGPT-mini Cell Embeddings (After Fine-tuning)",
        save_path=output_dir / "embeddings_finetuned.png",
    )
    plt.close(fig)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("END-TO-END WORKFLOW COMPLETE!")
    print("=" * 80)

    print("\nGenerated Files:")
    print(f"  📊 {output_dir / 'training_curves.png'}")
    print(f"  📊 {output_dir / 'embeddings_pretrained.png'}")
    print(f"  📊 {output_dir / 'embeddings_finetuned.png'}")
    print(f"  📄 {output_dir / 'annotation_report.txt'}")
    print(f"  💾 {checkpoint_dir / 'best_model.pt'}")
    print(f"  💾 {checkpoint_dir / 'annotation' / 'best_annotation_model.pt'}")

    print("\nPerformance Summary:")
    print(f"  Pretraining:")
    print(f"    - Best validation loss: {min(history['val_loss']):.4f}")
    print(f"  Embeddings:")
    print(f"    - Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"    - k-NN Accuracy: {metrics['knn_accuracy']:.4f}")
    print(f"  Annotation:")
    print(f"    - Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"    - Macro F1: {result['metrics']['macro_f1']:.4f}")

    print("\nNext Steps:")
    print("  - Explore the tutorial notebooks in notebooks/")
    print("  - Try with your own scRNA-seq datasets")
    print("  - Experiment with different hyperparameters")
    print("  - Compare with other methods (PCA, scanpy)")

    print("\n" + "=" * 80)
    print("Thank you for using scGPT-mini!")
    print("=" * 80)


if __name__ == "__main__":
    main()
