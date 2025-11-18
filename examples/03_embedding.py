"""
Example: Cell Embedding Extraction

This script demonstrates how to:
1. Load a pretrained model
2. Extract cell embeddings
3. Visualize embeddings with UMAP
4. Evaluate embedding quality
5. Save embeddings to AnnData

This covers Phase 4 functionality.
"""

import torch
import scanpy as sc
from pathlib import Path

# Import scGPT-mini modules
from scgpt_mini.tokenizer import GeneVocab, tokenize_batch
from scgpt_mini.data import preprocess_adata, create_dataloader
from scgpt_mini.model import TransformerModel
from scgpt_mini.tasks import (
    extract_cell_embeddings,
    save_embeddings_to_adata,
    embedding_eval_report,
)
from scgpt_mini.utils import plot_umap, plot_tsne


def main():
    print("=" * 70)
    print("scGPT-mini Phase 4: Cell Embedding Extraction Example")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load and preprocess data
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading and preprocessing PBMC 3k dataset...")
    adata = sc.datasets.pbmc3k()
    print(f"  Original: {adata.n_obs} cells x {adata.n_vars} genes")

    # Preprocess
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
    # Step 2: Load pretrained model
    # -------------------------------------------------------------------------
    print("\n[Step 2] Loading pretrained model...")

    model_path = Path("checkpoints/pretrained_model.pt")
    if not model_path.exists():
        print(f"  ERROR: Pretrained model not found at {model_path}")
        print("  Please run 02_training.py first to create a pretrained model.")
        return

    # Load vocabulary
    vocab = GeneVocab.from_json("scgpt_mini/tokenizer/default_vocab.json")

    # Load model
    model = TransformerModel.load_checkpoint(model_path, vocab=vocab)
    model.eval()

    print(f"  Model loaded successfully")
    print(f"  Model embedding dimension: {model.d_model}")

    # -------------------------------------------------------------------------
    # Step 3: Tokenize data
    # -------------------------------------------------------------------------
    print("\n[Step 3] Tokenizing data...")

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

    # -------------------------------------------------------------------------
    # Step 4: Create dataloader
    # -------------------------------------------------------------------------
    print("\n[Step 4] Creating dataloader...")

    dataloader = create_dataloader(
        tokenized_data=tokenized_data,
        vocab=vocab,
        batch_size=64,
        max_len=1001,
        shuffle=False,
        apply_masking=False,  # No masking for embedding extraction
    )

    print(f"  Dataloader created with {len(dataloader)} batches")

    # -------------------------------------------------------------------------
    # Step 5: Extract cell embeddings
    # -------------------------------------------------------------------------
    print("\n[Step 5] Extracting cell embeddings...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Extract with different modes
    print("\n  Extracting CLS token embeddings...")
    embeddings_cls = extract_cell_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        embedding_mode="cls",
    )

    print(f"  Embeddings shape: {embeddings_cls.shape}")

    # -------------------------------------------------------------------------
    # Step 6: Save embeddings to AnnData
    # -------------------------------------------------------------------------
    print("\n[Step 6] Saving embeddings to AnnData...")

    save_embeddings_to_adata(
        adata=adata,
        embeddings=embeddings_cls,
        key="X_scgpt",
    )

    print(f"  Embeddings saved to adata.obsm['X_scgpt']")

    # -------------------------------------------------------------------------
    # Step 7: Visualize with UMAP
    # -------------------------------------------------------------------------
    print("\n[Step 7] Visualizing embeddings with UMAP...")

    # Get cell type labels if available
    labels = None
    if "cell_type" in adata.obs.columns:
        labels = adata.obs["cell_type"].values
        print(f"  Found cell type labels: {len(set(labels))} unique types")
    elif "louvain" in adata.obs.columns:
        labels = adata.obs["louvain"].values
        print(f"  Using Louvain clusters as labels")
    else:
        # Compute clusters
        print("  No labels found, computing Louvain clusters...")
        sc.pp.neighbors(adata, use_rep="X_scgpt")
        sc.tl.louvain(adata)
        labels = adata.obs["louvain"].values

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Plot UMAP
    fig, ax, umap_coords = plot_umap(
        embeddings=embeddings_cls,
        labels=labels,
        title="scGPT-mini Cell Embeddings (UMAP)",
        save_path=output_dir / "embeddings_umap.png",
        n_neighbors=15,
        min_dist=0.1,
    )

    print(f"  UMAP plot saved to {output_dir / 'embeddings_umap.png'}")

    # -------------------------------------------------------------------------
    # Step 8: Visualize with t-SNE (optional)
    # -------------------------------------------------------------------------
    print("\n[Step 8] Visualizing embeddings with t-SNE...")

    fig, ax, tsne_coords = plot_tsne(
        embeddings=embeddings_cls,
        labels=labels,
        title="scGPT-mini Cell Embeddings (t-SNE)",
        save_path=output_dir / "embeddings_tsne.png",
        perplexity=30,
    )

    print(f"  t-SNE plot saved to {output_dir / 'embeddings_tsne.png'}")

    # -------------------------------------------------------------------------
    # Step 9: Evaluate embedding quality
    # -------------------------------------------------------------------------
    print("\n[Step 9] Evaluating embedding quality...")

    # Compute metrics
    metrics = embedding_eval_report(
        embeddings=embeddings_cls,
        labels=labels,
        n_neighbors=15,
    )

    print(f"\n  Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  Adjusted Rand Index: {metrics['ari']:.4f}")
    print(f"  k-NN Accuracy: {metrics['knn_accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # Step 10: Compare embedding modes
    # -------------------------------------------------------------------------
    print("\n[Step 10] Comparing different embedding modes...")

    # Mean pooling
    print("\n  Extracting mean pooling embeddings...")
    embeddings_mean = extract_cell_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        embedding_mode="mean",
    )

    metrics_mean = embedding_eval_report(
        embeddings=embeddings_mean,
        labels=labels,
        n_neighbors=15,
        verbose=False,
    )

    # Max pooling
    print("  Extracting max pooling embeddings...")
    embeddings_max = extract_cell_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        embedding_mode="max",
    )

    metrics_max = embedding_eval_report(
        embeddings=embeddings_max,
        labels=labels,
        n_neighbors=15,
        verbose=False,
    )

    # Compare
    print("\n  Embedding Mode Comparison:")
    print(f"    {'Mode':<12} {'Silhouette':<12} {'ARI':<12} {'k-NN Acc':<12}")
    print(f"    {'-'*48}")
    print(f"    {'CLS':<12} {metrics['silhouette_score']:<12.4f} "
          f"{metrics['ari']:<12.4f} {metrics['knn_accuracy']:<12.4f}")
    print(f"    {'Mean':<12} {metrics_mean['silhouette_score']:<12.4f} "
          f"{metrics_mean['ari']:<12.4f} {metrics_mean['knn_accuracy']:<12.4f}")
    print(f"    {'Max':<12} {metrics_max['silhouette_score']:<12.4f} "
          f"{metrics_max['ari']:<12.4f} {metrics_max['knn_accuracy']:<12.4f}")

    # -------------------------------------------------------------------------
    # Step 11: Save embeddings for downstream analysis
    # -------------------------------------------------------------------------
    print("\n[Step 11] Saving AnnData with embeddings...")

    # Save AnnData
    adata.write(output_dir / "adata_with_embeddings.h5ad")

    print(f"  AnnData saved to {output_dir / 'adata_with_embeddings.h5ad'}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 4 embedding extraction complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - {output_dir / 'embeddings_umap.png'}")
    print(f"  - {output_dir / 'embeddings_tsne.png'}")
    print(f"  - {output_dir / 'adata_with_embeddings.h5ad'}")

    print("\nEmbedding quality metrics:")
    print(f"  - Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  - Adjusted Rand Index: {metrics['ari']:.4f}")
    print(f"  - k-NN Accuracy: {metrics['knn_accuracy']:.4f}")

    print("\nNext steps:")
    print("  - Phase 5: Fine-tune for cell type annotation")
    print("  - Use embeddings for downstream analysis")
    print("  - Compare with PCA or other dimensionality reduction methods")
    print("=" * 70)


if __name__ == "__main__":
    main()
