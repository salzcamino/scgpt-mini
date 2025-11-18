"""
Example: Data Preprocessing for scGPT-mini

This script demonstrates how to:
1. Load single-cell RNA-seq data
2. Preprocess the data (filtering, normalization, HVG selection)
3. Create a gene vocabulary
4. Tokenize the data
5. Create a DataLoader for training

This covers all Phase 1 functionality.
"""

import scanpy as sc
import torch
from pathlib import Path

# Import scGPT-mini modules
from scgpt_mini.data import preprocess_adata
from scgpt_mini.tokenizer import GeneVocab, tokenize_batch
from scgpt_mini.data.collator import create_dataloader


def main():
    print("=" * 70)
    print("scGPT-mini Phase 1: Data Preprocessing Example")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load example dataset
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading PBMC 3k dataset...")
    adata = sc.datasets.pbmc3k()
    print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    # -------------------------------------------------------------------------
    # Step 2: Preprocess the data
    # -------------------------------------------------------------------------
    print("\n[Step 2] Preprocessing data...")
    adata = preprocess_adata(
        adata,
        filter_gene_by_counts=10,      # Min 10 total counts per gene
        filter_gene_by_cells=3,         # Expressed in at least 3 cells
        filter_cell_by_genes=200,       # At least 200 genes per cell
        normalize_total_target=1e4,     # Normalize to 10,000 counts
        log1p=True,                     # Log transformation
        subset_hvg=1000,                # Select 1000 HVGs
        hvg_flavor="seurat_v3",
        binning=51,                     # Bin into 51 levels
        inplace=False,
    )

    print(f"\n  Final dataset: {adata.n_obs} cells x {adata.n_vars} genes")

    # -------------------------------------------------------------------------
    # Step 3: Create gene vocabulary
    # -------------------------------------------------------------------------
    print("\n[Step 3] Creating gene vocabulary...")

    # Option 1: Use default vocabulary (recommended for transfer learning)
    vocab_path = Path("scgpt_mini/tokenizer/default_vocab.json")
    if vocab_path.exists():
        print(f"  Loading default vocabulary from {vocab_path}")
        vocab = GeneVocab.from_json(vocab_path)
        print(f"  Vocabulary size: {len(vocab)} genes")
    else:
        # Option 2: Create vocabulary from current genes
        print("  Creating vocabulary from current genes")
        gene_list = adata.var_names.tolist()
        vocab = GeneVocab(gene_list)
        print(f"  Vocabulary size: {len(vocab)} genes")

    # Show some vocabulary info
    print(f"  Special tokens: {vocab.special_tokens}")
    print(f"  Sample genes: {list(adata.var_names[:5])}")
    print(f"  Sample gene IDs: {[vocab[g] for g in adata.var_names[:5]]}")

    # -------------------------------------------------------------------------
    # Step 4: Tokenize the data
    # -------------------------------------------------------------------------
    print("\n[Step 4] Tokenizing cells...")

    # Get normalized and log-transformed data
    data_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    gene_names = adata.var_names.values

    # Tokenize all cells
    tokenized_data = tokenize_batch(
        data=data_matrix,
        gene_names=gene_names,
        vocab=vocab,
        append_cls=True,
        include_zero_genes=False,  # Only include non-zero genes
        return_pt=True,
    )

    print(f"  Tokenized {len(tokenized_data)} cells")

    # Show an example tokenized cell
    gene_ids, values = tokenized_data[0]
    print(f"\n  Example cell:")
    print(f"    Number of tokens: {len(gene_ids)}")
    print(f"    First 5 gene IDs: {gene_ids[:5].tolist()}")
    print(f"    First 5 values: {values[:5].tolist()}")

    # -------------------------------------------------------------------------
    # Step 5: Create DataLoader
    # -------------------------------------------------------------------------
    print("\n[Step 5] Creating DataLoader...")

    # Split data into train/val (simple 90/10 split for demo)
    n_cells = len(tokenized_data)
    n_train = int(0.9 * n_cells)

    train_data = tokenized_data[:n_train]
    val_data = tokenized_data[n_train:]

    print(f"  Train set: {len(train_data)} cells")
    print(f"  Val set: {len(val_data)} cells")

    # Create DataLoaders
    train_loader = create_dataloader(
        tokenized_data=train_data,
        vocab=vocab,
        batch_size=32,
        max_len=1001,  # 1000 genes + CLS token
        mask_ratio=0.15,  # Mask 15% of tokens for MLM
        shuffle=True,
        apply_masking=True,  # Apply MLM masking
    )

    val_loader = create_dataloader(
        tokenized_data=val_data,
        vocab=vocab,
        batch_size=32,
        max_len=1001,
        mask_ratio=0.15,
        shuffle=False,
        apply_masking=True,
    )

    print(f"  Train loader: {len(train_loader)} batches")
    print(f"  Val loader: {len(val_loader)} batches")

    # -------------------------------------------------------------------------
    # Step 6: Inspect a batch
    # -------------------------------------------------------------------------
    print("\n[Step 6] Inspecting a sample batch...")

    batch = next(iter(train_loader))

    print(f"\n  Batch keys: {list(batch.keys())}")
    print(f"  Batch shapes:")
    for key, value in batch.items():
        print(f"    {key}: {value.shape}")

    print(f"\n  Batch details:")
    print(f"    genes: {batch['genes'].dtype}")
    print(f"    values: {batch['values'].dtype}")
    print(f"    masked_values: {batch['masked_values'].dtype}")
    print(f"    attention_mask: {batch['attention_mask'].dtype}")
    print(f"    mask_positions: {batch['mask_positions'].dtype}")

    # Count masked positions
    n_masked = batch["mask_positions"].sum().item()
    n_total = batch["attention_mask"].sum().item()
    mask_ratio = n_masked / n_total if n_total > 0 else 0
    print(f"\n  Masking stats:")
    print(f"    Total tokens: {n_total}")
    print(f"    Masked tokens: {n_masked}")
    print(f"    Actual mask ratio: {mask_ratio:.3f}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 1 preprocessing complete!")
    print("=" * 70)
    print("\nYou now have:")
    print(f"  ✓ Preprocessed data: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  ✓ Gene vocabulary: {len(vocab)} genes")
    print(f"  ✓ Tokenized data: {len(tokenized_data)} cells")
    print(f"  ✓ Train DataLoader: {len(train_loader)} batches of size 32")
    print(f"  ✓ Val DataLoader: {len(val_loader)} batches of size 32")
    print("\nNext steps:")
    print("  - Phase 2: Implement the transformer model")
    print("  - Phase 3: Train the model with masked language modeling")
    print("=" * 70)


if __name__ == "__main__":
    main()
