"""
Example: Training scGPT-mini

This script demonstrates how to:
1. Load and preprocess data
2. Create train/val dataloaders
3. Set up the model and training components
4. Train the model with MLM
5. Evaluate and save the trained model

This covers Phase 3 functionality.
"""

import json
import torch
import torch.optim as optim
import scanpy as sc
from pathlib import Path

# Import scGPT-mini modules
from scgpt_mini.tokenizer import GeneVocab, tokenize_batch
from scgpt_mini.data import preprocess_adata, create_dataloader
from scgpt_mini.model import TransformerModel
from scgpt_mini.training import MLMLoss, Trainer


def main():
    print("=" * 70)
    print("scGPT-mini Phase 3: Training Example")
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
        subset_hvg=500,  # Use 500 HVGs for faster training
        binning=False,  # Use continuous values
        inplace=False,
    )

    print(f"  Preprocessed: {adata.n_obs} cells x {adata.n_vars} genes")

    # -------------------------------------------------------------------------
    # Step 2: Create vocabulary and tokenize
    # -------------------------------------------------------------------------
    print("\n[Step 2] Creating vocabulary and tokenizing...")

    # Create vocabulary from current genes
    gene_list = adata.var_names.tolist()
    vocab = GeneVocab(gene_list)
    print(f"  Vocabulary size: {len(vocab)}")

    # Tokenize all cells
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
    # Step 3: Split data into train/val
    # -------------------------------------------------------------------------
    print("\n[Step 3] Splitting data into train/val...")

    n_cells = len(tokenized_data)
    n_train = int(0.8 * n_cells)

    train_data = tokenized_data[:n_train]
    val_data = tokenized_data[n_train:]

    print(f"  Train: {len(train_data)} cells")
    print(f"  Val: {len(val_data)} cells")

    # -------------------------------------------------------------------------
    # Step 4: Create dataloaders
    # -------------------------------------------------------------------------
    print("\n[Step 4] Creating dataloaders...")

    train_loader = create_dataloader(
        tokenized_data=train_data,
        vocab=vocab,
        batch_size=32,
        max_len=1001,
        mask_ratio=0.15,
        shuffle=True,
        apply_masking=True,
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
    # Step 5: Create model
    # -------------------------------------------------------------------------
    print("\n[Step 5] Creating model...")

    # Load model config
    model_config_path = Path("scgpt_mini/model/model_config.json")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    # Remove non-parameter keys
    model_config.pop("notes", None)
    model_config.pop("model_name", None)
    model_config.pop("description", None)

    # Create model
    model = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        **model_config,
    )

    model.print_model_info()

    # -------------------------------------------------------------------------
    # Step 6: Set up training components
    # -------------------------------------------------------------------------
    print("\n[Step 6] Setting up training components...")

    # Load training config
    train_config_path = Path("scgpt_mini/training/training_config.json")
    with open(train_config_path, "r") as f:
        train_config = json.load(f)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config["optimizer"]["lr"],
        weight_decay=train_config["optimizer"]["weight_decay"],
        betas=train_config["optimizer"]["betas"],
    )

    print(f"  Optimizer: AdamW (lr={train_config['optimizer']['lr']})")

    # Loss function
    criterion = MLMLoss(
        value_mode=train_config["data"]["value_mode"],
        use_zero_prob=train_config["loss"]["use_zero_prob"],
        zero_prob_weight=train_config["loss"]["zero_prob_weight"],
    )

    print(f"  Loss: MLMLoss (value_mode={train_config['data']['value_mode']})")

    # Learning rate scheduler (optional)
    scheduler = None  # Can add scheduler here if needed

    # Trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        gradient_clip=train_config["training"]["gradient_clip"],
        log_interval=train_config["training"]["log_interval"],
        checkpoint_dir="checkpoints",
        value_mode=train_config["data"]["value_mode"],
    )

    # -------------------------------------------------------------------------
    # Step 7: Train the model
    # -------------------------------------------------------------------------
    print("\n[Step 7] Training the model...")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=train_config["training"]["num_epochs"],
        save_every=train_config["training"]["save_every"],
        eval_every=train_config["training"]["eval_every"],
        early_stopping_patience=train_config["training"].get("early_stopping_patience"),
    )

    # -------------------------------------------------------------------------
    # Step 8: Save training history
    # -------------------------------------------------------------------------
    print("\n[Step 8] Saving training history...")

    trainer.save_training_history("training_history.json")

    # -------------------------------------------------------------------------
    # Step 9: Load best model and evaluate
    # -------------------------------------------------------------------------
    print("\n[Step 9] Loading best model and final evaluation...")

    best_model_path = Path("checkpoints/best_model.pt")
    if best_model_path.exists():
        trainer.load_checkpoint(best_model_path, load_optimizer=False)

        from scgpt_mini.training import evaluate_model

        final_metrics = evaluate_model(
            model=trainer.model,
            dataloader=val_loader,
            device=device,
            value_mode=train_config["data"]["value_mode"],
        )

        from scgpt_mini.training import print_metrics
        print_metrics(final_metrics, prefix="Final Validation")

    # -------------------------------------------------------------------------
    # Step 10: Save the trained model for downstream tasks
    # -------------------------------------------------------------------------
    print("\n[Step 10] Saving final model...")

    final_model_path = Path("checkpoints/pretrained_model.pt")
    model.save_checkpoint(final_model_path)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 3 training complete!")
    print("=" * 70)
    print("\nTraining summary:")
    print(f"  Epochs completed: {len(history['train_loss'])}")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"  Best val loss: {trainer.best_val_loss:.4f}")

    print("\nSaved files:")
    print(f"  Best model: checkpoints/best_model.pt")
    print(f"  Final model: checkpoints/pretrained_model.pt")
    print(f"  Training history: checkpoints/training_history.json")

    print("\nNext steps:")
    print("  - Phase 4: Extract cell embeddings for visualization")
    print("  - Phase 5: Fine-tune for cell type annotation")
    print("  - Experiment with different hyperparameters")
    print("=" * 70)


if __name__ == "__main__":
    main()
