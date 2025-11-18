"""
Example: Model Creation for scGPT-mini

This script demonstrates how to:
1. Create a transformer model
2. Inspect model architecture
3. Test forward pass
4. Save and load checkpoints

This covers Phase 2 functionality.
"""

import json
import torch
from pathlib import Path

# Import scGPT-mini modules
from scgpt_mini.tokenizer import GeneVocab
from scgpt_mini.model import TransformerModel


def main():
    print("=" * 70)
    print("scGPT-mini Phase 2: Model Creation Example")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load vocabulary
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading vocabulary...")
    vocab_path = Path("scgpt_mini/tokenizer/default_vocab.json")
    vocab = GeneVocab.from_json(vocab_path)
    print(f"  Vocabulary size: {len(vocab)}")

    # -------------------------------------------------------------------------
    # Step 2: Create model with default configuration
    # -------------------------------------------------------------------------
    print("\n[Step 2] Creating model with default configuration...")

    # Load default config
    config_path = Path("scgpt_mini/model/model_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Remove notes from config
    config.pop("notes", None)
    config.pop("model_name", None)
    config.pop("description", None)

    print("\n  Model configuration:")
    for key, value in config.items():
        print(f"    {key}: {value}")

    # Create model
    model = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        **config,
    )

    print(f"\n  Model created successfully!")
    model.print_model_info()

    # -------------------------------------------------------------------------
    # Step 3: Test forward pass
    # -------------------------------------------------------------------------
    print("\n[Step 3] Testing forward pass...")

    # Create dummy batch
    batch_size = 4
    seq_len = 100

    gene_ids = torch.randint(3, len(vocab), (batch_size, seq_len))  # Skip special tokens
    gene_ids[:, 0] = vocab.cls_id  # Set first token to CLS

    values = torch.randn(batch_size, seq_len).abs()  # Positive expression values
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    print(f"  Input shapes:")
    print(f"    gene_ids: {gene_ids.shape}")
    print(f"    values: {values.shape}")
    print(f"    attention_mask: {attention_mask.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(gene_ids, values, attention_mask, return_embeddings=True)

    print(f"\n  Output keys: {list(output.keys())}")
    print(f"  Output shapes:")
    for key, value in output.items():
        print(f"    {key}: {value.shape}")

    # -------------------------------------------------------------------------
    # Step 4: Create model with classification
    # -------------------------------------------------------------------------
    print("\n[Step 4] Creating model with classification decoder...")

    model_with_cls = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        n_classes=10,  # 10 cell types
        **config,
    )

    print(f"  Model with classification created!")
    print(f"  Parameters: {model_with_cls.get_num_parameters():,}")

    # Test forward pass
    with torch.no_grad():
        output_cls = model_with_cls(gene_ids, values, attention_mask)

    print(f"\n  Output includes classification:")
    print(f"    expr_pred: {output_cls['expr_pred'].shape}")
    print(f"    cls_pred: {output_cls['cls_pred'].shape}")

    # -------------------------------------------------------------------------
    # Step 5: Create model with binned values
    # -------------------------------------------------------------------------
    print("\n[Step 5] Creating model with binned expression values...")

    model_binned = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        value_mode="binned",
        n_bins=51,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        d_hid=config["d_hid"],
        dropout=config["dropout"],
    )

    print(f"  Binned model created!")

    # Test with binned values
    binned_values = torch.randint(0, 51, (batch_size, seq_len))

    with torch.no_grad():
        output_binned = model_binned(gene_ids, binned_values, attention_mask)

    print(f"\n  Output for binned values:")
    print(f"    expr_pred (logits over bins): {output_binned['expr_pred'].shape}")

    # -------------------------------------------------------------------------
    # Step 6: Save and load checkpoint
    # -------------------------------------------------------------------------
    print("\n[Step 6] Testing checkpoint save/load...")

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / "model_test.pt"

    # Save checkpoint
    model.save_checkpoint(checkpoint_path)

    # Load checkpoint
    loaded_model = TransformerModel.load_checkpoint(checkpoint_path, vocab=vocab)

    # Verify loaded model
    print(f"\n  Loaded model parameters: {loaded_model.get_num_parameters():,}")
    print(f"  Original model parameters: {model.get_num_parameters():,}")
    print(f"  Parameters match: {loaded_model.get_num_parameters() == model.get_num_parameters()}")

    # Test that outputs match
    with torch.no_grad():
        output_original = model(gene_ids, values, attention_mask)
        output_loaded = loaded_model(gene_ids, values, attention_mask)

    max_diff = (output_original["expr_pred"] - output_loaded["expr_pred"]).abs().max()
    print(f"  Max difference in predictions: {max_diff.item():.10f}")

    # -------------------------------------------------------------------------
    # Step 7: Model variations
    # -------------------------------------------------------------------------
    print("\n[Step 7] Creating model variations...")

    # Tiny model for very fast prototyping
    tiny_model = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        d_model=16,
        nhead=2,
        num_layers=1,
        d_hid=32,
    )
    print(f"\n  Tiny model: {tiny_model.get_num_parameters():,} parameters")

    # Standard model (default config)
    standard_model = model
    print(f"  Standard model: {standard_model.get_num_parameters():,} parameters")

    # Larger model (still laptop-friendly)
    large_model = TransformerModel(
        vocab_size=len(vocab),
        vocab=vocab,
        d_model=64,
        nhead=4,
        num_layers=3,
        d_hid=128,
    )
    print(f"  Large model: {large_model.get_num_parameters():,} parameters")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 2 model creation complete!")
    print("=" * 70)
    print("\nYou now have:")
    print(f"  ✓ Transformer model with {model.get_num_parameters():,} parameters")
    print(f"  ✓ Gene and value encoders")
    print(f"  ✓ Expression decoder")
    print(f"  ✓ Optional classification decoder")
    print(f"  ✓ Checkpoint save/load functionality")
    print(f"  ✓ Support for continuous and binned values")
    print("\nModel variations:")
    print(f"  - Tiny (prototyping): {tiny_model.get_num_parameters():,} params")
    print(f"  - Standard (default): {standard_model.get_num_parameters():,} params")
    print(f"  - Large (more capacity): {large_model.get_num_parameters():,} params")
    print("\nNext steps:")
    print("  - Phase 3: Implement training infrastructure")
    print("  - Train the model with masked language modeling")
    print("  - Fine-tune for downstream tasks")
    print("=" * 70)


if __name__ == "__main__":
    main()
