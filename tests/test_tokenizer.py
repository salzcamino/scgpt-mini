"""
Unit tests for the tokenizer module.
"""

import json
import pytest
import numpy as np
import torch
from pathlib import Path

from scgpt_mini.tokenizer import (
    GeneVocab,
    create_vocab_from_genes,
    tokenize_genes,
    tokenize_cell,
    pad_sequences,
    create_mask,
)


class TestGeneVocab:
    """Tests for GeneVocab class."""

    def test_creation(self):
        """Test basic vocabulary creation."""
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        # Check size (3 genes + 3 special tokens)
        assert len(vocab) == 6

        # Check special tokens
        assert vocab.pad_token == "<pad>"
        assert vocab.cls_token == "<cls>"
        assert vocab.eoc_token == "<eoc>"
        assert vocab.pad_id == 0
        assert vocab.cls_id == 1
        assert vocab.eoc_id == 2

        # Check gene IDs
        assert vocab["GENE1"] == 3
        assert vocab["GENE2"] == 4
        assert vocab["GENE3"] == 5

    def test_get_gene(self):
        """Test reverse lookup (ID -> gene name)."""
        genes = ["GENE1", "GENE2"]
        vocab = GeneVocab(genes)

        assert vocab.get_gene(0) == "<pad>"
        assert vocab.get_gene(1) == "<cls>"
        assert vocab.get_gene(3) == "GENE1"

    def test_contains(self):
        """Test membership checking."""
        genes = ["GENE1", "GENE2"]
        vocab = GeneVocab(genes)

        assert "GENE1" in vocab
        assert "GENE2" in vocab
        assert "GENE3" not in vocab
        assert "<pad>" in vocab

    def test_unknown_gene(self):
        """Test handling of unknown genes."""
        genes = ["GENE1"]
        vocab = GeneVocab(genes)

        # Unknown gene should return default index (pad_id)
        assert vocab["UNKNOWN"] == vocab.pad_id

    def test_save_and_load(self, tmp_path):
        """Test saving and loading vocabulary."""
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        # Save
        file_path = tmp_path / "test_vocab.json"
        vocab.save_json(file_path)

        # Load
        loaded_vocab = GeneVocab.from_json(file_path)

        assert len(loaded_vocab) == len(vocab)
        assert loaded_vocab["GENE1"] == vocab["GENE1"]
        assert loaded_vocab["GENE2"] == vocab["GENE2"]


class TestTokenization:
    """Tests for tokenization functions."""

    def test_tokenize_genes(self):
        """Test gene name to ID conversion."""
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        gene_ids = tokenize_genes(["GENE1", "GENE3"], vocab)

        assert isinstance(gene_ids, torch.Tensor)
        assert gene_ids.tolist() == [3, 5]

    def test_tokenize_cell(self):
        """Test cell tokenization."""
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        gene_names = np.array(["GENE1", "GENE2", "GENE3"])
        values = np.array([1.5, 0.0, 2.3])

        gene_ids, expr_values = tokenize_cell(
            gene_names, values, vocab, append_cls=True, include_zero_genes=False
        )

        # Should have CLS + 2 non-zero genes
        assert len(gene_ids) == 3
        assert gene_ids[0] == vocab.cls_id  # CLS token
        assert gene_ids[1] == 3  # GENE1
        assert gene_ids[2] == 5  # GENE3

        assert expr_values[0] == 0.0  # CLS value
        assert expr_values[1] == 1.5  # GENE1 value
        assert expr_values[2] == 2.3  # GENE3 value

    def test_pad_sequences(self):
        """Test sequence padding."""
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        genes1 = torch.tensor([1, 3, 5])  # CLS + 2 genes
        values1 = torch.tensor([0.0, 1.5, 2.3])
        genes2 = torch.tensor([1, 4])  # CLS + 1 gene
        values2 = torch.tensor([0.0, 3.2])

        padded = pad_sequences(
            [genes1, genes2], [values1, values2], max_len=5, vocab=vocab
        )

        assert padded["genes"].shape == (2, 5)
        assert padded["values"].shape == (2, 5)
        assert padded["attention_mask"].shape == (2, 5)

        # Check first sequence (no padding needed)
        assert padded["genes"][0, 0] == 1  # CLS
        assert padded["attention_mask"][0, :3].all()

        # Check second sequence (needs padding)
        assert padded["genes"][1, 0] == 1  # CLS
        assert padded["genes"][1, 2] == vocab.pad_id  # Padded position
        assert padded["attention_mask"][1, :2].all()
        assert not padded["attention_mask"][1, 2:].any()

    def test_create_mask(self):
        """Test MLM masking."""
        values = torch.tensor([[0.0, 1.5, 2.3, 3.1], [0.0, 0.5, 1.0, 0.0]])
        mask = torch.tensor([[True, True, True, True], [True, True, True, False]])

        masked = create_mask(values, mask, mask_ratio=0.5, cls_appended=True)

        assert "masked_values" in masked
        assert "mask_positions" in masked
        assert "target_values" in masked

        # CLS token should not be masked
        assert not masked["mask_positions"][:, 0].any()

        # Some non-CLS, non-padding positions should be masked
        assert masked["mask_positions"][:, 1:].any()

        # Masked positions should have mask_value
        assert (masked["masked_values"][masked["mask_positions"]] == 0.0).all()


class TestDefaultVocab:
    """Tests for default vocabulary."""

    def test_default_vocab_exists(self):
        """Test that default vocabulary file exists."""
        vocab_path = Path("scgpt_mini/tokenizer/default_vocab.json")
        assert vocab_path.exists()

    def test_load_default_vocab(self):
        """Test loading default vocabulary."""
        vocab_path = Path("scgpt_mini/tokenizer/default_vocab.json")
        vocab = GeneVocab.from_json(vocab_path)

        # Should have ~8000 genes + 3 special tokens
        assert len(vocab) > 8000
        assert vocab.pad_token in vocab
        assert vocab.cls_token in vocab


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
