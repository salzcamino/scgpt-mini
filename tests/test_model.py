"""
Unit tests for the model module.
"""

import pytest
import torch
from pathlib import Path

from scgpt_mini.tokenizer import GeneVocab
from scgpt_mini.model import (
    GeneEncoder,
    ContinuousValueEncoder,
    CategoryValueEncoder,
    PositionalEncoding,
    ExpressionDecoder,
    ClassificationDecoder,
    BinnedExpressionDecoder,
    TransformerModel,
)


class TestEncoders:
    """Tests for encoder modules."""

    def test_gene_encoder(self):
        """Test GeneEncoder."""
        vocab_size = 1000
        d_model = 32
        encoder = GeneEncoder(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0)

        batch_size, seq_len = 4, 100
        gene_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = encoder(gene_ids)

        assert output.shape == (batch_size, seq_len, d_model)
        assert output.dtype == torch.float32

    def test_continuous_value_encoder(self):
        """Test ContinuousValueEncoder."""
        d_model = 32
        encoder = ContinuousValueEncoder(d_model=d_model, dropout=0.1)

        batch_size, seq_len = 4, 100
        values = torch.randn(batch_size, seq_len)

        output = encoder(values)

        assert output.shape == (batch_size, seq_len, d_model)
        assert output.dtype == torch.float32

    def test_category_value_encoder(self):
        """Test CategoryValueEncoder."""
        n_bins = 51
        d_model = 32
        encoder = CategoryValueEncoder(num_embeddings=n_bins, embedding_dim=d_model, padding_idx=0)

        batch_size, seq_len = 4, 100
        binned_values = torch.randint(0, n_bins, (batch_size, seq_len))

        output = encoder(binned_values)

        assert output.shape == (batch_size, seq_len, d_model)
        assert output.dtype == torch.float32

    def test_positional_encoding(self):
        """Test PositionalEncoding."""
        d_model = 32
        pos_enc = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=1000)

        batch_size, seq_len = 4, 100
        embeddings = torch.randn(batch_size, seq_len, d_model)

        output = pos_enc(embeddings)

        assert output.shape == (batch_size, seq_len, d_model)
        assert output.dtype == torch.float32


class TestDecoders:
    """Tests for decoder modules."""

    def test_expression_decoder(self):
        """Test ExpressionDecoder."""
        d_model = 32
        decoder = ExpressionDecoder(d_model=d_model)

        batch_size, seq_len = 4, 100
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        output = decoder(hidden_states)

        assert "pred" in output
        assert output["pred"].shape == (batch_size, seq_len)

    def test_expression_decoder_with_zero_prob(self):
        """Test ExpressionDecoder with explicit zero probability."""
        d_model = 32
        decoder = ExpressionDecoder(d_model=d_model, explicit_zero_prob=True)

        batch_size, seq_len = 4, 100
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        output = decoder(hidden_states)

        assert "pred" in output
        assert "zero_probs" in output
        assert output["pred"].shape == (batch_size, seq_len)
        assert output["zero_probs"].shape == (batch_size, seq_len)

    def test_classification_decoder(self):
        """Test ClassificationDecoder."""
        d_model = 32
        n_classes = 10
        decoder = ClassificationDecoder(d_model=d_model, n_classes=n_classes)

        batch_size = 4
        cls_embedding = torch.randn(batch_size, d_model)

        output = decoder(cls_embedding)

        assert output.shape == (batch_size, n_classes)

    def test_binned_expression_decoder(self):
        """Test BinnedExpressionDecoder."""
        d_model = 32
        n_bins = 51
        decoder = BinnedExpressionDecoder(d_model=d_model, n_bins=n_bins)

        batch_size, seq_len = 4, 100
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        output = decoder(hidden_states)

        assert output.shape == (batch_size, seq_len, n_bins)


class TestTransformerModel:
    """Tests for TransformerModel."""

    def test_model_creation(self):
        """Test basic model creation."""
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        model = TransformerModel(
            vocab_size=len(vocab),
            d_model=32,
            nhead=2,
            num_layers=2,
            d_hid=64,
            vocab=vocab,
        )

        assert model.d_model == 32
        assert model.value_mode == "continuous"

    def test_forward_pass_continuous(self):
        """Test forward pass with continuous values."""
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        model = TransformerModel(
            vocab_size=len(vocab),
            d_model=32,
            nhead=2,
            num_layers=2,
            d_hid=64,
            value_mode="continuous",
            vocab=vocab,
        )

        batch_size, seq_len = 4, 10
        gene_ids = torch.randint(0, len(vocab), (batch_size, seq_len))
        values = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = model(gene_ids, values, attention_mask)

        assert "expr_pred" in output
        assert output["expr_pred"].shape == (batch_size, seq_len)

    def test_forward_pass_binned(self):
        """Test forward pass with binned values."""
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        model = TransformerModel(
            vocab_size=len(vocab),
            d_model=32,
            nhead=2,
            num_layers=2,
            d_hid=64,
            value_mode="binned",
            n_bins=51,
            vocab=vocab,
        )

        batch_size, seq_len = 4, 10
        gene_ids = torch.randint(0, len(vocab), (batch_size, seq_len))
        binned_values = torch.randint(0, 51, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = model(gene_ids, binned_values, attention_mask)

        assert "expr_pred" in output
        assert output["expr_pred"].shape == (batch_size, seq_len, 51)

    def test_forward_pass_with_classification(self):
        """Test forward pass with classification."""
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        model = TransformerModel(
            vocab_size=len(vocab),
            d_model=32,
            nhead=2,
            num_layers=2,
            d_hid=64,
            n_classes=10,
            vocab=vocab,
        )

        batch_size, seq_len = 4, 10
        gene_ids = torch.randint(0, len(vocab), (batch_size, seq_len))
        values = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = model(gene_ids, values, attention_mask)

        assert "expr_pred" in output
        assert "cls_pred" in output
        assert output["cls_pred"].shape == (batch_size, 10)

    def test_return_embeddings(self):
        """Test returning embeddings."""
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        model = TransformerModel(
            vocab_size=len(vocab),
            d_model=32,
            nhead=2,
            num_layers=2,
            d_hid=64,
            n_classes=10,
            vocab=vocab,
        )

        batch_size, seq_len = 4, 10
        gene_ids = torch.randint(0, len(vocab), (batch_size, seq_len))
        values = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = model(gene_ids, values, attention_mask, return_embeddings=True)

        assert "embeddings" in output
        assert "cls_embedding" in output
        assert output["embeddings"].shape == (batch_size, seq_len, 32)
        assert output["cls_embedding"].shape == (batch_size, 32)

    def test_parameter_count(self):
        """Test parameter counting."""
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        model = TransformerModel(
            vocab_size=len(vocab),
            d_model=32,
            nhead=2,
            num_layers=2,
            d_hid=64,
            vocab=vocab,
        )

        n_params = model.get_num_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoints."""
        genes = ["GENE1", "GENE2", "GENE3"]
        vocab = GeneVocab(genes)

        model = TransformerModel(
            vocab_size=len(vocab),
            d_model=32,
            nhead=2,
            num_layers=2,
            d_hid=64,
            vocab=vocab,
        )

        # Save checkpoint
        checkpoint_path = tmp_path / "model_checkpoint.pt"
        model.save_checkpoint(checkpoint_path)

        # Load checkpoint
        loaded_model = TransformerModel.load_checkpoint(checkpoint_path, vocab=vocab)

        assert loaded_model.d_model == model.d_model
        assert loaded_model.get_num_parameters() == model.get_num_parameters()

        # Test that loaded model produces same output
        batch_size, seq_len = 2, 5
        gene_ids = torch.randint(0, len(vocab), (batch_size, seq_len))
        values = torch.randn(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        with torch.no_grad():
            output1 = model(gene_ids, values, attention_mask)
            output2 = loaded_model(gene_ids, values, attention_mask)

        torch.testing.assert_close(output1["expr_pred"], output2["expr_pred"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
