"""
Data collation for scGPT-mini.

This module provides the DataCollator class for batching and masking data
for masked language modeling training.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np

from scgpt_mini.tokenizer.vocabulary import GeneVocab
from scgpt_mini.tokenizer.gene_tokenizer import pad_sequences, create_mask


class DataCollator:
    """
    Data collator for batching and masking tokenized gene expression data.

    This collator:
    1. Takes a batch of tokenized cells (gene_ids, values)
    2. Pads sequences to a fixed length
    3. Optionally applies MLM masking
    4. Returns a dictionary suitable for model input

    Example:
        >>> vocab = GeneVocab(["GENE1", "GENE2", "GENE3"])
        >>> collator = DataCollator(vocab, max_len=10, mask_ratio=0.15)
        >>>
        >>> # Tokenized cells
        >>> batch = [
        ...     (torch.tensor([1, 3, 5]), torch.tensor([0.0, 1.5, 2.3])),
        ...     (torch.tensor([1, 4]), torch.tensor([0.0, 3.2])),
        ... ]
        >>>
        >>> collated = collator(batch)
        >>> print(collated["genes"].shape)
        torch.Size([2, 10])
    """

    def __init__(
        self,
        vocab: GeneVocab,
        max_len: int = 1001,
        mask_ratio: float = 0.15,
        mask_value: float = 0.0,
        pad_value: float = 0.0,
        cls_appended: bool = True,
        apply_masking: bool = True,
        truncate_method: str = "random",
    ):
        """
        Initialize the data collator.

        Args:
            vocab: GeneVocab instance
            max_len: Maximum sequence length (default: 1001 = 1000 genes + CLS)
            mask_ratio: Fraction of tokens to mask for MLM (default: 0.15)
            mask_value: Value to use for masked positions (default: 0.0)
            pad_value: Value to use for padding (default: 0.0)
            cls_appended: If True, sequences have CLS token prepended
            apply_masking: If True, apply MLM masking
            truncate_method: Method for truncating long sequences ("random" or "first")
        """
        self.vocab = vocab
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.pad_value = pad_value
        self.cls_appended = cls_appended
        self.apply_masking = apply_masking
        self.truncate_method = truncate_method

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
        labels: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of tokenized cells.

        Args:
            batch: List of tuples (gene_ids, expression_values)
            labels: Optional list of cell type labels for classification

        Returns:
            Dictionary containing:
            - "genes": Gene ID tensor (batch_size, max_len)
            - "values": Expression value tensor (batch_size, max_len)
            - "attention_mask": Boolean mask (batch_size, max_len)

            If apply_masking=True:
            - "masked_values": Values with masking applied
            - "mask_positions": Boolean tensor indicating masked positions
            - "target_values": Original values for masked positions

            If labels is provided:
            - "labels": Label tensor (batch_size,)
        """
        # Extract gene_ids and values from batch
        gene_ids_list = [item[0] for item in batch]
        values_list = [item[1] for item in batch]

        # Pad sequences
        padded = pad_sequences(
            gene_ids_list=gene_ids_list,
            values_list=values_list,
            max_len=self.max_len,
            vocab=self.vocab,
            pad_value=self.pad_value,
            cls_appended=self.cls_appended,
            truncate_method=self.truncate_method,
        )

        result = {
            "genes": padded["genes"],
            "values": padded["values"],
            "attention_mask": padded["attention_mask"],
        }

        # Apply MLM masking if requested
        if self.apply_masking:
            masked = create_mask(
                values=padded["values"],
                attention_mask=padded["attention_mask"],
                mask_ratio=self.mask_ratio,
                mask_value=self.mask_value,
                vocab=self.vocab,
                cls_appended=self.cls_appended,
            )

            result.update({
                "masked_values": masked["masked_values"],
                "mask_positions": masked["mask_positions"],
                "target_values": masked["target_values"],
            })

        # Add labels if provided
        if labels is not None:
            result["labels"] = torch.tensor(labels, dtype=torch.long)

        return result


class CellDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for single-cell gene expression data.

    This dataset wraps tokenized cell data for use with PyTorch DataLoader.

    Example:
        >>> import scanpy as sc
        >>> from scgpt_mini.data import preprocess_adata
        >>> from scgpt_mini.tokenizer import GeneVocab, tokenize_batch
        >>>
        >>> # Load and preprocess data
        >>> adata = sc.datasets.pbmc3k()
        >>> adata = preprocess_adata(adata, subset_hvg=1000)
        >>>
        >>> # Create vocabulary and tokenize
        >>> vocab = GeneVocab(adata.var_names.tolist())
        >>> tokenized = tokenize_batch(adata.X.toarray(), adata.var_names.values, vocab)
        >>>
        >>> # Create dataset and dataloader
        >>> dataset = CellDataset(tokenized)
        >>> collator = DataCollator(vocab, max_len=1001)
        >>> dataloader = torch.utils.data.DataLoader(
        ...     dataset, batch_size=32, collate_fn=collator
        ... )
    """

    def __init__(
        self,
        tokenized_data: List[Tuple[torch.Tensor, torch.Tensor]],
        labels: Optional[List[int]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            tokenized_data: List of (gene_ids, expression_values) tuples
            labels: Optional list of labels for classification tasks
        """
        self.tokenized_data = tokenized_data
        self.labels = labels

        if labels is not None and len(labels) != len(tokenized_data):
            raise ValueError(
                f"Number of labels ({len(labels)}) does not match "
                f"number of samples ({len(tokenized_data)})"
            )

    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> Union[Tuple, Tuple[Tuple, int]]:
        """
        Get a single cell's data.

        Args:
            idx: Index of the cell

        Returns:
            Tuple of (gene_ids, expression_values) or
            Tuple of ((gene_ids, expression_values), label) if labels provided
        """
        item = self.tokenized_data[idx]

        if self.labels is not None:
            return item, self.labels[idx]
        return item


def create_dataloader(
    tokenized_data: List[Tuple[torch.Tensor, torch.Tensor]],
    vocab: GeneVocab,
    batch_size: int = 32,
    max_len: int = 1001,
    mask_ratio: float = 0.15,
    labels: Optional[List[int]] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    apply_masking: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for training.

    This is a convenience function that creates a CellDataset and DataLoader
    with a DataCollator.

    Args:
        tokenized_data: List of (gene_ids, expression_values) tuples
        vocab: GeneVocab instance
        batch_size: Batch size for training
        max_len: Maximum sequence length
        mask_ratio: Masking ratio for MLM
        labels: Optional list of labels for classification
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading
        apply_masking: Whether to apply MLM masking

    Returns:
        PyTorch DataLoader

    Example:
        >>> dataloader = create_dataloader(
        ...     tokenized_data,
        ...     vocab,
        ...     batch_size=32,
        ...     max_len=1001,
        ...     mask_ratio=0.15,
        ... )
        >>>
        >>> for batch in dataloader:
        ...     genes = batch["genes"]
        ...     masked_values = batch["masked_values"]
        ...     # Train model...
    """
    dataset = CellDataset(tokenized_data, labels=labels)
    collator = DataCollator(
        vocab=vocab,
        max_len=max_len,
        mask_ratio=mask_ratio,
        apply_masking=apply_masking,
    )

    # Handle labels in collate function
    if labels is not None:
        def collate_with_labels(batch_items):
            batch_data = [item[0] for item in batch_items]
            batch_labels = [item[1] for item in batch_items]
            return collator(batch_data, labels=batch_labels)

        collate_fn = collate_with_labels
    else:
        collate_fn = collator

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return dataloader
