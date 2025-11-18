"""
Gene tokenization functions for scGPT-mini.

This module provides functions to:
- Tokenize gene names to IDs
- Tokenize cells with gene expression values
- Pad/truncate sequences
- Create masks for masked language modeling
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from scgpt_mini.tokenizer.vocabulary import GeneVocab


def tokenize_genes(
    gene_names: List[str],
    vocab: GeneVocab,
    return_pt: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Tokenize a list of gene names to IDs.

    Args:
        gene_names: List of gene names
        vocab: GeneVocab instance
        return_pt: If True, return torch.Tensor; otherwise return np.ndarray

    Returns:
        Array of gene IDs

    Example:
        >>> vocab = GeneVocab(["GENE1", "GENE2", "GENE3"])
        >>> gene_ids = tokenize_genes(["GENE1", "GENE3"], vocab)
        >>> print(gene_ids)
        tensor([3, 5])
    """
    gene_ids = np.array([vocab[gene] for gene in gene_names])

    if return_pt:
        return torch.from_numpy(gene_ids).long()
    return gene_ids


def tokenize_cell(
    gene_names: np.ndarray,
    expression_values: np.ndarray,
    vocab: GeneVocab,
    append_cls: bool = True,
    include_zero_genes: bool = False,
    return_pt: bool = True,
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """
    Tokenize a single cell's gene expression data.

    Args:
        gene_names: Array of gene names for the cell
        expression_values: Array of expression values (same length as gene_names)
        vocab: GeneVocab instance
        append_cls: If True, prepend CLS token
        include_zero_genes: If True, include genes with zero expression
        return_pt: If True, return torch.Tensor; otherwise return np.ndarray

    Returns:
        Tuple of (gene_ids, values)

    Example:
        >>> vocab = GeneVocab(["GENE1", "GENE2", "GENE3"])
        >>> genes = np.array(["GENE1", "GENE2", "GENE3"])
        >>> values = np.array([1.5, 0.0, 2.3])
        >>> gene_ids, expr_values = tokenize_cell(genes, values, vocab)
        >>> print(gene_ids)  # CLS token + non-zero genes
        tensor([1, 3, 5])
        >>> print(expr_values)
        tensor([0.0, 1.5, 2.3])
    """
    if len(gene_names) != len(expression_values):
        raise ValueError(
            f"Number of genes ({len(gene_names)}) does not match "
            f"number of values ({len(expression_values)})"
        )

    # Filter zero genes if requested
    if not include_zero_genes:
        nonzero_idx = np.nonzero(expression_values)[0]
        gene_names = gene_names[nonzero_idx]
        expression_values = expression_values[nonzero_idx]

    # Convert gene names to IDs
    gene_ids = np.array([vocab[gene] for gene in gene_names])

    # Append CLS token if requested
    if append_cls:
        gene_ids = np.insert(gene_ids, 0, vocab.cls_id)
        expression_values = np.insert(expression_values, 0, 0.0)

    # Convert to tensors if requested
    if return_pt:
        gene_ids = torch.from_numpy(gene_ids).long()
        expression_values = torch.from_numpy(expression_values).float()

    return gene_ids, expression_values


def pad_sequences(
    gene_ids_list: List[torch.Tensor],
    values_list: List[torch.Tensor],
    max_len: int,
    vocab: GeneVocab,
    pad_value: float = 0.0,
    cls_appended: bool = True,
    truncate_method: str = "random",
) -> Dict[str, torch.Tensor]:
    """
    Pad or truncate sequences to a fixed length.

    Args:
        gene_ids_list: List of gene ID tensors
        values_list: List of expression value tensors
        max_len: Maximum sequence length
        vocab: GeneVocab instance
        pad_value: Value to use for padding expression values
        cls_appended: If True, sequences already have CLS token prepended
        truncate_method: How to truncate long sequences ("random" or "first")

    Returns:
        Dictionary with keys:
        - "genes": Padded gene ID tensor of shape (batch_size, max_len)
        - "values": Padded expression value tensor of shape (batch_size, max_len)
        - "attention_mask": Boolean mask of shape (batch_size, max_len)

    Example:
        >>> genes1 = torch.tensor([1, 3, 5])  # CLS + 2 genes
        >>> values1 = torch.tensor([0.0, 1.5, 2.3])
        >>> genes2 = torch.tensor([1, 4])  # CLS + 1 gene
        >>> values2 = torch.tensor([0.0, 3.2])
        >>> batch = pad_sequences([genes1, genes2], [values1, values2], max_len=5, vocab=vocab)
        >>> print(batch["genes"].shape)
        torch.Size([2, 5])
    """
    if len(gene_ids_list) != len(values_list):
        raise ValueError("gene_ids_list and values_list must have the same length")

    batch_size = len(gene_ids_list)
    pad_id = vocab.pad_id

    # Initialize padded tensors
    genes_padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    values_padded = torch.full((batch_size, max_len), pad_value, dtype=torch.float)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, (gene_ids, values) in enumerate(zip(gene_ids_list, values_list)):
        seq_len = len(gene_ids)

        if seq_len > max_len:
            # Truncate sequence
            if truncate_method == "random":
                if cls_appended:
                    # Keep CLS token, randomly sample from the rest
                    indices = torch.randperm(seq_len - 1)[:max_len - 1] + 1
                    indices = torch.cat([torch.tensor([0]), indices])
                else:
                    indices = torch.randperm(seq_len)[:max_len]
                gene_ids = gene_ids[indices]
                values = values[indices]
            else:  # "first"
                gene_ids = gene_ids[:max_len]
                values = values[:max_len]
            seq_len = max_len

        # Fill in the sequence
        genes_padded[i, :seq_len] = gene_ids
        values_padded[i, :seq_len] = values
        attention_mask[i, :seq_len] = True

    return {
        "genes": genes_padded,
        "values": values_padded,
        "attention_mask": attention_mask,
    }


def create_mask(
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_ratio: float = 0.15,
    mask_value: float = 0.0,
    vocab: Optional[GeneVocab] = None,
    cls_appended: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Create masked inputs for masked language modeling (MLM).

    Args:
        values: Expression value tensor of shape (batch_size, seq_len)
        attention_mask: Boolean mask indicating valid positions
        mask_ratio: Fraction of non-padding tokens to mask
        mask_value: Value to use for masked positions
        vocab: GeneVocab instance (optional, used to avoid masking special tokens)
        cls_appended: If True, avoid masking the CLS token

    Returns:
        Dictionary with keys:
        - "masked_values": Values with some positions masked
        - "mask_positions": Boolean tensor indicating masked positions
        - "target_values": Original values at masked positions (for loss computation)

    Example:
        >>> values = torch.tensor([[0.0, 1.5, 2.3, 3.1], [0.0, 0.5, 0.0, 0.0]])
        >>> mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        >>> masked = create_mask(values, mask, mask_ratio=0.5)
        >>> print(masked["mask_positions"])
        tensor([[False,  True, False,  True],
                [False,  True, False, False]])
    """
    batch_size, seq_len = values.shape
    device = values.device

    # Create mask for positions that can be masked
    # Don't mask padding tokens or CLS token
    maskable_positions = attention_mask.clone()
    if cls_appended:
        # Don't mask the first position (CLS token)
        maskable_positions[:, 0] = False

    # Count maskable positions per sequence
    num_maskable = maskable_positions.sum(dim=1)

    # Determine how many positions to mask per sequence
    num_to_mask = (num_maskable.float() * mask_ratio).long()

    # Create mask positions tensor
    mask_positions = torch.zeros_like(values, dtype=torch.bool)

    for i in range(batch_size):
        if num_to_mask[i] > 0:
            # Get indices of maskable positions
            maskable_indices = torch.where(maskable_positions[i])[0]

            # Randomly select positions to mask
            perm = torch.randperm(len(maskable_indices))
            selected_indices = maskable_indices[perm[:num_to_mask[i]]]

            # Set mask positions
            mask_positions[i, selected_indices] = True

    # Create masked values
    masked_values = values.clone()
    masked_values[mask_positions] = mask_value

    # Store target values for masked positions
    target_values = values.clone()

    return {
        "masked_values": masked_values,
        "mask_positions": mask_positions,
        "target_values": target_values,
    }


def tokenize_batch(
    data: np.ndarray,
    gene_names: np.ndarray,
    vocab: GeneVocab,
    append_cls: bool = True,
    include_zero_genes: bool = False,
    return_pt: bool = True,
) -> List[Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]]:
    """
    Tokenize a batch of cells.

    Args:
        data: Expression data of shape (n_cells, n_genes)
        gene_names: Gene names array of shape (n_genes,)
        vocab: GeneVocab instance
        append_cls: If True, prepend CLS token to each cell
        include_zero_genes: If True, include genes with zero expression
        return_pt: If True, return torch.Tensor; otherwise return np.ndarray

    Returns:
        List of tuples (gene_ids, expression_values) for each cell

    Example:
        >>> data = np.array([[1.5, 0.0, 2.3], [0.5, 1.0, 0.0]])
        >>> genes = np.array(["GENE1", "GENE2", "GENE3"])
        >>> vocab = GeneVocab(genes)
        >>> batch = tokenize_batch(data, genes, vocab)
        >>> len(batch)
        2
    """
    if data.shape[1] != len(gene_names):
        raise ValueError(
            f"Number of genes in data ({data.shape[1]}) does not match "
            f"number of gene names ({len(gene_names)})"
        )

    tokenized_data = []
    for i in range(len(data)):
        gene_ids, values = tokenize_cell(
            gene_names=gene_names,
            expression_values=data[i],
            vocab=vocab,
            append_cls=append_cls,
            include_zero_genes=include_zero_genes,
            return_pt=return_pt,
        )
        tokenized_data.append((gene_ids, values))

    return tokenized_data
