"""
Gene vocabulary management for scGPT-mini.

This module provides a simplified GeneVocab class for managing gene name to ID mappings
without external dependencies (no torchtext required).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


class GeneVocab:
    """
    Vocabulary for genes that maps gene names to integer IDs.

    Special tokens:
    - <pad>: Padding token (ID: 0)
    - <cls>: CLS token for cell embedding (ID: 1)
    - <eoc>: End of cell token (ID: 2)

    Example:
        >>> vocab = GeneVocab(["GENE1", "GENE2", "GENE3"])
        >>> vocab["GENE1"]
        3
        >>> vocab.get_gene(3)
        'GENE1'
        >>> len(vocab)
        6
    """

    # Default special tokens
    DEFAULT_SPECIAL_TOKENS = ["<pad>", "<cls>", "<eoc>"]

    def __init__(
        self,
        gene_list: List[str],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ):
        """
        Initialize the vocabulary.

        Args:
            gene_list: List of gene names to include in vocabulary
            specials: List of special tokens (default: ["<pad>", "<cls>", "<eoc>"])
            special_first: If True, add special tokens at the beginning
        """
        if specials is None:
            specials = self.DEFAULT_SPECIAL_TOKENS

        # Create mappings
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        # Add tokens
        if special_first:
            # Add special tokens first
            for token in specials:
                self._add_token(token)
            # Then add gene tokens
            for gene in gene_list:
                if gene not in self._token_to_id:  # Avoid duplicates
                    self._add_token(gene)
        else:
            # Add gene tokens first
            for gene in gene_list:
                if gene not in self._token_to_id:
                    self._add_token(gene)
            # Then add special tokens
            for token in specials:
                self._add_token(token)

        # Store special tokens
        self._special_tokens = specials

        # Set default index for unknown tokens (use pad token)
        self._default_index = self._token_to_id.get("<pad>", 0)

    def _add_token(self, token: str) -> None:
        """Add a token to the vocabulary."""
        if token not in self._token_to_id:
            idx = len(self._token_to_id)
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token

    def __len__(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        """Check if a token is in the vocabulary."""
        return token in self._token_to_id

    def __getitem__(self, token: str) -> int:
        """
        Get the ID for a token.

        Args:
            token: Gene name or special token

        Returns:
            Integer ID for the token (default_index if token not found)
        """
        return self._token_to_id.get(token, self._default_index)

    def get_id(self, token: str) -> int:
        """
        Get the ID for a token (alias for __getitem__).

        Args:
            token: Gene name or special token

        Returns:
            Integer ID for the token
        """
        return self[token]

    def get_gene(self, idx: int) -> str:
        """
        Get the token for an ID.

        Args:
            idx: Integer ID

        Returns:
            Gene name or special token

        Raises:
            KeyError: If idx is not in the vocabulary
        """
        return self._id_to_token[idx]

    def get_stoi(self) -> Dict[str, int]:
        """
        Get string to integer mapping.

        Returns:
            Dictionary mapping tokens to IDs
        """
        return self._token_to_id.copy()

    def get_itos(self) -> Dict[int, str]:
        """
        Get integer to string mapping.

        Returns:
            Dictionary mapping IDs to tokens
        """
        return self._id_to_token.copy()

    @property
    def pad_token(self) -> str:
        """Get the padding token."""
        return "<pad>"

    @property
    def pad_id(self) -> int:
        """Get the padding token ID."""
        return self["<pad>"]

    @property
    def cls_token(self) -> str:
        """Get the CLS token."""
        return "<cls>"

    @property
    def cls_id(self) -> int:
        """Get the CLS token ID."""
        return self["<cls>"]

    @property
    def eoc_token(self) -> str:
        """Get the end-of-cell token."""
        return "<eoc>"

    @property
    def eoc_id(self) -> int:
        """Get the end-of-cell token ID."""
        return self["<eoc>"]

    @property
    def special_tokens(self) -> List[str]:
        """Get list of special tokens."""
        return self._special_tokens.copy()

    def set_default_index(self, idx: int) -> None:
        """
        Set the default index for unknown tokens.

        Args:
            idx: Default index to use
        """
        if idx not in self._id_to_token:
            raise ValueError(f"Index {idx} is not in the vocabulary.")
        self._default_index = idx

    def save_json(self, file_path: Union[Path, str]) -> None:
        """
        Save the vocabulary to a JSON file.

        Args:
            file_path: Path to save the vocabulary
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        with file_path.open("w") as f:
            json.dump(self._token_to_id, f, indent=2)

    @classmethod
    def from_json(cls, file_path: Union[Path, str]) -> "GeneVocab":
        """
        Load vocabulary from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            GeneVocab instance
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        with file_path.open("r") as f:
            token_to_id = json.load(f)

        return cls.from_dict(token_to_id)

    @classmethod
    def from_dict(cls, token_to_id: Dict[str, int]) -> "GeneVocab":
        """
        Create vocabulary from a dictionary mapping.

        Args:
            token_to_id: Dictionary mapping tokens to IDs

        Returns:
            GeneVocab instance
        """
        # Create an empty vocab
        vocab = cls([], specials=[])

        # Add tokens in order of their IDs
        for token, idx in sorted(token_to_id.items(), key=lambda x: x[1]):
            vocab._token_to_id[token] = idx
            vocab._id_to_token[idx] = token

        # Detect special tokens
        vocab._special_tokens = [
            token for token in cls.DEFAULT_SPECIAL_TOKENS
            if token in vocab._token_to_id
        ]

        # Set default index
        vocab._default_index = vocab._token_to_id.get("<pad>", 0)

        return vocab

    def __repr__(self) -> str:
        """String representation of the vocabulary."""
        return f"GeneVocab(size={len(self)}, special_tokens={self._special_tokens})"


def create_vocab_from_genes(
    gene_list: List[str],
    specials: Optional[List[str]] = None,
    special_first: bool = True,
) -> GeneVocab:
    """
    Create a gene vocabulary from a list of gene names.

    This is a convenience function that wraps the GeneVocab constructor.

    Args:
        gene_list: List of gene names
        specials: List of special tokens
        special_first: If True, add special tokens at the beginning

    Returns:
        GeneVocab instance

    Example:
        >>> genes = ["GENE1", "GENE2", "GENE3"]
        >>> vocab = create_vocab_from_genes(genes)
        >>> vocab["GENE1"]
        3
    """
    return GeneVocab(gene_list, specials=specials, special_first=special_first)
