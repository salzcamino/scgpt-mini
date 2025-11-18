"""
Data preprocessing functions for scGPT-mini.

This module provides functions for preprocessing single-cell RNA-seq data:
- Filtering genes and cells
- Normalization and log transformation
- Highly variable gene selection
- Expression value binning
"""

from typing import Optional, Union

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse


def filter_genes(
    adata: AnnData,
    min_cells: int = 3,
    min_counts: int = 10,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Filter genes based on expression criteria.

    Args:
        adata: AnnData object
        min_cells: Minimum number of cells expressing a gene
        min_counts: Minimum total counts for a gene
        inplace: If True, modify adata in place; otherwise return a copy

    Returns:
        Filtered AnnData object if inplace=False, otherwise None

    Example:
        >>> adata = filter_genes(adata, min_cells=3, min_counts=10)
    """
    if not inplace:
        adata = adata.copy()

    # Filter genes by number of cells
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Filter genes by total counts
    if min_counts > 0:
        gene_counts = np.array(adata.X.sum(axis=0)).flatten()
        gene_mask = gene_counts >= min_counts
        adata._inplace_subset_var(gene_mask)

    if not inplace:
        return adata


def filter_cells(
    adata: AnnData,
    min_genes: int = 200,
    max_genes: Optional[int] = None,
    min_counts: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_pct_mito: Optional[float] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Filter cells based on quality control metrics.

    Args:
        adata: AnnData object
        min_genes: Minimum number of genes expressed per cell
        max_genes: Maximum number of genes expressed per cell (optional)
        min_counts: Minimum total counts per cell (optional)
        max_counts: Maximum total counts per cell (optional)
        max_pct_mito: Maximum percentage of mitochondrial genes (optional)
        inplace: If True, modify adata in place; otherwise return a copy

    Returns:
        Filtered AnnData object if inplace=False, otherwise None

    Example:
        >>> adata = filter_cells(adata, min_genes=200, max_genes=10000)
    """
    if not inplace:
        adata = adata.copy()

    # Filter cells by number of genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes=max_genes)

    # Filter cells by counts
    if min_counts is not None:
        sc.pp.filter_cells(adata, min_counts=min_counts)
    if max_counts is not None:
        sc.pp.filter_cells(adata, max_counts=max_counts)

    # Filter by mitochondrial percentage
    if max_pct_mito is not None:
        # Calculate mitochondrial gene percentage
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito, :]

    if not inplace:
        return adata


def normalize_total(
    adata: AnnData,
    target_sum: float = 1e4,
    layer: Optional[str] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Normalize total counts per cell to a target sum.

    This is library-size normalization where each cell's counts are scaled
    to sum to target_sum (typically 10,000).

    Args:
        adata: AnnData object
        target_sum: Target sum for normalization
        layer: Layer to normalize (if None, normalizes .X)
        inplace: If True, modify adata in place; otherwise return a copy

    Returns:
        Normalized AnnData object if inplace=False, otherwise None

    Example:
        >>> adata = normalize_total(adata, target_sum=1e4)
    """
    if not inplace:
        adata = adata.copy()

    sc.pp.normalize_total(adata, target_sum=target_sum, layer=layer, inplace=True)

    if not inplace:
        return adata


def log_transform(
    adata: AnnData,
    base: Optional[float] = None,
    layer: Optional[str] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Apply log(x + 1) transformation to expression values.

    Args:
        adata: AnnData object
        base: Logarithm base (if None, uses natural log)
        layer: Layer to transform (if None, transforms .X)
        inplace: If True, modify adata in place; otherwise return a copy

    Returns:
        Transformed AnnData object if inplace=False, otherwise None

    Example:
        >>> adata = log_transform(adata)
    """
    if not inplace:
        adata = adata.copy()

    sc.pp.log1p(adata, base=base, layer=layer)

    if not inplace:
        return adata


def select_hvg(
    adata: AnnData,
    n_top_genes: int = 1000,
    flavor: str = "seurat_v3",
    layer: Optional[str] = None,
    batch_key: Optional[str] = None,
    subset: bool = True,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Select highly variable genes (HVGs).

    Args:
        adata: AnnData object
        n_top_genes: Number of top variable genes to select
        flavor: Method for HVG selection ("seurat_v3", "seurat", or "cell_ranger")
        layer: Layer to use for HVG calculation (if None, uses .X)
        batch_key: Key in .obs for batch information (for batch-aware HVG selection)
        subset: If True, subset the AnnData to selected genes
        inplace: If True, modify adata in place; otherwise return a copy

    Returns:
        AnnData object with HVGs if inplace=False, otherwise None

    Example:
        >>> adata = select_hvg(adata, n_top_genes=1000, flavor="seurat_v3")
    """
    if not inplace:
        adata = adata.copy()

    # Calculate highly variable genes
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        layer=layer,
        batch_key=batch_key,
        subset=subset,
        inplace=True,
    )

    if not inplace:
        return adata


def bin_expression(
    adata: AnnData,
    n_bins: int = 51,
    layer: Optional[str] = None,
    result_key: str = "X_binned",
    method: str = "uniform",
) -> AnnData:
    """
    Bin continuous expression values into discrete bins.

    Args:
        adata: AnnData object
        n_bins: Number of bins (default: 51, matching scGPT)
        layer: Layer to bin (if None, bins .X)
        result_key: Key to store binned data in .layers
        method: Binning method ("uniform" for equal-width or "quantile")

    Returns:
        AnnData object with binned data added to .layers[result_key]

    Example:
        >>> adata = bin_expression(adata, n_bins=51)
        >>> binned_data = adata.layers["X_binned"]
    """
    # Get data to bin
    if layer is None:
        data = adata.X
    else:
        data = adata.layers[layer]

    # Convert to dense if sparse
    if issparse(data):
        data = data.toarray()

    # Bin the data
    if method == "uniform":
        # Equal-width bins based on data range
        # Reserve bin 0 for values exactly equal to 0
        nonzero_mask = data > 0
        binned_data = np.zeros_like(data, dtype=np.int32)

        if nonzero_mask.any():
            nonzero_data = data[nonzero_mask]
            min_val, max_val = nonzero_data.min(), nonzero_data.max()

            # Create n_bins - 1 bins for non-zero values (bin 0 is for zeros)
            bins = np.linspace(min_val, max_val, n_bins)
            binned_nonzero = np.digitize(nonzero_data, bins)

            # Ensure binned values are in range [1, n_bins - 1]
            binned_nonzero = np.clip(binned_nonzero, 1, n_bins - 1)
            binned_data[nonzero_mask] = binned_nonzero

    elif method == "quantile":
        # Quantile-based binning
        nonzero_mask = data > 0
        binned_data = np.zeros_like(data, dtype=np.int32)

        if nonzero_mask.any():
            nonzero_data = data[nonzero_mask]

            # Create quantile bins for non-zero values
            quantiles = np.linspace(0, 100, n_bins)
            bins = np.percentile(nonzero_data, quantiles)
            binned_nonzero = np.digitize(nonzero_data, bins)

            # Ensure binned values are in range [1, n_bins - 1]
            binned_nonzero = np.clip(binned_nonzero, 1, n_bins - 1)
            binned_data[nonzero_mask] = binned_nonzero

    else:
        raise ValueError(f"Unknown binning method: {method}")

    # Store binned data
    adata.layers[result_key] = binned_data

    return adata


def preprocess_adata(
    adata: AnnData,
    filter_gene_by_counts: Union[int, bool] = 10,
    filter_cell_by_counts: Union[int, bool] = False,
    filter_gene_by_cells: Union[int, bool] = 3,
    filter_cell_by_genes: Union[int, bool] = 200,
    normalize_total_target: Union[float, bool] = 1e4,
    log1p: bool = True,
    subset_hvg: Union[int, bool] = 1000,
    hvg_flavor: str = "seurat_v3",
    binning: Union[int, bool] = 51,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Complete preprocessing pipeline for single-cell RNA-seq data.

    This function applies the following steps in order:
    1. Filter genes by counts and number of cells
    2. Filter cells by counts and number of genes
    3. Normalize total counts per cell
    4. Log1p transformation
    5. Select highly variable genes
    6. Bin expression values

    Args:
        adata: AnnData object
        filter_gene_by_counts: Minimum total counts per gene (False to skip)
        filter_cell_by_counts: Minimum total counts per cell (False to skip)
        filter_gene_by_cells: Minimum number of cells expressing a gene
        filter_cell_by_genes: Minimum number of genes per cell
        normalize_total_target: Target sum for normalization (False to skip)
        log1p: Whether to apply log1p transformation
        subset_hvg: Number of HVGs to select (False to skip)
        hvg_flavor: HVG selection method
        binning: Number of bins for expression values (False to skip)
        inplace: If True, modify adata in place; otherwise return a copy

    Returns:
        Preprocessed AnnData object if inplace=False, otherwise None

    Example:
        >>> adata = preprocess_adata(
        ...     adata,
        ...     filter_gene_by_counts=10,
        ...     normalize_total_target=1e4,
        ...     log1p=True,
        ...     subset_hvg=1000,
        ...     binning=51,
        ... )
    """
    if not inplace:
        adata = adata.copy()

    print(f"Starting preprocessing with {adata.n_obs} cells and {adata.n_vars} genes")

    # Step 1: Filter genes
    if filter_gene_by_counts or filter_gene_by_cells:
        min_counts = filter_gene_by_counts if isinstance(filter_gene_by_counts, int) else 0
        min_cells = filter_gene_by_cells if isinstance(filter_gene_by_cells, int) else 0
        filter_genes(adata, min_cells=min_cells, min_counts=min_counts, inplace=True)
        print(f"After gene filtering: {adata.n_vars} genes")

    # Step 2: Filter cells
    if filter_cell_by_counts or filter_cell_by_genes:
        min_counts = filter_cell_by_counts if isinstance(filter_cell_by_counts, int) else None
        min_genes = filter_cell_by_genes if isinstance(filter_cell_by_genes, int) else 0
        filter_cells(adata, min_genes=min_genes, min_counts=min_counts, inplace=True)
        print(f"After cell filtering: {adata.n_obs} cells")

    # Step 3: Normalize total counts
    if normalize_total_target:
        target = normalize_total_target if isinstance(normalize_total_target, float) else 1e4
        normalize_total(adata, target_sum=target, inplace=True)
        print(f"Normalized to target sum: {target}")

    # Step 4: Log transformation
    if log1p:
        log_transform(adata, inplace=True)
        print("Applied log1p transformation")

    # Step 5: Select HVGs
    if subset_hvg:
        n_hvg = subset_hvg if isinstance(subset_hvg, int) else 2000
        select_hvg(adata, n_top_genes=n_hvg, flavor=hvg_flavor, subset=True, inplace=True)
        print(f"Selected {adata.n_vars} highly variable genes")

    # Step 6: Bin expression values
    if binning:
        n_bins = binning if isinstance(binning, int) else 51
        bin_expression(adata, n_bins=n_bins)
        print(f"Binned expression into {n_bins} bins")

    print(f"Preprocessing complete: {adata.n_obs} cells x {adata.n_vars} genes")

    if not inplace:
        return adata
