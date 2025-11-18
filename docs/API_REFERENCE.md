# scGPT-mini API Reference

Complete API documentation for scGPT-mini v0.1.0

## Table of Contents

- [Tokenizer Module](#tokenizer-module)
  - [GeneVocab](#genevocab)
  - [Tokenization Functions](#tokenization-functions)
- [Data Module](#data-module)
  - [Preprocessing Functions](#preprocessing-functions)
  - [DataLoader Creation](#dataloader-creation)
- [Model Module](#model-module)
  - [TransformerModel](#transformermodel)
  - [Encoders](#encoders)
  - [Decoders](#decoders)
- [Training Module](#training-module)
  - [Trainer](#trainer)
  - [Loss Functions](#loss-functions)
  - [Metrics](#metrics)
- [Tasks Module](#tasks-module)
  - [Cell Embedding](#cell-embedding)
  - [Cell Type Annotation](#cell-type-annotation)
- [Utils Module](#utils-module)
  - [Visualization](#visualization)
  - [Logging](#logging)

---

## Tokenizer Module

### GeneVocab

**Class**: `scgpt_mini.tokenizer.GeneVocab`

A vocabulary class for mapping gene names to integer IDs.

#### Constructor

```python
GeneVocab(genes: List[str], special_tokens: Optional[List[str]] = None)
```

**Parameters:**
- `genes` (List[str]): List of gene names to include in vocabulary
- `special_tokens` (Optional[List[str]]): Custom special tokens. Defaults to `["<pad>", "<cls>", "<eoc>"]`

**Attributes:**
- `DEFAULT_SPECIAL_TOKENS`: `["<pad>", "<cls>", "<eoc>"]`
- `pad_token_id`: Always 0
- `cls_token_id`: Always 1
- `eoc_token_id`: Always 2

#### Methods

##### `__len__()`
Returns the total vocabulary size (genes + special tokens).

```python
vocab = GeneVocab(["GENE1", "GENE2"])
len(vocab)  # 5 (3 special tokens + 2 genes)
```

##### `get_id(gene: str) -> int`
Get the integer ID for a gene name.

```python
gene_id = vocab.get_id("GENE1")
```

**Raises:** `KeyError` if gene not in vocabulary

##### `get_gene(id: int) -> str`
Get the gene name for an integer ID.

```python
gene_name = vocab.get_gene(3)
```

**Raises:** `IndexError` if ID out of range

##### `__contains__(gene: str) -> bool`
Check if a gene is in the vocabulary.

```python
if "GENE1" in vocab:
    print("Found!")
```

##### `save(filepath: str)`
Save vocabulary to JSON file.

```python
vocab.save("my_vocab.json")
```

##### `from_json(filepath: str) -> GeneVocab` (classmethod)
Load vocabulary from JSON file.

```python
vocab = GeneVocab.from_json("default_vocab.json")
```

---

### Tokenization Functions

#### `tokenize_genes`

```python
tokenize_genes(
    genes: List[str],
    vocab: GeneVocab,
    handle_unknown: str = "skip"
) -> List[int]
```

Tokenize a list of gene names to IDs.

**Parameters:**
- `genes`: List of gene names
- `vocab`: GeneVocab instance
- `handle_unknown`: How to handle unknown genes. Options: `"skip"` or `"error"`

**Returns:** List of gene IDs

**Example:**
```python
gene_ids = tokenize_genes(["GENE1", "GENE2"], vocab)
```

---

#### `tokenize_batch`

```python
tokenize_batch(
    data: np.ndarray,
    gene_names: np.ndarray,
    vocab: GeneVocab,
    append_cls: bool = True,
    include_zero_genes: bool = False,
    return_pt: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]
```

Tokenize a batch of cells.

**Parameters:**
- `data`: Expression matrix (n_cells × n_genes)
- `gene_names`: Array of gene names matching columns
- `vocab`: GeneVocab instance
- `append_cls`: Whether to prepend CLS token
- `include_zero_genes`: Whether to include genes with zero expression
- `return_pt`: Return PyTorch tensors (True) or numpy arrays (False)

**Returns:** List of (gene_ids, expression_values) tuples

**Example:**
```python
tokenized = tokenize_batch(
    data=adata.X.toarray(),
    gene_names=adata.var_names.values,
    vocab=vocab,
)
# tokenized[0] = (gene_ids, values) for first cell
```

---

## Data Module

### Preprocessing Functions

#### `preprocess_adata`

```python
preprocess_adata(
    adata: sc.AnnData,
    filter_gene_by_counts: int = 10,
    filter_cell_by_genes: int = 200,
    filter_cell_max_genes: int = 10000,
    normalize_total_target: float = 1e4,
    log1p: bool = True,
    subset_hvg: Optional[int] = None,
    binning: Union[bool, int] = False,
    inplace: bool = True,
) -> sc.AnnData
```

Complete preprocessing pipeline for scRNA-seq data.

**Parameters:**
- `adata`: AnnData object
- `filter_gene_by_counts`: Minimum counts per gene
- `filter_cell_by_genes`: Minimum genes per cell
- `filter_cell_max_genes`: Maximum genes per cell
- `normalize_total_target`: Target sum for normalization (e.g., 10,000)
- `log1p`: Apply log(x+1) transformation
- `subset_hvg`: Number of highly variable genes to select (None = keep all)
- `binning`: Number of bins for expression discretization (False = continuous)
- `inplace`: Modify adata in-place

**Returns:** Preprocessed AnnData object

**Example:**
```python
adata = preprocess_adata(
    adata,
    subset_hvg=500,
    binning=51,
    inplace=False,
)
```

---

#### Individual Preprocessing Functions

##### `filter_genes`

```python
filter_genes(
    adata: sc.AnnData,
    min_cells: int = 3,
    min_counts: int = 10,
    inplace: bool = True,
) -> sc.AnnData
```

Filter genes by minimum cells and counts.

---

##### `filter_cells`

```python
filter_cells(
    adata: sc.AnnData,
    min_genes: int = 200,
    max_genes: int = 10000,
    inplace: bool = True,
) -> sc.AnnData
```

Filter cells by gene count thresholds.

---

##### `normalize_total`

```python
normalize_total(
    adata: sc.AnnData,
    target_sum: float = 1e4,
    inplace: bool = True,
) -> sc.AnnData
```

Normalize each cell to a target sum.

---

##### `log_transform`

```python
log_transform(
    adata: sc.AnnData,
    inplace: bool = True,
) -> sc.AnnData
```

Apply log(x + 1) transformation.

---

##### `select_hvg`

```python
select_hvg(
    adata: sc.AnnData,
    n_top_genes: int = 1000,
    inplace: bool = True,
) -> sc.AnnData
```

Select highly variable genes.

---

##### `bin_expression`

```python
bin_expression(
    adata: sc.AnnData,
    n_bins: int = 51,
    inplace: bool = True,
) -> sc.AnnData
```

Bin continuous expression values into discrete categories.

---

### DataLoader Creation

#### `create_dataloader`

```python
create_dataloader(
    tokenized_data: List[Tuple[torch.Tensor, torch.Tensor]],
    vocab: GeneVocab,
    batch_size: int = 32,
    max_len: int = 1001,
    shuffle: bool = True,
    apply_masking: bool = True,
    mask_ratio: float = 0.15,
    labels: Optional[List[int]] = None,
) -> torch.utils.data.DataLoader
```

Create a DataLoader for training or inference.

**Parameters:**
- `tokenized_data`: Output from `tokenize_batch()`
- `vocab`: GeneVocab instance
- `batch_size`: Batch size
- `max_len`: Maximum sequence length (pad/truncate to this)
- `shuffle`: Shuffle data
- `apply_masking`: Apply MLM masking (for pretraining)
- `mask_ratio`: Proportion of tokens to mask (if apply_masking=True)
- `labels`: Optional cell type labels for classification

**Returns:** PyTorch DataLoader

**Batch Output:**
- `genes`: (batch_size, max_len) - Gene IDs
- `values`: (batch_size, max_len) - Expression values
- `attention_mask`: (batch_size, max_len) - Valid positions (1) vs padding (0)
- `masked_values`: (batch_size, max_len) - Values with masking applied (if apply_masking=True)
- `mask_positions`: (batch_size, max_len) - Masked positions (if apply_masking=True)
- `target_values`: (batch_size, max_len) - Ground truth for masked positions (if apply_masking=True)
- `labels`: (batch_size,) - Cell type labels (if labels provided)

---

## Model Module

### TransformerModel

**Class**: `scgpt_mini.model.TransformerModel`

Main transformer model for single-cell gene expression.

#### Constructor

```python
TransformerModel(
    vocab_size: int,
    d_model: int = 32,
    nhead: int = 2,
    num_layers: int = 2,
    d_hid: int = 64,
    n_classes: Optional[int] = None,
    n_bins: int = 51,
    dropout: float = 0.1,
    max_seq_len: int = 1001,
    value_mode: str = "continuous",
    use_positional_encoding: bool = False,
    activation: str = "gelu",
    norm_first: bool = True,
    explicit_zero_prob: bool = False,
    vocab: Optional[GeneVocab] = None,
)
```

**Parameters:**
- `vocab_size`: Size of gene vocabulary
- `d_model`: Embedding dimension (32 for scGPT-mini)
- `nhead`: Number of attention heads
- `num_layers`: Number of transformer layers
- `d_hid`: Hidden dimension in feedforward network
- `n_classes`: Number of cell types (for classification). None = no classification head
- `n_bins`: Number of expression bins (if value_mode="binned")
- `dropout`: Dropout rate
- `max_seq_len`: Maximum sequence length
- `value_mode`: "continuous" or "binned"
- `use_positional_encoding`: Add positional encoding
- `activation`: "gelu" or "relu"
- `norm_first`: Pre-LN (True) or post-LN (False)
- `explicit_zero_prob`: Model zero expression separately
- `vocab`: GeneVocab instance (for padding index)

#### Methods

##### `forward()`

```python
forward(
    genes: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]
```

Forward pass through the model.

**Parameters:**
- `genes`: (batch_size, seq_len) - Gene IDs
- `values`: (batch_size, seq_len) - Expression values
- `attention_mask`: (batch_size, seq_len) - Attention mask

**Returns:** Dictionary with:
- `expr_pred`: (batch_size, seq_len) - Predicted expression values
- `cell_embeddings`: (batch_size, d_model) - CLS token embeddings
- `cls_pred`: (batch_size, n_classes) - Cell type logits (if n_classes provided)

**Example:**
```python
output = model(genes, values, attention_mask)
predictions = output['expr_pred']
embeddings = output['cell_embeddings']
```

---

##### `save_checkpoint()`

```python
save_checkpoint(
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    **kwargs
)
```

Save model checkpoint.

**Parameters:**
- `filepath`: Path to save checkpoint
- `optimizer`: Optimizer to save state
- `epoch`: Current epoch
- `**kwargs`: Additional metadata to save

---

##### `load_checkpoint()`

```python
load_checkpoint(
    filepath: str,
    device: str = "cpu",
    load_optimizer: bool = False,
) -> Dict
```

Load model checkpoint.

**Returns:** Checkpoint dictionary with metadata

---

### Encoders

#### `GeneEncoder`

```python
GeneEncoder(vocab_size: int, d_model: int, dropout: float = 0.1)
```

Embedding layer for gene IDs.

---

#### `ContinuousValueEncoder`

```python
ContinuousValueEncoder(d_model: int, dropout: float = 0.1)
```

Encoder for continuous expression values.

---

#### `CategoryValueEncoder`

```python
CategoryValueEncoder(n_bins: int, d_model: int, dropout: float = 0.1)
```

Encoder for binned expression values.

---

### Decoders

#### `ExpressionDecoder`

```python
ExpressionDecoder(d_model: int, dropout: float = 0.1)
```

Decoder for continuous expression prediction.

---

#### `BinnedExpressionDecoder`

```python
BinnedExpressionDecoder(d_model: int, n_bins: int, dropout: float = 0.1)
```

Decoder for binned expression prediction.

---

#### `ClassificationDecoder`

```python
ClassificationDecoder(
    d_model: int,
    n_classes: int,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.1,
)
```

Multi-layer MLP for cell type classification.

---

## Training Module

### Trainer

**Class**: `scgpt_mini.training.Trainer`

Training loop manager.

#### Constructor

```python
Trainer(
    model: TransformerModel,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    output_dir: Optional[str] = None,
    gradient_clip: float = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
)
```

#### Methods

##### `train()`

```python
train(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    eval_every: int = 1,
    save_every: int = 5,
    early_stopping_patience: Optional[int] = None,
) -> Dict
```

Main training loop.

**Returns:** Training history dictionary with metrics

---

##### `train_epoch()`

```python
train_epoch(train_loader: DataLoader) -> float
```

Train for one epoch.

**Returns:** Average training loss

---

##### `validate_epoch()`

```python
validate_epoch(val_loader: DataLoader) -> Dict
```

Validate on validation set.

**Returns:** Dictionary of validation metrics

---

### Loss Functions

#### `masked_mse_loss`

```python
masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor
```

MSE loss only on masked positions.

---

#### `masked_mae_loss`

```python
masked_mae_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor
```

MAE loss only on masked positions.

---

#### `combined_loss`

```python
combined_loss(
    expr_pred: torch.Tensor,
    expr_target: torch.Tensor,
    expr_mask: torch.Tensor,
    cls_pred: Optional[torch.Tensor] = None,
    cls_target: Optional[torch.Tensor] = None,
    mlm_weight: float = 1.0,
    cls_weight: float = 0.5,
) -> torch.Tensor
```

Combined MLM + classification loss.

---

### Metrics

#### `compute_masked_metrics`

```python
compute_masked_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]
```

Compute MSE, MAE, and Pearson correlation on masked positions.

**Returns:** Dictionary with `mse`, `mae`, `pearson`

---

## Tasks Module

### Cell Embedding

#### `extract_cell_embeddings`

```python
extract_cell_embeddings(
    model: TransformerModel,
    adata: sc.AnnData,
    vocab: GeneVocab,
    batch_size: int = 64,
    pool_strategy: str = "cls",
    device: str = "cpu",
) -> np.ndarray
```

Extract cell embeddings from a trained model.

**Parameters:**
- `model`: Trained TransformerModel
- `adata`: AnnData with preprocessed data
- `vocab`: GeneVocab instance
- `batch_size`: Batch size for inference
- `pool_strategy`: "cls", "mean", or "max"
- `device`: Device to run on

**Returns:** (n_cells, d_model) array of embeddings

**Example:**
```python
embeddings = extract_cell_embeddings(
    model=model,
    adata=adata,
    vocab=vocab,
    pool_strategy="cls",
)
adata.obsm['X_scgpt'] = embeddings
```

---

#### Embedding Quality Metrics

##### `compute_silhouette_score`

```python
compute_silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float
```

Compute silhouette score for cluster separation.

---

##### `compute_ari`

```python
compute_ari(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> float
```

Compute Adjusted Rand Index.

---

##### `knn_accuracy`

```python
knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    test_size: float = 0.3,
    random_state: int = 42,
) -> float
```

Compute k-NN classification accuracy.

---

### Cell Type Annotation

#### `finetune_for_annotation`

```python
finetune_for_annotation(
    model: TransformerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    freeze_encoder: bool = False,
    device: str = "cpu",
    output_dir: Optional[str] = None,
) -> Dict
```

Fine-tune model for cell type annotation.

**Parameters:**
- `freeze_encoder`: If True, only train classification head

**Returns:** Training history

---

#### `predict_cell_types`

```python
predict_cell_types(
    model: TransformerModel,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]
```

Predict cell types on unlabeled data.

**Returns:**
- `predictions`: (n_cells,) array of predicted class IDs
- `confidences`: (n_cells,) array of confidence scores (max softmax probability)

---

#### `annotate_adata`

```python
annotate_adata(
    model: TransformerModel,
    adata: sc.AnnData,
    vocab: GeneVocab,
    label_encoder: LabelEncoder,
    batch_size: int = 64,
    device: str = "cpu",
    confidence_threshold: float = 0.5,
) -> sc.AnnData
```

Annotate AnnData with predicted cell types.

Adds to adata.obs:
- `predicted_cell_type`: Predicted labels
- `prediction_confidence`: Confidence scores
- `low_confidence`: Boolean flag for confidence < threshold

---

#### Utility Functions

##### `freeze_encoder`

```python
freeze_encoder(model: TransformerModel)
```

Freeze all encoder parameters (only train classification head).

---

##### `unfreeze_encoder`

```python
unfreeze_encoder(model: TransformerModel)
```

Unfreeze all parameters for full fine-tuning.

---

## Utils Module

### Visualization

#### `plot_umap`

```python
plot_umap(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "UMAP",
    save_path: Optional[str] = None,
    **umap_kwargs
)
```

Compute and plot UMAP from embeddings.

---

#### `plot_training_curves`

```python
plot_training_curves(
    history: Dict,
    save_path: Optional[str] = None,
)
```

Plot training and validation curves.

---

### Logging

#### `setup_logger`

```python
setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger
```

Set up a logger with file and console handlers.

---

## Configuration Files

### Model Config

Location: `scgpt_mini/model/model_config.json`

```json
{
  "d_model": 32,
  "nhead": 2,
  "num_layers": 2,
  "d_hid": 64,
  "dropout": 0.1,
  "n_bins": 51,
  "max_seq_len": 1001,
  "value_mode": "continuous",
  "activation": "gelu",
  "norm_first": true
}
```

---

### Training Config

Location: `scgpt_mini/training/training_config.json`

```json
{
  "batch_size": 32,
  "learning_rate": 1e-4,
  "weight_decay": 0.01,
  "epochs": 20,
  "mask_ratio": 0.15,
  "gradient_clip": 1.0
}
```

---

## Common Workflows

### 1. Pretraining

```python
# Preprocess data
adata = preprocess_adata(adata, subset_hvg=500)

# Create vocabulary and tokenize
vocab = GeneVocab(adata.var_names.tolist())
tokenized = tokenize_batch(adata.X.toarray(), adata.var_names.values, vocab)

# Create DataLoader
train_loader = create_dataloader(tokenized, vocab, apply_masking=True)

# Create model
model = TransformerModel(vocab_size=len(vocab), d_model=32, num_layers=2)

# Train
trainer = Trainer(model, optimizer, device="cpu")
history = trainer.train(train_loader, epochs=20)
```

---

### 2. Embedding Extraction

```python
# Extract embeddings
embeddings = extract_cell_embeddings(
    model=model,
    adata=adata,
    vocab=vocab,
    pool_strategy="cls",
)

# Add to AnnData
adata.obsm['X_scgpt'] = embeddings

# Visualize with UMAP
import umap
reducer = umap.UMAP()
umap_coords = reducer.fit_transform(embeddings)
```

---

### 3. Cell Type Annotation

```python
# Prepare labeled data
train_loader = create_dataloader(
    tokenized_data, vocab, labels=cell_type_labels, apply_masking=False
)

# Fine-tune
model.n_classes = n_cell_types
history = finetune_for_annotation(
    model, train_loader, val_loader, freeze_encoder=True
)

# Predict
predictions, confidences = predict_cell_types(model, test_loader)
```

---

## Error Handling

Common errors and solutions:

### `KeyError: 'GENE_NAME'`
Gene not in vocabulary. Use `handle_unknown="skip"` in tokenization.

### `RuntimeError: size mismatch`
Model configuration doesn't match checkpoint. Verify `d_model`, `num_layers`, etc.

### `CUDA out of memory`
Reduce `batch_size` or use CPU with `device="cpu"`.

### `ValueError: n_classes not set`
For classification, set `n_classes` when creating model or before fine-tuning.

---

## Version History

### v0.1.0 (Current)
- Initial release
- MLM pretraining
- Cell embedding generation
- Cell type annotation
- CPU-optimized for laptops

---

## Citation

If you use scGPT-mini in your research, please cite:

```bibtex
@software{scgpt_mini,
  title={scGPT-mini: Educational Transformer for Single-Cell Analysis},
  author={scGPT-mini Contributors},
  year={2024},
  url={https://github.com/yourusername/scgpt-mini}
}
```

And the original scGPT paper:

```bibtex
@article{cui2023scgpt,
  title={scGPT: Foundation Model for Single-Cell Multi-omics},
  author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and others},
  journal={bioRxiv},
  year={2023}
}
```
