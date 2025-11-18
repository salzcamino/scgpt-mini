# scGPT-mini Troubleshooting Guide

This guide helps you resolve common issues when using scGPT-mini.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Data Preprocessing Errors](#data-preprocessing-errors)
3. [Tokenization Problems](#tokenization-problems)
4. [Model Training Issues](#model-training-issues)
5. [Memory Errors](#memory-errors)
6. [Performance Problems](#performance-problems)
7. [Embedding Extraction Issues](#embedding-extraction-issues)
8. [Annotation Errors](#annotation-errors)
9. [Common Questions](#common-questions)

---

## Installation Issues

### Problem: `scanpy` installation fails on Python 3.13+

**Error:**
```
ERROR: Could not build wheels for scanpy
```

**Solution:**
Use Python 3.8-3.12. Scanpy may not support the latest Python versions.

```bash
# Create environment with Python 3.10
conda create -n scgpt-mini python=3.10
conda activate scgpt-mini
pip install -r requirements.txt
```

### Problem: PyTorch installation issues

**Solution:**
Install CPU-only PyTorch explicitly:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Problem: Import errors after installation

**Error:**
```python
ModuleNotFoundError: No module named 'scgpt_mini'
```

**Solution:**
Install in editable mode:

```bash
cd scgpt-mini
pip install -e .
```

---

## Data Preprocessing Errors

### Problem: All genes filtered out

**Error:**
```
ValueError: No genes remaining after filtering
```

**Solution:**
Your filtering criteria are too strict. Relax the parameters:

```python
adata = preprocess_adata(
    adata,
    filter_gene_by_counts=3,  # Lower threshold
    filter_cell_by_genes=50,  # Lower threshold
    subset_hvg=500,
)
```

### Problem: All cells filtered out

**Error:**
```
ValueError: No cells remaining after filtering
```

**Solution:**
Check your data quality and adjust filters:

```python
# Check data before filtering
print(f"Genes per cell: {(adata.X > 0).sum(axis=1).mean()}")
print(f"Counts per cell: {adata.X.sum(axis=1).mean()}")

# Use more lenient filtering
adata = preprocess_adata(
    adata,
    filter_cell_by_genes=100,  # Lower minimum
    max_cell_genes=None,  # Remove upper limit
)
```

### Problem: Sparse matrix errors

**Error:**
```
TypeError: unsupported operand type(s) for +: 'numpy.ndarray' and 'scipy.sparse.csr_matrix'
```

**Solution:**
Convert sparse matrices to dense:

```python
if hasattr(adata.X, 'toarray'):
    adata.X = adata.X.toarray()
```

---

## Tokenization Problems

### Problem: Gene not in vocabulary

**Error:**
```
KeyError: 'GENENAME not in vocabulary'
```

**Solution:**
Use the default vocabulary which includes common genes, or create a custom vocabulary:

```python
# Option 1: Use default vocabulary (recommended)
vocab = GeneVocab.from_json("scgpt_mini/tokenizer/default_vocab.json")

# Option 2: Create custom vocabulary with your genes
gene_list = adata.var_names.tolist()
vocab = GeneVocab(gene_list)
vocab.save_json("my_vocab.json")
```

### Problem: Empty tokenized sequences

**Error:**
```
RuntimeError: All sequences are empty after tokenization
```

**Solution:**
Your genes don't match the vocabulary. Either:

1. Use genes that are in the vocabulary:
```python
# Keep only genes in vocabulary
valid_genes = [g for g in adata.var_names if g in vocab]
adata = adata[:, valid_genes]
```

2. Or include zero genes:
```python
tokenized_data = tokenize_batch(
    data=data_matrix,
    gene_names=gene_names,
    vocab=vocab,
    include_zero_genes=True,  # Include all genes
)
```

### Problem: Tokenization too slow

**Solution:**
```python
# Use fewer genes (HVG selection)
adata = preprocess_adata(adata, subset_hvg=500)

# Exclude zero genes to reduce sequence length
tokenized_data = tokenize_batch(
    ...,
    include_zero_genes=False,
)
```

---

## Model Training Issues

### Problem: Loss is NaN

**Error:**
```
Training loss: nan
```

**Possible Causes and Solutions:**

1. **Learning rate too high**:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Lower LR
```

2. **Gradient explosion**:
```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    gradient_clip=1.0,  # Add gradient clipping
)
```

3. **Bad initialization**:
```python
# Reinitialize model
model = TransformerModel(vocab_size=len(vocab), ...)
```

### Problem: Loss not decreasing

**Solutions:**

1. **Check masking ratio**:
```python
# Try different mask ratios
dataloader = create_dataloader(..., mask_ratio=0.15)
```

2. **Increase learning rate**:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

3. **Check data quality**:
```python
# Verify data has variation
batch = next(iter(dataloader))
print(f"Value range: {batch['values'].min():.2f} - {batch['values'].max():.2f}")
```

4. **Train longer**:
```python
history = trainer.train(..., num_epochs=50)
```

### Problem: Validation loss increases

**Solution:**
You may be overfitting. Try:

```python
# Increase dropout
model = TransformerModel(..., dropout=0.2)

# Reduce model size
model = TransformerModel(d_model=16, num_layers=1, ...)

# Use more training data
# or enable early stopping
trainer.train(..., patience=5)
```

### Problem: Training too slow

**Solutions:**

1. **Increase batch size**:
```python
dataloader = create_dataloader(..., batch_size=64)
```

2. **Use GPU if available**:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = Trainer(..., device=device)
```

3. **Reduce model size**:
```python
model = TransformerModel(
    d_model=16,  # Smaller
    num_layers=1,  # Fewer layers
    d_hid=32,  # Smaller FFN
)
```

4. **Use fewer cells**:
```python
# Subsample data for quick testing
adata = adata[:1000, :]
```

---

## Memory Errors

### Problem: Out of memory during training

**Error:**
```
RuntimeError: CUDA out of memory
```
or
```
MemoryError
```

**Solutions:**

1. **Reduce batch size**:
```python
dataloader = create_dataloader(..., batch_size=16)  # Smaller batches
```

2. **Reduce sequence length**:
```python
dataloader = create_dataloader(..., max_len=500)  # Shorter sequences
```

3. **Use fewer genes**:
```python
adata = preprocess_adata(..., subset_hvg=300)  # Fewer HVGs
```

4. **Reduce model size**:
```python
model = TransformerModel(
    d_model=16,  # Smallest
    num_layers=1,
    d_hid=32,
)
```

5. **Use gradient accumulation**:
```python
# Simulate larger batch size without memory cost
for i, batch in enumerate(dataloader):
    loss = ...
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Problem: Memory leak during training

**Solution:**
Ensure you're not accumulating gradients:

```python
# Clear cache periodically
import torch
torch.cuda.empty_cache()  # If using GPU

# Use no_grad for validation
with torch.no_grad():
    val_output = model(...)
```

---

## Performance Problems

### Problem: Training takes hours on CPU

**Solution:**
This is expected for CPU training. To speed up:

1. Use smaller dataset (1000-5000 cells)
2. Reduce model size (d_model=16, num_layers=1)
3. Train for fewer epochs (5-10)
4. Use a GPU if available

```python
# Quick training configuration
model = TransformerModel(d_model=16, num_layers=1, ...)
history = trainer.train(..., num_epochs=5)
```

### Problem: Preprocessing is slow

**Solution:**

```python
# Use inplace operations
adata = preprocess_adata(..., inplace=True)

# Reduce gene count early
sc.pp.filter_genes(adata, min_cells=10)
```

---

## Embedding Extraction Issues

### Problem: Embeddings are all similar

**Solution:**
Your model may not be trained well:

```python
# Train longer
history = trainer.train(..., num_epochs=20)

# Check training loss decreased
print(f"Final loss: {history['train_loss'][-1]}")

# Verify embeddings have variance
print(f"Embedding std: {embeddings.std(axis=0).mean()}")
```

### Problem: UMAP/t-SNE fails

**Error:**
```
ValueError: n_neighbors must be less than n_samples
```

**Solution:**
```python
# Reduce n_neighbors for small datasets
fig, ax, coords = plot_umap(
    embeddings,
    labels=labels,
    n_neighbors=5,  # Smaller value
)
```

### Problem: Poor embedding quality

**Solution:**
```python
# 1. Ensure model is pretrained
# 2. Check silhouette score
metrics = embedding_eval_report(embeddings, labels)
print(f"Silhouette: {metrics['silhouette_score']}")

# 3. Try different embedding modes
embeddings_mean = extract_cell_embeddings(..., embedding_mode="mean")
embeddings_max = extract_cell_embeddings(..., embedding_mode="max")

# 4. Compare with PCA baseline
from sklearn.decomposition import PCA
pca = PCA(n_components=32)
pca_embeddings = pca.fit_transform(adata.X.toarray())
```

---

## Annotation Errors

### Problem: Fine-tuning fails immediately

**Error:**
```
RuntimeError: Expected classification decoder
```

**Solution:**
Ensure you're calling `finetune_for_annotation()` which automatically adds the classification head:

```python
history = finetune_for_annotation(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    n_classes=label_encoder["n_classes"],
    ...
)
```

### Problem: Low annotation accuracy

**Solutions:**

1. **Train longer**:
```python
history = finetune_for_annotation(..., num_epochs=20)
```

2. **Try full fine-tuning**:
```python
history = finetune_for_annotation(
    ...,
    freeze_encoder=False,  # Fine-tune entire model
    learning_rate=1e-5,  # Lower LR for stability
)
```

3. **Use more training data**:
```python
split_indices, label_encoder = create_classification_dataset(
    adata,
    train_split=0.8,  # More training data
    val_split=0.1,
    test_split=0.1,
)
```

4. **Check class balance**:
```python
# Check label distribution
print(adata.obs['cell_type'].value_counts())

# Use balanced_accuracy instead of accuracy
from scgpt_mini.tasks import annotation_eval_report
result = annotation_eval_report(...)
print(f"Balanced accuracy: {result['metrics']['balanced_accuracy']}")
```

### Problem: Predictions are all the same class

**Solution:**
Your model is predicting the majority class:

```python
# Use class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)

# Apply to loss function (manual implementation needed)
```

---

## Common Questions

### Q: How many cells do I need for pretraining?

**A:** Minimum 1000 cells, recommended 5000-10000. More data = better model, but also longer training.

### Q: How long should I train?

**A:**
- Pretraining: 10-20 epochs (watch validation loss)
- Fine-tuning: 5-15 epochs (watch validation accuracy)
- Stop if validation metrics stop improving

### Q: Should I use binning or continuous values?

**A:** Continuous values (binning=False) work better for scGPT-mini. Binning can help with very large expression ranges.

### Q: How do I know if my model is good?

**A:**
- **Pretraining**: Validation loss should decrease and converge
- **Embeddings**: Silhouette score > 0.3, k-NN accuracy > 0.7
- **Annotation**: Accuracy > 0.8 for well-separated cell types

### Q: Can I use my own dataset?

**A:** Yes! Load it as an AnnData object:

```python
import anndata as ad

# From CSV
adata = ad.read_csv("my_data.csv")

# From 10x
adata = sc.read_10x_mtx("path/to/filtered_gene_bc_matrices/")

# From h5ad
adata = ad.read_h5ad("my_data.h5ad")

# Then preprocess normally
adata = preprocess_adata(adata, ...)
```

### Q: How do I compare with other methods?

**A:**
```python
# PCA baseline
from sklearn.decomposition import PCA
pca = PCA(n_components=32)
pca_embeddings = pca.fit_transform(adata.X.toarray())

# Scanpy baseline
sc.pp.neighbors(adata, use_rep="X")
sc.tl.umap(adata)
sc.pl.umap(adata, color="cell_type")

# Compare with scGPT-mini embeddings
sc.pp.neighbors(adata, use_rep="X_scgpt")
sc.tl.umap(adata)
sc.pl.umap(adata, color="cell_type")
```

### Q: How do I save and load models?

**A:**
```python
# Save
model.save_checkpoint("my_model.pt")

# Load
model = TransformerModel.load_checkpoint("my_model.pt", vocab=vocab)
```

### Q: Can I use GPU?

**A:** Yes, if you have one:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = Trainer(..., device=device)
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check examples**: Look at `examples/` for working code
2. **Run tests**: `pytest tests/ -v` to verify installation
3. **Check documentation**: Read `CLAUDE.md` for detailed specifications
4. **Open an issue**: https://github.com/yourusername/scgpt-mini/issues

---

## Debug Checklist

When encountering errors, check:

- [ ] Python version is 3.8-3.12
- [ ] All dependencies are installed: `pip install -r requirements.txt`
- [ ] Data has non-zero cells and genes after preprocessing
- [ ] Genes match vocabulary (or using default vocabulary)
- [ ] Model parameters are reasonable (d_model <= 64)
- [ ] Learning rate is not too high (1e-5 to 1e-3)
- [ ] Batch size fits in memory (try 16 or 32)
- [ ] Training loss is decreasing (not NaN)

---

**scGPT-mini**: Making transformer models accessible! If all else fails, start with the smallest possible test case and gradually increase complexity.
