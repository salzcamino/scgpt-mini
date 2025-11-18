# scGPT-mini User Guide

A comprehensive guide to using scGPT-mini for single-cell RNA-seq analysis.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Concepts](#basic-concepts)
3. [Data Preparation](#data-preparation)
4. [Model Pretraining](#model-pretraining)
5. [Cell Embedding Generation](#cell-embedding-generation)
6. [Cell Type Annotation](#cell-type-annotation)
7. [Advanced Topics](#advanced-topics)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [FAQs](#faqs)

---

## Getting Started

### Installation

```bash
# From source (recommended for development)
git clone https://github.com/yourusername/scgpt-mini.git
cd scgpt-mini
pip install -e .

# With optional visualization dependencies
pip install -e ".[viz]"

# With development dependencies
pip install -e ".[dev]"
```

### Quick Start

```python
import scanpy as sc
from scgpt_mini.data import preprocess_adata
from scgpt_mini.tokenizer import GeneVocab
from scgpt_mini.model import TransformerModel

# Load data
adata = sc.datasets.pbmc3k()

# Preprocess
adata = preprocess_adata(adata, subset_hvg=500)

# Create vocabulary
vocab = GeneVocab(adata.var_names.tolist())

# Create model
model = TransformerModel(vocab_size=len(vocab))

# Now you're ready to train!
```

---

## Basic Concepts

### What is scGPT-mini?

scGPT-mini is an educational implementation of transformer models for single-cell RNA-seq analysis. It's designed to:

- **Teach**: Help you understand how transformers work with genomics data
- **Run anywhere**: Works on laptops with CPU only (no GPU required)
- **Scale down**: ~50-150K parameters vs millions in full models
- **Stay simple**: Minimal dependencies, clear code structure

### Key Differences from scGPT

| Feature | scGPT (Full) | scGPT-mini |
|---------|--------------|------------|
| Model size | ~2-5M params | ~50-150K params |
| Hardware | GPU required | CPU friendly |
| Memory | 8-16GB GPU | <4GB RAM |
| Training data | 33M cells | 1K-50K cells |
| Genes | ~19K vocab | ~5-10K vocab |
| Use case | Production | Education |

### Architecture Overview

```
Input: Gene IDs + Expression Values
    ↓
[Gene Embedding] + [Value Embedding]
    ↓
[Transformer Encoder Layers]
    ↓
[CLS Token] → Cell Embedding
[All Tokens] → Expression Prediction
```

---

## Data Preparation

### Understanding AnnData

scGPT-mini uses AnnData, the standard format for single-cell data:

```python
import scanpy as sc

adata = sc.datasets.pbmc3k()

# Key components:
# adata.X - Expression matrix (cells × genes)
# adata.obs - Cell metadata
# adata.var - Gene metadata
# adata.obsm - Cell embeddings
```

### Preprocessing Pipeline

#### Step 1: Quality Control

```python
from scgpt_mini.data import filter_genes, filter_cells

# Filter low-quality genes
adata = filter_genes(
    adata,
    min_cells=3,      # Gene must be in ≥3 cells
    min_counts=10,    # Gene must have ≥10 total counts
)

# Filter low-quality cells
adata = filter_cells(
    adata,
    min_genes=200,    # Cell must express ≥200 genes
    max_genes=10000,  # Cell must express ≤10K genes (filter doublets)
)
```

**Why?**
- Removes noise and low-quality data
- Reduces computational burden
- Improves model performance

---

#### Step 2: Normalization

```python
from scgpt_mini.data import normalize_total

# Normalize to 10,000 counts per cell
adata = normalize_total(adata, target_sum=1e4)
```

**Why?**
- Accounts for differences in sequencing depth
- Makes cells comparable
- Standard practice in scRNA-seq

---

#### Step 3: Log Transformation

```python
from scgpt_mini.data import log_transform

# Apply log(x + 1) transformation
adata = log_transform(adata)
```

**Why?**
- Stabilizes variance
- Reduces impact of outliers
- Helps with neural network training

---

#### Step 4: Feature Selection

```python
from scgpt_mini.data import select_hvg

# Select 500 highly variable genes
adata = select_hvg(adata, n_top_genes=500)
```

**Why?**
- Focuses on informative genes
- Reduces computational cost
- Removes uninformative/noisy genes

**How many genes?**
- Small datasets: 300-500 genes
- Medium datasets: 500-1000 genes
- Large datasets: 1000-2000 genes

---

#### All-in-One Preprocessing

```python
from scgpt_mini.data import preprocess_adata

adata = preprocess_adata(
    adata,
    filter_gene_by_counts=10,
    filter_cell_by_genes=200,
    normalize_total_target=1e4,
    log1p=True,
    subset_hvg=500,
    binning=False,  # Use continuous values
    inplace=False,  # Return new object
)
```

---

### Gene Tokenization

#### Creating a Vocabulary

```python
from scgpt_mini.tokenizer import GeneVocab

# From your genes
gene_list = adata.var_names.tolist()
vocab = GeneVocab(gene_list)

# Or load default vocabulary
vocab = GeneVocab.from_json("scgpt_mini/tokenizer/default_vocab.json")

print(f"Vocabulary size: {len(vocab)}")
# Output: Vocabulary size: 503 (500 genes + 3 special tokens)
```

**Special Tokens:**
- `<pad>` (ID=0): Padding for variable-length sequences
- `<cls>` (ID=1): Classification token (cell embedding)
- `<eoc>` (ID=2): End of cell marker

---

#### Tokenizing Cells

```python
from scgpt_mini.tokenizer import tokenize_batch

data_matrix = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
gene_names = adata.var_names.values

tokenized_data = tokenize_batch(
    data=data_matrix,
    gene_names=gene_names,
    vocab=vocab,
    append_cls=True,  # Prepend CLS token
    include_zero_genes=False,  # Only non-zero genes
)

# tokenized_data[i] = (gene_ids, expression_values) for cell i
```

---

### Creating DataLoaders

```python
from scgpt_mini.data import create_dataloader

# For pretraining (with MLM masking)
train_loader = create_dataloader(
    tokenized_data=tokenized_data,
    vocab=vocab,
    batch_size=32,
    max_len=1001,  # 1000 genes + 1 CLS token
    shuffle=True,
    apply_masking=True,  # Enable MLM
    mask_ratio=0.15,  # Mask 15% of genes
)

# For classification (no masking)
train_loader = create_dataloader(
    tokenized_data=tokenized_data,
    vocab=vocab,
    batch_size=32,
    apply_masking=False,  # No masking
    labels=cell_type_labels,  # Add labels
)
```

---

## Model Pretraining

### Understanding Masked Language Modeling (MLM)

MLM is a self-supervised learning technique:

1. **Mask** random gene expression values
2. **Predict** the masked values using context
3. Model learns **gene relationships** and **biological patterns**

```python
# Original: [0.5, 1.2, 0.0, 2.3, 0.8]
# Masked:   [0.5, MASK, 0.0, MASK, 0.8]
# Target:   predict 1.2 and 2.3
```

---

### Creating a Model

```python
from scgpt_mini.model import TransformerModel

model = TransformerModel(
    vocab_size=len(vocab),
    d_model=32,         # Embedding dimension
    nhead=2,            # Number of attention heads
    num_layers=2,       # Number of transformer layers
    d_hid=64,           # Hidden dimension in FFN
    dropout=0.1,        # Dropout rate
    value_mode="continuous",  # Use continuous expression values
)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")
```

**Configuration Guide:**

| Dataset Size | d_model | num_layers | nhead | Expected Params |
|--------------|---------|------------|-------|-----------------|
| <5K cells | 32 | 2 | 2 | ~50K |
| 5-20K cells | 64 | 2-3 | 2-4 | ~150K |
| >20K cells | 128 | 3-4 | 4 | ~500K |

---

### Training Setup

```python
import torch
from scgpt_mini.training import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
)

# Learning rate scheduler (optional)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
)

# Trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device=device,
    output_dir='./checkpoints',
    gradient_clip=1.0,
    scheduler=scheduler,
)
```

---

### Running Training

```python
# Train
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    eval_every=1,  # Evaluate every epoch
    save_every=5,  # Save checkpoint every 5 epochs
    early_stopping_patience=5,  # Stop if no improvement for 5 epochs
)

# Training history contains:
# - train_loss, val_loss
# - val_mse, val_mae, val_pearson
# - lr (if scheduler used)
```

---

### Monitoring Training

```python
import matplotlib.pyplot as plt

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history['val_pearson'])
plt.xlabel('Epoch')
plt.ylabel('Pearson R')
plt.title('Validation Correlation')

plt.tight_layout()
plt.show()
```

**What to look for:**
- **Decreasing loss**: Model is learning
- **Val loss < Train loss**: Underfitting (train longer or increase model size)
- **Train loss << Val loss**: Overfitting (use more data or regularization)
- **Pearson R > 0.5**: Good performance on MLM

---

### Saving and Loading

```python
# Save checkpoint
trainer.save_checkpoint(
    'best_model.pt',
    epoch=20,
    is_best=True,
)

# Load checkpoint
checkpoint = torch.load('best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Cell Embedding Generation

### Extracting Embeddings

```python
from scgpt_mini.tasks import extract_cell_embeddings

# Extract with CLS token (recommended)
embeddings = extract_cell_embeddings(
    model=model,
    adata=adata,
    vocab=vocab,
    batch_size=64,
    pool_strategy='cls',  # 'cls', 'mean', or 'max'
    device=device,
)

# Store in AnnData
adata.obsm['X_scgpt'] = embeddings
```

**Pooling Strategies:**

- **CLS token** (default): Uses the [CLS] token embedding. Works like BERT.
- **Mean pooling**: Average over all gene embeddings. Robust to outliers.
- **Max pooling**: Maximum across gene embeddings. Emphasizes highly expressed genes.

---

### Visualization with UMAP

```python
import umap

# Compute UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_coords = reducer.fit_transform(embeddings)

# Store
adata.obsm['X_umap_scgpt'] = umap_coords

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
for cell_type in adata.obs['cell_type'].unique():
    mask = adata.obs['cell_type'] == cell_type
    plt.scatter(
        umap_coords[mask, 0],
        umap_coords[mask, 1],
        label=cell_type,
        s=10,
        alpha=0.6,
    )
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend()
plt.title('scGPT Cell Embeddings')
plt.show()
```

---

### Evaluating Embedding Quality

```python
from scgpt_mini.tasks.embedding import (
    compute_silhouette_score,
    compute_ari,
    knn_accuracy,
)

# Silhouette score (cluster separation)
silhouette = compute_silhouette_score(
    embeddings,
    labels=adata.obs['cell_type'].values,
)
print(f"Silhouette score: {silhouette:.3f}")
# > 0.3 = good, > 0.5 = excellent

# Adjusted Rand Index (clustering agreement)
ari = compute_ari(
    embeddings,
    labels=adata.obs['cell_type'].values,
    n_clusters=len(adata.obs['cell_type'].unique()),
)
print(f"ARI: {ari:.3f}")
# > 0.5 = good, > 0.7 = excellent

# k-NN accuracy (label transfer)
knn_acc = knn_accuracy(
    embeddings,
    labels=adata.obs['cell_type'].values,
    k=5,
)
print(f"k-NN accuracy: {knn_acc:.3f}")
# Should outperform random baseline
```

---

### Comparing to PCA

```python
from sklearn.decomposition import PCA

# PCA embeddings
pca = PCA(n_components=32)
pca_embeddings = pca.fit_transform(adata.X.toarray())

# Compare metrics
print("scGPT-mini vs PCA:")
print(f"  Silhouette: {silhouette_scgpt:.3f} vs {silhouette_pca:.3f}")
print(f"  k-NN Acc:   {knn_scgpt:.3f} vs {knn_pca:.3f}")
```

---

## Cell Type Annotation

### Understanding Transfer Learning

Transfer learning leverages pretrained knowledge:

**Phase 1: Pretraining** (unsupervised)
- Task: Predict masked gene expression
- Data: All cells (no labels needed)
- Goal: Learn general gene relationships

**Phase 2: Fine-tuning** (supervised)
- Task: Classify cell types
- Data: Labeled cells only
- Goal: Adapt model to specific task

---

### Preparing Labeled Data

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode labels
label_encoder = LabelEncoder()
cell_type_ids = label_encoder.fit_transform(adata.obs['cell_type'])
n_classes = len(label_encoder.classes_)

# Split data
train_idx, test_idx = train_test_split(
    range(len(cell_type_ids)),
    test_size=0.2,
    stratify=cell_type_ids,
    random_state=42,
)

train_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.25,
    stratify=cell_type_ids[train_idx],
    random_state=42,
)
```

---

### Fine-Tuning Strategies

#### Strategy 1: Frozen Encoder (Recommended for <10K cells)

```python
from scgpt_mini.tasks.annotation import freeze_encoder

# Add classification head
model.n_classes = n_classes

# Freeze encoder
freeze_encoder(model)

# Train only classification head
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,  # Higher LR for head only
)

# Train for fewer epochs
history = finetune_for_annotation(
    model,
    train_loader,
    val_loader,
    epochs=10,
    learning_rate=1e-3,
    freeze_encoder=True,
)
```

**Advantages:**
- Faster training
- Less data needed
- Prevents overfitting

---

#### Strategy 2: Full Fine-Tuning (Recommended for >10K cells)

```python
from scgpt_mini.tasks.annotation import unfreeze_encoder

# Unfreeze all parameters
unfreeze_encoder(model)

# Train entire model
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,  # Lower LR for full model
)

history = finetune_for_annotation(
    model,
    train_loader,
    val_loader,
    epochs=20,
    learning_rate=1e-4,
    freeze_encoder=False,
)
```

**Advantages:**
- Better final performance
- Adapts representations to task

---

### Making Predictions

```python
from scgpt_mini.tasks.annotation import predict_cell_types

# Predict on test set
predictions, confidences = predict_cell_types(
    model=model,
    dataloader=test_loader,
    device=device,
)

# Convert to labels
predicted_labels = label_encoder.inverse_transform(predictions)
true_labels = label_encoder.inverse_transform(cell_type_ids[test_idx])
```

---

### Evaluating Performance

```python
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# Accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)

print(f"Accuracy: {accuracy:.3f}")
print(f"Balanced Accuracy: {balanced_acc:.3f}")

# Per-class metrics
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels))

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
```

---

### Analyzing Confidence

```python
import numpy as np

# Separate correct and incorrect predictions
correct_mask = predictions == cell_type_ids[test_idx]
correct_conf = confidences[correct_mask]
incorrect_conf = confidences[~correct_mask]

print(f"Mean confidence (correct): {correct_conf.mean():.3f}")
print(f"Mean confidence (incorrect): {incorrect_conf.mean():.3f}")

# Flag low-confidence predictions
low_confidence = confidences < 0.5
print(f"Low-confidence predictions: {low_confidence.sum()} / {len(confidences)}")
```

---

### Annotating New Data

```python
from scgpt_mini.tasks.annotation import annotate_adata

# Annotate entire dataset
adata = annotate_adata(
    model=model,
    adata=adata,
    vocab=vocab,
    label_encoder=label_encoder,
    batch_size=64,
    device=device,
    confidence_threshold=0.5,
)

# Adds to adata.obs:
# - predicted_cell_type
# - prediction_confidence
# - low_confidence (boolean flag)
```

---

## Advanced Topics

### Custom Datasets

```python
# Load your own data
adata = sc.read_h5ad('your_data.h5ad')

# Ensure gene names are in adata.var_names
# Ensure raw or normalized counts in adata.X

# Preprocess
adata = preprocess_adata(adata, subset_hvg=500)

# Rest is the same!
```

---

### Batch Correction

scGPT-mini doesn't include advanced batch correction, but you can:

```python
# Option 1: Preprocess batches separately
for batch in adata.obs['batch'].unique():
    mask = adata.obs['batch'] == batch
    adata[mask] = preprocess_adata(adata[mask])

# Option 2: Use Scanpy's batch correction
import scanpy as sc
sc.pp.combat(adata, key='batch')

# Then proceed with scGPT-mini
```

---

### Multi-Task Learning

Train MLM and classification simultaneously:

```python
from scgpt_mini.training.losses import combined_loss

# Model with classification head
model = TransformerModel(
    vocab_size=len(vocab),
    n_classes=n_classes,
    ...
)

# Use combined loss
loss = combined_loss(
    expr_pred=output['expr_pred'],
    expr_target=targets,
    expr_mask=mask,
    cls_pred=output['cls_pred'],
    cls_target=labels,
    mlm_weight=1.0,
    cls_weight=0.5,
)
```

---

### Hyperparameter Tuning

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'd_model': [32, 64],
    'num_layers': [2, 3],
    'learning_rate': [1e-4, 5e-4],
    'mask_ratio': [0.10, 0.15, 0.20],
}

best_score = 0
best_params = None

for params in ParameterGrid(param_grid):
    model = TransformerModel(
        vocab_size=len(vocab),
        d_model=params['d_model'],
        num_layers=params['num_layers'],
    )

    # Train and evaluate
    history = trainer.train(...)
    score = history['val_pearson'][-1]

    if score > best_score:
        best_score = score
        best_params = params

print(f"Best parameters: {best_params}")
```

---

## Best Practices

### Data Quality

✅ **DO:**
- Filter low-quality cells and genes
- Normalize and log-transform
- Use 300-1000 HVGs depending on data size
- Check for batch effects

❌ **DON'T:**
- Use raw counts without normalization
- Include too many genes (increases computation)
- Mix different preprocessing pipelines

---

### Model Selection

✅ **DO:**
- Start small (d_model=32, num_layers=2)
- Increase size if underfitting
- Use validation set to tune hyperparameters
- Save best checkpoint based on validation loss

❌ **DON'T:**
- Make model too large for your data
- Overtrain (watch for overfitting)
- Ignore validation metrics

---

### Training

✅ **DO:**
- Use 80/20 or 70/15/15 train/val/test splits
- Monitor training curves
- Use early stopping
- Save checkpoints regularly
- Use gradient clipping (1.0)

❌ **DON'T:**
- Train on test set
- Use too high learning rate (>1e-3)
- Skip validation
- Train for too long without improvement

---

### Evaluation

✅ **DO:**
- Use multiple metrics (accuracy, silhouette, ARI)
- Compare to baselines (PCA, logistic regression)
- Check per-class performance
- Analyze errors and edge cases

❌ **DON'T:**
- Rely only on overall accuracy
- Ignore class imbalance
- Trust low-confidence predictions blindly

---

## Troubleshooting

See [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) for detailed solutions.

### Common Issues

**"CUDA out of memory"**
- Reduce batch_size
- Use smaller model (d_model=32)
- Use CPU instead: `device='cpu'`

**"Training loss not decreasing"**
- Check learning rate (try 1e-4)
- Verify data preprocessing
- Check for NaN values
- Increase model capacity

**"Poor embedding quality"**
- Train longer (20-50 epochs)
- Use more data (>5K cells)
- Increase HVGs (500-1000)
- Check preprocessing steps

**"Low classification accuracy"**
- Use more labeled data
- Try full fine-tuning instead of frozen
- Check class balance
- Verify labels are correct

---

## FAQs

**Q: Do I need a GPU?**
A: No! scGPT-mini is designed for CPU-only training on laptops.

**Q: How much data do I need?**
A: Minimum 1K cells for pretraining, 500 labeled cells for annotation. More is better.

**Q: Can I use this for production?**
A: scGPT-mini is educational. For production, consider the full scGPT or other tools.

**Q: How long does training take?**
A: On CPU: ~10-30 minutes for 10K cells, 20 epochs. Depends on your hardware.

**Q: Can I use my own gene vocabulary?**
A: Yes! Create custom vocab from your genes: `GeneVocab(adata.var_names.tolist())`

**Q: What about other species (mouse, etc.)?**
A: Works fine! Just use gene names from your species.

**Q: Can I transfer between datasets?**
A: Yes! Pretrain on large dataset, fine-tune on smaller labeled dataset.

**Q: Should I use binned or continuous values?**
A: Continuous is simpler and works well for most cases.

---

## Next Steps

1. **Run tutorials**: Complete all 4 notebooks in `notebooks/`
2. **Try examples**: Run scripts in `examples/`
3. **Apply to your data**: Follow this guide with your own datasets
4. **Read the paper**: Understand the original scGPT
5. **Explore code**: scGPT-mini is designed to be readable!

---

## Resources

- **GitHub**: https://github.com/yourusername/scgpt-mini
- **Original scGPT**: https://github.com/bowang-lab/scGPT
- **Scanpy**: https://scanpy.readthedocs.io/
- **PyTorch**: https://pytorch.org/docs/

---

## Getting Help

- Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
- Read [API_REFERENCE.md](API_REFERENCE.md)
- Open an issue on GitHub
- Read tutorial notebooks

---

**Happy analyzing! 🧬🤖**
