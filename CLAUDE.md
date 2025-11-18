# scGPT-mini: Implementation Plan

## Project Overview

**scGPT-mini** is a miniature, educational version of the scGPT foundation model for single-cell RNA-seq analysis. This implementation focuses on core transformer architecture and essential features while maintaining laptop-friendly resource requirements.

### Original scGPT
- **Repository**: https://github.com/bowang-lab/scGPT
- **Paper**: "scGPT: Foundation Model for Single-Cell Multi-omics" (Cui et al., 2023)
- **Architecture**: Transformer-based foundation model pretrained on 33M+ human cells
- **Size**: Default 128-dim embeddings, 4 layers, 4 heads, ~1200 genes

### scGPT-mini Goals
- **Primary Use**: Learning and education about transformer models for genomics
- **Target Hardware**: 16GB RAM, CPU-only (no GPU required)
- **Dataset Size**: 1,000-10,000 cells
- **Model Size**: ~10-50x smaller than full scGPT

---

## Key Design Decisions

### What We're Keeping
1. **Core Transformer Architecture**
   - Gene tokenization and vocabulary system
   - Self-attention mechanism for gene relationships
   - Masked Language Modeling (MLM) pretraining objective
   - Cell embedding generation via CLS token

2. **Essential Features** (Priority Order)
   - ✅ Core pretraining (MLM on gene expression)
   - ✅ Cell embedding generation for visualization
   - ✅ Cell type annotation/classification

3. **Data Processing Pipeline**
   - Gene filtering and normalization
   - Highly Variable Gene (HVG) selection
   - Expression value binning
   - Tokenization and masking

### What We're Removing/Simplifying
1. **Removed Components**
   - ❌ Flash-attention (GPU-dependent, complex installation)
   - ❌ Domain-Specific Batch Normalization (batch correction not needed)
   - ❌ Adversarial discriminator (batch correction related)
   - ❌ Multi-omic model (out of scope)
   - ❌ Gene Regulatory Network (GRN) inference (advanced task)
   - ❌ Perturbation prediction (advanced task)
   - ❌ Weights & Biases integration (optional, can add later)

2. **Simplifications**
   - Use standard PyTorch attention (no custom kernels)
   - Minimal dependencies (remove scvi-tools, torchtext, etc.)
   - Simplified vocabulary management (no torchtext.vocab)
   - Single-GPU/CPU training only (no distributed training)

### Technical Specifications

| Component | scGPT (default) | scGPT-mini |
|-----------|-----------------|------------|
| Embedding dimension | 128 | 32 |
| Transformer layers | 4 | 2-3 |
| Attention heads | 4 | 2 |
| Hidden dimension | 512 | 64 |
| Max genes (HVGs) | 1200 | 500-1000 |
| Gene vocabulary | ~19,000 | ~5,000-10,000 |
| Expression bins | 51 | 51 |
| Pretrain data | 33M cells | 10K-50K cells |
| Model parameters | ~2-5M | ~50-150K |
| Memory (inference) | ~2-4GB GPU | <1GB RAM |
| Memory (training) | ~8-16GB GPU | <4GB RAM |

---

## Dependencies

### Core Dependencies (Minimal)
```
python >= 3.8, <=3.12    # Avoid 3.13+ due to potential scanpy conflicts
torch >= 1.13.0          # PyTorch (CPU version)
numpy >= 1.20.0
pandas >= 1.3.0
scikit-learn >= 1.0.0    # For metrics, train/test split
```

### Genomics Dependencies
```
scanpy >= 1.9.0          # Single-cell analysis (includes anndata)
anndata >= 0.8.0         # Annotated data structures
```

### Visualization (Optional)
```
matplotlib >= 3.5.0
seaborn >= 0.11.0
umap-learn >= 0.5.0      # For dimensionality reduction
```

### Development (Optional)
```
pytest >= 7.0.0          # Testing
jupyter >= 1.0.0         # Notebooks
tqdm >= 4.60.0          # Progress bars
```

---

## Project Structure

```
scgpt-mini/
├── scgpt_mini/                    # Main package
│   ├── __init__.py
│   ├── tokenizer/                 # Gene tokenization
│   │   ├── __init__.py
│   │   ├── vocabulary.py          # GeneVocab class
│   │   └── gene_tokenizer.py      # Tokenization functions
│   │
│   ├── model/                     # Model architecture
│   │   ├── __init__.py
│   │   ├── encoder.py             # Gene & value encoders
│   │   ├── decoder.py             # Expression & classification decoders
│   │   └── transformer.py         # Main TransformerModel
│   │
│   ├── data/                      # Data processing
│   │   ├── __init__.py
│   │   ├── preprocess.py          # Preprocessing functions
│   │   └── collator.py            # Data collation & masking
│   │
│   ├── training/                  # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop
│   │   └── metrics.py             # Evaluation metrics
│   │
│   ├── tasks/                     # Downstream tasks
│   │   ├── __init__.py
│   │   ├── embedding.py           # Cell embedding generation
│   │   └── annotation.py          # Cell type annotation
│   │
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── logging.py             # Simple logging
│       └── visualization.py       # Plotting functions
│
├── examples/                      # Usage examples
│   ├── data/                      # Small example datasets
│   ├── 01_pretraining.py          # MLM pretraining example
│   ├── 02_embedding.py            # Cell embedding example
│   └── 03_annotation.py           # Cell type annotation example
│
├── notebooks/                     # Jupyter tutorials
│   ├── Tutorial_01_Preprocessing.ipynb
│   ├── Tutorial_02_Pretraining.ipynb
│   ├── Tutorial_03_Embeddings.ipynb
│   └── Tutorial_04_Annotation.ipynb
│
├── tests/                         # Unit tests
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_training.py
│
├── CLAUDE.md                      # This file
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
└── setup.py                       # Package installation
```

---

## Implementation Phases

### Phase Dependencies Overview

```
Phase 1 (Foundation) → Phase 2 (Model) → Phase 3 (Training)
                                              ↓
                                         ┌────┴────┐
                                         ↓         ↓
                                    Phase 4    Phase 5
                                   (Embedding) (Annotation)
                                         ↓         ↓
                                         └────┬────┘
                                              ↓
                                         Phase 6
                                      (Integration)
                                              ↓
                                         Phase 7
                                       (Package)
```

**Sequential Phases**: 1 → 2 → 3 → 6 → 7
**Concurrent Phases**: 4 and 5 (both depend on Phase 3, can run in parallel)

---

## PHASE 1: Foundation & Data Processing
**Status**: Must complete first (prerequisite for all others)
**Estimated Time**: 1-2 sessions
**Dependencies**: None

### Objectives
Build the foundational data processing pipeline including gene tokenization, vocabulary management, and data preprocessing.

### Tasks

#### 1.1 Gene Vocabulary System
**File**: `scgpt_mini/tokenizer/vocabulary.py`

- [ ] Implement `GeneVocab` class
  - Store gene name → ID mapping (dict)
  - Store ID → gene name mapping (reverse dict)
  - Special tokens: `<pad>` (0), `<cls>` (1), `<eoc>` (2)
  - Methods: `__len__()`, `__getitem__()`, `get_id()`, `get_gene()`

- [ ] Implement vocabulary creation
  - `create_vocab_from_genes(gene_list)` - Build vocab from gene names
  - Support for custom ordering (e.g., by frequency)

- [ ] Implement vocabulary persistence
  - `save_vocab(filepath)` - Save to JSON
  - `load_vocab(filepath)` - Load from JSON

- [ ] Create default vocabulary
  - Extract top 5,000-10,000 genes from scGPT's default vocab
  - Save as `default_vocab.json` in package data

**Success Criteria**:
- Can create vocab from list of gene names
- Can map gene names ↔ IDs bidirectionally
- Can save/load vocab from JSON
- Special tokens work correctly

#### 1.2 Gene Tokenization
**File**: `scgpt_mini/tokenizer/gene_tokenizer.py`

- [ ] Implement `tokenize_genes()`
  - Input: List of gene names, GeneVocab
  - Output: List of gene IDs
  - Handle unknown genes (skip or map to UNK token)

- [ ] Implement `tokenize_cell()`
  - Input: Gene names + expression values for one cell, GeneVocab
  - Output: Gene IDs + expression values (only non-zero)
  - Prepend CLS token

- [ ] Implement `pad_sequences()`
  - Pad/truncate to max_length
  - Create attention masks
  - Handle both gene IDs and expression values

- [ ] Implement `create_mask()`
  - Random masking for MLM (default 15% mask rate)
  - Return mask positions and masked values
  - Optionally save ground truth for evaluation

**Success Criteria**:
- Can tokenize individual cells with gene expression
- Padding works correctly with attention masks
- Masking creates appropriate MLM training targets

#### 1.3 Data Preprocessing
**File**: `scgpt_mini/data/preprocess.py`

- [ ] Implement `filter_genes()`
  - Remove genes expressed in < N cells (default: 3)
  - Remove genes with < M total counts (default: 10)

- [ ] Implement `filter_cells()`
  - Remove cells with < K genes (default: 200)
  - Remove cells with > L genes (default: 10000)
  - Optional: Filter by mitochondrial percentage

- [ ] Implement `normalize_total()`
  - Normalize each cell to target sum (default: 10,000)
  - Equivalent to scanpy's `sc.pp.normalize_total()`

- [ ] Implement `log_transform()`
  - Apply log1p transformation: log(x + 1)

- [ ] Implement `select_hvg()`
  - Select top N highly variable genes (default: 500-1000)
  - Simplified version (by variance or dispersion)
  - Return selected gene names

- [ ] Implement `bin_expression()`
  - Bin continuous expression into N bins (default: 51)
  - Use equal-width or quantile binning
  - Special bin for zero values (bin 0)

- [ ] Implement `preprocess_adata()`
  - Combined pipeline function
  - Takes AnnData object, returns preprocessed AnnData
  - Configurable parameters

**Success Criteria**:
- Can load and preprocess small scRNA-seq dataset (e.g., PBMC 3k)
- Output has normalized, log-transformed, and binned values
- HVG selection reduces to target gene count
- Preprocessing is reproducible

#### 1.4 Data Collator
**File**: `scgpt_mini/data/collator.py`

- [ ] Implement `DataCollator` class
  - Initialize with vocab, max_length, mask_ratio
  - `__call__()` method for batching

- [ ] Implement batch collation
  - Input: List of (gene_ids, expression_values) tuples
  - Output: Batched tensors with padding
  - Create attention masks

- [ ] Implement MLM masking
  - Apply masking to expression values
  - Create masked input + labels for MLM
  - Handle special tokens (don't mask CLS, PAD)

- [ ] Support for classification labels (optional)
  - Include cell type labels if available
  - Create label tensor for batch

**Success Criteria**:
- Can collate variable-length sequences into batches
- Masking creates valid MLM targets
- Works with PyTorch DataLoader

### Deliverables
- `scgpt_mini/tokenizer/` module (vocabulary.py, gene_tokenizer.py)
- `scgpt_mini/data/` module (preprocess.py, collator.py)
- `default_vocab.json` with 5-10K genes
- Unit tests for tokenization and preprocessing
- Example preprocessing script

### Validation
- Process PBMC 3k dataset successfully
- Tokenize and batch 1000 cells without errors
- Memory usage < 2GB for preprocessing

---

## PHASE 2: Core Model Architecture
**Status**: Sequential (requires Phase 1)
**Estimated Time**: 1-2 sessions
**Dependencies**: Phase 1 complete

### Objectives
Implement the transformer model architecture with gene/value encoders and expression decoder.

### Tasks

#### 2.1 Gene and Value Encoders
**File**: `scgpt_mini/model/encoder.py`

- [ ] Implement `GeneEncoder`
  - Embedding layer: vocab_size → d_model
  - Layer normalization
  - Dropout

- [ ] Implement `ContinuousValueEncoder`
  - Linear layer: 1 → d_model
  - Activation (ReLU or GELU)
  - Dropout
  - Input: log-normalized expression values

- [ ] Implement `CategoryValueEncoder` (optional)
  - Embedding layer: n_bins → d_model
  - For binned expression values
  - Alternative to continuous encoder

- [ ] Implement embedding combination
  - Add or multiply gene + value embeddings
  - Test both strategies

**Success Criteria**:
- Gene encoder creates d_model dimensional embeddings
- Value encoder handles continuous expression values
- Combined embeddings have correct shape: (batch, seq_len, d_model)

#### 2.2 Expression Decoder
**File**: `scgpt_mini/model/decoder.py`

- [ ] Implement `ExpressionDecoder`
  - Linear layer: d_model → 1 (continuous values)
  - OR: Linear layer: d_model → n_bins (classification)
  - Layer normalization
  - Activation function

- [ ] Implement `ClassificationDecoder`
  - Multi-layer MLP for cell type classification
  - Input: CLS token embedding (d_model)
  - Hidden layers with dropout
  - Output: n_cell_types logits

- [ ] Support both decoder modes
  - Continuous: Predict expression value directly
  - Binned: Predict expression bin (classification)

**Success Criteria**:
- Expression decoder outputs correct shape for MLM loss
- Classification decoder works with CLS token
- Both decoders train without numerical issues

#### 2.3 Transformer Model
**File**: `scgpt_mini/model/transformer.py`

- [ ] Implement `TransformerModel` class
  - Initialization with hyperparameters
  - Register all sub-modules

- [ ] Implement model components
  - Gene encoder (from 2.1)
  - Value encoder (from 2.1)
  - Positional encoding (optional, test with/without)
  - Transformer encoder (PyTorch nn.TransformerEncoder)
  - Expression decoder (from 2.2)
  - Classification decoder (from 2.2, optional)

- [ ] Implement `forward()` method
  - Input: gene_ids, values, attention_mask, (optional) value_labels
  - Encode genes and values
  - Combine embeddings
  - Apply transformer encoder
  - Decode for MLM and/or classification
  - Return predictions dict

- [ ] Implement configuration
  - Hyperparameters: d_model, nhead, num_layers, d_hid, dropout
  - Save/load model config

- [ ] Implement checkpoint management
  - `save_checkpoint(path)` - Save model + config
  - `load_checkpoint(path)` - Load model + config
  - Save optimizer state (optional)

**Success Criteria**:
- Forward pass works with dummy data
- Model outputs correct shapes for MLM and classification
- Can save and load checkpoints
- Model fits in <500MB memory for inference

### Configuration

**Recommended Hyperparameters** (scGPT-mini):
```python
MODEL_CONFIG = {
    "d_model": 32,           # Embedding dimension
    "nhead": 2,              # Number of attention heads
    "num_layers": 2,         # Number of transformer layers
    "d_hid": 64,             # Hidden dimension in FFN
    "dropout": 0.1,          # Dropout rate
    "n_bins": 51,            # Number of expression bins
    "max_seq_len": 1001,     # Max genes + 1 for CLS
    "activation": "gelu",    # Activation function
    "norm_first": True,      # Pre-LN (more stable)
}
```

### Deliverables
- `scgpt_mini/model/` module (encoder.py, decoder.py, transformer.py)
- Model configuration file (model_config.json)
- Unit tests for model components
- Example model creation script
- Model parameter count script (<500K params target)

### Validation
- Forward pass on batch of 32 cells completes in <1 second (CPU)
- Backward pass works without NaN/Inf
- Model memory footprint < 500MB
- All shapes match expected dimensions

---

## PHASE 3: Training Infrastructure
**Status**: Sequential (requires Phase 2)
**Estimated Time**: 1-2 sessions
**Dependencies**: Phases 1 and 2 complete

### Objectives
Build the training loop for MLM pretraining with evaluation and checkpointing.

### Tasks

#### 3.1 Loss Functions
**File**: `scgpt_mini/training/losses.py`

- [ ] Implement `masked_mse_loss()`
  - MSE loss only on masked positions
  - Input: predictions, targets, mask
  - Ignore padding tokens

- [ ] Implement `masked_cross_entropy_loss()`
  - For binned expression values
  - Cross-entropy on masked positions

- [ ] Implement `classification_loss()`
  - Cross-entropy for cell type classification
  - Input: CLS token predictions, labels

- [ ] Implement combined loss
  - Weighted combination of MLM + classification
  - Configurable weights

**Success Criteria**:
- Losses compute correctly with masking
- No NaN/Inf during training
- Loss values are reasonable (decrease over time)

#### 3.2 Metrics
**File**: `scgpt_mini/training/metrics.py`

- [ ] Implement MLM metrics
  - `masked_mse` - MSE on masked positions
  - `masked_mae` - MAE on masked positions
  - `masked_pearson` - Correlation on masked positions

- [ ] Implement classification metrics
  - `accuracy` - Overall accuracy
  - `balanced_accuracy` - For imbalanced classes
  - `f1_score` - Macro F1

- [ ] Implement evaluation function
  - `evaluate_mlm()` - Run model on validation set
  - Compute all metrics
  - Return metrics dict

**Success Criteria**:
- Metrics compute correctly on validation data
- Can track metrics over training epochs
- Metrics are interpretable

#### 3.3 Training Loop
**File**: `scgpt_mini/training/trainer.py`

- [ ] Implement `Trainer` class
  - Initialize with model, optimizer, config
  - Store hyperparameters

- [ ] Implement `train_epoch()`
  - Iterate over training DataLoader
  - Forward pass → loss → backward → optimizer step
  - Track batch losses
  - Progress bar (tqdm)

- [ ] Implement `validate_epoch()`
  - Iterate over validation DataLoader
  - Forward pass (no gradients)
  - Compute metrics
  - Return validation metrics

- [ ] Implement `train()`
  - Main training loop over epochs
  - Call train_epoch() and validate_epoch()
  - Learning rate scheduling (optional)
  - Early stopping (optional)
  - Checkpoint saving
  - Logging (simple print or file)

- [ ] Implement checkpoint management
  - Save best model based on validation loss
  - Save periodic checkpoints (every N epochs)
  - Resume training from checkpoint

- [ ] Implement simple logging
  - Log to stdout and/or file
  - Track: epoch, train_loss, val_loss, metrics
  - No external dependencies (no wandb)

**Success Criteria**:
- Can train model for multiple epochs without errors
- Training and validation losses decrease
- Checkpoints save and load correctly
- Training completes on 10K cells in reasonable time (<30 min)

### Training Configuration

**Recommended Training Config**:
```python
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "epochs": 20,
    "warmup_steps": 100,
    "mask_ratio": 0.15,          # MLM masking ratio
    "mlm_weight": 1.0,           # Loss weight
    "cls_weight": 0.5,           # Classification loss weight (if using)
    "eval_every": 1,             # Evaluate every N epochs
    "save_every": 5,             # Save checkpoint every N epochs
    "gradient_clip": 1.0,        # Gradient clipping value
}
```

### Deliverables
- `scgpt_mini/training/` module (losses.py, metrics.py, trainer.py)
- Training configuration file (training_config.json)
- Simple logging utility
- Example training script
- Unit tests for training loop

### Validation
- Can complete training on 1K cells in <5 minutes
- Can complete training on 10K cells in <30 minutes
- Validation metrics improve over training
- Memory usage < 8GB during training
- Training is reproducible with fixed random seed

---

## PHASE 4: Cell Embedding Generation
**Status**: Concurrent with Phase 5 (requires Phase 3)
**Estimated Time**: 1 session
**Dependencies**: Phases 1, 2, 3 complete

### Objectives
Implement cell embedding extraction and visualization utilities.

### Tasks

#### 4.1 Embedding Extraction
**File**: `scgpt_mini/tasks/embedding.py`

- [ ] Implement `extract_cell_embeddings()`
  - Input: Pretrained model, AnnData, device
  - Forward pass through model
  - Extract CLS token embeddings
  - Return: (n_cells, d_model) array

- [ ] Support different embedding modes
  - CLS token (default)
  - Mean pooling over all tokens
  - Max pooling over all tokens

- [ ] Implement batch processing
  - Process large datasets in batches
  - Track progress with tqdm
  - Efficient memory usage

- [ ] Save embeddings to AnnData
  - Store in `adata.obsm['X_scgpt']`
  - Compatible with scanpy workflows

**Success Criteria**:
- Can extract embeddings for 10K cells in <2 minutes
- Embeddings have correct shape (n_cells, d_model)
- Memory efficient (streaming, no OOM)

#### 4.2 Visualization Utilities
**File**: `scgpt_mini/utils/visualization.py`

- [ ] Implement `plot_umap()`
  - Compute UMAP from embeddings
  - Color by cell type, batch, etc.
  - Save to file

- [ ] Implement `plot_tsne()` (optional)
  - Alternative to UMAP
  - Similar interface

- [ ] Implement `plot_embedding_metrics()`
  - Silhouette score by cell type
  - Neighborhood preservation
  - Batch mixing (if applicable)

- [ ] Integration with scanpy
  - Use scanpy's plotting functions
  - Store embeddings in AnnData.obsm

**Success Criteria**:
- Can visualize embeddings with UMAP
- Plots are clear and informative
- Compatible with scanpy visualization

#### 4.3 Embedding Quality Metrics
**File**: `scgpt_mini/tasks/embedding.py`

- [ ] Implement `compute_silhouette_score()`
  - Measure cluster separation by cell type
  - Higher is better

- [ ] Implement `compute_ari()` (Adjusted Rand Index)
  - Compare clustering to ground truth
  - Cluster embeddings with k-means or leiden

- [ ] Implement `knn_accuracy()`
  - k-NN classification on embeddings
  - Measure label transfer quality

- [ ] Implement `embedding_eval_report()`
  - Compute all metrics
  - Print formatted report
  - Return metrics dict

**Success Criteria**:
- Metrics compute correctly on test data
- Can compare different model checkpoints
- Metrics align with expected quality

### Deliverables
- `scgpt_mini/tasks/embedding.py` - Embedding extraction
- `scgpt_mini/utils/visualization.py` - Plotting utilities
- Example embedding extraction script
- Tutorial notebook for visualization
- Unit tests for embedding functions

### Validation
- Embeddings separate cell types in UMAP
- Silhouette score > 0.3 for well-separated cell types
- k-NN accuracy > random baseline
- Visualization completes in <1 minute for 10K cells

---

## PHASE 5: Cell Type Annotation
**Status**: Concurrent with Phase 4 (requires Phase 3)
**Estimated Time**: 1 session
**Dependencies**: Phases 1, 2, 3 complete

### Objectives
Implement fine-tuning for cell type annotation and inference pipeline.

### Tasks

#### 5.1 Fine-tuning for Classification
**File**: `scgpt_mini/tasks/annotation.py`

- [ ] Implement `create_classification_dataset()`
  - Input: AnnData with cell type labels
  - Output: Dataset with labels
  - Train/val/test split

- [ ] Implement `finetune_for_annotation()`
  - Load pretrained model
  - Add/replace classification head
  - Fine-tune on labeled data
  - Use classification loss
  - Save fine-tuned model

- [ ] Implement layer freezing options
  - Freeze encoder, train only classifier
  - Full fine-tuning
  - Gradual unfreezing

- [ ] Implement data augmentation (optional)
  - Random masking during fine-tuning
  - Expression value noise

**Success Criteria**:
- Can fine-tune on labeled dataset
- Fine-tuning improves classification accuracy
- Converges in <10 epochs

#### 5.2 Cell Type Prediction
**File**: `scgpt_mini/tasks/annotation.py`

- [ ] Implement `predict_cell_types()`
  - Input: Fine-tuned model, unlabeled AnnData
  - Forward pass through model
  - Extract CLS predictions
  - Return predicted cell types + confidence scores

- [ ] Implement batch inference
  - Process large datasets efficiently
  - Return predictions as array or add to AnnData

- [ ] Implement confidence thresholding
  - Flag low-confidence predictions
  - Return "unknown" for uncertain cells

**Success Criteria**:
- Predictions work on unseen data
- Can process 10K cells in <1 minute
- Confidence scores are calibrated

#### 5.3 Annotation Metrics
**File**: `scgpt_mini/training/metrics.py`

- [ ] Implement classification evaluation
  - Accuracy, balanced accuracy
  - Per-class precision/recall/F1
  - Confusion matrix

- [ ] Implement `annotation_eval_report()`
  - Compute all classification metrics
  - Print formatted report
  - Plot confusion matrix

- [ ] Implement cross-validation helper
  - K-fold CV for small datasets
  - Return average metrics

**Success Criteria**:
- Metrics compute correctly
- Can evaluate model on test set
- Reports are clear and interpretable

### Deliverables
- `scgpt_mini/tasks/annotation.py` - Fine-tuning and prediction
- Example annotation script
- Tutorial notebook for cell type annotation
- Fine-tuned model checkpoint (optional)
- Unit tests for annotation functions

### Validation
- Accuracy > 80% on well-separated cell types (e.g., PBMC major types)
- Fine-tuning completes in <10 minutes on 5K labeled cells
- Predictions generalize to held-out test set
- Confusion matrix shows interpretable errors

---

## PHASE 6: Integration & Testing
**Status**: Sequential (requires Phases 3, 4, 5)
**Estimated Time**: 1 session
**Dependencies**: Phases 1-5 complete

### Objectives
Create end-to-end examples, comprehensive tests, and documentation.

### Tasks

#### 6.1 End-to-End Examples
**Directory**: `examples/`

- [ ] Create `01_pretraining.py`
  - Download/load example dataset (PBMC 3k)
  - Preprocess data
  - Create vocabulary
  - Train model with MLM
  - Save checkpoints
  - Visualize training curves

- [ ] Create `02_embedding.py`
  - Load pretrained model
  - Extract cell embeddings
  - Compute UMAP
  - Visualize by cell type
  - Compute embedding metrics

- [ ] Create `03_annotation.py`
  - Load pretrained model
  - Fine-tune for cell type annotation
  - Evaluate on test set
  - Predict on unlabeled cells
  - Visualize results

- [ ] Add example data loader
  - Download PBMC 3k from scanpy
  - Or include tiny synthetic dataset
  - Document data format requirements

**Success Criteria**:
- All examples run without errors
- Examples complete in reasonable time (<30 min total)
- Results are reproducible
- Code is well-commented

#### 6.2 Tutorial Notebooks
**Directory**: `notebooks/`

- [ ] Create `Tutorial_01_Preprocessing.ipynb`
  - Load scRNA-seq data
  - Explain preprocessing steps
  - Visualize QC metrics
  - Show tokenization process

- [ ] Create `Tutorial_02_Pretraining.ipynb`
  - Explain MLM objective
  - Train small model
  - Monitor training
  - Interpret results

- [ ] Create `Tutorial_03_Embeddings.ipynb`
  - Extract embeddings
  - Visualize with UMAP/t-SNE
  - Compare to PCA
  - Evaluate embedding quality

- [ ] Create `Tutorial_04_Annotation.ipynb`
  - Fine-tune for classification
  - Evaluate predictions
  - Compare to baseline methods
  - Interpret errors

**Success Criteria**:
- Notebooks run top-to-bottom without errors
- Explanations are clear and educational
- Visualizations are informative
- Suitable for beginners

#### 6.3 Unit Tests
**Directory**: `tests/`

- [ ] Create `test_tokenizer.py`
  - Test vocab creation
  - Test tokenization
  - Test padding/masking

- [ ] Create `test_preprocessing.py`
  - Test filtering
  - Test normalization
  - Test HVG selection
  - Test binning

- [ ] Create `test_model.py`
  - Test encoder/decoder shapes
  - Test forward pass
  - Test save/load

- [ ] Create `test_training.py`
  - Test loss functions
  - Test training loop
  - Test checkpointing

- [ ] Create `test_tasks.py`
  - Test embedding extraction
  - Test annotation pipeline

- [ ] Set up pytest configuration
  - Add pytest.ini or pyproject.toml config
  - Configure test discovery
  - Add fixtures for common data

**Success Criteria**:
- All tests pass
- Test coverage > 70%
- Tests run in <1 minute total
- Tests are reproducible

#### 6.4 Documentation
**Files**: `README.md`, docstrings

- [ ] Update README.md
  - Project overview
  - Installation instructions
  - Quick start guide
  - Link to tutorials
  - Citation information

- [ ] Add comprehensive docstrings
  - All public functions
  - NumPy/Google style
  - Include examples

- [ ] Create API reference (optional)
  - Auto-generate with sphinx
  - Or simple markdown docs

- [ ] Add troubleshooting guide
  - Common errors
  - FAQ
  - Performance tips

**Success Criteria**:
- README is clear and comprehensive
- New users can get started easily
- All public APIs documented
- Examples work as shown

### Deliverables
- 3 end-to-end example scripts
- 4 tutorial notebooks
- Comprehensive test suite (>70% coverage)
- Updated README.md
- Full docstring coverage
- Troubleshooting guide

### Validation
- All examples and notebooks run successfully
- All tests pass
- Documentation is clear and accurate
- New user can complete tutorials in <2 hours

---

## PHASE 7: Package & Distribution (Optional)
**Status**: Sequential (requires Phase 6)
**Estimated Time**: 1 session
**Dependencies**: All previous phases complete

### Objectives
Package scGPT-mini for distribution and installation.

### Tasks

#### 7.1 Package Configuration

- [ ] Create `setup.py`
  - Package metadata
  - Dependencies
  - Entry points (optional CLI)

- [ ] Create `pyproject.toml`
  - Modern packaging with Poetry or setuptools
  - Specify build system
  - Development dependencies

- [ ] Create `requirements.txt`
  - Pin dependency versions
  - Separate requirements-dev.txt

- [ ] Create `MANIFEST.in`
  - Include data files (vocab, configs)
  - Include LICENSE, README

**Success Criteria**:
- Can install with `pip install -e .`
- Dependencies install correctly
- Package imports work

#### 7.2 Distribution

- [ ] Test installation in clean environment
  - Create virtual environment
  - Install package
  - Run examples

- [ ] Create distribution files
  - `python -m build` (sdist + wheel)
  - Test installation from wheel

- [ ] Upload to PyPI (optional)
  - Test upload to TestPyPI first
  - Upload to PyPI if desired

- [ ] Create GitHub release (optional)
  - Tag version
  - Upload distribution files
  - Write release notes

**Success Criteria**:
- Package installs cleanly
- All examples work after installation
- No missing dependencies

#### 7.3 Documentation Site (Optional)

- [ ] Set up Sphinx documentation
  - Configure sphinx
  - Auto-generate API docs
  - Include tutorials

- [ ] Deploy to Read the Docs
  - Link GitHub repo
  - Configure builds

- [ ] Add badges to README
  - PyPI version
  - License
  - Documentation

**Success Criteria**:
- Documentation builds successfully
- Available online
- Searchable and navigable

### Deliverables
- `setup.py` and/or `pyproject.toml`
- `requirements.txt`
- Distribution files (wheel + sdist)
- Optional: PyPI package
- Optional: Documentation site

### Validation
- Package installs in fresh Python 3.8+ environment
- All dependencies resolve correctly
- Examples work after pip install
- Documentation is accessible

---

## Success Metrics

### Overall Project Success Criteria

#### Functionality
- ✅ Can preprocess scRNA-seq data (filter, normalize, tokenize)
- ✅ Can train transformer model with MLM objective
- ✅ Can extract meaningful cell embeddings
- ✅ Can fine-tune for cell type annotation
- ✅ Achieves >80% accuracy on PBMC cell type classification

#### Performance
- ✅ Preprocessing: <2 minutes for 10K cells
- ✅ Training: <30 minutes for 10K cells, 20 epochs (CPU)
- ✅ Embedding extraction: <2 minutes for 10K cells
- ✅ Inference: <1 minute for 10K cells
- ✅ Memory: <8GB RAM during training, <2GB during inference

#### Code Quality
- ✅ All unit tests pass
- ✅ Test coverage >70%
- ✅ No major bugs in core functionality
- ✅ Code follows PEP 8 style guide
- ✅ Comprehensive docstrings

#### Usability
- ✅ Clear README with quick start
- ✅ 4 tutorial notebooks that run end-to-end
- ✅ 3 example scripts for common tasks
- ✅ Installable with pip
- ✅ Works on Windows, Mac, Linux

#### Educational Value
- ✅ Code is readable and well-commented
- ✅ Tutorials explain key concepts
- ✅ Architecture is understandable
- ✅ Easy to experiment and modify

---

## Benchmarks & Baselines

### Test Datasets
1. **PBMC 3k** (Recommended for testing)
   - 2,700 cells, ~32,000 genes
   - Well-characterized cell types
   - Available via scanpy: `scanpy.datasets.pbmc3k()`

2. **PBMC 10k** (Optional, larger test)
   - ~10,000 cells
   - More challenging for laptop

### Expected Performance

| Task | Metric | Target | Baseline |
|------|--------|--------|----------|
| MLM Pretraining | Masked MSE | <0.5 | Random: ~1.0 |
| | Masked Pearson | >0.5 | Random: ~0.0 |
| Cell Embedding | Silhouette score | >0.3 | PCA: ~0.2 |
| | k-NN accuracy | >0.7 | Random: 1/n_types |
| Cell Type Annotation | Accuracy | >0.80 | Logistic: ~0.70 |
| | Macro F1 | >0.75 | Logistic: ~0.65 |

*Note: Targets are approximate, based on simplified model. Full scGPT achieves higher performance.*

### Comparison to Baselines

Test against simple baselines:
- **PCA + k-NN**: For cell type annotation
- **PCA + k-means**: For clustering
- **Logistic regression**: On normalized counts

scGPT-mini should outperform these on embedding quality and annotation accuracy.

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Model doesn't fit in 16GB RAM | High | Low | Use smaller batch size (16 vs 32), reduce model size |
| Training too slow on CPU | Medium | Medium | Profile code, optimize bottlenecks, reduce dataset size |
| Poor embedding quality | Medium | Low | Tune hyperparameters, increase training data, check preprocessing |
| Dependencies conflict | Low | Medium | Pin versions, test in clean env, minimal dependencies |
| Code too complex for learning | Medium | Low | Add extensive comments, simplify where possible, provide tutorials |

### Process Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Scope creep | Medium | Medium | Stick to plan, defer advanced features to future |
| Phase dependencies unclear | Low | Low | This document clarifies dependencies |
| Insufficient testing | Medium | Low | Phase 6 ensures comprehensive testing |
| Poor documentation | High | Low | Documentation is part of Phase 6 |

---

## Future Extensions

**After completing Phases 1-7**, consider these additions:

### Advanced Features
- **Batch correction**: Add simple batch normalization or adversarial training
- **Multi-omic integration**: Add protein/ATAC modalities
- **Perturbation prediction**: Generative model for gene knockout effects
- **Gene regulatory networks**: Attention-based GRN inference
- **Transfer learning**: Easy domain adaptation

### Performance Optimizations
- **GPU support**: Add CUDA compatibility
- **Mixed precision**: FP16 training for memory efficiency
- **Flash attention**: Add as optional dependency
- **Quantization**: INT8 inference for faster predictions
- **Distributed training**: Multi-GPU support

### Usability Improvements
- **CLI tool**: Command-line interface for common tasks
- **Weights & Biases**: Optional experiment tracking
- **AutoML**: Hyperparameter tuning with Optuna
- **Pre-trained checkpoints**: Host models on HuggingFace
- **Web demo**: Streamlit or Gradio interface

### Additional Downstream Tasks
- **Cell trajectory inference**: Pseudotime analysis
- **Spatial transcriptomics**: Add spatial coordinates
- **Cell-cell communication**: Ligand-receptor analysis
- **Drug response prediction**: Pharmacogenomics

---

## Timeline Estimate

**Assuming 1-2 sessions per phase, 2-4 hours per session:**

| Phase | Sessions | Cumulative Time |
|-------|----------|-----------------|
| Phase 1: Foundation | 1-2 sessions | 2-8 hours |
| Phase 2: Model | 1-2 sessions | 4-16 hours |
| Phase 3: Training | 1-2 sessions | 6-24 hours |
| Phase 4: Embedding | 1 session | 8-28 hours |
| Phase 5: Annotation | 1 session | 10-32 hours |
| Phase 6: Integration | 1 session | 12-36 hours |
| Phase 7: Package (optional) | 1 session | 14-40 hours |

**Total: 7-9 sessions, 14-40 hours of active work**

Note: Phases 4 and 5 can run concurrently, saving 1 session.

---

## Confirmed Specifications

**✅ Decisions Made:**

1. **Dataset**: PBMC 3k (~2,700 cells) from scanpy
2. **Model size**:
   - d_model: **32** (compact for educational use)
   - num_layers: 2-3
   - n_hvgs: 500-1000
3. **Python version**: 3.8-3.12 (avoid 3.13+ due to scanpy conflicts)
4. **Output format**: AnnData objects (standard in single-cell field)
5. **Testing strategy**: Comprehensive unit tests with >70% coverage

---

## Getting Started

**To begin Phase 1 in the next session:**

1. Confirm this plan looks good
2. Answer any clarifying questions above
3. Set up development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install torch numpy pandas scikit-learn scanpy
   ```
4. Start with Phase 1.1: Gene Vocabulary System

**First concrete task**: Implement `GeneVocab` class in `scgpt_mini/tokenizer/vocabulary.py`

---

## References

1. **scGPT Paper**: Cui et al. (2023). "scGPT: Foundation Model for Single-Cell Multi-omics"
   - bioRxiv: https://www.biorxiv.org/content/10.1101/2023.04.30.538439

2. **scGPT Repository**: https://github.com/bowang-lab/scGPT

3. **Transformer Architecture**: Vaswani et al. (2017). "Attention is All You Need"

4. **Single-cell Analysis**:
   - Scanpy: https://scanpy.readthedocs.io/
   - AnnData: https://anndata.readthedocs.io/

5. **PyTorch Documentation**: https://pytorch.org/docs/

---

## Appendix: File Checklist

Use this checklist to track implementation progress:

### Phase 1: Foundation
- [ ] `scgpt_mini/tokenizer/__init__.py`
- [ ] `scgpt_mini/tokenizer/vocabulary.py`
- [ ] `scgpt_mini/tokenizer/gene_tokenizer.py`
- [ ] `scgpt_mini/tokenizer/default_vocab.json`
- [ ] `scgpt_mini/data/__init__.py`
- [ ] `scgpt_mini/data/preprocess.py`
- [ ] `scgpt_mini/data/collator.py`

### Phase 2: Model
- [ ] `scgpt_mini/model/__init__.py`
- [ ] `scgpt_mini/model/encoder.py`
- [ ] `scgpt_mini/model/decoder.py`
- [ ] `scgpt_mini/model/transformer.py`
- [ ] `scgpt_mini/model/model_config.json`

### Phase 3: Training
- [ ] `scgpt_mini/training/__init__.py`
- [ ] `scgpt_mini/training/losses.py`
- [ ] `scgpt_mini/training/metrics.py`
- [ ] `scgpt_mini/training/trainer.py`
- [ ] `scgpt_mini/training/training_config.json`

### Phase 4: Embedding
- [ ] `scgpt_mini/tasks/__init__.py`
- [ ] `scgpt_mini/tasks/embedding.py`
- [ ] `scgpt_mini/utils/__init__.py`
- [ ] `scgpt_mini/utils/visualization.py`

### Phase 5: Annotation
- [ ] `scgpt_mini/tasks/annotation.py`

### Phase 6: Integration
- [ ] `examples/01_pretraining.py`
- [ ] `examples/02_embedding.py`
- [ ] `examples/03_annotation.py`
- [ ] `notebooks/Tutorial_01_Preprocessing.ipynb`
- [ ] `notebooks/Tutorial_02_Pretraining.ipynb`
- [ ] `notebooks/Tutorial_03_Embeddings.ipynb`
- [ ] `notebooks/Tutorial_04_Annotation.ipynb`
- [ ] `tests/test_tokenizer.py`
- [ ] `tests/test_preprocessing.py`
- [ ] `tests/test_model.py`
- [ ] `tests/test_training.py`
- [ ] `tests/test_tasks.py`
- [ ] `README.md`

### Phase 7: Package
- [ ] `setup.py`
- [ ] `pyproject.toml`
- [ ] `requirements.txt`
- [ ] `requirements-dev.txt`
- [ ] `MANIFEST.in`
- [ ] `LICENSE`

---

*This implementation plan is designed to be executed across multiple Claude sessions, with each phase building on the previous ones. Phases 4 and 5 can be completed concurrently once Phase 3 is finished.*

**Ready to start Phase 1? Let's build scGPT-mini! 🚀**
