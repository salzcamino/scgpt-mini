# scGPT-mini

A miniature, educational version of [scGPT](https://github.com/bowang-lab/scGPT) for single-cell RNA-seq analysis.

## Overview

**scGPT-mini** is a simplified transformer-based model designed for learning and education about foundation models in genomics. It focuses on core transformer architecture and essential features while running efficiently on standard laptops.

### Key Features

- **Educational Focus**: Simplified, well-documented code for learning transformer models
- **Laptop-Friendly**: Runs on 16GB RAM, CPU-only (no GPU required)
- **Compact Model**: 32-dim embeddings, 2-3 transformer layers (~50-150K parameters)
- **Essential Tasks**: MLM pretraining, cell embedding generation, cell type annotation
- **Minimal Dependencies**: PyTorch, NumPy, Pandas, Scanpy

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scgpt-mini.git
cd scgpt-mini

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

See `examples/00_preprocessing_example.py` for a complete walkthrough of Phase 1 functionality:

```python
import scanpy as sc
from scgpt_mini.data import preprocess_adata
from scgpt_mini.tokenizer import GeneVocab, tokenize_batch
from scgpt_mini.data.collator import create_dataloader

# Load data
adata = sc.datasets.pbmc3k()

# Preprocess
adata = preprocess_adata(
    adata,
    filter_gene_by_counts=10,
    normalize_total_target=1e4,
    log1p=True,
    subset_hvg=1000,
    binning=51,
)

# Create vocabulary
vocab = GeneVocab(adata.var_names.tolist())

# Tokenize
tokenized_data = tokenize_batch(
    data=adata.X.toarray(),
    gene_names=adata.var_names.values,
    vocab=vocab,
)

# Create DataLoader
dataloader = create_dataloader(
    tokenized_data,
    vocab,
    batch_size=32,
    max_len=1001,
    mask_ratio=0.15,
)
```

## Project Structure

```
scgpt-mini/
├── scgpt_mini/           # Main package
│   ├── tokenizer/        # Gene vocabulary and tokenization
│   ├── model/            # Transformer architecture (Phase 2)
│   ├── data/             # Data preprocessing and collation
│   ├── training/         # Training infrastructure (Phase 3)
│   ├── tasks/            # Downstream tasks (Phases 4-5)
│   └── utils/            # Utility functions
├── examples/             # Usage examples
├── tests/                # Unit tests
├── notebooks/            # Tutorial notebooks (coming soon)
├── CLAUDE.md             # Detailed implementation plan
└── README.md             # This file
```

## Implementation Status

### ✅ Phase 1: Foundation & Data Processing (COMPLETE)
- [x] Gene vocabulary system
- [x] Gene tokenization
- [x] Data preprocessing (filtering, normalization, HVG selection)
- [x] Data collator with MLM masking
- [x] Default vocabulary (8,000 genes from scGPT)
- [x] Unit tests
- [x] Example script

### ✅ Phase 2: Core Model Architecture (COMPLETE)
- [x] Gene and value encoders
- [x] Expression decoder (continuous and binned)
- [x] Classification decoder
- [x] Transformer model
- [x] Model configuration
- [x] Checkpoint save/load
- [x] Unit tests
- [x] Example script

### ✅ Phase 3: Training Infrastructure (COMPLETE)
- [x] Loss functions (MSE, MAE, cross-entropy, combined)
- [x] Training metrics (MSE, MAE, Pearson, classification)
- [x] Training loop with Trainer class
- [x] Checkpointing and model persistence
- [x] Learning rate scheduling support
- [x] Early stopping support
- [x] Unit tests
- [x] Example training script

### 🚧 Phase 4: Cell Embedding Generation (TODO)
- [ ] Embedding extraction
- [ ] Visualization utilities
- [ ] Embedding quality metrics

### 🚧 Phase 5: Cell Type Annotation (TODO)
- [ ] Fine-tuning for classification
- [ ] Cell type prediction
- [ ] Evaluation metrics

### 🚧 Phase 6: Integration & Testing (TODO)
- [ ] End-to-end examples
- [ ] Tutorial notebooks
- [ ] Comprehensive test suite
- [ ] Documentation

### 🚧 Phase 7: Package & Distribution (TODO)
- [ ] Package configuration
- [ ] Distribution files
- [ ] Documentation site (optional)

## Model Specifications

| Component | scGPT (default) | scGPT-mini |
|-----------|-----------------|------------|
| Embedding dimension | 128 | 32 |
| Transformer layers | 4 | 2-3 |
| Attention heads | 4 | 2 |
| Hidden dimension | 512 | 64 |
| Max genes (HVGs) | 1200 | 500-1000 |
| Gene vocabulary | ~19,000 | ~8,000 |
| Model parameters | ~2-5M | ~50-150K |
| Memory (training) | ~8-16GB GPU | <4GB RAM |
| Memory (inference) | ~2-4GB GPU | <1GB RAM |

## Requirements

- Python 3.8-3.12 (avoid 3.13+ due to potential scanpy conflicts)
- PyTorch >= 1.13.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Scanpy >= 1.9.0
- scikit-learn >= 1.0.0

See `requirements.txt` for full list.

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Detailed implementation plan with all phases
- **[examples/](examples/)**: Usage examples for each phase
- **[tests/](tests/)**: Unit tests for all components

## Comparison with scGPT

### What We're Keeping
- Core transformer architecture
- Gene tokenization and vocabulary system
- Masked language modeling (MLM) pretraining
- Cell embedding generation
- Cell type annotation

### What We're Removing/Simplifying
- Flash-attention (use standard PyTorch)
- Domain-Specific Batch Normalization
- Multi-omic integration
- Advanced tasks (GRN inference, perturbation prediction)
- Heavy dependencies (scvi-tools, torchtext, wandb)

## Contributing

This is an educational project. Contributions are welcome! Please:
1. Follow the phase structure in CLAUDE.md
2. Add unit tests for new features
3. Document code thoroughly
4. Keep it simple and educational

## License

MIT License

## Citation

If you use scGPT-mini in your work, please cite the original scGPT paper:

```bibtex
@article{cui2023scgpt,
  title={scGPT: Foundation Model for Single-Cell Multi-omics},
  author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and others},
  journal={bioRxiv},
  year={2023}
}
```

## Acknowledgments

- Original [scGPT](https://github.com/bowang-lab/scGPT) by Bo Wang Lab
- [Scanpy](https://scanpy.readthedocs.io/) for single-cell analysis tools
- PBMC 3k dataset from 10x Genomics

## Contact

For questions or issues, please open an issue on GitHub.

---

**scGPT-mini**: Making transformer models for genomics accessible and understandable! 🧬🤖
