"""
Setup script for scGPT-mini.

Install with:
    pip install -e .  # Development mode
    pip install .     # Regular installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read version from __init__.py
version = "0.1.0"  # Initial version

setup(
    name="scgpt-mini",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="A miniature, educational version of scGPT for single-cell RNA-seq analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scgpt-mini",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/scgpt-mini/issues",
        "Documentation": "https://github.com/yourusername/scgpt-mini#readme",
        "Source Code": "https://github.com/yourusername/scgpt-mini",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "notebooks"]),
    package_data={
        "scgpt_mini": [
            "tokenizer/default_vocab.json",
            "model/model_config.json",
            "training/training_config.json",
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    python_requires=">=3.8,<3.13",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "single-cell",
        "genomics",
        "transformer",
        "deep-learning",
        "bioinformatics",
        "scRNA-seq",
        "foundation-model",
        "educational",
    ],
    entry_points={
        "console_scripts": [
            # Optional CLI tools can be added here
            # "scgpt-mini=scgpt_mini.cli:main",
        ],
    },
)
