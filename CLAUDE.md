# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**tnqeet** is an Arabic text diacritization (dotting) library developed by the Allen Institute for AI. The project focuses on restoring dots to undotted Arabic text (Rasm), where multiple letters share the same basic shape without diacritical marks. The repository implements and evaluates three different approaches to the Arabic dotting problem.

### Core Concept: Arabic Rasm and Dotting

Arabic text without dots uses simplified letter forms called "Rasm". Multiple letters can map to the same Rasm:
- `ب ت ث ن` all map to BAA_RASM (`\u066e`)
- `ج ح خ` all map to JEEM_RASM (`ح`)
- `د ذ` map to DAL_RASM (`د`)
- And so on for other letter groups

The dotting process restores the original dotted letters based on linguistic context.

## Architecture

The codebase is organized into three main dotting model approaches:

### 1. Sequence Labeling Models (`tnqeet/dotting_models/sequence_labeling/`)
Uses PyTorch Lightning with LSTM-based neural networks for sequence-to-sequence dotting:
- [models.py](tnqeet/dotting_models/sequence_labeling/models.py): `LSTMDottingModel` implementation with bidirectional LSTM
- [data.py](tnqeet/dotting_models/sequence_labeling/data.py): `DottingDataModule` for PyTorch Lightning data loading
- [trainer.py](tnqeet/dotting_models/sequence_labeling/trainer.py): Training configuration with checkpointing and early stopping
- [train.py](tnqeet/dotting_models/sequence_labeling/train.py): Main training script with WandB logging

### 2. N-gram Language Models (`tnqeet/dotting_models/ngrams/`)
Uses KenLM for n-gram based dotting with beam search:
- Beam search decoding to find the most likely dotted sequence
- Character-level language model scoring
- Training notebooks in `train.ipynb`

### 3. LLM-based Models (`tnqeet/dotting_models/llms/`)
Uses DSPy framework with large language models (Claude, GPT) via OpenRouter:
- [models.py](tnqeet/dotting_models/llms/models.py): `OpenRouterArabicDotter` class with signature definitions
- Supports few-shot learning with `ArabicDottingSignature` and `DetailedArabicDotingSignature`
- Direct LLM API integration for dotting via DSPy

### Core Utilities

- [constants.py](tnqeet/constants.py): Arabic letter mappings (`LETTERS_MAPPING`), Rasm definitions, Unicode normalization tables
- [__init__.py](tnqeet/__init__.py): `remove_dots()` function - core utility to convert dotted text to Rasm
- [data/__init__.py](tnqeet/data/__init__.py): Dataset loading from HuggingFace Hub
  - `train_dataset`: MagedSaeed/tnqeet-training-datasets
  - `test_dataset`: MagedSaeed/tnqeet-testing-datasets
  - `val_dataset` and `fewshot_val_dataset`: Stratified validation splits
- [evaluate/metrics.py](tnqeet/evaluate/metrics.py): Evaluation metrics (`wer`, `cer`, `doer`)
  - WER: Word Error Rate
  - CER: Character Error Rate
  - DOER: Dot Error Rate (specific to ambiguous Rasms)

### Tokenizer

The project uses a custom tokenizer from HuggingFace Hub:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("MagedSaeed/tnqeet-tokenizer", trust_remote_code=True)
```

## Development Commands

### Environment Setup
```bash
# Install in editable mode with dev dependencies
pip install -U pip setuptools wheel
pip install -e .[dev]
```

### Code Quality

```bash
# Format code
isort .
black .

# Lint and type-check
ruff check .
mypy .
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest -v tests/hello_test.py

# Run with coverage
pytest --cov=tnqeet tests/
```

### Training Models

**Sequence Labeling:**
```bash
# Train LSTM model (modify tnqeet/dotting_models/sequence_labeling/train.py for configuration)
python tnqeet/dotting_models/sequence_labeling/train.py
```

Note: Training uses PyTorch Lightning with:
- WandB logging (project: "dotting_models")
- Automatic checkpointing to `tnqeet/dotting_models/sequence_labeling/trained_models/`
- Early stopping (patience=4)
- GPU auto-detection via `accelerator="auto"`

**N-gram Models:**
- Use Jupyter notebook: `tnqeet/dotting_models/ngrams/train.ipynb`

### Evaluation

Each model type has evaluation scripts in its `evaluate/` subdirectory:

```bash
# Sequence labeling evaluation (validation set)
python tnqeet/dotting_models/sequence_labeling/evaluate/val_dataset.py

# Sequence labeling evaluation (test set)
python tnqeet/dotting_models/sequence_labeling/evaluate/test_dataset.py

# N-gram evaluation
python tnqeet/dotting_models/ngrams/evaluate/val_dataset.py
python tnqeet/dotting_models/ngrams/evaluate/test_dataset.py

# LLM evaluation
python tnqeet/dotting_models/llms/evaluate/val_dataset.py
python tnqeet/dotting_models/llms/evaluate/test_dataset.py
```

Evaluation scripts save results as JSON files with per-example predictions and metrics.

### Documentation

```bash
# Build documentation
make docs

# The documentation uses Sphinx with autodoc extension
```

## Model Inference API

All dotting models implement a consistent `restore_dots()` interface:

**Sequence Labeling:**
```python
from tnqeet.dotting_models.sequence_labeling.models import LSTMDottingModel
model = LSTMDottingModel.load_from_checkpoint(checkpoint_path)
dotted_text = model.restore_dots(dotless_text, resolve_ambiguous_rasms_only=False)
```

**N-gram:**
```python
from tnqeet.dotting_models.ngrams.evaluate.val_dataset import NgramDotter
import kenlm
model = kenlm.LanguageModel("path/to/ngrams_15.binary")
dotter = NgramDotter(model=model, beam_size=10)
dotted_text = dotter.restore_dots(dotless_text)
```

**LLM:**
```python
from tnqeet.dotting_models.llms.models import OpenRouterArabicDotter
dotter = OpenRouterArabicDotter(
    model="anthropic/claude-sonnet-4",
    num_fewshot=5
)
dotted_text = dotter.restore_dots(dotless_text)
```

## Important Configuration Notes

### PyTorch Lightning Configuration
- Training uses `val_check_interval=0.25` (4 evaluations per epoch)
- Gradient clipping: `gradient_clip_val=1`
- Deterministic mode enabled for reproducibility
- Checkpoints save both best model (by `val_loss`) and last model

### Environment Variables for LLM Models
Create a `.env` file with:
```
OPENROUTER_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here  # for direct Anthropic API
OPENAI_API_KEY=your_key_here     # for direct OpenAI API
```

### Dataset Stratification
Validation datasets are stratified by source to ensure diverse representation:
- 15 examples per source for `val_dataset` (from last samples)
- 15 examples per source for `fewshot_val_dataset` (from first samples)
- Both shuffled with `RANDOM_SEED=42`

## Code Formatting Standards

- Line length: 100 characters (black), 115 characters (ruff)
- Import sorting: `isort` with black profile
- Type hints preferred but not strictly enforced (mypy ignores missing imports)
- Python 3.10+ required

## Key Design Patterns

1. **Lazy Loading**: `LazyDottingDataset` loads text on-demand to reduce memory usage
2. **Character-level Processing**: N-gram models tokenize to characters with `<SPACE>` markers
3. **Rasm Ambiguity Resolution**: Models can optionally only restore ambiguous Rasms, preserving unambiguous letters
4. **Evaluation Persistence**: Evaluation scripts save intermediate results every 5 examples to enable resumption
