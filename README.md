# tnqeet

Arabic text diacritization (dotting) library. **tnqeet** restores dots to
undotted Arabic text (*Rasm*), where multiple letters share the same basic
shape without diacritical marks — for example `ب ت ث ن` all collapse to the
same dotless form.

The library implements and evaluates several approaches to the dotting problem:
sequence-labeling (BiLSTM), Transformer, CANINE, n-gram language models (KenLM),
and LLM-based models.

## Installation

```bash
pip install tnqeet
```

This installs everything needed for the neural dotters (BiLSTM, Transformer,
CANINE). The **n-gram** method additionally needs KenLM, which is not installed
automatically because it compiles from source — see
[N-gram method (KenLM)](#n-gram-method-kenlm) below.

## Usage

Convert dotted text to *Rasm*:

```python
from tnqeet import remove_dots

remove_dots("لسان الفتى شطر وشطر فؤاده")  # -> dotless (Rasm) form
```

Restore dots with a pretrained model. Weights are downloaded from the Hugging
Face Hub on first use and cached locally:

```python
from tnqeet.dotting_models.transformer.models import TransformerDottingModel

model = TransformerDottingModel.from_pretrained()        # default size (6L)
model.restore_dots("لسٮٯ الڡٮ９ سطر وسطر ڡؤاده")
```

Each method exposes `from_pretrained` with friendly size keys:

| Method | Class | Sizes (default in bold) |
| --- | --- | --- |
| BiLSTM | `LSTMDottingModel` | `1L`–`6L` (**`4L`**) |
| Transformer | `TransformerDottingModel` | `3L`, **`6L`**, `9L`, `12L` |
| CANINE | `CanineDottingModel` | `c`, **`s`** |
| n-gram | `NgramDotter` | order `2`–`8` (**`6`**) |

### N-gram method (KenLM)

The n-gram dotter is backed by [KenLM](https://github.com/kpu/kenlm). Install it
separately, setting `MAX_ORDER` to at least the n-gram order you intend to load
(published orders go up to 8):

```bash
MAX_ORDER=8 pip install "git+https://github.com/kpu/kenlm.git"
```

```python
from tnqeet.dotting_models.ngrams.models import NgramDotter

dotter = NgramDotter.from_pretrained(order=6, beam_size=10)
dotter.restore_dots("لسٮٯ الڡٮ９ سطر وسطر ڡؤاده")
```

Calling `NgramDotter.from_pretrained(...)` without KenLM installed raises an
`ImportError` with this install command.

See [CLAUDE.md](CLAUDE.md) for the full model inference APIs and project layout.

## Development

Clone this repo first, then, to install KenLM (used by the n-gram models) with larger ngrams for development, set the `MAX_ORDER` environment
variable to your preferred n-gram order before installing, e.g.

This project uses [uv](https://docs.astral.sh/uv/) for dependency management
and packaging.

```bash
# Set up the environment (installs the project and the dev dependency group)
uv sync

# Run the quality checks
uv run isort --check .
uv run black --check .
uv run ruff check .
uv run mypy .
uv run pytest

# Build and publish
uv build
uv publish
```

## License

Apache-2.0. See [LICENSE](LICENSE).
