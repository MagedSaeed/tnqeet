# tnqeet

Arabic text diacritization (dotting) library. **tnqeet** restores dots to
undotted Arabic text (*Rasm*), where multiple letters share the same basic
shape without diacritical marks — for example `ب ت ث ن` all collapse to the
same dotless form.

The library implements and evaluates three approaches to the dotting problem:
sequence-labeling (LSTM), n-gram language models (KenLM), and LLM-based models.

## Installation

```bash
pip install tnqeet
```:

## Usage

```python
from tnqeet import remove_dots

remove_dots("لسان الفتى شطر وشطر فؤاده")  # convert dotted text to Rasm
```

See [CLAUDE.md](CLAUDE.md) for the model inference APIs and the project layout.

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
