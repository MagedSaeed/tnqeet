# tnqeet

**tnqeet** is an Arabic dots restoration (*dotting*) library. It restores dots
to dotless Arabic text (*Rasm*), where several letters collapse to the same
basic shape once their dots are removed — for example `ب ت ث` all share one
dotless form. Restoring the dots requires resolving this ambiguity from context.

For instance, the dotted sentence:

```
لسان الفتى شطر وشطر فؤاده
```

has the dotless (Rasm) form:

```
لساں الڡٮى سطر وسطر ڡواده
```

and the dotting task is to recover the former from the latter.

The library implements and evaluates several approaches to dotting:
sequence-labeling (BiLSTM), Transformer, CANINE, n-gram language models (KenLM),
and LLM-based models.

## Installation

```bash
pip install tnqeet
```

This installs everything needed for the neural dotters (BiLSTM, Transformer,
CANINE). The **n-gram** method additionally requires KenLM, which is not
installed automatically because it compiles from source — see
[N-gram method (KenLM)](#n-gram-method-kenlm).

### CPU vs. GPU (torch backend)

`tnqeet` depends on PyTorch. The default install above pulls the **CUDA** build
of torch (the standard PyPI wheel on Linux), which brings in the `nvidia-*`
packages. That is the right choice on a machine with an NVIDIA GPU.

On a CPU-only machine you'll want the **CPU** build, which skips those packages.
The CPU wheel of torch is not published on PyPI — it lives on PyTorch's own
index — so point the installer at that index with the `cpu` extra:

```bash
pip install "tnqeet[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu
```

If you install with **uv** rather than pip, add `--index-strategy unsafe-best-match`:

```bash
uv pip install "tnqeet[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match
```

> The extra flag is needed because the PyTorch index also hosts old pinned
> copies of other packages (e.g. `torchmetrics`). pip merges indexes and picks
> the best version automatically; uv defaults to a stricter "first index wins"
> strategy (a dependency-confusion safeguard), so it needs the flag to consider
> PyPI's newer versions. Both indexes here are official and trusted.

> From a repo clone you don't deal with any of this: the CPU/CUDA indexes are
> configured in `pyproject.toml` and scoped to torch only, so `uv sync --extra
> cpu` (or `--extra cuda`) just works.

## Usage

### Remove dots

Convert dotted Arabic text to its dotless *Rasm* form:

```python
from tnqeet import remove_dots

remove_dots("لسان الفتى شطر وشطر فؤاده")
# -> "لساں الڡٮى سطر وسطر ڡواده"
```

### Restore dots

Load a pretrained model and restore the dots. Weights are downloaded from the
Hugging Face Hub on first use and cached locally:

```python
from tnqeet.dotting_models.transformer.models import TransformerDottingModel

model = TransformerDottingModel.from_pretrained()  # default size (6L)

model.restore_dots("لساں الڡٮى سطر وسطر ڡواده")
# -> "لسان الفتى شطر وشطر فؤاده"
```

`restore_dots` accepts a single string (returns a string) or a list of strings
(returns a list).

### Available models

Each method exposes `from_pretrained(size)` with friendly size keys (the default
is used when no size is given):

| Method | Class | Sizes (default in **bold**) |
| --- | --- | --- |
| BiLSTM | `LSTMDottingModel` | `1L`–`6L` (**`4L`**) |
| Transformer | `TransformerDottingModel` | `3L`, **`6L`**, `9L`, `12L` |
| CANINE | `CanineDottingModel` | `c`, **`s`** |
| n-gram | `NgramDotter` | order `2`–`8` (**`6`**) |

```python
from tnqeet.dotting_models.canine.models import CanineDottingModel

model = CanineDottingModel.from_pretrained("c")   # pick a specific size
```

The three neural models are PyTorch modules; move them to a GPU with the usual
`.to("cuda")` and switch to eval mode with `.eval()` for inference.

### N-gram method (KenLM)

The n-gram dotter is backed by [KenLM](https://github.com/kpu/kenlm), which
compiles from source and is therefore **not** a dependency of `tnqeet`. Install
it yourself, setting `MAX_ORDER` to at least the highest n-gram order you intend
to load. tnqeet publishes orders up to 8, so build with `MAX_ORDER=8` (KenLM's
own default is 6, which cannot load orders 7–8):

```bash
MAX_ORDER=8 pip install "git+https://github.com/kpu/kenlm.git"
```

Then:

```python
from tnqeet.dotting_models.ngrams.models import NgramDotter

dotter = NgramDotter.from_pretrained(order=6, beam_size=10)

dotter.restore_dots("لساں الڡٮى سطر وسطر ڡواده")
# -> "لسان الفتى شطر وشطر فؤاده"
```

Calling `NgramDotter.from_pretrained(...)` without KenLM installed raises an
`ImportError` containing the install command above.

See [CLAUDE.md](CLAUDE.md) for the full inference APIs and project layout.

## Development

The project uses [uv](https://docs.astral.sh/uv/) for dependency management and
packaging. Clone the repository and set up the environment:

```bash
git clone https://github.com/MagedSaeed/tnqeet
cd tnqeet
uv sync   # installs the project plus the dev dependency group
```

`uv sync` installs everything needed to train, evaluate, and run the notebooks
— including KenLM (built from source via `MAX_ORDER`, double check its installation) and the training/visualization tools (wandb, matplotlib, seaborn, ipython) that the published package omits. To build KenLM for a higher order, set `MAX_ORDER` before syncing:

```bash
MAX_ORDER=8 uv sync
```

Run the quality checks and tests:

```bash
uv run isort --check .
uv run black --check .
uv run ruff check .
uv run mypy .
uv run pytest
```

Building and publishing is handled by CI on a version tag (see
`.github/workflows/publish.yml`); to build locally:

```bash
uv build
```

## License

Apache-2.0. See [LICENSE](LICENSE).
