"""On-demand resolution of trained model weights.

Published weights live in a single Hugging Face Hub model repo
(``MagedSaeed/tnqeet-models``) under size-keyed names, e.g.
``lstm/4L.ckpt`` or ``ngrams/order_6.binary``. ``resolve_weight`` returns a
local filesystem path to a single weight file, downloading it from the Hub
(cached, one file at a time) when needed.

For development, pass ``weights_dir`` pointing at a method's local
``trained_models`` tree (the raw training-output layout). The resolver then
reads directly from disk instead of the Hub: for the neural models it picks
the latest ``epoch=*.ckpt`` under the matching ``layers_N`` / CANINE run dir,
and for n-grams it returns the requested ``ngrams_N.binary``.
"""

import glob
import os
import re

HF_WEIGHTS_REPO = "MagedSaeed/tnqeet-models"

# method -> {friendly size -> Hub filename}
WEIGHTS = {
    "lstm": {f"{n}L": f"lstm/{n}L.ckpt" for n in (1, 2, 3, 4, 5, 6)},
    "transformer": {f"{n}L": f"transformer/{n}L.ckpt" for n in (3, 6, 9, 12)},
    "canine": {"c": "canine/c.ckpt", "s": "canine/s.ckpt"},
    "ngram": {n: f"ngrams/order_{n}.binary" for n in range(2, 9)},  # orders 2-8
}

# default size per method, used when the caller does not specify one
DEFAULTS = {"lstm": "4L", "transformer": "6L", "canine": "s", "ngram": 6}

# Default local ``trained_models`` path per method (from the training scripts).
# These are NOT used automatically — ``resolve_weight`` always goes to
# the Hub unless ``weights_dir`` is passed explicitly. They are provided as a
# convenience so dev code can do, e.g.::
#
#     resolve_weight("lstm", "4L", weights_dir=LOCAL_WEIGHTS_DIRS["lstm"])
#
# to read the in-repo checkpoints instead of downloading.
_MODELS_ROOT = "tnqeet/dotting_models"
LOCAL_WEIGHTS_DIRS = {
    "lstm": f"{_MODELS_ROOT}/sequence_labeling/trained_models",
    "transformer": f"{_MODELS_ROOT}/transformer/trained_models",
    "canine": f"{_MODELS_ROOT}/canine/trained_models",
    "ngram": f"{_MODELS_ROOT}/ngrams/trained_models",
}


def _local_run_dir(method, size, weights_dir):
    """Return the local run directory holding a neural method's checkpoints.

    The raw training layout (``LSTM/layers_4/`` etc.) differs from the canonical
    Hub layout, so the mapping is kept here rather than in WEIGHTS. Not used for
    n-grams, which are single files rather than run directories.
    """
    if method == "lstm":
        return os.path.join(weights_dir, "LSTM", f"layers_{size[:-1]}")
    if method == "transformer":
        return os.path.join(weights_dir, "Transformer", f"layers_{size[:-1]}")
    if method == "canine":
        return os.path.join(weights_dir, f"CANINE-canine-{size}")
    raise ValueError(f"Unknown method: {method!r}")


def best_checkpoint(run_dir):
    """Return the validation-best ``epoch=*.ckpt`` in ``run_dir``, or None.

    Checkpoints are named ``epoch=NN-val_loss=0.XXXX.ckpt``; selection minimizes
    the embedded ``val_loss`` (falling back to mtime if it can't be parsed).
    ``last.ckpt`` is excluded by the glob.
    """
    candidates = glob.glob(os.path.join(run_dir, "epoch=*.ckpt"))
    if not candidates:
        return None

    def val_loss(path):
        match = re.search(r"val_loss=(\d+(?:\.\d+)?)", os.path.basename(path))
        return (float(match.group(1)), 0.0) if match else (float("inf"), -os.path.getmtime(path))

    return min(candidates, key=val_loss)


def _local_path(method, size, weights_dir):
    """Resolve a weight file from a local ``trained_models`` tree.

    ``weights_dir`` is the root of the method's training output.
    """
    if method == "ngram":
        return os.path.join(weights_dir, f"ngrams_{size}.binary")

    run_dir = _local_run_dir(method, size, weights_dir)
    checkpoint = best_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(
            f"No epoch=*.ckpt found in {run_dir!r} for {method} size {size!r}."
        )
    return checkpoint


def resolve_weight(method, size=None, revision=None, weights_dir=None):
    """Return a local path to the weight file for ``(method, size)``.

    Args:
        method: One of ``"lstm"``, ``"transformer"``, ``"canine"``, ``"ngram"``.
        size: Friendly size key (e.g. ``"4L"``, ``"s"``, or an n-gram order
            like ``6``). Defaults to ``DEFAULTS[method]``.
        revision: Optional Hub revision (tag/branch/commit) for reproducible
            downloads. Ignored when ``weights_dir`` is given.
        weights_dir: If set, read from this local ``trained_models`` tree
            instead of downloading from the Hub.

    Returns:
        Filesystem path to a single weight file.
    """
    if method not in WEIGHTS:
        raise ValueError(
            f"Unknown method {method!r}. Available: {sorted(WEIGHTS)}."
        )
    if size is None:
        size = DEFAULTS[method]
    if size not in WEIGHTS[method]:
        raise ValueError(
            f"Unknown size {size!r} for method {method!r}. "
            f"Available: {sorted(WEIGHTS[method])}."
        )

    if weights_dir is not None:
        return _local_path(method, size, weights_dir)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=HF_WEIGHTS_REPO,
        filename=WEIGHTS[method][size],
        revision=revision,
    )
