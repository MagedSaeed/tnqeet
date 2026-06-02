"""Upload trained weights to the Hugging Face Hub.

Pushes the local ``trained_models`` weights to ``MagedSaeed/tnqeet-models``
under the size-keyed names consumed by :mod:`tnqeet.weights`, e.g.
``lstm/4L.ckpt`` or ``ngrams/order_6.binary``.

Published subset:
    - lstm:        layers 1-6           -> lstm/{n}L.ckpt
    - transformer: layers 3, 6, 9, 12   -> transformer/{n}L.ckpt
    - canine:      c, s                 -> canine/{c,s}.ckpt
    - ngram:       orders 2-8           -> ngrams/order_{n}.binary

For each neural tier the validation-best ``epoch=*.ckpt`` is selected;
``last.ckpt`` is ignored. Requires ``huggingface-cli login``. The target repo
is ``HF_WEIGHTS_REPO`` in :mod:`tnqeet.weights`; edit it there to change it.

Usage:
    python tnqeet_scripts/upload_weights.py
"""

import os

from tqdm.auto import tqdm

from tnqeet.weights import (
    HF_WEIGHTS_REPO,
    LOCAL_WEIGHTS_DIRS,
    WEIGHTS,
    _local_run_dir,
    best_checkpoint,
)


def _local_source(method, size):
    """Return the local file to publish for ``(method, size)``, or None."""
    root = LOCAL_WEIGHTS_DIRS[method]
    if method == "ngram":
        path = os.path.join(root, f"ngrams_{size}.binary")
        return path if os.path.exists(path) else None
    return best_checkpoint(_local_run_dir(method, size, root))


def main():
    from huggingface_hub import HfApi

    api = HfApi()

    # Pre-flight: if the repo already exists, show what's there and confirm
    # before pushing (uploads overwrite files at the same paths).
    if api.repo_exists(repo_id=HF_WEIGHTS_REPO, repo_type="model"):
        existing = api.list_repo_files(repo_id=HF_WEIGHTS_REPO, repo_type="model")
        weights = [f for f in existing if f.endswith((".ckpt", ".binary"))]
        print(f"Repo {HF_WEIGHTS_REPO!r} already exists with {len(weights)} weight file(s).")
        for f in sorted(weights):
            print(f"  - {f}")
        reply = input("\nPush anyway? Existing files will be overwritten. [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return
    else:
        api.create_repo(repo_id=HF_WEIGHTS_REPO, repo_type="model", exist_ok=True)
        print(f"Created repo {HF_WEIGHTS_REPO!r}.")

    # Per-file byte progress is shown by huggingface_hub itself; the tqdm here
    # tracks overall progress across the published tiers.
    tiers = [(m, s, c) for m, sizes in WEIGHTS.items() for s, c in sizes.items()]
    for method, size, canonical in tqdm(tiers, desc="Uploading weights", unit="file"):
        src = _local_source(method, size)
        if src is None:
            tqdm.write(f"  SKIP {method} {size}: not found locally")
            continue
        tqdm.write(f"  {method} {size}: {src} -> {canonical}")
        api.upload_file(
            path_or_fileobj=src,
            path_in_repo=canonical,
            repo_id=HF_WEIGHTS_REPO,
            repo_type="model",
        )
    print("Done.")


if __name__ == "__main__":
    main()
