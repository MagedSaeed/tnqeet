"""Evaluate the best model of each (non-LLM) method on a BiGRU test set (default: ATB).

For each method we take the best configuration from the paper's Table 1 (lowest
test-set WER):

    method       best config        Table-1 WER
    n-gram       order 8, beam 60    13.32
    BiLSTM       6 layers             8.43
    Transformer  12 layers            6.97
    CANINE       canine-s             7.48

This is a thin orchestrator: it builds a Hugging Face dataset from the BiGRU gold
lines (``text`` + ``source`` columns, exactly the shape the in-repo evaluators
expect) and hands it to each method's own ``evaluate/test_dataset.py:evaluate_model``.
Reusing those functions verbatim means the Rasm reduction, chunking, inference, and
WER/CER/DotER computation are byte-for-byte identical to the ``val_dataset`` /
``test_dataset`` runs.

Orthographic folding (why raw scores mislead on these datasets)
---------------------------------------------------------------
The BiGRU corpora ("Automatic dottization of Arabic text (Rasms) using deep
recurrent neural networks", Pattern Recognition Letters 162, 2022;
https://doi.org/10.1016/j.patrec.2022.09.001) ship gold text that has been run
through a lossy orthographic normalization. Precisly, they did the following
two main normalizations:

    * hamza/madda on alef  ->  bare alef   (أ إ آ ٱ  ->  ا)
    * taa-marbuta          ->  haa         (ة        ->  ه)

These are the two Arabic normalizations known as ``normalize_alef``
and ``normalize_teh_marbuta`` from CAMeL (Obeid et al., "CAMeL Tools", LREC 2020;
https://aclanthology.org/2020.lrec-1.868/). Our models correctly *restore* hamza
and taa-marbuta -- so scoring their output raw against their normalized gold
counts every such correct restoration as an error.

We therefore report BOTH:
    * raw     -- prediction vs. gold verbatim for record.
    * folded  -- both sides passed through the same two normalizations above, so
                 only dotting decisions are scored.

Per-example results are kept with this script (separated from the package's own
evaluation results) at::

    tnqeet_scripts/bigru_evaluation/evaluation_results/<source>/<method>/<config>/*.json

Usage:
    uv run python tnqeet_scripts/bigru_evaluation/evaluate.py
    uv run python tnqeet_scripts/bigru_evaluation/evaluate.py --source ATB --models transformer canine
    uv run python tnqeet_scripts/bigru_evaluation/evaluate.py --limit 50   # smoke test
"""

import argparse
import glob
import json
from pathlib import Path

import pyarabic.araby as araby
from tabulate import tabulate

import datasets
from tnqeet.evaluate.metrics import cer, doer, wer

REPO_ROOT = Path(__file__).resolve().parents[2]
BIGRU_ROOT = REPO_ROOT / "BiGRUDatasets"

# source -> subdirectory holding "*_Y_test.txt" (the dotted gold references)
BIGRU_SOURCES = {"ATB": "ATB", "Quran": "Quran", "poem": "poem", "tshkela": "tshkela"}


# --- per-method runners (best config from Table 1) --------------------------
#
# Each runner loads the best model (neural weights via ``from_pretrained``; the
# n-gram evaluator loads its own local KenLM binary) and delegates to that
# method's in-repo ``evaluate_model``. All return {avg_wer, avg_cer, avg_doer}.


def run_ngram(dataset, dataset_name, overwrite, device, results_dir):
    from tnqeet.dotting_models.ngrams.evaluate.test_dataset import evaluate_model

    return evaluate_model(
        ngrams=8,
        beam_size=60,
        dataset=dataset,
        dataset_name=dataset_name,
        overwrite=overwrite,
        results_dir=results_dir,
    )


def run_bilstm(dataset, dataset_name, overwrite, device, results_dir):
    from tnqeet.dotting_models.sequence_labeling.evaluate.test_dataset import (
        evaluate_model,
    )
    from tnqeet.dotting_models.sequence_labeling.models import LSTMDottingModel

    model = LSTMDottingModel.from_pretrained("6L").to(device)
    model.eval()
    return evaluate_model(
        model=model,
        dataset=dataset,
        dataset_name=dataset_name,
        overwrite=overwrite,
        n_layers=6,
        results_dir=results_dir,
    )


def run_transformer(dataset, dataset_name, overwrite, device, results_dir):
    from tnqeet.dotting_models.transformer.evaluate.test_dataset import evaluate_model
    from tnqeet.dotting_models.transformer.models import TransformerDottingModel

    model = TransformerDottingModel.from_pretrained("12L").to(device)
    model.eval()
    return evaluate_model(
        model=model,
        dataset=dataset,
        dataset_name=dataset_name,
        overwrite=overwrite,
        n_layers=12,
        results_dir=results_dir,
    )


def run_canine(dataset, dataset_name, overwrite, device, results_dir):
    from tnqeet.dotting_models.canine.evaluate.test_dataset import evaluate_model
    from tnqeet.dotting_models.canine.models import CanineDottingModel

    model = CanineDottingModel.from_pretrained("s").to(device)
    model.eval()
    return evaluate_model(
        model=model,
        dataset=dataset,
        dataset_name=dataset_name,
        overwrite=overwrite,
        model_name="CANINE-canine-s",
        results_dir=results_dir,
    )


# ``config`` is the last path segment (evaluation_results/<source>/<method>/<config>/).
BEST_MODELS = {
    "ngram": {"label": "8-gram (beam=60)", "config": "order8_beam60", "runner": run_ngram},
    "bilstm": {"label": "BiLSTM 6L", "config": "6L", "runner": run_bilstm},
    "transformer": {"label": "Transformer 12L", "config": "12L", "runner": run_transformer},
    "canine": {"label": "CANINE-s", "config": "canine-s", "runner": run_canine},
}


# --- data loading -----------------------------------------------------------


def load_gold_dataset(source: str, limit: int | None) -> datasets.Dataset:
    """Build a {text, source} HF dataset from a BiGRU source's dotted gold lines."""
    subdir = BIGRU_SOURCES[source]
    matches = sorted(glob.glob(str(BIGRU_ROOT / subdir / "*_Y_test.txt")))
    if not matches:
        raise FileNotFoundError(f"No *_Y_test.txt found under {BIGRU_ROOT / subdir}")
    lines: list[str] = []
    for path in matches:
        with open(path, encoding="utf-8") as handle:
            lines.extend(line.strip() for line in handle if line.strip())
    if limit:
        lines = lines[:limit]
    return datasets.Dataset.from_dict({"text": lines, "source": [source] * len(lines)})


# --- orthographic folding ---------------------------------------------------
#
# The two lossy normalizations ATB (and the other BiGRU sets) apply to their gold
# -- CAMeL Tools' normalize_alef + normalize_teh_marbuta. Applied identically to
# prediction and reference so only genuine dotting decisions are scored. See the
# module docstring for the rationale and references.

_FOLD_TABLE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه"})


def fold(text: str) -> str:
    """Normalize hamza-on-alef -> alef and taa-marbuta -> haa (ATB's convention)."""
    return araby.strip_tatweel(araby.strip_tashkeel(text)).translate(_FOLD_TABLE)


def augment_with_folded(results_dir: str) -> dict:
    """Add folded WER/CER/DotER to each saved record in-place; return folded averages.

    Reads the per-example JSON the reused evaluator just wrote, computes the folded
    metrics from the stored gold/prediction text (no re-inference), writes them back
    alongside the raw ones, and returns the averages for the summary table.
    """
    files = glob.glob(str(Path(results_dir) / "*.json"))
    if not files:
        return {}
    path = files[0]
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    fw = fc = fd = 0.0
    for r in records:
        g, p = fold(r["original_dotted_text"]), fold(r["predicted_dotted_text"])
        r["wer_folded"], r["cer_folded"], r["doer_folded"] = wer(g, p), cer(g, p), doer(g, p)
        fw, fc, fd = fw + r["wer_folded"], fc + r["cer_folded"], fd + r["doer_folded"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4, default=str)
    n = len(records) or 1
    return {"avg_wer_folded": fw / n, "avg_cer_folded": fc / n, "avg_doer_folded": fd / n}


# --- summary table ----------------------------------------------------------


def print_summary(summaries: list[dict], source: str, n_lines: int) -> None:
    headers = [
        "Method",
        "Setting",
        "WER %",
        "CER %",
        "DotER %",
        "WER % (fold)",
        "CER % (fold)",
        "DotER % (fold)",
    ]
    rows = [
        [
            s["method"],
            s["label"],
            f"{100 * s['avg_wer']:.2f}",
            f"{100 * s['avg_cer']:.2f}",
            f"{100 * s['avg_doer']:.2f}",
            f"{100 * s['avg_wer_folded']:.2f}",
            f"{100 * s['avg_cer_folded']:.2f}",
            f"{100 * s['avg_doer_folded']:.2f}",
        ]
        for s in summaries
    ]
    print()
    print(
        f"{source} evaluation ({n_lines} lines) -- lower is better; "
        "(fold) = hamza/taa-marbuta normalized on both sides"
    )
    print(
        tabulate(
            rows, headers=headers, tablefmt="fancy_grid", colalign=("left", "left") + ("right",) * 6
        )
    )


# --- driver -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=list(BIGRU_SOURCES), default="ATB")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(BEST_MODELS),
        default=list(BEST_MODELS),
        help="which methods to evaluate (default: all four)",
    )
    parser.add_argument(
        "--device", default=None, help="torch device for neural models (default: auto)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only score the first N lines (smoke test)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="ignore cached per-example results"
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_gold_dataset(args.source, args.limit)
    # Results live with this script, not in the package's evaluation_results tree
    # (separation of concern): evaluation_results/<source>/<method>/<config>/*.json
    results_root = Path(__file__).resolve().parent / "evaluation_results" / args.source
    print(f"Evaluating on {args.source}: {len(dataset):,} lines (device={device})")
    print(f"Per-example results go to {results_root}/<method>/<config>/")

    summaries: list[dict] = []
    for method in args.models:
        spec = BEST_MODELS[method]
        results_dir = str(results_root / method / spec["config"])
        print(f"\n=== {spec['label']} ===")
        try:
            summary = spec["runner"](dataset, args.source, args.overwrite, device, results_dir)
        except Exception as exc:  # keep going if one model fails (e.g. KenLM not built)
            print(f"  [{method}] SKIPPED: {type(exc).__name__}: {exc}")
            continue
        # Add the orthography-folded metrics from the just-written predictions.
        folded = augment_with_folded(results_dir)
        summaries.append({"method": method, "label": spec["label"], **summary, **folded})

    if summaries:
        print_summary(summaries, args.source, len(dataset))


main()
