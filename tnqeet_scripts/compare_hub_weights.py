"""Compare Hub-downloaded weights against the stored local evaluation results.

For each configured model this script downloads the weights from the Hugging
Face Hub (via ``<Model>.from_pretrained``), runs inference over the chosen
dataset, and tabulates its WER / CER / DotER and average time-per-character
against the matching ``evaluation_results`` JSON produced from the local
checkpoints. Matching metrics confirm the uploaded weights are identical to the
local ones.

Edit ``CONFIGS`` below to choose which models to test (2-3 per method by
default). Run on the validation set by default; pass ``--dataset test`` for the
test set, and ``--limit N`` to only score the first N examples (handy for a
quick check — the stored baseline is sliced to the same N for a fair compare).

Usage:
    python tnqeet_scripts/compare_hub_weights.py
    python tnqeet_scripts/compare_hub_weights.py --dataset test --limit 100
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

import json
import torch
from tabulate import tabulate
from tqdm.auto import tqdm

from tnqeet import remove_dots
from tnqeet.evaluate.metrics import cer, doer, wer
from tnqeet.dotting_models.sequence_labeling.utils import split_text_by_threshold

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "tnqeet" / "dotting_models"

# --------------------------------------------------------------------------
# Models to test. Comment out / add rows freely (2-3 per method recommended).
#   method:  lstm | transformer | canine | ngram
#   size:    Hub size key  ("2L".."6L", "3L".."12L", "c"/"s")  [neural]
#   order:   n-gram order (2-8)  +  beam: beam size            [ngram]
#   label:   display name in the table
# --------------------------------------------------------------------------
CONFIGS = [
    {"method": "lstm", "size": "2L", "label": "BiLSTM 2L"},
    {"method": "lstm", "size": "4L", "label": "BiLSTM 4L"},
    {"method": "lstm", "size": "6L", "label": "BiLSTM 6L"},
    {"method": "transformer", "size": "3L", "label": "Transformer 3L"},
    {"method": "transformer", "size": "6L", "label": "Transformer 6L"},
    {"method": "transformer", "size": "12L", "label": "Transformer 12L"},
    {"method": "canine", "size": "c", "label": "CANINE-c"},
    {"method": "canine", "size": "s", "label": "CANINE-s"},
    {"method": "ngram", "order": 4, "beam": 10, "label": "4-gram (beam 10)"},
    {"method": "ngram", "order": 6, "beam": 10, "label": "6-gram (beam 10)"},
    # {"method": "ngram", "order": 8, "beam": 10, "label": "8-gram (beam 10)"},
]


def stored_results_path(cfg, dataset_name):
    """Path to the stored evaluation JSON matching ``cfg`` for ``dataset_name``."""
    method = cfg["method"]
    if method == "lstm":
        return (
            RESULTS_ROOT / "sequence_labeling" / "evaluation_results" / dataset_name
            / f"LSTM_layers_{cfg['size'][:-1]}_results.json"
        )
    if method == "transformer":
        return (
            RESULTS_ROOT / "transformer" / "evaluation_results" / dataset_name
            / f"Transformer_layers_{cfg['size'][:-1]}_results.json"
        )
    if method == "canine":
        return (
            RESULTS_ROOT / "canine" / "evaluation_results" / dataset_name
            / f"CANINE-canine-{cfg['size']}_results.json"
        )
    if method == "ngram":
        return (
            RESULTS_ROOT / "ngrams" / "evaluation_results" / dataset_name
            / f"beam_size_{cfg['beam']}" / f"ngrams_{cfg['order']}.json"
        )
    raise ValueError(f"Unknown method: {method!r}")


def load_dotter(cfg, device):
    """Download weights from the Hub and return (dotter, threshold_offset).

    ``threshold_offset`` is subtracted from ``max_sequence_length`` when
    chunking (CANINE reserves 2 positions for [CLS]/[SEP]); None for n-grams.
    """
    method = cfg["method"]
    if method == "lstm":
        from tnqeet.dotting_models.sequence_labeling.models import LSTMDottingModel

        return LSTMDottingModel.from_pretrained(cfg["size"]).to(device).eval(), 0
    if method == "transformer":
        from tnqeet.dotting_models.transformer.models import TransformerDottingModel

        return TransformerDottingModel.from_pretrained(cfg["size"]).to(device).eval(), 0
    if method == "canine":
        from tnqeet.dotting_models.canine.models import CanineDottingModel

        return CanineDottingModel.from_pretrained(cfg["size"]).to(device).eval(), 2
    if method == "ngram":
        from tnqeet.dotting_models.ngrams.models import NgramDotter

        return NgramDotter.from_pretrained(order=cfg["order"], beam_size=cfg["beam"]), None
    raise ValueError(f"Unknown method: {cfg['method']!r}")


def predict(cfg, dotter, offset, dotted_text):
    """Replicate the per-method inference used to produce the stored results."""
    if cfg["method"] == "ngram":
        return dotter.restore_dots(remove_dots(dotted_text))
    predicted = ""
    for partial in split_text_by_threshold(
        dotted_text, threshold=dotter.max_sequence_length - offset
    ):
        predicted += dotter.restore_dots(remove_dots(partial))
    return predicted.strip()


def run_hub_eval(cfg, dataset, device):
    """Download the model, run inference, return aggregated metrics."""
    dotter, offset = load_dotter(cfg, device)
    wers, cers, doers, ms_per_char = [], [], [], []
    for example in tqdm(dataset, desc=cfg["label"], leave=False):
        dotted = example["text"]
        dotless = remove_dots(dotted)
        t0 = datetime.now()
        predicted = predict(cfg, dotter, offset, dotted)
        seconds = (datetime.now() - t0).total_seconds()
        wers.append(wer(dotted, predicted))
        cers.append(cer(dotted, predicted))
        doers.append(doer(dotted, predicted))
        if dotless:
            ms_per_char.append(seconds * 1000 / len(dotless))
    return {
        "wer": mean(wers),
        "cer": mean(cers),
        "doer": mean(doers),
        "ms_per_char": mean(ms_per_char) if ms_per_char else float("nan"),
        "n": len(wers),
    }


def _parse_dotting_time(value):
    h, m, s = value.split(":")
    return timedelta(hours=int(h), minutes=int(m), seconds=float(s)).total_seconds()


def load_stored_eval(path, limit):
    """Aggregate the stored per-example JSON (sliced to ``limit`` if given)."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if limit:
        records = records[:limit]
    if not records:
        return None
    ms_per_char = [
        _parse_dotting_time(r["dotting_time"]) * 1000 / len(r["dotless_text"])
        for r in records
        if r.get("dotting_time") and r.get("dotless_text")
    ]
    return {
        "wer": mean(r["wer"] for r in records),
        "cer": mean(r["cer"] for r in records),
        "doer": mean(r["doer"] for r in records),
        "ms_per_char": mean(ms_per_char) if ms_per_char else float("nan"),
        "n": len(records),
    }


def build_table(comparisons):
    headers = ["Config", "Source", "WER %", "CER %", "DotER %", "ms/char", "N"]
    rows = []
    for idx, (cfg, hub, stored) in enumerate(comparisons):
        if idx > 0:
            rows.append(["", "", "", "", "", "", ""])
        rows.append([
            cfg["label"], "Hub",
            f"{hub['wer'] * 100:.2f}", f"{hub['cer'] * 100:.2f}",
            f"{hub['doer'] * 100:.2f}", f"{hub['ms_per_char']:.4f}", hub["n"],
        ])
        if stored is None:
            rows.append(["", "Local", "—", "—", "—", "—", "(no stored file)"])
            continue
        rows.append([
            "", "Local",
            f"{stored['wer'] * 100:.2f}", f"{stored['cer'] * 100:.2f}",
            f"{stored['doer'] * 100:.2f}", f"{stored['ms_per_char']:.4f}", stored["n"],
        ])
        rows.append([
            "", "Δ (hub-local)",
            f"{(hub['wer'] - stored['wer']) * 100:+.2f}",
            f"{(hub['cer'] - stored['cer']) * 100:+.2f}",
            f"{(hub['doer'] - stored['doer']) * 100:+.2f}",
            f"{hub['ms_per_char'] - stored['ms_per_char']:+.4f}",
            "",
        ])
    return tabulate(
        rows, headers=headers, tablefmt="fancy_grid",
        colalign=("left", "left", "right", "right", "right", "right", "right"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N examples (stored baseline sliced to match).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = f"{args.dataset}_dataset"

    from tnqeet.data import val_dataset, test_dataset

    dataset = val_dataset if args.dataset == "val" else test_dataset
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset)))) # type: ignore

    print(f"Comparing {len(CONFIGS)} model(s) on {dataset_name} "
          f"({len(dataset)} examples) | device={device}")  # type: ignore
    print("Selected configs:", ", ".join(c["label"] for c in CONFIGS))
    print()

    comparisons = []
    for cfg in CONFIGS:
        hub = run_hub_eval(cfg, dataset, device)
        stored = load_stored_eval(stored_results_path(cfg, dataset_name), args.limit)
        comparisons.append((cfg, hub, stored))

    print()
    print(f"Hub-vs-local comparison on {dataset_name} "
          f"(lower is better; Δ near zero ⇒ uploaded weights match)")
    print()
    print(build_table(comparisons))


if __name__ == "__main__":
    main()
