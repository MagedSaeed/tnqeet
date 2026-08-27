"""How much context does Arabic dot restoration need?

Feeds each model only the first ``N`` words of every dev example, scores the
prediction against those ``N`` words, and sweeps ``N`` over a range of context
lengths -- a falling curve means longer context helps. A thin orchestrator (like
``bigru_evaluation/evaluate.py``): it hands truncated ``{text, source}`` data to
each method's own ``evaluate/val_dataset.py:evaluate_model``, so Rasm reduction,
chunking, and WER/CER/DotER stay identical to the package's real eval runs.

Outputs under ``results/<dataset>/``: per-example JSON at ``<method>/`` (resumable),
the grid JSON, one CSV per metric, and ``context_length_tables.txt``.

    python tnqeet_scripts/context_length_experiment/run.py
    python tnqeet_scripts/context_length_experiment/run.py --models bilstm transformer
    python tnqeet_scripts/context_length_experiment/run.py --tabulate-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabulate import tabulate

import datasets
from tnqeet import constants
from tnqeet.data import val_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"

# Context lengths (in words) to probe -- the rows of the final table. The dev set
# tops out at 3889 words but only ~13 examples exceed 1000, so beyond ~500 the
# curve is fully saturated; extend with --context-lengths for the (noisy) tail.
CONTEXT_LENGTHS = [1, 2, 3, 5, 8, 10, 20, 30, 50, 80, 100, 200, 300, 500, 1000]

# Summary keys returned by every reused evaluate_model, mapped to display labels.
METRICS = {"avg_wer": "WER", "avg_cer": "CER", "avg_doer": "DotER"}
METRIC_NAMES = ("wer", "cer", "doer")

MODEL_KEYS = ["ngram", "bilstm", "transformer", "canine"]


# --------------------------------------------------------------------------- #
# Per-method runners
#
# Each builder loads the best model ONCE and returns (run, label), where
# run(dataset, results_dir, overwrite) delegates to that method's own
# evaluate_model. Defaults are the paper's best configs (matching
# bigru_evaluation): 8-gram/beam60, BiLSTM-6L, Transformer-12L, CANINE-s.
# --------------------------------------------------------------------------- #


def build_ngram_runner(device, order: int, beam_size: int):
    from tnqeet.dotting_models.ngrams.evaluate.val_dataset import evaluate_model

    def run(dataset, results_dir, overwrite):
        return evaluate_model(
            ngrams=order,
            beam_size=beam_size,
            dataset=dataset,
            dataset_name="context_length",
            results_dir=results_dir,
            overwrite=overwrite,
        )

    return run, f"{order}-gram (beam={beam_size})"


def build_bilstm_runner(device, num_layers: int):
    from tnqeet.dotting_models.sequence_labeling.evaluate.val_dataset import evaluate_model
    from tnqeet.dotting_models.sequence_labeling.models import LSTMDottingModel

    model = LSTMDottingModel.from_pretrained(f"{num_layers}L").to(device)
    model.eval()

    def run(dataset, results_dir, overwrite):
        return evaluate_model(
            model=model,
            dataset=dataset,
            dataset_name="context_length",
            n_layers=num_layers,
            results_dir=results_dir,
            overwrite=overwrite,
        )

    return run, f"BiLSTM ({num_layers}L)"


def build_transformer_runner(device, num_layers: int):
    from tnqeet.dotting_models.transformer.evaluate.val_dataset import evaluate_model
    from tnqeet.dotting_models.transformer.models import TransformerDottingModel

    model = TransformerDottingModel.from_pretrained(f"{num_layers}L").to(device)
    model.eval()

    def run(dataset, results_dir, overwrite):
        return evaluate_model(
            model=model,
            dataset=dataset,
            dataset_name="context_length",
            n_layers=num_layers,
            results_dir=results_dir,
            overwrite=overwrite,
        )

    return run, f"Transformer ({num_layers}L)"


def build_canine_runner(device, variant: str):
    from tnqeet.dotting_models.canine.evaluate.val_dataset import evaluate_model
    from tnqeet.dotting_models.canine.models import CanineDottingModel

    model = CanineDottingModel.from_pretrained(variant.replace("canine-", "")).to(device)
    model.eval()

    def run(dataset, results_dir, overwrite):
        return evaluate_model(
            model=model,
            dataset=dataset,
            dataset_name="context_length",
            model_name=f"CANINE-{variant}",
            results_dir=results_dir,
            overwrite=overwrite,
        )

    return run, f"CANINE ({variant})"


def build_runner(model_key: str, args, device):
    if model_key == "ngram":
        return build_ngram_runner(device, args.ngram_order, args.ngram_beam)
    if model_key == "bilstm":
        return build_bilstm_runner(device, args.bilstm_layers)
    if model_key == "transformer":
        return build_transformer_runner(device, args.transformer_layers)
    if model_key == "canine":
        return build_canine_runner(device, args.canine)
    raise KeyError(model_key)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #


def load_examples(dataset, min_words: int, limit: int | None):
    """Return (texts, sources) for examples with at least ``min_words`` words."""
    texts, sources = [], []
    for example in dataset:
        if len(example["text"].split()) >= min_words:
            texts.append(example["text"])
            sources.append(example["source"])
    if limit is not None:
        texts, sources = texts[:limit], sources[:limit]
    return texts, sources


def build_flat_dataset(texts, sources, context_lengths):
    """Flatten (example x context length) into one deduplicated dataset.

    A word-prefix equals the whole text once the length reaches its word count, so
    each distinct prefix is emitted once. ``row_for[(example_idx, prefix_len)]`` locates
    a prefix's row; ``prefix_len_for[(example_idx, context_len)] = min(context_len, word_count)``
    -- together they map every (example, context length) cell back to its computed row.
    """
    prefix_texts: list[str] = []
    prefix_sources: list[str] = []
    row_for: dict[tuple[int, int], int] = {}
    prefix_len_for: dict[tuple[int, int], int] = {}
    for example_idx, (text, source) in enumerate(zip(texts, sources)):
        words = text.split()
        word_count = len(words)
        for context_len in context_lengths:
            prefix_len = min(context_len, word_count)
            prefix_len_for[(example_idx, context_len)] = prefix_len
            if (example_idx, prefix_len) not in row_for:
                # len() before append == the index this prefix will occupy.
                row_for[(example_idx, prefix_len)] = len(prefix_texts)
                prefix_texts.append(" ".join(words[:prefix_len]))
                prefix_sources.append(source)
    dataset = datasets.Dataset.from_dict({"text": prefix_texts, "source": prefix_sources})
    return dataset, row_for, prefix_len_for


def has_ambiguous_rasm(dotless_text: str) -> bool:
    """True if the text has any letter with a dotting decision to make.

    Prefixes made only of unambiguous characters (digits, punctuation, undottable
    letters like ك ل م ء) carry no dotting signal, so they are excluded from the
    averages -- e.g. a one-word prefix ")3(" contributes nothing to score.
    """
    return any(constants.is_ambigous_rasm(char) for char in dotless_text)


def aggregate_column(records, row_for, prefix_len_for, context_lengths, num_examples):
    """Fold flat per-row records into one model's {N: {avg_wer, avg_cer, avg_doer, num_samples}}.

    Only prefixes containing an ambiguous rasm are averaged; ``num_samples`` reports
    how many qualified at each context length (it varies with N, but is the same for
    every model since the dotless text is identical)."""
    column: dict[str, dict] = {}
    for context_len in context_lengths:
        totals = {metric: 0.0 for metric in METRIC_NAMES}
        num_samples = 0
        for example_idx in range(num_examples):
            prefix_len = prefix_len_for[(example_idx, context_len)]
            record = records[row_for[(example_idx, prefix_len)]]
            if not has_ambiguous_rasm(record["dotless_text"]):
                continue
            num_samples += 1
            for metric in totals:
                totals[metric] += record[metric]
        entry = {
            f"avg_{metric}": (totals[metric] / num_samples if num_samples else None)
            for metric in METRIC_NAMES
        }
        entry["num_samples"] = num_samples
        column[str(context_len)] = entry
    return column


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def render_metric_table(grid, model_keys, labels, context_lengths, metric_key):
    """Return (table_str, headers, rows) for one metric: rows = context words, cols = models.

    A ``#samples`` column reports how many prefixes carried a dotting decision at each
    context length (model-independent, so read from the first present model)."""
    present_models = [key for key in model_keys if key in grid]
    sample_counts = grid[present_models[0]] if present_models else {}
    headers = ["ctx words", "#samples"] + [labels[key] for key in present_models]
    rows = []
    for context_len in context_lengths:
        num_samples = sample_counts.get(str(context_len), {}).get("num_samples")
        cells = [str(context_len), str(num_samples) if num_samples is not None else "-"]
        for key in present_models:
            value = grid.get(key, {}).get(str(context_len), {}).get(metric_key)
            cells.append(f"{value * 100:6.2f}" if value is not None else "   -  ")
        rows.append(cells)
    table = tabulate(rows, headers=headers, tablefmt="fancy_grid", colalign=("right",) * len(headers))
    return table, headers, rows


def write_csv(path: Path, headers, rows):
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def write_report(grid, model_keys, labels, context_lengths, results_dir: Path, num_examples):
    """Save the grid JSON, one CSV per metric, and the rendered tables; also print them."""
    (results_dir / "context_length_grid.json").write_text(
        json.dumps(
            {"num_examples": num_examples, "context_lengths": context_lengths, "grid": grid},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    title = "Context-length ablation on val_dataset (lower is better; prefixes without an ambiguous rasm excluded)"
    if num_examples is not None:
        title += f" -- {num_examples} examples"
    sections = [title]
    for metric_key, metric_label in METRICS.items():
        table, headers, rows = render_metric_table(grid, model_keys, labels, context_lengths, metric_key)
        sections.append(f"\n### {metric_label} %  (rows = context words, cols = models)\n\n{table}")
        write_csv(results_dir / f"context_length_{metric_label.lower()}.csv", headers, rows)
    report = "\n".join(sections)
    print("\n" + report)
    (results_dir / "context_length_tables.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved grid, per-metric CSVs, and tables to {results_dir}")


def read_evaluation_records(model_dir: Path):
    """Read back the single per-row JSON that evaluate_model wrote into ``model_dir``."""
    result_paths = [p for p in model_dir.glob("*.json") if p.name != "context_length_grid.json"]
    return json.loads(sorted(result_paths)[0].read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=MODEL_KEYS)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=CONTEXT_LENGTHS,
        help="context word counts (rows of the table)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=1,
        help="only evaluate examples with at least this many words (default: 1, the whole dev set)",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap number of examples (for quick smoke tests)")
    parser.add_argument("--dataset-name", type=str, default="val_dataset")
    parser.add_argument("--overwrite", action="store_true", help="recompute instead of loading cached results")
    parser.add_argument("--tabulate-only", action="store_true", help="only print/save tables from cached results")
    # per-method configuration (paper's best configs by default)
    parser.add_argument("--ngram-order", type=int, default=8)
    parser.add_argument("--ngram-beam", type=int, default=60)
    parser.add_argument("--bilstm-layers", type=int, default=6)
    parser.add_argument("--transformer-layers", type=int, default=12)
    parser.add_argument("--canine", type=str, default="canine-s")
    parser.add_argument("--device", type=str, default=None, help="torch device for neural models (default: auto)")
    return parser.parse_args()


def resolve_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def model_labels(args) -> dict[str, str]:
    return {
        "ngram": f"{args.ngram_order}-gram",
        "bilstm": f"BiLSTM {args.bilstm_layers}L",
        "transformer": f"Transformer {args.transformer_layers}L",
        "canine": f"CANINE {args.canine}",
    }


def main() -> None:
    args = parse_args()
    context_lengths = args.context_lengths
    labels = model_labels(args)
    results_dir = RESULTS_ROOT / args.dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.tabulate_only:
        grid_file = results_dir / "context_length_grid.json"
        if not grid_file.exists():
            raise SystemExit(f"No cached grid at {grid_file}; run without --tabulate-only first.")
        cached = json.loads(grid_file.read_text(encoding="utf-8"))
        write_report(cached["grid"], args.models, labels, context_lengths, results_dir, cached.get("num_examples"))
        return

    device = resolve_device(args.device)
    texts, sources = load_examples(val_dataset, args.min_words, args.limit)
    # One deduplicated pass per model: each distinct word-prefix is scored once.
    flat_dataset, row_for, prefix_len_for = build_flat_dataset(texts, sources, context_lengths)
    print(
        f"Evaluating {args.models} over context lengths {context_lengths} on "
        f"{len(texts)} examples (>= {args.min_words} words each), device={device}."
    )
    print(
        f"Flattened to {len(flat_dataset)} distinct prefixes "
        f"(vs {len(texts) * len(context_lengths)} naive). Results -> {results_dir}"
    )

    grid: dict[str, dict[str, dict]] = {}
    for model_key in args.models:
        print(f"\n=== Loading '{model_key}' ===")
        try:
            run, display = build_runner(model_key, args, device)
        except Exception as error:
            print(f"[skip] could not load '{model_key}': {error}")
            continue
        print(f"=== Evaluating {display} on {len(flat_dataset)} prefixes ===")
        model_dir = results_dir / model_key
        run(flat_dataset, str(model_dir), args.overwrite)
        records = read_evaluation_records(model_dir)
        column = aggregate_column(records, row_for, prefix_len_for, context_lengths, len(texts))
        grid[model_key] = column
        for context_len in context_lengths:
            scores = column[str(context_len)]
            metric_str = "  ".join(
                f"{label}={scores[key] * 100:5.2f}%" if scores[key] is not None else f"{label}=  -  "
                for key, label in METRICS.items()
            )
            print(f"  {display} ctx={context_len:>4} (n={scores['num_samples']:>4}): {metric_str}")

    write_report(grid, args.models, labels, context_lengths, results_dir, num_examples=len(texts))


if __name__ == "__main__":
    main()
