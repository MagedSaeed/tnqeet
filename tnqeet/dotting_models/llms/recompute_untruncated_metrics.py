"""Recompute LLM evaluation metrics from stored predictions, without truncation.

The evaluation scripts (``evaluate/test_dataset.py`` / ``evaluate/val_dataset.py``)
store the *full*, untruncated ``predicted_dotted_text`` for every example, but the
reported WER/CER/DOER are computed on ``predicted_dotted_text[: len(reference)]`` --
i.e. the hypothesis is truncated to the reference length before scoring.

This script re-scores the already-stored predictions **without any new API calls**,
so it costs nothing to run. It can report the untruncated numbers, the truncated
numbers (matching the paper), or both side by side.

Usage
-----
    # Test set, truncated vs untruncated side by side (defaults)
    python -m tnqeet.dotting_models.llms.recompute_untruncated_metrics

    # Dev/validation set instead of test
    python -m tnqeet.dotting_models.llms.recompute_untruncated_metrics --dataset dev

    # Only the untruncated numbers
    python -m tnqeet.dotting_models.llms.recompute_untruncated_metrics --mode untruncated

    # Filter to specific configurations
    python -m tnqeet.dotting_models.llms.recompute_untruncated_metrics \
        --dataset dev --evaluation-type fewshot_8 --prompt-type default_prompt
"""

import argparse
import json
from pathlib import Path

from tnqeet.evaluate.metrics import wer, cer, doer

RESULTS_ROOT = Path(__file__).resolve().parent / "evaluation_results"

# Friendly dataset aliases -> results subdirectory name.
DATASET_DIRS = {
    "test": "test_dataset",
    "dev": "val_dataset",
    "val": "val_dataset",
    "noon": "val_dataset_noon_comparison",
}

METRIC_FNS = {"wer": wer, "cer": cer, "doer": doer}

# Metrics are edit-distance ratios in [0, 1]; report them as percentages (out of 100).
SCALE = 100.0


def score_file(path, truncate):
    """Return {'wer': .., 'cer': .., 'doer': .., 'n': ..} averaged over a results file.

    WER/CER/DOER are scaled to percentages (out of 100).
    """
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not results:
        return None
    totals = {name: 0.0 for name in METRIC_FNS}
    for r in results:
        ref = r["original_dotted_text"]
        pred = r["predicted_dotted_text"]
        hyp = pred[: len(ref)] if truncate else pred
        for name, fn in METRIC_FNS.items():
            totals[name] += fn(ref, hyp)
    n = len(results)
    scores = {name: SCALE * totals[name] / n for name in METRIC_FNS}
    scores["n"] = n
    return scores


def discover_configs(dataset_dir, evaluation_type=None, prompt_type=None):
    """Yield (evaluation_type, prompt_type, model_name, json_path) tuples.

    ``evaluation_type`` / ``prompt_type`` here are the intermediate directory names
    (``fewshot_8``, ``default_prompt``, ...). For ``val_dataset_noon_comparison`` the
    first level is ``with_noon`` / ``without_noon`` instead of a fewshot type; the
    same discovery logic still applies.
    """
    for eval_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        if evaluation_type and eval_dir.name != evaluation_type:
            continue
        for prompt_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
            if prompt_type and prompt_dir.name != prompt_type:
                continue
            for json_path in sorted(prompt_dir.glob("*.json")):
                yield eval_dir.name, prompt_dir.name, json_path.stem, json_path


def format_table(rows, mode):
    """Render rows as a fixed-width text table. ``rows`` are dicts."""
    if mode == "compare":
        header = ["model", "n", "wer_t", "wer_u", "cer_t", "cer_u", "doer_t", "doer_u"]
    else:
        header = ["model", "n", "wer", "cer", "doer"]

    def cell(row, key):
        if key == "model":
            return row["model"]
        if key == "n":
            return str(row["n"])
        return f"{row[key]:.2f}"

    table = [header] + [[cell(r, k) for k in header] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = []
    for idx, row in enumerate(table):
        lines.append("  ".join(val.ljust(widths[i]) for i, val in enumerate(row)))
        if idx == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(header))))
    return "\n".join(lines)


def build_rows(configs, mode):
    """Score each config file and return sorted rows (best WER first)."""
    rows = []
    for _eval, _prompt, model_name, json_path in configs:
        row = {"model": model_name}
        if mode in ("truncated", "compare"):
            t = score_file(json_path, truncate=True)
            if t is None:
                continue
            row["n"] = t["n"]
            if mode == "truncated":
                row.update({"wer": t["wer"], "cer": t["cer"], "doer": t["doer"]})
            else:
                row.update({"wer_t": t["wer"], "cer_t": t["cer"], "doer_t": t["doer"]})
        if mode in ("untruncated", "compare"):
            u = score_file(json_path, truncate=False)
            if u is None:
                continue
            row["n"] = u["n"]
            if mode == "untruncated":
                row.update({"wer": u["wer"], "cer": u["cer"], "doer": u["doer"]})
            else:
                row.update({"wer_u": u["wer"], "cer_u": u["cer"], "doer_u": u["doer"]})
        rows.append(row)
    sort_key = "wer" if mode in ("truncated", "untruncated") else "wer_u"
    rows.sort(key=lambda r: r[sort_key])
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Recompute LLM metrics from stored predictions (no API calls).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_DIRS.keys()),
        default="test",
        help="Which stored results to score.",
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "truncated", "untruncated"],
        default="compare",
        help="Report untruncated, truncated (paper), or both side by side.",
    )
    parser.add_argument(
        "--evaluation-type",
        default=None,
        help="Filter to one config dir, e.g. 'fewshot_8', 'zeroshot', 'with_noon'.",
    )
    parser.add_argument(
        "--prompt-type",
        default=None,
        help="Filter to one prompt dir, e.g. 'default_prompt', 'detailed_prompt'.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text tables.",
    )
    args = parser.parse_args()

    dataset_dir = RESULTS_ROOT / DATASET_DIRS[args.dataset]
    if not dataset_dir.is_dir():
        parser.error(f"No stored results at {dataset_dir}")

    configs = list(discover_configs(dataset_dir, args.evaluation_type, args.prompt_type))
    if not configs:
        parser.error("No result files matched the given filters.")

    # Group configs by (evaluation_type, prompt_type) so each table is homogeneous.
    grouped = {}
    for eval_name, prompt_name, model_name, json_path in configs:
        grouped.setdefault((eval_name, prompt_name), []).append(
            (eval_name, prompt_name, model_name, json_path)
        )

    if args.json:
        payload = {}
        for (eval_name, prompt_name), group in grouped.items():
            payload[f"{eval_name}/{prompt_name}"] = build_rows(group, args.mode)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    trunc_note = {
        "compare": "columns: _t = truncated (paper), _u = untruncated",
        "truncated": "truncated to reference length (as reported in the paper)",
        "untruncated": "full model output, no truncation",
    }[args.mode]
    print(f"Dataset: {args.dataset} ({DATASET_DIRS[args.dataset]})   mode: {args.mode}")
    print(f"{trunc_note}   (WER/CER/DOER as %, out of 100)")
    for (eval_name, prompt_name), group in grouped.items():
        rows = build_rows(group, args.mode)
        if not rows:
            continue
        print()
        print(f"### {eval_name} / {prompt_name}")
        print(format_table(rows, args.mode))


if __name__ == "__main__":
    main()
