"""Aggregate per-example test-dataset evaluation JSONs into a terminal table.

Walks each model family's `evaluation_results/test_dataset/` directory, computes
mean WER / CER / DotER / inference-time across the 5000 test examples, and prints
a grouped table (via `tabulate`) with one section per family. The lowest WER,
CER, DotER, and inference time across ALL rows are marked with a leading `*`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from statistics import mean

from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO_ROOT / "tnqeet" / "dotting_models"


@dataclass
class Row:
    method: str
    configuration: str
    wer: float
    cer: float
    doer: float
    # Inference cost: ms per source character for local models, or
    # (input_tokens, output_tokens) per example for token-billed LLMs.
    # Exactly one is set.
    ms_per_char: float | None = None
    tokens: tuple[float, float] | None = None


def parse_dotting_time(value: str) -> float:
    """Parse a `H:MM:SS.ffffff` timedelta string into seconds."""
    h, m, s = value.split(":")
    return timedelta(hours=int(h), minutes=int(m), seconds=float(s)).total_seconds()


def aggregate_time(records: list[dict]) -> tuple[float, float, float, float]:
    wer = mean(r["wer"] for r in records)
    cer = mean(r["cer"] for r in records)
    doer = mean(r["doer"] for r in records)
    per_char_ms = [
        parse_dotting_time(r["dotting_time"]) * 1000 / len(r["dotless_text"])
        for r in records
        if r.get("dotting_time") and r.get("dotless_text")
    ]
    inf = mean(per_char_ms) if per_char_ms else float("nan")
    return wer, cer, doer, inf


def aggregate_tokens(records: list[dict]) -> tuple[float, float, float, float, float]:
    wer = mean(r["wer"] for r in records)
    cer = mean(r["cer"] for r in records)
    doer = mean(r["doer"] for r in records)
    prompt = [r["tokens"]["prompt_tokens"] for r in records if r.get("tokens")]
    completion = [r["tokens"]["completion_tokens"] for r in records if r.get("tokens")]
    in_tok = mean(prompt) if prompt else float("nan")
    out_tok = mean(completion) if completion else float("nan")
    return wer, cer, doer, in_tok, out_tok


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def natural_key(text: str) -> list:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", text)]


def collect_ngrams() -> list[Row]:
    base = MODELS_ROOT / "ngrams" / "evaluation_results" / "test_dataset"
    rows: list[Row] = []
    if not base.exists():
        return rows
    for beam_dir in sorted(base.iterdir(), key=lambda p: natural_key(p.name)):
        if not beam_dir.is_dir():
            continue
        m_beam = re.search(r"(\d+)", beam_dir.name)
        beam = m_beam.group(1) if m_beam else beam_dir.name
        for json_file in sorted(beam_dir.glob("ngrams_*.json"), key=lambda p: natural_key(p.name)):
            m_n = re.search(r"ngrams_(\d+)", json_file.stem)
            n = m_n.group(1) if m_n else json_file.stem
            wer, cer, doer, ms_pc = aggregate_time(load_json(json_file))
            rows.append(Row(f"{n}-gram", f"beam={beam}", wer, cer, doer, ms_pc))
    return rows


BILSTM_LAYERS_TO_REPORT = {"2", "4", "6"}


def collect_sequence_labeling() -> list[Row]:
    base = MODELS_ROOT / "sequence_labeling" / "evaluation_results" / "test_dataset"
    rows: list[Row] = []
    if not base.exists():
        return rows
    for json_file in sorted(base.glob("LSTM_layers_*_results.json"), key=lambda p: natural_key(p.name)):
        m = re.search(r"LSTM_layers_(\d+)_results", json_file.stem)
        layers = m.group(1) if m else "?"
        if layers not in BILSTM_LAYERS_TO_REPORT:
            continue
        wer, cer, doer, ms_pc = aggregate_time(load_json(json_file))
        suffix = "layer" if layers == "1" else "layers"
        rows.append(Row("BiLSTM", f"{layers} {suffix}", wer, cer, doer, ms_pc))
    return rows


def collect_transformer() -> list[Row]:
    base = MODELS_ROOT / "transformer" / "evaluation_results" / "test_dataset"
    rows: list[Row] = []
    if not base.exists():
        return rows
    for json_file in sorted(base.glob("Transformer_layers_*_results.json"), key=lambda p: natural_key(p.name)):
        m = re.search(r"Transformer_layers_(\d+)_results", json_file.stem)
        layers = m.group(1) if m else "?"
        wer, cer, doer, ms_pc = aggregate_time(load_json(json_file))
        rows.append(Row("Transformer", f"{layers} layers", wer, cer, doer, ms_pc))
    return rows


def collect_canine() -> list[Row]:
    base = MODELS_ROOT / "canine" / "evaluation_results" / "test_dataset"
    rows: list[Row] = []
    if not base.exists():
        return rows
    for json_file in sorted(base.glob("CANINE-*_results.json"), key=lambda p: natural_key(p.name)):
        m = re.search(r"CANINE-(.+)_results", json_file.stem)
        variant = m.group(1) if m else json_file.stem
        wer, cer, doer, ms_pc = aggregate_time(load_json(json_file))
        rows.append(Row("CANINE", variant, wer, cer, doer, ms_pc))
    return rows


LLM_DISPLAY_NAMES = {
    "claude-sonnet-4": "Claude Sonnet 4",
    "gemini-2.5-flash-preview": "Gemini 2.5 Flash Preview",
    "gpt-4o": "GPT-4o",
}


def collect_llms() -> list[Row]:
    base = MODELS_ROOT / "llms" / "evaluation_results" / "test_dataset"
    rows: list[Row] = []
    if not base.exists():
        return rows
    for fewshot_dir in sorted(base.iterdir(), key=lambda p: natural_key(p.name)):
        if not fewshot_dir.is_dir():
            continue
        m = re.search(r"fewshot_(\d+)", fewshot_dir.name)
        shots = m.group(1) if m else "?"
        for prompt_dir in sorted(fewshot_dir.iterdir(), key=lambda p: natural_key(p.name)):
            if not prompt_dir.is_dir():
                continue
            for json_file in sorted(prompt_dir.glob("*.json"), key=lambda p: natural_key(p.name)):
                model_id = json_file.stem
                display = LLM_DISPLAY_NAMES.get(model_id, model_id)
                wer, cer, doer, in_tok, out_tok = aggregate_tokens(load_json(json_file))
                rows.append(
                    Row(
                        display,
                        f"{shots}-shot",
                        wer,
                        cer,
                        doer,
                        tokens=(in_tok, out_tok),
                    )
                )
    return rows


SECTIONS: list[tuple[str, callable]] = [
    ("N-gram Language Models", collect_ngrams),
    ("BiLSTM Sequence Labeling", collect_sequence_labeling),
    ("Transformer Sequence Labeling", collect_transformer),
    ("CANINE", collect_canine),
    ("Large Language Models", collect_llms),
]


def format_metric(value: float, is_best: bool, fmt: str) -> str:
    cell = format(value, fmt)
    return f"*{cell}" if is_best else f" {cell}"


def build_table(sections: list[tuple[str, list[Row]]]) -> str:
    all_rows = [r for _, rs in sections for r in rs]
    if not all_rows:
        return "No evaluation results found."

    best_wer = min(r.wer for r in all_rows)
    best_cer = min(r.cer for r in all_rows)
    best_doer = min(r.doer for r in all_rows)
    ms_values = [r.ms_per_char for r in all_rows if r.ms_per_char is not None]
    best_ms = min(ms_values) if ms_values else None

    table_rows: list[list[str]] = []
    headers = ["Method", "Setting", "WER %", "CER %", "DotER %", "Inference cost"]

    for idx, (section_title, rows) in enumerate(sections):
        if not rows:
            continue
        if idx > 0 and table_rows:
            table_rows.append(["", "", "", "", "", ""])
        table_rows.append([f"-- {section_title} --", "", "", "", "", ""])
        for r in rows:
            wer_cell = format_metric(r.wer * 100, r.wer == best_wer, "5.2f")
            cer_cell = format_metric(r.cer * 100, r.cer == best_cer, "5.2f")
            doer_cell = format_metric(r.doer * 100, r.doer == best_doer, "5.2f")
            if r.tokens is not None:
                in_tok, out_tok = r.tokens
                cost_cell = f"{in_tok:,.0f} in / {out_tok:,.0f} out tok"
            elif r.ms_per_char is not None:
                ms = format(r.ms_per_char, ".4f")
                marker = "*" if best_ms is not None and r.ms_per_char == best_ms else " "
                cost_cell = f"{marker}{ms} ms/char"
            else:
                cost_cell = ""
            table_rows.append([r.method, r.configuration, wer_cell, cer_cell, doer_cell, cost_cell])

    return tabulate(
        table_rows,
        headers=headers,
        tablefmt="fancy_grid",
        colalign=("left", "left", "right", "right", "right", "right"),
    )


def main() -> None:
    sections = [(title, collector()) for title, collector in SECTIONS]
    print("Test-dataset performance comparison (lower is better; * marks best across all rows)")
    print()
    print(build_table(sections))


if __name__ == "__main__":
    main()
