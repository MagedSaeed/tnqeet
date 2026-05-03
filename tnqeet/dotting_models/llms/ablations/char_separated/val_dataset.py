"""
Ablation: character-separated input/output format.

Hypothesis: BPE tokenizers merge multiple Arabic characters into single tokens,
so the LLM never "sees" individual glyphs. Forcing one-character-per-token via
single-space separation (with '|' marking word boundaries) should give the
model a cleaner view of each Rasm when deciding which dotted letter it maps to.

Zero-shot only. Runs on the 120-sample llms_val_dataset. Same 3 models that
were evaluated on the test set: claude-sonnet-4, gpt-4o, gemini-2.5-flash-preview.
"""
import json
import os
import re
import unicodedata
from datetime import datetime

import dspy
from tqdm.auto import tqdm

from tnqeet import remove_dots
from tnqeet.data import llms_val_dataset
from tnqeet.dotting_models.llms.models import OpenRouterArabicDotter
from tnqeet.evaluate.metrics import cer, doer, wer

WORD_SEP = "|"
CHAR_SEP = " "
WORD_SEP_WRAPPED = f" {WORD_SEP} "

OPEN_ROUTER_MODELS = {
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "gpt-4o": "openai/gpt-4o-2024-11-20",
    "gemini-2.5-flash-preview": "google/gemini-2.5-flash",
}


def _iter_grapheme_clusters(word):
    """Yield clusters where combining marks (category M*) stay attached to the preceding base char."""
    cluster = ""
    for ch in word:
        if cluster and unicodedata.category(ch).startswith("M"):
            cluster += ch
        else:
            if cluster:
                yield cluster
            cluster = ch
    if cluster:
        yield cluster


def to_char_separated(text):
    """"مرحبا يا" -> "م ر ح ب ا | ي ا"."""
    words = re.split(r"\s+", text.strip())
    return WORD_SEP_WRAPPED.join(CHAR_SEP.join(_iter_grapheme_clusters(w)) for w in words)


def from_char_separated(text):
    """"م ر ح ب ا | ي ا" -> "مرحبا يا". Tolerant of extra whitespace around the sentinel."""
    words = [w.replace(" ", "") for w in text.split(WORD_SEP)]
    return " ".join(w for w in words if w)


# The format spec + example pair lives in the input field `desc` (not a class
# docstring) for two reasons: (1) we need to interpolate WORD_SEP, and Python
# only treats a *plain* string literal as the first statement of a class body
# as __doc__ -- an f-string there is a discarded expression and would leave
# DSPy falling back to its default "Given the fields..." instructions;
# (2) it matches the pattern already used by `DetailedArabicDotingSignature`
# in ../../models.py, which similarly carries its extra context on the input
# field rather than via class-level instructions.
class CharSeparatedArabicDottingSignature(dspy.Signature):
    dotless_text = dspy.InputField(
        desc=(
            f"Arabic Rasm text in character-separated form. Every character is separated by a single "
            f"space; word boundaries are marked by the sentinel '{WORD_SEP}' (surrounded by spaces). "
            f"Your output must use the same layout: same word boundaries, same number of characters "
            f"in the same order - only the identity of each dotless letter (Rasm) changes to its "
            f"correct dotted form.\n"
            f"Example input:  \"م ر ح ٮ ا {WORD_SEP} ى ا {WORD_SEP} ع ا ل م\"\n"
            f"Example output: \"م ر ح ب ا {WORD_SEP} ي ا {WORD_SEP} ع ا ل م\""
        )
    )
    dotted_text = dspy.OutputField(
        desc=f"Dotted Arabic text in the same space-separated format with '{WORD_SEP}' word boundaries."
    )


class CharSeparatedArabicDotter(OpenRouterArabicDotter):
    """OpenRouterArabicDotter with char-separation pre/postprocessing around the LM call."""

    def restore_dots(self, dotless_text):
        formatted_input = to_char_separated(dotless_text)
        prediction = self.dotter(dotless_text=formatted_input)
        return from_char_separated(prediction.dotted_text)


def evaluate_model(
    model_name,
    dataset=llms_val_dataset,
    dataset_name="val_dataset",
    overwrite=False,
    save_every=5,
    retry=5,
):
    evaluation_type = "zeroshot"
    results_dir = (
        f"tnqeet/dotting_models/llms/ablation_results/char_separated/"
        f"{dataset_name}/{evaluation_type}"
    )
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{model_name}.json")

    per_example_results = []
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0 and not overwrite:
        print(f"Results for {model_name} already exist. Loading from {results_file}")
        with open(results_file, "r", encoding="utf-8") as f:
            per_example_results = json.load(f)

    model = f"openrouter/{OPEN_ROUTER_MODELS[model_name]}"
    dotter = CharSeparatedArabicDotter(
        model=model,
        dspy_cache=False,
        signature=CharSeparatedArabicDottingSignature, # type: ignore
        num_fewshot=0,
    )

    if len(per_example_results) < len(dataset):  # type:ignore
        for i, example in tqdm(
            enumerate(dataset.select(range(len(per_example_results), len(dataset)))),  # type:ignore
            desc=f"Evaluating {model_name}",
            initial=len(per_example_results),
            total=len(dataset),  # type:ignore
        ):
            original_dotted_text = example["text"]  # type:ignore
            dotless_text = remove_dots(original_dotted_text)
            retry_count = 0
            predicted_dotted_text = None
            dotting_time = None
            while not predicted_dotted_text and retry_count < retry:
                if retry_count > 0:
                    print(
                        f"Failed to restore dots for example index {len(per_example_results)} "
                        f"after {retry_count} retries. Retrying..."
                    )
                try:
                    t0 = datetime.now()
                    predicted_dotted_text = dotter.restore_dots(dotless_text)
                    dotting_time = datetime.now() - t0
                except Exception as e:
                    print(f"Error during dot restoration: {e}")
                    predicted_dotted_text = None
                raw_dspy_logs = dotter.lm.history[-1].copy() if dotter.lm.history else {}
                retry_count += 1

            final_prediction = predicted_dotted_text if predicted_dotted_text else ""
            per_example_results.append(
                {
                    "original_dotted_text": original_dotted_text,
                    "dotless_text": dotless_text,
                    "predicted_dotted_text": final_prediction,
                    "text_source": example["source"],  # type:ignore
                    "wer": wer(original_dotted_text, final_prediction),
                    "cer": cer(original_dotted_text, final_prediction),
                    "doer": doer(original_dotted_text, final_prediction),
                    "dotting_time": dotting_time or float("inf"),
                    "tokens": raw_dspy_logs.get("usage") if raw_dspy_logs else None, # type: ignore
                    "raw_dspy_logs": raw_dspy_logs or {}, # type: ignore
                }
            )
            if i > 0 and (i + 1) % save_every == 0:
                json.dump(
                    per_example_results,
                    open(results_file, "w", encoding="utf-8"),
                    ensure_ascii=False,
                    indent=4,
                    default=str,
                )
    else:
        print(f"Skipping evaluation for {model_name} as results already exist for all examples.")

    json.dump(
        per_example_results,
        open(results_file, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
        default=str,
    )
    summary = {
        "avg_wer": sum(r["wer"] for r in per_example_results) / len(per_example_results),
        "avg_cer": sum(r["cer"] for r in per_example_results) / len(per_example_results),
        "avg_doer": sum(r["doer"] for r in per_example_results) / len(per_example_results),
    }
    return summary


if __name__ == "__main__":
    for model_name in OPEN_ROUTER_MODELS:
        summary = evaluate_model(model_name=model_name)
        print(f"Summary for {model_name} (char-separated, zeroshot): {summary}")
        print("-" * 120)
