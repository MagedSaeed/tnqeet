"""Check whether the misplaced ``ن`` in the detailed prompt changed anything.

``models.mapping_desc`` used to list BAA_RASM as ``ب ت ث ن``. The ``ن`` does not belong
there, since it has its own NOON_RASM entry, so it was dropped. Only
``DetailedArabicDotingSignature`` reads ``mapping_desc``, so no other run is affected.

To see whether it mattered, this runs the detailed prompt over the dev set twice for the
three models we report on the test set: once with the ``ن`` put back, once with it gone,
then compares the scores. The two prompts differ by that one letter and nothing else, so
any difference in WER, CER or DOER comes from it. Running this file runs only the
comparison; it does not import the regular ``val_dataset.py`` runner.

Results are written to their own directory, leaving the existing ones alone:

    evaluation_results/val_dataset_noon_comparison/{with,without}_noon/detailed_prompt/<model>.json

The numbers already committed under ``val_dataset/fewshot_8/detailed_prompt/`` came from
the old mapping, so they double as a sanity check on ``with_noon``.
"""

import os
import json
from datetime import datetime

import dspy
from tqdm.auto import tqdm

from tnqeet import remove_dots
from tnqeet.data import llms_val_dataset
from tnqeet.dotting_models.llms import models as llm_models
from tnqeet.dotting_models.llms.models import OpenRouterArabicDotter
from tnqeet.evaluate.metrics import wer, cer, doer

# --- The three models that were evaluated on the test set -----------------
# Routing mirrors tnqeet/dotting_models/llms/evaluate/val_dataset.py (OpenRouter,
# the model string is prefixed with ``openrouter/`` before being handed to dspy).
MODELS = {
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "gemini-2.5-flash-preview": "google/gemini-2.5-flash",
    "gpt-4o": "openai/gpt-4o-2024-11-20",
}

NUM_FEWSHOT = 8

# --- Build the two mapping variants ---------------------------------------
# ``llm_models.mapping_desc`` is the FIXED mapping (no ``ن`` under BAA_RASM).
# The buggy variant is reconstructed by re-inserting ``ن`` on the BAA line,
# guaranteeing the two prompts differ by exactly that one letter.
MAPPING_WITHOUT_NOON = llm_models.mapping_desc
MAPPING_WITH_NOON = MAPPING_WITHOUT_NOON.replace(
    "(BAA_RASM) → ب ت ث",
    "(BAA_RASM) → ب ت ث ن",
)
assert MAPPING_WITH_NOON != MAPPING_WITHOUT_NOON, "failed to build the buggy mapping variant"


def _make_detailed_signature(mapping_desc):
    """Build a DetailedArabicDotingSignature clone using ``mapping_desc``."""

    class _DetailedSignature(dspy.Signature):
        dotless_text = dspy.InputField(
            desc=(
                "Arabic text without dots (Rasm) - simplified letter forms using "
                f"direct character mapping as in the following:\n\n{mapping_desc}"
            )
        )
        dotted_text = dspy.OutputField(
            desc=(
                "Properly dotted Arabic text with correct diacritical marks restored "
                "based on context and meaning"
            )
        )

    return _DetailedSignature


CONDITIONS = {
    "with_noon": _make_detailed_signature(MAPPING_WITH_NOON),
    "without_noon": _make_detailed_signature(MAPPING_WITHOUT_NOON),
}


def evaluate_model(model_name, condition, signature, dataset=llms_val_dataset, save_every=5, retry=5):
    results_dir = (
        f"tnqeet/dotting_models/llms/evaluation_results/val_dataset_noon_comparison/"
        f"{condition}/detailed_prompt"
    )
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{model_name}.json")

    per_example_results = []
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        print(f"Results for [{condition}] {model_name} already exist. Loading from {results_file}")
        with open(results_file, "r", encoding="utf-8") as f:
            per_example_results = json.load(f)

    model = f"openrouter/{MODELS[model_name]}"
    dotter = OpenRouterArabicDotter(
        model=model,
        dspy_cache=False,
        signature=signature,
        num_fewshot=NUM_FEWSHOT,
    )
    if len(per_example_results) < len(dataset):
        for i, example in tqdm(
            enumerate(dataset.select(range(len(per_example_results), len(dataset)))),
            desc=f"Evaluating [{condition}] {model_name}",
            initial=len(per_example_results),
            total=len(dataset),
        ):
            original_dotted_text = example["text"]
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
                    time_before_prediction = datetime.now()
                    predicted_dotted_text = dotter.restore_dots(dotless_text)
                    time_after_prediction = datetime.now()
                    dotting_time = time_after_prediction - time_before_prediction
                except Exception as e:
                    print(f"Error during dot restoration: {e}")
                    predicted_dotted_text = None
                raw_dspy_logs = dotter.lm.history[-1].copy() if dotter.lm.history else {}
                retry_count += 1
            per_example_results.append(
                {
                    "original_dotted_text": original_dotted_text,
                    "dotless_text": dotless_text,
                    "predicted_dotted_text": predicted_dotted_text if predicted_dotted_text else "",
                    "text_source": example["source"],
                    "wer": wer(original_dotted_text, predicted_dotted_text[: len(original_dotted_text)]),
                    "cer": cer(original_dotted_text, predicted_dotted_text[: len(original_dotted_text)]),
                    "doer": doer(original_dotted_text, predicted_dotted_text[: len(original_dotted_text)]),
                    "dotting_time": dotting_time or float("inf"),
                    "tokens": raw_dspy_logs["usage"] if raw_dspy_logs else None,
                    "raw_dspy_logs": raw_dspy_logs or {},
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
        print(f"Skipping [{condition}] {model_name}; results already exist for all examples.")

    json.dump(
        per_example_results,
        open(results_file, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
        default=str,
    )
    return {
        "avg_wer": sum(r["wer"] for r in per_example_results) / len(per_example_results),
        "avg_cer": sum(r["cer"] for r in per_example_results) / len(per_example_results),
        "avg_doer": sum(r["doer"] for r in per_example_results) / len(per_example_results),
    }


summaries = {}
for condition, signature in CONDITIONS.items():
    summaries[condition] = {}
    for model_name in MODELS:
        summary = evaluate_model(model_name=model_name, condition=condition, signature=signature)
        summaries[condition][model_name] = summary
        print(f"[{condition}] {model_name}: {summary}")
        print("-" * 120)
    print("=" * 120)

# --- Before/after diff table --------------------------------------------------
# Metrics are reported as percentages (x100) so the small deltas are easy to read.
print("\nBefore (with ن) → After (without ن)  [detailed prompt, dev/val set]  (values are %)\n")
header = f"{'model':<26} {'metric':<6} {'with_noon':>10} {'without_noon':>13} {'delta':>10}"
print(header)
print("-" * len(header))
for model_name in MODELS:
    before = summaries["with_noon"][model_name]
    after = summaries["without_noon"][model_name]
    for metric in ("avg_wer", "avg_cer", "avg_doer"):
        b, a = before[metric] * 100, after[metric] * 100
        print(f"{model_name:<26} {metric.replace('avg_', ''):<6} {b:>10.4f} {a:>13.4f} {a - b:>+10.4f}")
    print("-" * len(header))
