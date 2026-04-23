"""
Recompute WER, CER, and DOER metrics over saved per-example LLM results using
the full (un-sliced) predicted_dotted_text.

An earlier version of the evaluation scripts sliced the prediction to
len(original_dotted_text) before scoring, on the assumption that excess length
was "inflating" the metrics. That assumption was wrong: WER/CER/DOER use
Levenshtein distance (divided by reference length), which already handles
length mismatches correctly -- trailing characters cost one insertion each.
Slicing instead tends to *increase* error, because when the model's length
drift comes from a mid-sequence insertion, slicing chops off correctly
predicted tail characters and turns them into substitution/deletion errors.

This script rewrites every saved results JSON under evaluation_results/
(val_dataset and test_dataset) so its stored wer/cer/doer match the standard
un-sliced Levenshtein-based definition.
"""

import json
import os

from tqdm.auto import tqdm

from tnqeet.evaluate.metrics import cer, doer, wer

OPEN_ROUTER_MODELS = {
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "calude-haiku-3.5": "anthropic/claude-3.5-haiku",
    "gpt-4o": "openai/gpt-4o-2024-11-20",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gemini-2.5-flash-preview": "google/gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite-preview-06-17",
    "deepseek-r1": "deepseek/deepseek-r1",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "qwen-3-235b": "qwen/qwen3-235b-a22b",
    "qwen-3-32b": "qwen/qwen3-32b",
    "gemma-3-27b": "google/gemma-3-27b-it",
}


def recompute_file(
    model_name,
    dataset_name,
    num_fewshot,
    prompt_type,
):
    if num_fewshot == 0:
        evaluation_type = "zeroshot"
    elif num_fewshot > 0:
        evaluation_type = f"fewshot_{num_fewshot}"
    else:
        raise ValueError(f"Unknown fewshot value: {num_fewshot}")
    results_file = (
        f"tnqeet/dotting_models/llms/evaluation_results/{dataset_name}/"
        f"{evaluation_type}/{prompt_type}_prompt/{model_name}.json"
    )
    if not (os.path.exists(results_file) and os.path.getsize(results_file) > 0):
        return None
    with open(results_file, "r", encoding="utf-8") as f:
        per_example_results = json.load(f)
    fixed_results = []
    for example in tqdm(
        per_example_results,
        desc=f"{dataset_name}/{evaluation_type}/{prompt_type}_prompt/{model_name}",
    ):
        original_dotted_text = example["original_dotted_text"]
        predicted_dotted_text = example["predicted_dotted_text"]
        fixed_results.append(
            {
                "original_dotted_text": original_dotted_text,
                "dotless_text": example["dotless_text"],
                "predicted_dotted_text": predicted_dotted_text,
                "text_source": example["text_source"],
                "wer": wer(original_dotted_text, predicted_dotted_text),
                "cer": cer(original_dotted_text, predicted_dotted_text),
                "doer": doer(original_dotted_text, predicted_dotted_text),
                "dotting_time": example["dotting_time"],
                "tokens": example["tokens"],
                "raw_dspy_logs": example["raw_dspy_logs"],
            }
        )
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(fixed_results, f, ensure_ascii=False, indent=4, default=str)
    return {
        "avg_wer": sum(r["wer"] for r in fixed_results) / len(fixed_results),
        "avg_cer": sum(r["cer"] for r in fixed_results) / len(fixed_results),
        "avg_doer": sum(r["doer"] for r in fixed_results) / len(fixed_results),
    }


if __name__ == "__main__":
    for dataset_name in ("val_dataset", "test_dataset"):
        for prompt_type in ("default", "detailed"):
            for fewshot in (0, 1, 3, 5, 8, 10):
                for model in OPEN_ROUTER_MODELS:
                    summary = recompute_file(
                        model_name=model,
                        dataset_name=dataset_name,
                        num_fewshot=fewshot,
                        prompt_type=prompt_type,
                    )
                    if summary is None:
                        continue
                    print(
                        f"[{dataset_name}] {model} | {prompt_type} prompt | "
                        f"{fewshot}-shot -> {summary}"
                    )
