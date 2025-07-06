"""
This file is used to fix the following:

when calculating WER, CER, and DOER metrics, \
the predicted dotted text was not sliced to match the length of the original dotted text. \
This file re-evaluates the results for all models calculating metrics for the predicted dotted text \
up to the length of the original dotted text.
"""

import os
import json
from tqdm.auto import tqdm
from tnqeet.evaluate.metrics import wer, cer, doer

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
    # "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
    "qwen-3-32b": "qwen/qwen3-32b",
    "gemma-3-27b": "google/gemma-3-27b-it",
}


def evaluate_model(
    model_name,
    dataset_name="val_dataset",
    num_fewshot=0,
    evaluation_type="zeroshot",
    prompt_type="default",
):
    if num_fewshot == 0:
        evaluation_type = "zeroshot"
    elif num_fewshot > 0:
        evaluation_type = f"fewshot_{num_fewshot}"
    else:
        raise ValueError(f"Unknown fewshot value: {num_fewshot}")
    results_dir = (
        f"tnqeet/dotting_models/llms/evaluation_results/{dataset_name}/{evaluation_type}/{prompt_type}_prompt"
    )
    results_file = os.path.join(results_dir, f"{model_name}.json")
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        print(f"Results for {model_name} already exist. Loading from {results_file}")
        with open(results_file, "r", encoding="utf-8") as f:
            per_example_results = json.load(f)
    else:
        raise FileNotFoundError(
            f"Results file {results_file} does not exist or is empty. Please run the evaluation first."
        )
    fixed_results = []
    for i, example in tqdm(
        enumerate(per_example_results),
        desc=f"Evaluating {model_name}",
    ):
        original_dotted_text = example["original_dotted_text"]  # type:ignore
        predicted_dotted_text = example["predicted_dotted_text"]
        fixed_results.append(
            {
                "original_dotted_text": original_dotted_text,
                "dotless_text": example["dotless_text"],
                "predicted_dotted_text": predicted_dotted_text,
                "text_source": example["text_source"],  # type:ignore
                "wer": wer(original_dotted_text, predicted_dotted_text[: len(original_dotted_text)]),  # type:ignore
                "cer": cer(original_dotted_text, predicted_dotted_text[: len(original_dotted_text)]),  # type:ignore
                "doer": doer(original_dotted_text, predicted_dotted_text[: len(original_dotted_text)]),  # type:ignore
                "dotting_time": example["dotting_time"],  # type:ignore
                "tokens": example["tokens"],  # type:ignore
                "raw_dspy_logs": example["raw_dspy_logs"],  # type:ignore
            }
        )
    # Save final results to file
    json.dump(
        fixed_results,
        open(
            results_file,
            "w",
            encoding="utf-8",
        ),
        ensure_ascii=False,
        indent=4,
        default=str,
    )
    summary = {
        "avg_wer": sum(result["wer"] for result in fixed_results) / len(fixed_results),
        "avg_cer": sum(result["cer"] for result in fixed_results) / len(fixed_results),
        "avg_doer": sum(result["doer"] for result in fixed_results) / len(fixed_results),
    }
    return summary


# print(evaluate_model("claude-sonnet-4"))
# print(evaluate_model("gemini-2.0"))
# print(evaluate_model("qwen-3"))
# print(evaluate_model('gemma-3'))

# for prompt_type in ("default", "detailed"):
for prompt_type in ("default", "detailed"):
    for fewshot in (0, 1, 3, 5, 8, 10):
        for model in list(OPEN_ROUTER_MODELS.keys()):
            summary = evaluate_model(
                model_name=model,
                prompt_type=prompt_type,
                num_fewshot=fewshot,
            )
            print(f"Summary for {model} with {prompt_type} prompt and {fewshot} fewshot: {summary}")
            print("-" * 120)
        print("=" * 120)
# print("=" * 120)
# for model in list(OPEN_ROUTER_MODELS.keys()):
#     summary = evaluate_model(
#         model_name=model,
#         prompt_type="detailed",
#     )
#     print(f"Summary for {model} with detailed prompt: {summary}")
#     print("-" * 120)


# TODO:
# - check failed requests
# - add cost (input/output tokens)
