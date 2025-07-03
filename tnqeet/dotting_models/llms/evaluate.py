import os
import json
from datetime import datetime
from tqdm.auto import tqdm
from tnqeet.dotting_models.llms.models import (
    OpenRouterArabicDotter,
    ArabicDottingSignature,
    DetailedArabicDotingSignature,
)
from tnqeet.data import val_dataset
from tnqeet import remove_dots
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
    dataset=val_dataset,
    dataset_name="val_dataset",
    num_fewshot=0,
    evaluation_type="zeroshot",
    prompt_type="default",
    overwrite=False,
    save_every=5,
    retry=5,
):
    if prompt_type == "default":
        signature = ArabicDottingSignature
    elif prompt_type == "detailed":
        signature = DetailedArabicDotingSignature
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    if num_fewshot == 0:
        evaluation_type = "zeroshot"
    elif num_fewshot > 0:
        evaluation_type = f"fewshot_{num_fewshot}"
    else:
        raise ValueError(f"Unknown fewshot value: {num_fewshot}")
    assert retry > 0 and isinstance(retry, int), "retry must be an integer greater than 0"
    results_dir = f"tnqeet/dotting_models/llms/test_results/{dataset_name}/{evaluation_type}/{prompt_type}_prompt"
    os.makedirs(results_dir, exist_ok=True)
    # make sure {model_name}.json file exists in the results_dir
    results_file = os.path.join(results_dir, f"{model_name}.json")
    per_example_results = []
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0 and not overwrite:
        print(f"Results for {model_name} already exist. Loading from {results_file}")
        with open(results_file, "r", encoding="utf-8") as f:
            per_example_results = json.load(f)
    model = OPEN_ROUTER_MODELS[model_name]
    dotter = OpenRouterArabicDotter(
        model=model,
        dspy_cache=False,
        signature=signature,  # type: ignore
        num_fewshot=num_fewshot,
    )
    if len(per_example_results) < len(dataset):
        for i, example in tqdm(
            enumerate(dataset.select(range(len(per_example_results), len(dataset)))),
            desc=f"Evaluating {model_name}",
            initial=len(per_example_results),
            total=len(dataset),
        ):
            original_dotted_text = example["text"]  # type:ignore
            dotless_text = remove_dots(original_dotted_text)
            retry_count = 0
            predicted_dotted_text = None
            dotting_time = None
            while not predicted_dotted_text and retry_count < retry:
                if retry_count > 0:
                    print(
                        f"Failed to restore dots for example index {len(per_example_results)} after {retry_count} retries. Retrying..."
                    )
                try:
                    time_before_prediction = datetime.now()
                    predicted_dotted_text = dotter.restore_dots(dotless_text)
                    time_after_prediction = datetime.now()
                    dotting_time = time_after_prediction - time_before_prediction
                except Exception as e:
                    print(f"Error during dot restoration: {e}")
                    predicted_dotted_text = None
                    pass
                raw_dspy_logs = dotter.lm.history[-1].copy() if dotter.lm.history else {}
                retry_count += 1
            per_example_results.append(
                {
                    "original_dotted_text": original_dotted_text,
                    "dotless_text": dotless_text,
                    "predicted_dotted_text": predicted_dotted_text if predicted_dotted_text else "",
                    "text_source": example["source"],  # type:ignore
                    "wer": wer(original_dotted_text, predicted_dotted_text),
                    "cer": cer(original_dotted_text, predicted_dotted_text),
                    "doer": doer(original_dotted_text, predicted_dotted_text),
                    "dotting_time": dotting_time or float("inf"),  # type:ignore
                    "tokens": raw_dspy_logs["usage"] if raw_dspy_logs else None,  # type:ignore
                    "raw_dspy_logs": raw_dspy_logs or {},  # type:ignore
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
    # Save final results to file
    json.dump(
        per_example_results,
        open(results_file, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
        default=str,
    )
    summary = {
        "avg_wer": sum(result["wer"] for result in per_example_results) / len(per_example_results),
        "avg_cer": sum(result["cer"] for result in per_example_results) / len(per_example_results),
        "avg_doer": sum(result["doer"] for result in per_example_results) / len(per_example_results),
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
