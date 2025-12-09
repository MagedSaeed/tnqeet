import os
import json
from datetime import datetime
from tqdm.auto import tqdm
from tnqeet.dotting_models.llms.models import (
    OpenRouterArabicDotter,
    ArabicDottingSignature,
    DetailedArabicDotingSignature,
)
from tnqeet.data import test_dataset
from tnqeet import remove_dots
from tnqeet.evaluate.metrics import wer, cer, doer

OPEN_ROUTER_MODELS = {
    "gemini-2.5-flash-preview": "openai/google/gemini-2.5-flash-preview-05-20",
    "claude-sonnet-4": "openai/anthropic/claude-sonnet-4",
    "gpt-4o": "openai/openai/gpt-4o-2024-11-20",
}

ORIGINAL_MODELS = {
    "claude-sonnet-4": "anthropic/claude-sonnet-4-20250514",  # anthropic model equivalent to openrouter model: (anthropic/claude-4-sonnet-20250522)
    "gpt-4o": "openai/gpt-4o-2024-11-20",
}


def evaluate_model(
    model_name,
    dataset=test_dataset,
    dataset_name="test_dataset",
    num_fewshot=0,
    evaluation_type="zeroshot",
    prompt_type="default",
    overwrite=False,
    save_every=5,
    retry=5,
    use_openrouter_model=True,
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
    results_dir = (
        f"tnqeet/dotting_models/llms/evaluation_results/{dataset_name}/{evaluation_type}/{prompt_type}_prompt"
    )
    os.makedirs(results_dir, exist_ok=True)
    # make sure {model_name}.json file exists in the results_dir
    results_file = os.path.join(results_dir, f"{model_name}.json")
    per_example_results = []
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0 and not overwrite:
        print(f"Results for {model_name} already exist. Loading from {results_file}")
        with open(results_file, "r", encoding="utf-8") as f:
            per_example_results = json.load(f)
    model = OPEN_ROUTER_MODELS[model_name] if use_openrouter_model else ORIGINAL_MODELS[model_name]
    dotter = OpenRouterArabicDotter(
        model=model,
        dspy_cache=False,
        signature=signature,  # type: ignore
        num_fewshot=num_fewshot,
        use_openrouter_model=use_openrouter_model,
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
            predicted_dotted_text = ""
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
                    predicted_dotted_text = ""
                    pass
                raw_dspy_logs = dotter.lm.history[-1].copy() if dotter.lm.history else {}
                retry_count += 1
            per_example_results.append(
                {
                    "original_dotted_text": original_dotted_text,
                    "dotless_text": dotless_text,
                    "predicted_dotted_text": predicted_dotted_text,
                    "text_source": example["source"],  # type:ignore
                    "wer": wer(original_dotted_text, predicted_dotted_text[: len(original_dotted_text)]),  # type:ignore
                    "cer": cer(original_dotted_text, predicted_dotted_text[: len(original_dotted_text)]),  # type:ignore
                    "doer": doer(original_dotted_text, predicted_dotted_text[: len(original_dotted_text)]),  # type:ignore
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


# for model_name in list(ORIGINAL_MODELS.keys()):
for model_name in list(OPEN_ROUTER_MODELS.keys()):
    summary = evaluate_model(
        num_fewshot=8,
        model_name=model_name,
        # use_openrouter_model=False,
    )
    print(f"Summary for {model_name} with 8 fewshot: {summary}")
    print("-" * 120)


"""
for reference, gemini was called via openrouter, but others are called directly from the provider.
"""
