import argparse
import json
import os
from datetime import datetime

from tqdm.auto import tqdm

from tnqeet import remove_dots
from tnqeet.data import val_dataset
from tnqeet.dotting_models.canine.models import CANINE_MODEL_NAME, CanineDottingModel
from tnqeet.dotting_models.sequence_labeling.utils import split_text_by_threshold
from tnqeet.evaluate.metrics import cer, doer, wer


CHECKPOINTS_ROOT = "tnqeet/dotting_models/canine/trained_models"


def run_name_from_model(model_name: str) -> str:
    return f"CANINE-{model_name.split('/')[-1]}"


def evaluate_model(
    model=None,
    dataset=val_dataset,
    dataset_name: str = "val_dataset",
    overwrite: bool = False,
    save_every: int = 5,
    model_name: str = run_name_from_model(CANINE_MODEL_NAME),
    results_dir: str | None = None,
):
    if results_dir is None:
        results_dir = f"tnqeet/dotting_models/canine/evaluation_results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{model_name}_results.json")

    per_example_results = []
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0 and not overwrite:
        print(f"Results for {model_name} already exist. Loading from {results_file}")
        with open(results_file, "r", encoding="utf-8") as f:
            per_example_results = json.load(f)

    if model is None:
        checkpoints_dir = os.path.join(CHECKPOINTS_ROOT, model_name)
        checkpoint_name = [c for c in os.listdir(checkpoints_dir) if c.startswith("epoch=")][0]
        model = CanineDottingModel.load_from_checkpoint(
            checkpoint_path=os.path.join(checkpoints_dir, checkpoint_name),
        )
    dotter = model

    if len(per_example_results) < len(dataset):
        for i, example in tqdm(
            enumerate(dataset.select(range(len(per_example_results), len(dataset)))),
            desc=f"Evaluating {model_name}..",
            initial=len(per_example_results),
            total=len(dataset),
        ):
            time_before = datetime.now()
            original_dotted_text = example["text"]  # type: ignore
            predicted_dotted_text = ""
            for partial_dotted_text in split_text_by_threshold(
                original_dotted_text,
                threshold=dotter.max_sequence_length - 2,
            ):
                partial_dotless_text = remove_dots(partial_dotted_text)
                partial_pred = dotter.restore_dots(partial_dotless_text)
                predicted_dotted_text += partial_pred  # type: ignore
            predicted_dotted_text = predicted_dotted_text.strip()
            dotting_time = datetime.now() - time_before

            per_example_results.append(
                {
                    "original_dotted_text": original_dotted_text,
                    "dotless_text": remove_dots(original_dotted_text),
                    "predicted_dotted_text": predicted_dotted_text,
                    "text_source": example["source"],  # type: ignore
                    "wer": wer(original_dotted_text, predicted_dotted_text),
                    "cer": cer(original_dotted_text, predicted_dotted_text),
                    "doer": doer(original_dotted_text, predicted_dotted_text),
                    "dotting_time": dotting_time,
                }
            )
            if i > 0 and i % save_every == 0:
                json.dump(
                    per_example_results,
                    open(results_file, "w", encoding="utf-8"),
                    ensure_ascii=False,
                    indent=4,
                    default=str,
                )
    else:
        print(f"Skipping evaluation for {model_name} as results already exist.")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    if args.model_name is None:
        run_names = sorted(
            d for d in os.listdir(CHECKPOINTS_ROOT)
            if os.path.isdir(os.path.join(CHECKPOINTS_ROOT, d))
        )
    else:
        run_names = [run_name_from_model(args.model_name)]

    for run_name in run_names:
        results = evaluate_model(model_name=run_name)
        print(f"Summary for {run_name}: {results}")
