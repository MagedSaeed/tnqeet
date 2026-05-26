import argparse
import json
import os
from datetime import datetime

from tqdm.auto import tqdm

from tnqeet import remove_dots
from tnqeet.data import test_dataset
from tnqeet.dotting_models.sequence_labeling.utils import split_text_by_threshold
from tnqeet.dotting_models.transformer.models import TransformerDottingModel
from tnqeet.evaluate.metrics import cer, doer, wer


MODEL_NAME = "Transformer"
CHECKPOINTS_ROOT = "tnqeet/dotting_models/transformer/trained_models"


def evaluate_model(
    model=None,
    dataset=test_dataset,
    dataset_name: str = "test_dataset",
    overwrite: bool = False,
    save_every: int = 5,
    model_name: str = MODEL_NAME,
    n_layers: int = 6,
):
    results_dir = f"tnqeet/dotting_models/transformer/evaluation_results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(
        results_dir, f"{model_name}_layers_{n_layers}_results.json"
    )

    per_example_results = []
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0 and not overwrite:
        print(f"Results for {model_name} already exist. Loading from {results_file}")
        with open(results_file, "r", encoding="utf-8") as f:
            per_example_results = json.load(f)

    if model is None:
        checkpoints_dir = os.path.join(CHECKPOINTS_ROOT, model_name, f"layers_{n_layers}")
        checkpoint_name = [c for c in os.listdir(checkpoints_dir) if c.startswith("epoch=")][0]
        model = TransformerDottingModel.load_from_checkpoint(
            checkpoint_path=os.path.join(checkpoints_dir, checkpoint_name),
        )
    dotter = model

    if len(per_example_results) < len(dataset):  # type: ignore
        for i, example in tqdm(
            enumerate(dataset.select(range(len(per_example_results), len(dataset)))),  # type: ignore
            desc=f"Evaluating {model_name} layers={n_layers}..",
            initial=len(per_example_results),
            total=len(dataset),  # type: ignore
        ):
            time_before = datetime.now()
            original_dotted_text = example["text"]  # type: ignore
            predicted_dotted_text = ""
            for partial_dotted_text in split_text_by_threshold(
                original_dotted_text,
                threshold=dotter.max_sequence_length,
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
    parser.add_argument("--n_layers", type=int, default=None)
    args = parser.parse_args()

    if args.n_layers is None:
        model_dir = os.path.join(CHECKPOINTS_ROOT, MODEL_NAME)
        layers_to_run = sorted(
            int(d.replace("layers_", ""))
            for d in os.listdir(model_dir)
            if d.startswith("layers_") and os.path.isdir(os.path.join(model_dir, d))
        )
    else:
        layers_to_run = [args.n_layers]

    for n_layers in layers_to_run:
        results = evaluate_model(n_layers=n_layers)
        print(f"Summary for Transformer with {n_layers} layers: {results}")
