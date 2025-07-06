import os
import json
from datetime import datetime
from tqdm.auto import tqdm
from tnqeet.data import val_dataset
from tnqeet import remove_dots
from tnqeet.evaluate.metrics import wer, cer, doer
from tnqeet.dotting_models.sequence_labeling.models import LSTMDottingModel
from tnqeet.dotting_models.sequence_labeling.utils import split_text_by_threshold


def get_model():
    checkpoints_dir = "tnqeet/dotting_models/sequence_labeling/trained_models/LSTM"
    checkpoint_name = [c for c in os.listdir(checkpoints_dir) if c.startswith("epoch=")][0]
    model = LSTMDottingModel.load_from_checkpoint(
        checkpoint_path=os.path.join(checkpoints_dir, checkpoint_name),
        strict=False,
    )
    return model


def evaluate_model(
    model=None,
    dataset=val_dataset,
    dataset_name="val_dataset",
    overwrite=True,
    save_every=5,
    model_name="LSTM",
    n_layers=2,
):
    results_dir = f"tnqeet/dotting_models/sequence_labeling/evaluation_results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{model_name}_layers_{n_layers}_results.json")
    per_example_results = []
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0 and not overwrite:
        print(f"Results for {model_name} already exist. Loading from {results_file}")
        with open(results_file, "r", encoding="utf-8") as f:
            per_example_results = json.load(f)
    # load the model
    if model is None:
        checkpoints_dir = f"tnqeet/dotting_models/sequence_labeling/trained_models/LSTM/layers_{n_layers}"
        checkpoint_name = [c for c in os.listdir(checkpoints_dir) if c.startswith("epoch=")][0]
        model = LSTMDottingModel.load_from_checkpoint(
            checkpoint_path=os.path.join(checkpoints_dir, checkpoint_name),
            # strict=False,
        )
    dotter = model
    if len(per_example_results) < len(dataset):
        for i, example in tqdm(
            enumerate(dataset.select(range(len(per_example_results), len(dataset)))),
            desc=f"Evaluating {model_name}..",
            initial=len(per_example_results),
            total=len(dataset),
        ):
            time_before_prediction = datetime.now()
            original_dotted_text = example["text"]  # type:ignore
            predicted_dotted_text = ""
            for partial_dotted_text in split_text_by_threshold(
                original_dotted_text,
                threshold=dotter.max_sequence_length,
            ):
                partial_dotless_text = remove_dots(partial_dotted_text)
                partial_predicted_dotted_text = dotter.restore_dots(partial_dotless_text)
                predicted_dotted_text += partial_predicted_dotted_text.lstrip()  # type:ignore
                if not predicted_dotted_text[-1].isspace():
                    predicted_dotted_text += " "  # Ensure space at the end of each segment
            predicted_dotted_text = predicted_dotted_text.strip()
            time_after_prediction = datetime.now()
            dotting_time = time_after_prediction - time_before_prediction
            per_example_results.append(
                {
                    "original_dotted_text": original_dotted_text,
                    "dotless_text": remove_dots(original_dotted_text),
                    "predicted_dotted_text": predicted_dotted_text,
                    "text_source": example["source"],  # type:ignore
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


for n_layers in [1, 2, 3, 4, 5]:
    results = evaluate_model(n_layers=n_layers)
    print(f"Summary for LSTM with {n_layers} layers: {results}")
