import os
import json
from datetime import datetime
from tqdm.auto import tqdm
import kenlm
from tnqeet.data import test_dataset
from tnqeet import remove_dots
from tnqeet.evaluate.metrics import wer, cer, doer
from tnqeet.dotting_models.ngrams.models import NgramDotter


def evaluate_model(
    ngrams=15,
    dataset=test_dataset,
    dataset_name="test_dataset",
    beam_size=10,
    overwrite=False,
    save_every=5,
):
    results_dir = f"tnqeet/dotting_models/ngrams/evaluation_results/{dataset_name}/beam_size_{beam_size}"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"ngrams_{ngrams}.json")
    per_example_results = []
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0 and not overwrite:
        print(f"Results for {ngrams} already exist. Loading from {results_file}")
        with open(results_file, "r", encoding="utf-8") as f:
            per_example_results = json.load(f)
    model = kenlm.LanguageModel(f"tnqeet/dotting_models/ngrams/trained_models/ngrams_{ngrams}.binary")
    dotter = NgramDotter(model=model, beam_size=beam_size)
    if len(per_example_results) < len(dataset):  # type:ignore
        for i, example in tqdm(
            enumerate(dataset.select(range(len(per_example_results), len(dataset)))),  # type:ignore
            desc=f"Evaluating {ngrams} grams with beam size {beam_size}",
            initial=len(per_example_results),
            total=len(dataset),  # type:ignore
        ):
            original_dotted_text = example["text"]  # type:ignore
            dotless_text = remove_dots(original_dotted_text)
            time_before_prediction = datetime.now()
            predicted_dotted_text = dotter.restore_dots(dotless_text)
            time_after_prediction = datetime.now()
            dotting_time = time_after_prediction - time_before_prediction
            per_example_results.append(
                {
                    "original_dotted_text": original_dotted_text,
                    "dotless_text": dotless_text,
                    "predicted_dotted_text": predicted_dotted_text,
                    "text_source": example["source"],  # type:ignore
                    "wer": wer(original_dotted_text, predicted_dotted_text),
                    "cer": cer(original_dotted_text, predicted_dotted_text),
                    "doer": doer(original_dotted_text, predicted_dotted_text),
                    "dotting_time": dotting_time,
                }
            )
            # break
            if i > 0 and i % save_every == 0:
                json.dump(
                    per_example_results,
                    open(results_file, "w", encoding="utf-8"),
                    ensure_ascii=False,
                    indent=4,
                    default=str,
                )

    else:
        print(f"Skipping evaluation for {ngrams} as results already exist for all examples.")
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


ngram_beam = {
    3: 15,
    # 4: 50,
    4: 30,
    6: 50,
    8: 60,
    # 11: 90,
    # 14: 100,
}

for ngrams, beam_size in ngram_beam.items():
    summary = evaluate_model(ngrams=ngrams, beam_size=beam_size)
    print(f"Summary for beam size {beam_size} and {ngrams} ngrams: {summary}")
    print("-" * 120)
