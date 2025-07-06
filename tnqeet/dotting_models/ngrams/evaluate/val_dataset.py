import os
import json
from datetime import datetime
from tqdm.auto import tqdm
import kenlm
from tnqeet.data import val_dataset
from tnqeet import remove_dots
from tnqeet.evaluate.metrics import wer, cer, doer
from tnqeet import constants
from collections import defaultdict


def to_chars(text):
    chars = list()
    for c in text:
        if c.isspace():
            chars.append("<SPACE>")
        else:
            chars.append(c)
    chars = " ".join(chars)
    return chars


class NgramDotter:
    def __init__(self, model, beam_size: int = 10):
        self.model = model
        self.beam_size = beam_size
        # Create reverse mapping from rasm to original letters
        self.rasm_to_letters = defaultdict(list)
        for dotted_letter, rasm in constants.LETTERS_MAPPING.items():
            self.rasm_to_letters[rasm].append(dotted_letter)
        # self.rasm_to_letters[constants.NOON_RASM] += ["ن"]
        # self.rasm_to_letters[constants.BAA_RASM] += ["ي"]
        # self.rasm_to_letters[constants.QAF_RASM] += ["ق"]
        # yaa can be mapped to baa if it comes in the beginning of a word.
        # self.rasm_to_letters[constants.YAA_RASM] += ["ب"]
        # convert back to dict
        self.rasm_to_letters = dict(self.rasm_to_letters)

    def restore_dots(self, dotless_text: str) -> str:
        tokens = to_chars(dotless_text)
        beam = [([], 0.0)]
        for char in tokens.split():
            new_beam = []
            candidates = self.rasm_to_letters.get(char, [char])
            for sequence, score in beam:
                for candidate in candidates:
                    new_sequence = sequence + [candidate]
                    context = " ".join(new_sequence).strip()
                    new_score = score + self.model.score(context, bos=len(sequence) == 0)
                    new_beam.append((new_sequence, new_score))
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[: self.beam_size]
        best_sequence = max(beam, key=lambda x: x[1])[0]
        return "".join(best_sequence).replace(" ", "").replace("<SPACE>", " ")


def evaluate_model(
    ngrams=15,
    dataset=val_dataset,
    dataset_name="val_dataset",
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
    if len(per_example_results) < len(dataset):
        for i, example in tqdm(
            enumerate(dataset.select(range(len(per_example_results), len(dataset)))),
            desc=f"Evaluating {ngrams} grams with beam size {beam_size}",
            initial=len(per_example_results),
            total=len(dataset),
        ):
            original_dotted_text = example["text"]  # type:ignore
            dotless_text = remove_dots(original_dotted_text)
            dotter = NgramDotter(model=model, beam_size=beam_size)
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


for ngrams in range(2, 16):
    for beam_size in [1, 3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]:
        summary = evaluate_model(ngrams=ngrams, beam_size=beam_size)
        print(f"Summary for beam size {beam_size} and {ngrams} ngrams: {summary}")
        print("-" * 120)
