from transformers import AutoTokenizer
from tnqeet.data import train_dataset
from tqdm.auto import tqdm
from tnqeet import remove_dots
from datasets import concatenate_datasets

DO_TRAIN = False
DO_PUSH = False

if DO_TRAIN:
    tokenizer = AutoTokenizer.from_pretrained(
        "MagedSaeed/APCD-Plus-meter-classification-model",
        trust_remote_code=True,
    )

    print("original tokenizer vocab size:", tokenizer.vocab_size)

    tokenizer.train(
        texts=tqdm(
            concatenate_datasets(
                [
                    train_dataset,  # type: ignore
                    train_dataset.map(lambda example: {"text": remove_dots(example["text"])}),  # type: ignore
                ]
            ).map(lambda example: {"text": example["text"].replace("\xa0", "")})["text"],
            desc="Training tokenizer on dotted and dotless text...",
        )
    )

else:
    tokenizer = AutoTokenizer.from_pretrained(
        "MagedSaeed/tnqeet-tokenizer",
        trust_remote_code=True,
    )

print("vocab size after training:", tokenizer.vocab_size)

# do not forget to push this file to the hub too:
# https://huggingface.co/MagedSaeed/APCD-Plus-meter-classification-model/blob/main/tokenizer_script.py

if DO_PUSH:
    tokenizer.register_for_auto_class(auto_class="AutoTokenizer")
    tokenizer.push_to_hub("MagedSaeed/tnqeet-tokenizer")
