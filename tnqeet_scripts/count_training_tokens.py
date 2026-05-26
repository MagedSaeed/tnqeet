"""Count tokens in the training dataset using the ALLaM tokenizer."""

import statistics

from tqdm import tqdm
from transformers import AutoTokenizer

from tnqeet.data import train_dataset

TOKENIZER_ID = "humain-ai/ALLaM-7B-Instruct-preview"
BATCH_SIZE = 2048


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    print(f"Tokenizer: {TOKENIZER_ID}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Training examples: {len(train_dataset):,}")  # type: ignore

    texts = train_dataset["text"]  # type: ignore
    total_tokens = 0
    total_chars = 0
    per_example_lengths: list[int] = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Tokenizing"):
        batch = texts[start : start + BATCH_SIZE]
        encodings = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for text, ids in zip(batch, encodings):
            n = len(ids)
            per_example_lengths.append(n)
            total_tokens += n
            total_chars += len(text)

    mean_len = statistics.mean(per_example_lengths)
    median_len = statistics.median(per_example_lengths)
    p95_len = statistics.quantiles(per_example_lengths, n=20)[18]

    print()
    print(f"Total tokens:           {total_tokens:,}")
    print(f"Total characters:       {total_chars:,}")
    print(f"Tokens / char fertility:{total_tokens / total_chars:.4f}")
    print(f"Mean tokens / example:  {mean_len:.2f}")
    print(f"Median tokens / example:{median_len:.2f}")
    print(f"p95 tokens / example:   {p95_len:.2f}")
    print(f"Max tokens / example:   {max(per_example_lengths)}")


if __name__ == "__main__":
    main()
