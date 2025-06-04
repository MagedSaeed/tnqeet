from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from tnqeet import constants

# Dataset configurations with text column mapping and processing functions
DATASETS = {
    "wasm": ("MagedSaeed/wasm", "Tweet", None),
    "iwslt": (("iwslt2017", "iwslt2017-ar-en"), "translation", lambda x: x["ar"]),
    "ashaar": ("arbml/ashaar", "poem verses", lambda x: " ".join(x) if isinstance(x, list) else str(x)), 
    "tashkeela": ("community-datasets/tashkeela", "text", None),
    "sanad": ("arbml/SANAD", "Article", None),
    "oscar_small": (("nthngdy/oscar-small","unshuffled_deduplicated_ar"), "text", None),
    "arabic_wikipedia": ("SaiedAlshahrani/Arabic_Wikipedia_20230101_bots", "text", None),
    "annotated_aoc": ("arbml/annotated_aoc", "Sentence1", None),  # New dataset added
    # "arabic_english_cs": ("MohamedRashad/arabic-english-code-switching", "sentence", None)
}

def load_dataset_train(name, config_info):
    """Load train split of dataset."""
    config, text_col, processor = config_info
    
    if isinstance(config, tuple):
        ds = load_dataset(config[0], config[1], split="train")
    else:
        ds = load_dataset(config, split="train")
    
    print(f"✓ {name}: {len(ds)} samples") # type: ignore
    return ds, text_col, processor

def normalize_unicode(text):
    """Apply unicode normalization mapping to text."""
    for old_char, new_char in constants.UNICODE_LETTERS_MAPPING.items():
        text = text.replace(old_char, new_char)
    return text

def extract_text(example, text_col, processor):
    """Extract and process text based on processor function."""
    text = example[text_col]
    text = processor(text) if processor else str(text)
    return normalize_unicode(text)

def split_by_newlines(text):
    """Split text by newlines and return non-empty lines."""
    lines = text.split('\n')
    return [line.strip() for line in lines if line.strip()]

def standardize(ds, text_col, processor, source_name, minimum_words_threshold=10):
    if f'standardize_{source_name}' in globals() and callable(globals()[f'standardize_{source_name}']):
        return globals()[f'standardize_{source_name}'](ds, minimum_words_threshold)
    
    # Standard processing for other datasets
    ds = ds.map(lambda x: {
        "text": extract_text(x, text_col, processor),
        "source": source_name
    }).remove_columns([col for col in ds.column_names if col not in ["text", "source"]])
    
    split_samples = []
    for example in ds:
        lines = split_by_newlines(example["text"])
        for line in lines:
            if len(line.split()) >= minimum_words_threshold:
                split_samples.append({
                    "text": line,
                    "source": example["source"]
                })
    
    return Dataset.from_list(split_samples)

def standardize_annotated_aoc(ds, minimum_words_threshold=10):
    """Custom standardization for annotated_aoc dataset to extract all Sentence# columns."""
    samples = []
    
    for example in ds:
        # Extract all sentence columns (Sentence1 to Sentence12)
        for i in range(1, 13):  # Sentence1 to Sentence12
            sentence_col = f"Sentence{i}"
            if sentence_col in example and example[sentence_col] is not None:
                text = normalize_unicode(str(example[sentence_col]))
                # Split by newlines and filter
                lines = split_by_newlines(text)
                for line in lines:
                    if len(line.split()) >= minimum_words_threshold:
                        samples.append({
                            "text": line,
                            "source": "annotated_aoc"
                        })
    
    print(f"  → Annotated AOC: extracted {len(samples)} samples from all Sentence columns")
    
    return Dataset.from_list(samples)

def standardize_tashkeela(ds, minimum_words_threshold=10):    
    text_samples = []
    diacritized_samples = []
    
    # Extract samples from both columns
    for example in ds:
        # Process 'text' column
        if 'text' in example:
            text_lines = split_by_newlines(normalize_unicode(str(example['text'])))
            for line in text_lines:
                if len(line.split()) >= minimum_words_threshold:
                    text_samples.append({
                        "text": line,
                        "source": "tashkeela"
                    })
        
        # Process 'diacritized' column  
        if 'diacritized' in example:
            diac_lines = split_by_newlines(normalize_unicode(str(example['diacritized'])))
            for line in diac_lines:
                if len(line.split()) >= minimum_words_threshold:
                    diacritized_samples.append({
                        "text": line,
                        "source": "tashkeela"
                    })
    
    # Calculate 50/50 split
    min_count = min(len(text_samples), len(diacritized_samples))
    half_count = min_count // 2
    
    # Take equal amounts from each column
    final_samples = text_samples[:half_count] + diacritized_samples[:half_count]
    
    print(f"  → Tashkeela: {half_count} from 'text' + {half_count} from 'diacritized' = {len(final_samples)} total")
    
    return Dataset.from_list(final_samples)

def main():
    print("Loading datasets...")
    
    # Load all datasets
    datasets = {}
    for name, config_info in DATASETS.items():
        print(f"Loading {name} dataset...")
        ds, text_col, processor = load_dataset_train(name, config_info)
        if ds:
            standardized_ds = standardize(ds, text_col, processor, name)
            datasets[name] = standardized_ds
            if name not in ["tashkeela", "annotated_aoc"]:  # Already printed for these
                print(f"  → Dataset Samples (after processing): {len(standardized_ds)} samples")
    
    # Create combined shuffled dataset
    all_datasets = list(datasets.values())
    combined = concatenate_datasets(all_datasets).shuffle(seed=constants.RANDOM_SEED)
    datasets["all_shuffled"] = combined
    
    # Create DatasetDict and save
    dataset_dict = DatasetDict(datasets)
    dataset_dict.save_to_disk("./tnqeet_training_datasets")
    
    print(f"\n✓ Saved {len(datasets)-1} datasets + all_shuffled to ./tnqeet_training_datasets")
    print(f"Total samples in all_shuffled: {len(combined)}")

if __name__ == "__main__":
    main()