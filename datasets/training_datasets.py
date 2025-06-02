from datasets import load_dataset, DatasetDict, concatenate_datasets
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
    "arabic_english_cs": ("MohamedRashad/arabic-english-code-switching", "sentence", None)
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

def standardize(ds, text_col, processor, source_name):
    """Standardize to text + source columns."""
    return ds.map(lambda x: {
        "text": extract_text(x, text_col, processor),
        "source": source_name
    }).remove_columns([col for col in ds.column_names if col not in ["text", "source"]])

def main():
    print("Loading datasets...")
    
    # Load all datasets
    datasets = {}
    for name, config_info in DATASETS.items():
        print(f"Loading {name} dataset...")
        ds, text_col, processor = load_dataset_train(name, config_info)
        if ds:
            datasets[name] = standardize(ds, text_col, processor, name)
    
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