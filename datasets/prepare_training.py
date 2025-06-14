import os
import re
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from huggingface_hub import login
from tnqeet import constants
from tqdm.auto import tqdm

# Dataset configurations with text column mapping and processing functions
DATASETS = {
    "wasm": ("MagedSaeed/wasm", "Tweet", None),
    "iwslt": (("iwslt2017", "iwslt2017-ar-en"), "translation", lambda x: x["ar"]),
    "ashaar": ("arbml/ashaar", "poem verses", lambda x: "، ".join(x) if isinstance(x, list) else str(x)), 
    "tashkeela": ("asas-ai/Tashkeela", "text", None),
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
        ds = load_dataset(config[0], config[1], split="train", trust_remote_code=True)
    else:
        ds = load_dataset(config, split="train", trust_remote_code=True)
    
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
    text = ''.join(c for c in text if c.isprintable())
    # remove multi spaces using re
    text = re.sub(r'\s+', ' ', text).strip()
    return normalize_unicode(text)

def split_by_newlines(text):
    """Split text by newlines and return non-empty lines."""
    lines = text.split('\n')
    return [line.strip() for line in lines if line.strip()]

def standardize(ds, text_col, processor, source_name, minimum_words_threshold=10):
    if callable(globals().get(f'standardize_{source_name}')):
        return globals()[f'standardize_{source_name}'](ds, minimum_words_threshold)
    
    # Standard processing for other datasets
    ds = ds.map(lambda x: {
        "text": extract_text(x, text_col, processor),
        "source": source_name
    }).remove_columns([col for col in ds.column_names if col not in ["text", "source"]])
    
    def generate_split_samples():
        for example in ds:
            lines = split_by_newlines(example["text"])
            for line in lines:
                if len(line.split()) >= minimum_words_threshold:
                    yield{
                        "text": line,
                        "source": example["source"]
                    }
    
    return Dataset.from_generator(generate_split_samples)

def standardize_annotated_aoc(ds, minimum_words_threshold=10):
    """Custom standardization for annotated_aoc dataset to extract all Sentence# columns."""
    
    def generate_samples():
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
                            yield{
                                "text": line,
                                "source": "annotated_aoc"
                            }
    generated_dataset = Dataset.from_generator(generate_samples)
    print(f"  → Annotated AOC: extracted {len(generated_dataset)} samples from all Sentence columns") # type: ignore
    
    return generated_dataset

def standardize_tashkeela(ds, minimum_words_threshold=10):    
    text_samples = []
    diacritized_samples = []
    
    # Extract samples from both columns
    for example in tqdm(ds):
        # Process 'text' column
        if 'text_no_taskheel' in example:
            text_lines = split_by_newlines(normalize_unicode(str(example['text_no_taskheel'])))
            for line in text_lines:
                if len(line.split()) >= minimum_words_threshold:
                    text_samples.append({
                        "text": line,
                        "source": "tashkeela"
                    })
        
        # Process 'diacritized' column
        if 'text' in example:
            diac_lines = split_by_newlines(normalize_unicode(str(example['text'])))
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

def main(hf_repo_name="MagedSaeed/tnqeet-training-datasets", push_to_hub=True):
    print("Loading datasets...")
    
    # Load and process all datasets
    processed_datasets = {}
    for name, config_info in DATASETS.items():
        print(f"Loading {name} dataset...")
        ds, text_col, processor = load_dataset_train(name, config_info)
        if ds:
            standardized_ds = standardize(ds, text_col, processor, name)
            processed_datasets[name] = standardized_ds
            if not callable(globals().get(f'standardize_{name}')):  # Already printed for these
                print(f"  → Dataset Samples (after processing): {len(standardized_ds)} samples") # type: ignore
    
    # Create combined shuffled dataset
    all_datasets = list(processed_datasets.values())
    combined = concatenate_datasets(all_datasets).shuffle(seed=constants.RANDOM_SEED)
    processed_datasets["all_shuffled"] = combined
    
    # Save locally - each dataset as a separate config
    local_path = "./tnqeet_training_datasets"
    print(f"\n✓ Processed {len(processed_datasets)} datasets (including all_shuffled)")
    print(f"Total samples in all_shuffled: {len(combined)}")
    
    # Push to Hugging Face Hub if requested
    if push_to_hub:
        print(f"\nPushing datasets as configs to Hugging Face Hub: {hf_repo_name}")
        
        # Login to HF
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        login(token=hf_token)
        print("✓ Successfully logged in to Hugging Face")
        
        try:
            # Push each dataset as a separate config
            for config_name, dataset in processed_datasets.items():
                print(f"  Pushing config: {config_name}")
                
                # Create DatasetDict with just train split for this config
                config_dict = DatasetDict({"train": dataset})
                
                # Push this specific config
                config_dict.push_to_hub(
                    repo_id=hf_repo_name,
                    config_name=config_name,  # This makes it a config, not a split
                    private=True,
                    commit_message=f"Upload {config_name} config"
                )
                print(f"    ✓ {config_name} config uploaded")
            
            print(f"\n✓ Successfully pushed all configs to https://huggingface.co/datasets/{hf_repo_name}")
            print("You can now load datasets as:")
            for config_name in processed_datasets.keys():
                print(f"  load_dataset('{hf_repo_name}', '{config_name}', split='train')")
                
        except Exception as e:
            print(f"✗ Error pushing to Hub: {e}")
            print("You can still save datasets locally if needed.")
    
    else:
        # Just save locally as separate files
        for config_name, dataset in processed_datasets.items():
            config_path = f"{local_path}_{config_name}"
            config_dict = DatasetDict({"train": dataset})
            config_dict.save_to_disk(config_path)
            print(f"✓ Saved {config_name} to {config_path}")

if __name__ == "__main__":
    # Modify these parameters as needed
    HF_REPO_NAME = "MagedSaeed/tnqeet-training-datasets"  # Change this to your desired repo name
    
    main(hf_repo_name=HF_REPO_NAME, push_to_hub=True)