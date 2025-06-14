import os
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from huggingface_hub import login
from tnqeet import constants
import re
from tqdm.auto import tqdm

# Dataset configurations with text column mapping and processing functions
TEST_DATASETS = {
    "wasm": ("MagedSaeed/wasm", "Tweet", None, "test"),
    "iwslt": (("iwslt2017", "iwslt2017-ar-en"), "translation", lambda x: x["ar"], "test"),
    "quran": ("ReySajju742/Quran", "verse", None, "train"),
    "tashkeela": ("asas-ai/Tashkeela", "text", None, "train"),  # Special processing
    "kind": ("KIND-Dataset/KIND", "strippedText", None, "train"),
    "poetry": ("omkarthawakar/FannOrFlop", "poem_verses", None, "train"),  # Special processing
    "arabic_english_code_switching": ("MagedSaeed/arabic-english-code-switching-text", "text", None, "train"),
    "arasum": ("arbml/AraSum", "article", None, "train"),
    "social_media": ("KFUPM-JRCAI/arabic-generated-social-media-posts", "original_post", None, "train"),
    "LLMs_abstracts": ("KFUPM-JRCAI/arabic-generated-abstracts", None, None, None),  # Special processing
}

def load_dataset_split(name, config_info):
    """Load specified split of dataset."""
    config, text_col, processor, split = config_info
    
    if isinstance(config, tuple):
        ds = load_dataset(config[0], config[1], split=split, trust_remote_code=True)
    elif name.lower()=='kind':
        ds = load_dataset(config, split=split, data_files='https://huggingface.co/datasets/KIND-Dataset/KIND/raw/main/KIND-Dialectal-Sentences.csv',trust_remote_code=True)
    else:
        ds = load_dataset(config, split=split,trust_remote_code=True)
    
    print(f"✓ {name}: {len(ds)} samples from {split} split") # type: ignore
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

def clean_poetry_text(text):
    """Special processing for poetry: remove number-only lines and normalize whitespace."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip lines that contain only numbers (and possibly whitespace/punctuation)
        if re.match(r'^[\d\s\.\-\،\؍]*$', line):
            continue
        if line:  # Skip empty lines
            # Replace multiple spaces, newlines, and tabs with single space
            cleaned_line = re.sub(r'[\s\n\t]+', ' ', line).strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
    
    return '، '.join(cleaned_lines).strip().removesuffix('،').strip()

def standardize(ds, text_col, processor, source_name, minimum_words_threshold=10):
    """Standardize dataset processing with special cases."""
    if callable(globals().get(f'standardize_{source_name}')):
        return globals()[f'standardize_{source_name}'](ds, minimum_words_threshold)
    
    # Standard processing for other datasets
    def process_example(example):
        if source_name == "poetry":
            # Special processing for poetry
            raw_text = extract_text(example, text_col, processor)
            cleaned_text = clean_poetry_text(raw_text)
            return {
                "text": cleaned_text,
                "source": source_name
            }
        else:
            return {
                "text": extract_text(example, text_col, processor),
                "source": source_name
            }
    
    ds = ds.map(process_example).remove_columns([col for col in ds.column_names if col not in ["text", "source"]])
    
    def generate_split_samples():
        for example in ds:
            lines = split_by_newlines(example["text"])
            for line in lines:
                if len(line.split()) >= minimum_words_threshold:
                    yield {
                        "text": line,
                        "source": example["source"]
                    }
    
    return Dataset.from_generator(generate_split_samples)

def standardize_tashkeela(ds, minimum_words_threshold=10):
    """Custom standardization for tashkeela dataset - reverse the ratio from training."""
    diacritized_samples = []
    text_samples = []
    
    # Extract samples from both columns (reversed priority from training)
    for example in tqdm(ds):
        # Process 'diacritized' column first (more priority than in training)
        if 'text' in example:
            diac_lines = split_by_newlines(normalize_unicode(str(example['text'])))
            for line in diac_lines:
                if len(line.split()) >= minimum_words_threshold:
                    diacritized_samples.append({
                        "text": line,
                        "source": "tashkeela"
                    })
        
        # Process 'text' column second
        if 'text_no_taskheel' in example:
            text_lines = split_by_newlines(normalize_unicode(str(example['text_no_taskheel'])))
            for line in text_lines:
                if len(line.split()) >= minimum_words_threshold:
                    text_samples.append({
                        "text": line,
                        "source": "tashkeela"
                    })
    
    # Calculate 50/50 split (reversed from training)
    min_count = min(len(text_samples), len(diacritized_samples))
    half_count = min_count // 2
    
    # Take equal amounts from each column (reversed priority)
    final_samples = diacritized_samples[:half_count] + text_samples[:half_count]
    
    print(f"  → Tashkeela: {half_count} from 'diacritized' + {half_count} from 'text' = {len(final_samples)} total")
    
    return Dataset.from_list(final_samples)

def standardize_LLMs_abstracts(ds=None, minimum_words_threshold=10):
    """Custom standardization for arabic-generated-abstracts dataset."""
    # Load all three splits
    by_polishing = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts", split="by_polishing")
    from_title = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts", split="from_title")
    from_title_and_content = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts", split="from_title_and_content")
    
    print(f"  → LLMs Abstracts: by_polishing={len(by_polishing)}, from_title={len(from_title)}, from_title_and_content={len(from_title_and_content)}") # type: ignore
    
    all_samples = []
    
    # AI-generated columns to extract (excluding original_abstract which is human text)
    ai_columns = ['allam_generated_abstract', 'jais_generated_abstract', 'llama_generated_abstract', 'openai_generated_abstract']
    
    # Process all three splits
    for split_name, split_ds in [("by_polishing", by_polishing), ("from_title", from_title), ("from_title_and_content", from_title_and_content)]:
        for example in split_ds:
            # Get AI-generated abstracts from all AI columns
            for col_name in ai_columns:
                if col_name in example and example[col_name] and isinstance(example[col_name], str): # type: ignore
                    ai_text = normalize_unicode(str(example[col_name])) # type: ignore
                    lines = split_by_newlines(ai_text)
                    for line in lines:
                        if len(line.split()) >= minimum_words_threshold:
                            all_samples.append({
                                "text": line,
                                "source": "LLMs_abstracts"
                            })
    
    print(f"  → LLMs Abstracts: extracted {len(all_samples)} samples from all AI-generated columns and splits")
    
    return Dataset.from_list(all_samples)

def sample_datasets(datasets, n_samples=500):
    """Sample n_samples from each dataset after shuffling."""
    sampled_datasets = {}
    
    for name, ds in datasets.items():
        # Shuffle the dataset
        shuffled_ds = ds.shuffle(seed=constants.RANDOM_SEED)
        
        # Sample n_samples or take all if dataset is smaller
        n_to_sample = min(n_samples, len(shuffled_ds))
        sampled_ds = shuffled_ds.select(range(n_to_sample))
        
        sampled_datasets[name] = sampled_ds
        print(f"  → {name}: sampled {len(sampled_ds)} samples")
    
    return sampled_datasets

def main(hf_repo_name="MagedSaeed/tnqeet-testing-datasets", push_to_hub=True, n_samples=500):
    print("Loading test datasets...")
    
    # Load all datasets
    datasets = {}
    for name, config_info in TEST_DATASETS.items():
        print(f"Loading {name} dataset...")
        ds, text_col, processor = load_dataset_split(name, config_info)
        if ds:
            standardized_ds = standardize(ds, text_col, processor, name)
            datasets[name] = standardized_ds
            if not callable(globals().get(f'standardize_{name}')):
                print(f"  → Dataset Samples (after processing): {len(standardized_ds)} samples") # type: ignore
    
    print(f"\n=== Sampling {n_samples} samples from each dataset ===")
    
    # Sample n_samples from each dataset
    sampled_datasets = sample_datasets(datasets, n_samples=n_samples)
    
    # Create combined shuffled dataset
    all_datasets = list(sampled_datasets.values())
    combined = concatenate_datasets(all_datasets).shuffle(seed=constants.RANDOM_SEED)
    sampled_datasets["all_shuffled"] = combined
    
    # Save locally
    local_path = "./tnqeet_testing_datasets"
    print(f"\n✓ Processed {len(sampled_datasets)} datasets (including all_shuffled)")
    print(f"Total samples in all_shuffled: {len(combined)}")
    
    # Print summary
    print("\n=== Final Dataset Summary ===")
    for name, ds in sampled_datasets.items():
        if name != "all_shuffled":
            print(f"{name}: {len(ds)} samples")
    print(f"all_shuffled: {len(combined)} samples")
    
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
            for config_name, dataset in sampled_datasets.items():
                print(f"  Pushing config: {config_name}")
                
                # Create DatasetDict with test split
                config_dict = DatasetDict({"test": dataset})
                
                # Push this specific config
                config_dict.push_to_hub(
                    repo_id=hf_repo_name,
                    config_name=config_name,  # This makes it a config, not a split
                    private=True,
                    commit_message=f"Upload {config_name} config with {len(dataset)} samples"
                )
                print(f"    ✓ {config_name} config uploaded ({len(dataset)} samples)")
            
            print(f"\n✓ Successfully pushed all configs to https://huggingface.co/datasets/{hf_repo_name}")
            print("You can now load datasets as:")
            for config_name in sampled_datasets.keys():
                print(f"  load_dataset('{hf_repo_name}', '{config_name}', split='test')")
                
        except Exception as e:
            print(f"✗ Error pushing to Hub: {e}")
            print("Datasets are still processed and available locally if needed.")
    
    else:
        # Just save locally as separate files
        for config_name, dataset in sampled_datasets.items():
            config_path = f"{local_path}_{config_name}"
            config_dict = DatasetDict({"test": dataset})
            config_dict.save_to_disk(config_path)
            print(f"✓ Saved {config_name} to {config_path}")

if __name__ == "__main__":
    # Parameters - modify as needed
    HF_REPO_NAME = "MagedSaeed/tnqeet-testing-datasets"  # Using your username
    N_SAMPLES = 500  # Number of samples to take from each dataset
    
    main(hf_repo_name=HF_REPO_NAME, push_to_hub=True, n_samples=N_SAMPLES)