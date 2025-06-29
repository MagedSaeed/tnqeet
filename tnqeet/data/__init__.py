import datasets
from tnqeet import constants
from collections import defaultdict

train_dataset = datasets.load_dataset("MagedSaeed/tnqeet-training-datasets", "all_shuffled", split="train")
test_dataset = datasets.load_dataset("MagedSaeed/tnqeet-testing-datasets", "all_shuffled", split="test")

# create validation datasets:

# first, list all sources
source_groups = defaultdict(list)
for i, example in enumerate(train_dataset.select(range(10_000))):  # type:ignore
    source_groups[example["source"]].append(i)  # type:ignore

# Sample 15 examples from each source for each validation set
val_indices = []
fewshot_val_indices = []

for source, indices in source_groups.items():
    # Take last 15 for val_dataset, first 15 for fewshot_val_dataset
    val_indices.extend(indices[-15:])
    fewshot_val_indices.extend(indices[:15])

# Create the validation datasets
val_dataset = train_dataset.select(val_indices)  # type:ignore
fewshot_val_dataset = train_dataset.select(fewshot_val_indices)  # type:ignore

# shuffle val datasets
val_dataset = val_dataset.shuffle(seed=constants.RANDOM_SEED)
fewshot_val_dataset = fewshot_val_dataset.shuffle(seed=constants.RANDOM_SEED)

# print(f"val_dataset size: {len(val_dataset)}")
# print(f"fewshot_val_dataset size: {len(fewshot_val_dataset)}")
