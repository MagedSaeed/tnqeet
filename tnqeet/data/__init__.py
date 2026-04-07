import datasets
from tnqeet import constants
from collections import defaultdict

train_dataset = datasets.load_dataset(
    "MagedSaeed/tnqeet-training-datasets", "all_shuffled", split="train"
)
test_dataset = datasets.load_dataset(
    "MagedSaeed/tnqeet-testing-datasets", "all_shuffled", split="test"
)

# create validation datasets:

# first, list all sources
source_groups = defaultdict(list)
for i, example in enumerate(train_dataset.select(range(10_000))):  # type:ignore
    source_groups[example["source"]].append(i)  # type:ignore

# Sample 15 examples from each source for each validation set
val_indices = []
llms_val_indices = []
fewshot_val_indices = []

for source, indices in source_groups.items():
    # Take last 150 for val_dataset, first 15 for fewshot_val_dataset
    val_indices.extend(indices[-150:])
    llms_val_indices.extend(indices[-15:])
    fewshot_val_indices.extend(indices[:15])

# Create the validation datasets
val_dataset = train_dataset.select(val_indices)  # type:ignore
llms_val_dataset = train_dataset.select(llms_val_indices)  # type:ignore
fewshot_val_dataset = train_dataset.select(fewshot_val_indices)  # type:ignore

# shuffle val datasets
val_dataset = val_dataset.shuffle(seed=constants.RANDOM_SEED)
llms_val_dataset = llms_val_dataset.shuffle(seed=constants.RANDOM_SEED)
fewshot_val_dataset = fewshot_val_dataset.shuffle(seed=constants.RANDOM_SEED)

# Drop validation sets from train set to avoid data leakage
all_val_indices = set(val_indices + fewshot_val_indices)
train_indices = [i for i in range(len(train_dataset)) if i not in all_val_indices]  # type:ignore
train_dataset = train_dataset.select(train_indices)  # type:ignore

# print('train dataset size:', len(train_dataset))
# print('val dataset size:', len(val_dataset))
# print('llms val dataset size:', len(llms_val_dataset))
# print('fewshot val dataset size:', len(fewshot_val_dataset))

# print('fewshot val dataset number of words:', sum(len(example['text'].split()) for example in fewshot_val_dataset))  # type:ignore
# print("llms val dataset number of words:", sum(len(example['text'].split()) for example in llms_val_dataset))  # type:ignore
# print('val dataset number of words:', sum(len(example['text'].split()) for example in val_dataset))  # type:ignore
# print('train dataset number of words:', sum(len(example['text'].split()) for example in train_dataset))  # type:ignore
# print(f"llms_val_dataset size: {len(val_dataset)}")
# print(f"fewshot_val_dataset size: {len(fewshot_val_dataset)}")
