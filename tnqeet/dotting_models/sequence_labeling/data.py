import torch
import re
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from typing import Optional, Dict, List, Union
import pandas as pd
from tnqeet.data import train_dataset, test_dataset
from tnqeet import remove_dots, constants
from transformers import AutoTokenizer

# train_dataset = train_dataset.select(range(50_000))  # type:ignore

tokenizer = AutoTokenizer.from_pretrained(
    "MagedSaeed/tnqeet-tokenizer",
    trust_remote_code=True,
)


class LazyDottingDataset(Dataset):
    def __init__(
        self,
        max_length,
        data_source: Union[pd.DataFrame, List[str]],
        tokenizer=tokenizer,
        text_column: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column

        # Handle both DataFrame and List inputs
        if isinstance(data_source, pd.DataFrame):
            self.data_frame = data_source
            self.is_dataframe = True
            self._length = len(data_source)
        elif isinstance(data_source, list):
            self.texts = data_source
            self.is_dataframe = False
            self._length = len(data_source)
        else:
            raise ValueError("data_source must be either a pandas DataFrame or a list of strings")

    def __len__(self) -> int:
        return self._length

    def _get_text_at_index(self, idx: int) -> str:
        if self.is_dataframe:
            return self.data_frame.iloc[idx][self.text_column]
        else:
            return self.texts[idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Lazily load the text only when requested
        original_text = self._get_text_at_index(idx)
        original_text = " ".join(re.split(r"\s+", original_text))
        source_text = remove_dots(original_text)
        target_text = original_text

        # Tokenize source text (input)
        source_encoded = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target text (labels)
        target_encoded = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Extract tensors and remove batch dimension
        input_ids = source_encoded["input_ids"].squeeze(0)  # type: ignore
        labels = target_encoded["input_ids"].squeeze(0)  # type: ignore

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class DottingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer=tokenizer,
        max_length: int = 1024,
        batch_size: int = 512,
        num_workers: int = 4,
        val_split: float = 0.05,
        stratify_column: str = "source",
        text_column: str = "text",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.stratify_column = stratify_column
        self.text_column = text_column

        # These will hold the split indices, not the actual data
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self._train_df = None
        self._test_df = None

        assert 0 < val_split < 1, (
            "Validation split must be between 0 and 1, preferably very small number between 0 and 0.1"
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            if self._train_df is None:
                # Load train dataset lazily only when needed
                self._train_df = train_dataset.to_pandas()  # type: ignore

            # Create train/validation split if not already done
            if self.train_data is None or self.val_data is None:
                train_split, val_split = train_test_split(
                    self._train_df,
                    test_size=self.val_split,
                    random_state=constants.RANDOM_SEED,
                    stratify=self._train_df[self.stratify_column],  # type: ignore
                )

                # Reset indices to ensure proper indexing
                self.train_data = train_split.reset_index(drop=True)
                self.val_data = val_split.reset_index(drop=True)

        if stage == "test" or stage is None:
            if self._test_df is None:
                # Load test dataset lazily only when needed
                self._test_df = test_dataset.to_pandas()  # type: ignore
                self.test_data = self._test_df.reset_index(drop=True)  # type: ignore

        # Print dataset sizes for debugging
        if stage == "fit" or stage is None:
            print(f"Train dataset size: {len(self.train_data)}")  # type: ignore
            print(f"Validation dataset size: {len(self.val_data)}")  # type: ignore
        if stage == "test" or stage is None:
            print(f"Test dataset size: {len(self.test_data)}")  # type: ignore

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with lazy dataset."""
        if self.train_data is None:
            raise RuntimeError("Train data not set up. Call setup('fit') first.")

        dataset = LazyDottingDataset(
            self.max_length,
            self.train_data,
            self.tokenizer,
            self.text_column,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # For faster GPU transfer
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader with lazy dataset."""
        if self.val_data is None:
            raise RuntimeError("Validation data not set up. Call setup('fit') first.")

        dataset = LazyDottingDataset(
            self.max_length,
            self.val_data,
            self.tokenizer,
            self.text_column,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_data is None:
            raise RuntimeError("Test data not set up. Call setup('test') first.")

        dataset = LazyDottingDataset(
            self.max_length,
            self.test_data,
            self.tokenizer,
            self.text_column,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
