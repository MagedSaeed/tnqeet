import re
from typing import Dict, List, Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from tnqeet import constants, remove_dots
from tnqeet.data import test_dataset, train_dataset
from tnqeet.dotting_models.canine.models import (
    PAD_LABEL_ID,
    canine_tokenizer,
    encode_labels,
)


class LazyCanineDottingDataset(Dataset):
    def __init__(
        self,
        max_length: int,
        data_source: Union[pd.DataFrame, List[str]],
        tokenizer=None,
        text_column: str = "text",
    ):
        self.max_length = max_length
        self.text_column = text_column
        self.tokenizer = tokenizer if tokenizer is not None else canine_tokenizer

        if isinstance(data_source, pd.DataFrame):
            self.data_frame = data_source
            self.is_dataframe = True
            self._length = len(data_source)
        elif isinstance(data_source, list):
            self.texts = data_source
            self.is_dataframe = False
            self._length = len(data_source)
        else:
            raise ValueError(
                "data_source must be either a pandas DataFrame or a list of strings"
            )

    def __len__(self) -> int:
        return self._length

    def _get_text_at_index(self, idx: int) -> str:
        if self.is_dataframe:
            return self.data_frame.iloc[idx][self.text_column]
        return self.texts[idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        original_text = self._get_text_at_index(idx)
        original_text = " ".join(re.split(r"\s+", original_text))
        # Truncate at character level so source/target stay aligned (CANINE
        # adds [CLS]+[SEP], so leave room for those two specials).
        char_budget = self.max_length - 2
        target_text = original_text[:char_budget]
        source_text = remove_dots(target_text)

        source_encoded = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = source_encoded["input_ids"].squeeze(0)
        attention_mask = source_encoded["attention_mask"].squeeze(0)

        # Build labels aligned to input_ids: [CLS] + per-char labels + [SEP] + pads
        target_label_ids = encode_labels(target_text)
        labels = torch.full(
            (self.max_length,), PAD_LABEL_ID, dtype=torch.long
        )
        # Position 0 is [CLS] -> ignore. Positions 1..len(target)+1 are body.
        body_len = min(len(target_label_ids), self.max_length - 2)
        if body_len > 0:
            labels[1 : 1 + body_len] = torch.tensor(
                target_label_ids[:body_len], dtype=torch.long
            )
        # [SEP] position and remaining tail stay as PAD_LABEL_ID (ignored).

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CanineDottingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer=None,
        max_length: int = 2048,
        batch_size: int = 16,
        num_workers: int = 4,
        val_split: float = 0.05,
        stratify_column: str = "source",
        text_column: str = "text",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "kwargs"])

        self.tokenizer = canine_tokenizer if tokenizer is None else tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.stratify_column = stratify_column
        self.text_column = text_column

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self._train_df = None
        self._test_df = None

        assert 0 < val_split < 1, "Validation split must be between 0 and 1"

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            if self._train_df is None:
                self._train_df = train_dataset.to_pandas()  # type: ignore
            if self.train_data is None or self.val_data is None:
                train_split, val_split = train_test_split(
                    self._train_df,
                    test_size=self.val_split,
                    random_state=constants.RANDOM_SEED,
                    stratify=self._train_df[self.stratify_column],  # type: ignore
                )
                self.train_data = train_split.reset_index(drop=True)
                self.val_data = val_split.reset_index(drop=True)

        if stage == "test" or stage is None:
            if self._test_df is None:
                self._test_df = test_dataset.to_pandas()  # type: ignore
                self.test_data = self._test_df.reset_index(drop=True)  # type: ignore

        if stage == "fit" or stage is None:
            print(f"Train dataset size: {len(self.train_data)}")  # type: ignore
            print(f"Validation dataset size: {len(self.val_data)}")  # type: ignore
        if stage == "test" or stage is None:
            print(f"Test dataset size: {len(self.test_data)}")  # type: ignore

    def _make_loader(self, data, shuffle: bool) -> DataLoader:
        dataset = LazyCanineDottingDataset(
            self.max_length, data, self.tokenizer, self.text_column
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_data is None:
            raise RuntimeError("Train data not set up. Call setup('fit') first.")
        return self._make_loader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_data is None:
            raise RuntimeError("Validation data not set up. Call setup('fit') first.")
        return self._make_loader(self.val_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_data is None:
            raise RuntimeError("Test data not set up. Call setup('test') first.")
        return self._make_loader(self.test_data, shuffle=False)
