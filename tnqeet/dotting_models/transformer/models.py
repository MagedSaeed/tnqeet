import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from transformers import AutoTokenizer

from tnqeet import constants


def _sinusoidal_position_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


tnqeet_tokenizer = AutoTokenizer.from_pretrained(
    "MagedSaeed/tnqeet-tokenizer",
    trust_remote_code=True,
)


class TransformerDottingModel(pl.LightningModule):
    position_encoding: torch.Tensor

    def __init__(
        self,
        vocab_size=None,
        output_size=None,
        pad_id=1,
        max_sequence_length=2048,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        vocab_size = vocab_size or tnqeet_tokenizer.vocab_size
        output_size = output_size or vocab_size

        self.vocab_size = vocab_size
        self.output_size = output_size
        self.pad_id = pad_id
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.learning_rate = learning_rate

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_id
        )
        self.register_buffer(
            "position_encoding",
            _sinusoidal_position_encoding(max_sequence_length, d_model),
            persistent=False,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),  # final norm; pre-LN convention
        )
        self.fc = nn.Linear(d_model, output_size)

        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=output_size, ignore_index=pad_id
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=output_size, ignore_index=pad_id
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=output_size, ignore_index=pad_id
        )

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.token_embedding(input_ids) + self.position_encoding[:seq_len]
        x = self.embedding_dropout(x)
        # True at pad positions -> masked out of attention.
        pad_mask = input_ids == self.pad_id
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.fc(x)

    def step(self, inputs, labels):
        assert torch.sum(inputs == self.pad_id) == torch.sum(labels == self.pad_id), (
            f"pad ids and their target tags do not match: "
            f"{torch.sum(inputs == self.pad_id)} != {torch.sum(labels == self.pad_id)}"
        )
        outputs = self(inputs)
        outputs = outputs.view(-1, self.output_size)
        labels = labels.view(-1)
        return outputs, labels

    def training_step(self, batch, batch_idx):
        outputs, labels = self.step(batch["input_ids"], batch["labels"])
        loss = F.cross_entropy(outputs, labels, ignore_index=self.pad_id)
        acc = self.train_accuracy(outputs, labels)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, labels = self.step(batch["input_ids"], batch["labels"])
        loss = F.cross_entropy(outputs, labels, ignore_index=self.pad_id)
        acc = self.val_accuracy(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        outputs, labels = self.step(batch["input_ids"], batch["labels"])
        loss = F.cross_entropy(outputs, labels, ignore_index=self.pad_id)
        acc = self.test_accuracy(outputs, labels)
        self.log_dict({"test_acc": acc, "test_loss": loss}, prog_bar=True)
        return outputs

    def predict_step(self, batch, batch_idx):
        outputs, labels = self.step(batch["input_ids"], batch["labels"])
        predictions = torch.argmax(outputs, dim=-1)
        return predictions, labels

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    @torch.no_grad()
    def restore_dots(
        self,
        dotless_texts,
        tokenizer=None,
        resolve_ambiguous_rasms_only=True,
    ):
        self.eval()
        is_single_text = False
        if isinstance(dotless_texts, str):
            dotless_texts = [dotless_texts]
            is_single_text = True
        tokenizer = tokenizer or tnqeet_tokenizer
        dotless_samples = tokenizer(
            dotless_texts,
            padding=True,
            return_tensors="pt",
        ).to(self.device)["input_ids"]
        outputs = self(dotless_samples)
        predictions = torch.argmax(outputs, dim=-1)
        predictions = tokenizer.batch_decode(predictions)
        predictions = [
            prediction[: len(dotless_texts[j])]
            for j, prediction in enumerate(predictions)
        ]
        if resolve_ambiguous_rasms_only:
            predictions = [
                "".join(
                    prediction[i]
                    if constants.is_ambigous_rasm(dotless_texts[j][i])
                    else dotless_texts[j][i]
                    for i in range(len(dotless_texts[j]))
                )
                for j, prediction in enumerate(predictions)
            ]
        return predictions[0] if is_single_text else predictions
