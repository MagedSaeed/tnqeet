import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from tnqeet import constants


tnqeet_tokenizer = AutoTokenizer.from_pretrained(
    "MagedSaeed/tnqeet-tokenizer",
    trust_remote_code=True,
)


class TransformerDottingModel(pl.LightningModule):
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
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        total_training_steps=100_000,
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
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.total_training_steps = total_training_steps

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_id
        )
        self.position_embedding = nn.Embedding(max_sequence_length, d_model)
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
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
        no_decay = ("bias", "LayerNorm.weight", "norm.weight")
        decay_params = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        no_decay_params = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )
        warmup_steps = int(self.total_training_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
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
