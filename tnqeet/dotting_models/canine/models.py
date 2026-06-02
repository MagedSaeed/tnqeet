import string

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from transformers import CanineModel, CanineTokenizer, get_linear_schedule_with_warmup

from tnqeet import constants
from tnqeet.weights import resolve_weight


CANINE_MODEL_NAME = "google/canine-c"

# Label vocabulary: every character that may appear in dotted text.
# This is the character-level analogue of the LSTM's "full tokenizer vocab"
# label space. Order is fixed so checkpoints are reproducible.
_DOTTED_ARABIC_CHARS = sorted(
    set(constants.LETTERS_MAPPING.keys())
    | set(constants.ARABIC_RASMS)
    | set(constants.ARABIC_LETTERS)
)
_ASCII_CHARS = list(string.printable)  # letters, digits, punctuation, whitespace
_EXTRA_CHARS = ["\u060c", "\u061b", "\u061f", "\u0640", "\xa0"]  # ، ؛ ؟ tatweel nbsp

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

LABEL_CHARS = list(
    dict.fromkeys(
        [PAD_TOKEN, UNK_TOKEN] + _DOTTED_ARABIC_CHARS + _ASCII_CHARS + _EXTRA_CHARS
    )
)
CHAR_TO_LABEL = {ch: i for i, ch in enumerate(LABEL_CHARS)}
LABEL_TO_CHAR = {i: ch for i, ch in enumerate(LABEL_CHARS)}
PAD_LABEL_ID = CHAR_TO_LABEL[PAD_TOKEN]
UNK_LABEL_ID = CHAR_TO_LABEL[UNK_TOKEN]
LABEL_VOCAB_SIZE = len(LABEL_CHARS)


def encode_labels(text: str) -> list:
    return [CHAR_TO_LABEL.get(ch, UNK_LABEL_ID) for ch in text]


def decode_labels(label_ids, source_chars=None) -> str:
    out = []
    for idx, label_id in enumerate(label_ids):
        ch = LABEL_TO_CHAR.get(int(label_id), "")
        if ch == PAD_TOKEN:
            continue
        if ch in (UNK_TOKEN, ""):
            # Copy through the original source character for unknown labels.
            if source_chars is not None and idx < len(source_chars):
                out.append(source_chars[idx])
            continue
        out.append(ch)
    return "".join(out)


canine_tokenizer = CanineTokenizer.from_pretrained(CANINE_MODEL_NAME)


class CanineDottingModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = CANINE_MODEL_NAME,
        num_labels: int = LABEL_VOCAB_SIZE,
        pad_label_id: int = PAD_LABEL_ID,
        max_sequence_length: int = 2048,
        dropout: float = 0.1,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        total_training_steps: int = 100_000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.num_labels = num_labels
        self.pad_label_id = pad_label_id
        self.max_sequence_length = max_sequence_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.total_training_steps = total_training_steps

        self.backbone = CanineModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_labels)

        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_labels, ignore_index=pad_label_id
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_labels, ignore_index=pad_label_id
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_labels, ignore_index=pad_label_id
        )

    @classmethod
    def from_pretrained(cls, size=None, revision=None, weights_dir=None, **kwargs):
        """Load a pretrained CANINE dotter by friendly size (``"c"`` or ``"s"``).

        Downloads the checkpoint from the Hugging Face Hub on demand, or reads
        from a local ``trained_models`` tree when ``weights_dir`` is given.
        """
        checkpoint_path = resolve_weight(
            "canine", size=size, revision=revision, weights_dir=weights_dir
        )
        return cls.load_from_checkpoint(checkpoint_path, **kwargs)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits

    def step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        logits = self(input_ids, attention_mask=attention_mask)
        flat_logits = logits.view(-1, self.num_labels)
        flat_labels = labels.view(-1)
        return flat_logits, flat_labels

    def training_step(self, batch, batch_idx):
        logits, labels = self.step(batch)
        loss = F.cross_entropy(logits, labels, ignore_index=self.pad_label_id)
        acc = self.train_accuracy(logits, labels)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels = self.step(batch)
        loss = F.cross_entropy(logits, labels, ignore_index=self.pad_label_id)
        acc = self.val_accuracy(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        logits, labels = self.step(batch)
        loss = F.cross_entropy(logits, labels, ignore_index=self.pad_label_id)
        acc = self.test_accuracy(logits, labels)
        self.log_dict({"test_acc": acc, "test_loss": loss}, prog_bar=True)
        return logits

    def predict_step(self, batch, batch_idx):
        logits, labels = self.step(batch)
        predictions = torch.argmax(logits, dim=-1)
        return predictions, labels

    def configure_optimizers(self):  # type: ignore
        no_decay = ("bias", "LayerNorm.weight")
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
        pad_char=" ",
    ):
        """Restore dots to undotted (Rasm) Arabic text.

        When resolve_ambiguous_rasms_only=True (default), the model's
        predictions are only used at ambiguous Rasm positions — all other
        characters (spaces, harakat, punctuation, unambiguous letters) are
        copied from the input unchanged. This is the recommended mode because
        CANINE's label vocabulary may not cover every possible Unicode
        character (e.g. harakat, Quranic marks). With False, the model
        predicts at every position, which can introduce unnecessary errors on
        non-Rasm characters and lose characters not in the label vocabulary.
        """
        self.eval()
        is_single_text = False
        if isinstance(dotless_texts, str):
            dotless_texts = [dotless_texts]
            is_single_text = True
        # CANINE's downsampling uses MaxPool1d with kernel_size=downsampling_rate,
        # so the input sequence (excluding [CLS]/[SEP]) must be at least that long.
        min_chars = self.backbone.config.downsampling_rate
        padded_texts = [
            t + pad_char * (min_chars - len(t)) if len(t) < min_chars else t
            for t in dotless_texts
        ]
        tokenizer = tokenizer or canine_tokenizer
        encoded = tokenizer(
            padded_texts,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self(encoded["input_ids"], attention_mask=encoded["attention_mask"])
        predictions = torch.argmax(logits, dim=-1).cpu().tolist()
        attention = encoded["attention_mask"].cpu().tolist()

        results = []
        for j, dotless in enumerate(dotless_texts):
            # CANINE adds [CLS] at position 0 and [SEP] at the end; the body
            # corresponds to the input characters in order.
            valid_len = sum(attention[j])
            body = predictions[j][1 : valid_len - 1]
            decoded = decode_labels(body, source_chars=dotless)
            decoded = decoded[: len(dotless)]
            # If the model truncated/produced fewer chars, pad with original.
            if len(decoded) < len(dotless):
                decoded = decoded + dotless[len(decoded) :]
            if resolve_ambiguous_rasms_only:
                decoded = "".join(
                    decoded[i] if constants.is_ambigous_rasm(dotless[i]) else dotless[i]
                    for i in range(len(dotless))
                )
            results.append(decoded)
        return results[0] if is_single_text else results
