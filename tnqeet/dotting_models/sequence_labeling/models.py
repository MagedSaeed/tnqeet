import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer
from tnqeet import constants
import torchmetrics

tnqeet_tokenizer = AutoTokenizer.from_pretrained(
    "MagedSaeed/tnqeet-tokenizer",
    trust_remote_code=True,
)


class LSTMDottingModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size=None,
        output_size=None,
        pad_id=1,
        seq_len=1024,
        hidden_size=512,
        embedding_size=512,
        dropout=0.33,
        learning_rate=0.001,
        n_layers=1,
        bidirectional=True,
    ):
        super().__init__()

        self.save_hyperparameters()

        vocab_size = vocab_size or tnqeet_tokenizer.vocab_size
        output_size = output_size or vocab_size
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.dropout_prop = dropout
        self.pad_id = pad_id
        self.max_sequence_length = seq_len
        self.output_size = output_size

        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=output_size,
            ignore_index=self.pad_id,
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=output_size,
            ignore_index=self.pad_id,
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=output_size,
            ignore_index=self.pad_id,
        )

        # embedding layer is useful to map input into vector representation
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=pad_id,
        )

        # LSTM layer preserved by PyTorch library
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # self.embedding_dropout = nn.Dropout(embedding_dropout)

        # Linear layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        outputs = self.embedding(inputs)
        # outputs = self.embedding_dropout(outputs) # apply dropout on embedding
        inputs_lengths = torch.sum(inputs != self.pad_id, axis=-1).cpu()  # type: ignore
        packed_outputs = nn.utils.rnn.pack_padded_sequence(
            outputs,
            inputs_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        if not self.bidirectional:
            # pass forward to lstm
            packed_outputs, _ = self.lstm(packed_outputs)
            outputs, lengths = nn.utils.rnn.pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=self.max_sequence_length,
            )
        else:
            bidirectional_packed_outputs, _ = self.lstm(packed_outputs)
            bidirectional_outputs, lengths = nn.utils.rnn.pad_packed_sequence(
                bidirectional_packed_outputs,
                batch_first=True,
                # padding_value=self.pad_id,
                total_length=self.max_sequence_length,
            )
            outputs = (
                bidirectional_outputs[:, :, : self.hidden_size] + bidirectional_outputs[:, :, self.hidden_size :]
            )
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        # softmax will be done in the loss calculation
        return outputs

    def step(self, inputs, labels):
        assert torch.sum(inputs == self.pad_id) == torch.sum(labels == self.pad_id), (
            f"pad ids and their target tags does not match: {torch.sum(inputs == self.pad_id):=} != {torch.sum(labels == self.pad_id):=}"
        )
        outputs = self(inputs)
        outputs = outputs.squeeze()
        outputs = outputs.view(outputs.shape[0] * outputs.shape[1], -1)
        labels = labels.view(-1)
        return outputs, labels

    def training_step(self, batch, batch_idx):
        inputs = batch["input_ids"]
        labels = batch["labels"]
        outputs, labels = self.step(inputs, labels)
        loss = F.cross_entropy(
            outputs,
            labels,
            ignore_index=self.pad_id,
        )
        train_accuracy = self.train_accuracy(outputs, labels)
        self.log(
            "loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train_acc",
            train_accuracy,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["input_ids"]
        labels = batch["labels"]
        outputs, labels = self.step(inputs, labels)
        loss = F.cross_entropy(
            outputs,
            labels,
            ignore_index=self.pad_id,
        )
        val_accuracy = self.val_accuracy(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log(
            "val_acc",
            val_accuracy,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        inputs = batch["input_ids"]
        labels = batch["labels"]
        outputs, labels = self.step(inputs, labels)
        loss = F.cross_entropy(
            outputs,
            labels,
            ignore_index=self.pad_id,
        )
        test_accuracy = self.test_accuracy(outputs, labels)
        metrics = {"test_acc": test_accuracy, "test_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        return outputs

    def predict_step(self, batch, batch_idx):
        inputs = batch["input_ids"]
        labels = batch["labels"]
        outputs, labels = self.step(inputs, labels)
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
            verbose=True,  # type: ignore
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
        tokenizer=tnqeet_tokenizer,
        resolve_ambiguous_rasms_only=False,
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
        predictions = [prediction[: len(dotless_texts[j])] for j, prediction in enumerate(predictions)]
        if resolve_ambiguous_rasms_only:
            predictions = [
                "".join(
                    prediction[i] if constants.is_ambigous_rasm(dotless_texts[j][i]) else dotless_texts[j][i]
                    for i in range(len(dotless_texts[j]))
                )
                for j, prediction in enumerate(predictions)
            ]
        return predictions[0] if is_single_text else predictions
