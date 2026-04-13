import argparse
import os

from pytorch_lightning.loggers import WandbLogger

from tnqeet.dotting_models.sequence_labeling.data import DottingDataModule
from tnqeet.dotting_models.sequence_labeling.models import LSTMDottingModel
from tnqeet.dotting_models.sequence_labeling.trainer import get_trainer

parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=5)
args = parser.parse_args()

model_name = "LSTM"
checkpoint_dir = os.path.join(
    "tnqeet/dotting_models/sequence_labeling/trained_models/", model_name, f"layers_{args.num_layers}"
)
training_done_marker = os.path.join(checkpoint_dir, "training_complete")

if os.path.exists(training_done_marker):
    print(f"Training already completed for {model_name} layers={args.num_layers} (marker found). Skipping.")
else:
    logger = WandbLogger(project="dotting_models", name=model_name)

    datamodule = DottingDataModule()
    datamodule.setup()

    model = LSTMDottingModel(
        vocab_size=datamodule.tokenizer.vocab_size,
        output_size=datamodule.tokenizer.vocab_size,
        pad_id=datamodule.tokenizer.pad_token_id,  # type: ignore
        max_sequence_length=datamodule.max_length,
        n_layers=args.num_layers,
    )
    trainer, checkpoint_path = get_trainer(
        model_name=model_name,
        logger=logger,
        n_layers=args.num_layers,
    )

    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    else:
        print("No checkpoint found, starting from scratch.")
        trainer.validate(model=model, datamodule=datamodule)
        trainer.fit(model=model, datamodule=datamodule)

    # Write training-complete marker so future runs skip training.
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(training_done_marker, "w") as f:
        f.write("done\n")
    print(f"Training complete. Marker written to {training_done_marker}")

    trainer.test(model=model, datamodule=datamodule)