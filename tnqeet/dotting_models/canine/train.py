import argparse
import math
import os

from pytorch_lightning.loggers import WandbLogger

from tnqeet.dotting_models.canine.data import CanineDottingDataModule
from tnqeet.dotting_models.canine.models import (
    CANINE_MODEL_NAME,
    LABEL_VOCAB_SIZE,
    PAD_LABEL_ID,
    CanineDottingModel,
)
from tnqeet.dotting_models.canine.trainer import get_trainer


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=CANINE_MODEL_NAME)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epochs", type=int, default=10)
args = parser.parse_args()

run_name = f"CANINE-{args.model_name.split('/')[-1]}"
checkpoint_dir = os.path.join("tnqeet/dotting_models/canine/trained_models/", run_name)
training_done_marker = os.path.join(checkpoint_dir, "training_complete")

if os.path.exists(training_done_marker):
    print(f"Training already completed (marker found at {training_done_marker}). Skipping.")
else:
    logger = WandbLogger(project="dotting_models", name=run_name)

    datamodule = CanineDottingDataModule(
        batch_size=args.batch_size,
    )
    datamodule.setup()

    # Estimate total training steps for the LR scheduler.
    steps_per_epoch = math.ceil(
        len(datamodule.train_data) / args.batch_size  # type: ignore
    )
    total_training_steps = steps_per_epoch * args.max_epochs

    model = CanineDottingModel(
        model_name=args.model_name,
        total_training_steps=total_training_steps,
    )

    trainer, checkpoint_path = get_trainer(
        model_name=run_name,
        logger=logger,
        max_epochs=args.max_epochs,
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
