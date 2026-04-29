import argparse
import math
import os

from pytorch_lightning.loggers import WandbLogger

from tnqeet.dotting_models.transformer.data import TransformerDottingDataModule
from tnqeet.dotting_models.transformer.models import TransformerDottingModel
from tnqeet.dotting_models.transformer.trainer import get_trainer


parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epochs", type=int, default=10)
args = parser.parse_args()

model_name = "Transformer"
checkpoint_dir = os.path.join(
    "tnqeet/dotting_models/transformer/trained_models/",
    model_name,
    f"layers_{args.num_layers}",
)
training_done_marker = os.path.join(checkpoint_dir, "training_complete")

if os.path.exists(training_done_marker):
    print(
        f"Training already completed for {model_name} layers={args.num_layers} "
        f"(marker found). Skipping."
    )
else:
    run_name = f"{model_name}-layers_{args.num_layers}"
    logger = WandbLogger(project="dotting_models", name=run_name)

    datamodule = TransformerDottingDataModule(batch_size=args.batch_size)
    datamodule.setup()

    steps_per_epoch = math.ceil(len(datamodule.train_data) / args.batch_size)  # type: ignore
    total_training_steps = steps_per_epoch * args.max_epochs

    model = TransformerDottingModel(
        vocab_size=datamodule.tokenizer.vocab_size,
        output_size=datamodule.tokenizer.vocab_size,
        pad_id=datamodule.tokenizer.pad_token_id,  # type: ignore
        max_sequence_length=datamodule.max_length,
        num_layers=args.num_layers,
        total_training_steps=total_training_steps,
    )

    trainer, checkpoint_path = get_trainer(
        model_name=model_name,
        logger=logger,
        n_layers=args.num_layers,
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

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(training_done_marker, "w") as f:
        f.write("done\n")
    print(f"Training complete. Marker written to {training_done_marker}")

    trainer.test(model=model, datamodule=datamodule)
