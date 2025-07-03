import os
from pytorch_lightning.loggers import WandbLogger
from tnqeet.dotting_models.sequence_labeling.trainer import get_trainer
from tnqeet.dotting_models.sequence_labeling.data import DottingDataModule
from tnqeet.dotting_models.sequence_labeling.models import LSTMDottingModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_name = "LSTM"
num_layers = 5

logger = WandbLogger(project="dotting_models", name=model_name)

datamodule = DottingDataModule()
datamodule.setup()

model = LSTMDottingModel(
    vocab_size=datamodule.tokenizer.vocab_size,
    output_size=datamodule.tokenizer.vocab_size,
    pad_id=datamodule.tokenizer.pad_token_id,  # type: ignore
    max_sequence_length=datamodule.max_length,
    n_layers=num_layers,
)
trainer, checkpoint_path = get_trainer(
    model_name=model_name,
    logger=logger,
    n_layers=num_layers,
)

if checkpoint_path:
    print(f"Resuming from checkpoint: {checkpoint_path}")
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    trainer.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
else:
    print("No checkpoint found, starting from scratch.")
    trainer.validate(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
