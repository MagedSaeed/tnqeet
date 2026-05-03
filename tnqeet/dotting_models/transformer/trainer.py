import glob
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar


def get_trainer(
    model_name: str,
    logger=None,
    n_layers: int = 6,
    max_epochs: int = 10,
    checkpoint_dir: str = "tnqeet/dotting_models/transformer/trained_models/",
    resume_from_checkpoint: bool = True,
    accumulate_grad_batches: int = 1,
):
    checkpoint_dir = os.path.join(checkpoint_dir, model_name, f"layers_{n_layers}")

    latest_checkpoint = None
    if resume_from_checkpoint and os.path.exists(checkpoint_dir):
        last_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last_checkpoint):
            latest_checkpoint = last_checkpoint
        else:
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=4,
        mode="min",
        verbose=True,
        min_delta=0.0001,
        check_finite=True,
    )

    trainer = pl.Trainer(
        devices="auto",
        logger=logger,
        accelerator="auto",
        deterministic=False,  # SDPA kernels aren't fully deterministic.
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        max_epochs=max_epochs,
        val_check_interval=0.25,
        precision="bf16-mixed",
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[checkpoint_callback, early_stopping, TQDMProgressBar(refresh_rate=50)],
    )

    return trainer, latest_checkpoint
