import os
import glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def get_trainer(
    model_name,
    logger=None,
    n_layers=1,
    max_epochs: int = 100,
    checkpoint_dir: str = "tnqeet/dotting_models/sequence_labeling/trained_models/",
    resume_from_checkpoint: bool = True,
):
    # Model checkpoint - save only the best
    checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    checkpoint_dir = os.path.join(checkpoint_dir, f"layers_{n_layers}")

    # Find the latest checkpoint if resuming is enabled
    latest_checkpoint = None
    if resume_from_checkpoint and os.path.exists(checkpoint_dir):
        # Look for the last.ckpt file first (most recent)
        last_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last_checkpoint):
            latest_checkpoint = last_checkpoint
        else:
            # Fallback to finding the best checkpoint by filename pattern
            checkpoint_pattern = os.path.join(checkpoint_dir, "*.ckpt")
            checkpoint_files = glob.glob(checkpoint_pattern)
            if checkpoint_files:
                # Sort by modification time to get the most recent
                latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # Early stopping after 4 evals
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=4,
        mode="min",
        verbose=True,
        min_delta=0.00001,
        check_finite=True,
    )

    trainer = pl.Trainer(
        devices="auto",
        logger=logger,
        accelerator="auto",
        deterministic=True,
        gradient_clip_val=1,
        max_epochs=max_epochs,
        val_check_interval=0.25,  # 4 evals per epoch
        callbacks=[checkpoint_callback, early_stopping],
    )

    return trainer, latest_checkpoint
