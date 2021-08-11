import sys
import os
import re
import json

import argparse
import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from mlcore.callbacks import ModelCheckpoint, ProgressPrinter
import dense_model
from mlcore.config import Config, config_add_options, ConfigEntry
from pytorch_lightning.loggers import TensorBoardLogger

from formula_pipeline import build_train_dataloader, build_val_dataloader


@config_add_options("trainer")
def config_trainer():
    return {
        "output_path": ConfigEntry(default=os.getcwd()),
        "val_check_interval": ConfigEntry(default=500, type=int),
        "log_save_interval": ConfigEntry(default=200, type=int),
        "checkpoint_save_interval": ConfigEntry(default=5000, type=int),
        "gradient_clip_val": ConfigEntry(default=0, type=int),
        "precision": ConfigEntry(default=32, type=int),
        "progress_refresh_rate": ConfigEntry(default=100, type=int),
        "max_steps": ConfigEntry(default=100000, type=int),
    }


def main():
    config = Config()

    level = logging.ERROR
    if config.to_args().verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    if pl.utilities.distributed.rank_zero_only.rank == 0:
        os.makedirs(config.to_args().trainer.output_path, exist_ok=True)

        with open(os.path.join(config.to_args().trainer.output_path, "config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    # default logger used by trainer
    logger = TensorBoardLogger(save_dir=config.to_args().trainer.output_path, name="summary")

    model = dense_model.DenseModel(config.to_args(), flat_params=config.to_flat_args())

    checkpoint_callback = ModelCheckpoint(
        checkpoint_save_interval=config.to_args().trainer.checkpoint_save_interval,
        filepath=os.path.join(config.to_args().trainer.output_path, "model_{global_step:06d}-{val_loss:.4f}"),
        save_top_k=-1,
        verbose=True,
        monitor="val_loss",
    )

    callbacks = [
        ProgressPrinter(refresh_rate=config.to_args().trainer.progress_refresh_rate),
        pl.callbacks.LearningRateLogger(),
    ]

    train_dataloader = build_train_dataloader(**config.to_args().train_dataloader)

    val_dataloader = build_val_dataloader(**config.to_args().val_dataloader)
    # most basic trainer, uses good defaults
    trainer = Trainer(
        callbacks=callbacks,
        gpus=-1,
        logger=logger,
        max_steps=config.to_args().trainer.max_steps,
        weights_save_path=config.to_args().trainer.output_path,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=config.to_args().trainer.val_check_interval,
        distributed_backend="ddp",
        log_save_interval=config.to_args().trainer.log_save_interval,
        gradient_clip_val=config.to_args().trainer.gradient_clip_val,
        precision=config.to_args().trainer.precision,
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    sys.exit(main())
