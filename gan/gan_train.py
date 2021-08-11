import sys
import os
import re
import json
import copy

import argparse
import logging

import torch
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger

from mlcore.config import Config, config_add_options, ConfigEntry, str2bool

from mlcore.callbacks import ProgressPrinter, ModelCheckpoint, LogModelWeightCallback
from callbacks import LogImageCallback, LogAttentionWeightCallback
from formula_pipeline import build_gan_dataloader

from gan_model import GANModel


@config_add_options("trainer")
def config_trainer():
    return {
        "output_path": ConfigEntry(default=os.getcwd()),
        "val_check_interval": ConfigEntry(default=1000, type=int),
        "log_save_interval": ConfigEntry(default=200, type=int),
        "gradient_clip_val": ConfigEntry(default=0, type=float),
        "precision": ConfigEntry(default=32, choices=(16, 32), type=int),
        "progress_refresh_rate": ConfigEntry(default=100),
        "checkpoint_save_interval": ConfigEntry(default=1000),
        "resume_from_checkpoint": ConfigEntry(default=None),
    }


def main():

    config = Config()

    level = logging.ERROR
    if config.to_args().verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    if pl.utilities.distributed.rank_zero_only.rank == 0:
        print(json.dumps(config.to_dict(), indent=2))

        os.makedirs(config.to_args().trainer.output_path, exist_ok=True)

        with open(os.path.join(config.to_args().trainer.output_path, "config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    # default logger used by trainer
    logger = TensorBoardLogger(save_dir=config.to_args().trainer.output_path, name="summary")

    callbacks = [
        ProgressPrinter(refresh_rate=config.to_args().trainer.progress_refresh_rate),
        LogImageCallback(),
        LogAttentionWeightCallback(),
        LogModelWeightCallback(),
    ]

    train_dataloader = build_gan_dataloader(**config.to_args().gan_dataloader)

    # sample = next(iter(train_dataloader))
    # print(sample["source_domain"])
    # print(sample["target_domain"])
    # exit()
    checkpoint_callback = ModelCheckpoint(
        checkpoint_save_interval=config.to_args().trainer.checkpoint_save_interval,
        filepath=os.path.join(config.to_args().trainer.output_path, "model_{global_step:06d}"),
        save_top_k=-1,
        verbose=True,
        # monitor="val_loss",
        period=0,
    )
    # most basic trainer, uses good defaults
    trainer = Trainer(
        callbacks=callbacks,
        # early_stop_callback=early_stop_callback,
        gpus=-1,
        logger=logger,
        weights_save_path=config.to_args().trainer.output_path,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=config.to_args().trainer.val_check_interval,
        distributed_backend="ddp",
        log_save_interval=config.to_args().trainer.log_save_interval,
        gradient_clip_val=config.to_args().trainer.gradient_clip_val,
        precision=config.to_args().trainer.precision,
        resume_from_checkpoint=config.to_args().trainer.resume_from_checkpoint
        # amp_level="02",
    )

    pl_model = GANModel(config.to_args())
    trainer.fit(pl_model, train_dataloader=train_dataloader)


if __name__ == "__main__":
    sys.exit(main())
