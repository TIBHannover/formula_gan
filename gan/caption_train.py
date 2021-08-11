import sys
import os
import re

import argparse
import logging

import torch
from pytorch_lightning import Trainer
import wap_model
import caption_model
from config import Config, config_add_options, ConfigEntry
from pytorch_lightning.loggers import TensorBoardLogger




@config_add_options("trainer")
def config_trainer():
    return {
        "output_path": ConfigEntry(default=os.getcwd()),  # learning rate for encoder if fine-tuning
    }

def main():

    config = Config()

    print(config.to_args())

    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=config.to_args().trainer.output_path,
        name='lightning_logs'
    )

    model = caption_model.CaptionModel(config.to_args(), flat_params=config.to_flat_args())

    # most basic trainer, uses good defaults
    trainer = Trainer(gpus=-1, logger=logger, weights_save_path=config.to_args().trainer.output_path)
    trainer.fit(model)


if __name__ == "__main__":
    sys.exit(main())
