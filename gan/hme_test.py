import sys
import os
import re

import argparse
import logging

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hme_model
from config import Config, config_add_options, ConfigEntry
from pytorch_lightning.loggers import TensorBoardLogger




@config_add_options("trainer")
def config_trainer():
    return {
        "output_path": ConfigEntry(default=os.getcwd()),  # learning rate for encoder if fine-tuning
        "checkpoint_path": ConfigEntry(default=os.getcwd()),  # learning rate for encoder if fine-tuning
    }

def main():

    config = Config()

    print(config.to_args())


    # most basic trainer, uses good defaults
    model = hme_model.HMEModel.load_from_checkpoint(checkpoint_path=config.to_args().trainer.checkpoint_path,params=config.to_args(), flat_params=config.to_flat_args())
    trainer=Trainer(gpus=-1, distributed_backend='dp',show_progress_bar=False)
    trainer.test(model)

if __name__ == "__main__":
    sys.exit(main())
