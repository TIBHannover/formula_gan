import sys
import os
import re
import json

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
        "output_path": ConfigEntry(default=os.getcwd()), 
        "val_check_interval": ConfigEntry(default=500, type=int),
        "log_save_interval": ConfigEntry(default=200, type=int),
        "gradient_clip_val": ConfigEntry(default=0.5, type=float),
    }

def main():

    config = Config()

    print(config.to_args())

    os.makedirs(config.to_args().trainer.output_path, exist_ok=True)

    with open(os.path.join(config.to_args().trainer.output_path,'config.json'), "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=config.to_args().trainer.output_path,
        name='summary'
    )

    model = hme_model.HMEModel(config.to_args(), flat_params=config.to_flat_args())

    checkpoint_callback = ModelCheckpoint(
         filepath=os.path.join(config.to_args().trainer.output_path,'model_{global_step:06d}-{val_loss:.4f}'),
         save_top_k = -1,
         verbose =True,
         monitor='val_loss',

     )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=16,
        verbose=True,
        mode='min'
    )

    # most basic trainer, uses good defaults
    trainer = Trainer(early_stop_callback=early_stop_callback, progress_bar_refresh_rate=1, gpus=-1, logger=logger, weights_save_path=config.to_args().trainer.output_path,checkpoint_callback=checkpoint_callback, val_check_interval=config.to_args().trainer.val_check_interval, distributed_backend='dp', log_save_interval=config.to_args().trainer.log_save_interval, gradient_clip_val=config.to_args().trainer.gradient_clip_val)
    trainer.fit(model)


if __name__ == "__main__":
    sys.exit(main())
