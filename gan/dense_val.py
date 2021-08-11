import sys
import os
import re
import json

import argparse
import logging

import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from mlcore.callbacks import ModelCheckpoint, ProgressPrinter
import dense_model
from mlcore.config import Config, config_add_options, ConfigEntry
from pytorch_lightning.loggers import TensorBoardLogger

from formula_pipeline import build_train_dataloader, build_val_dataloader

from pytorch_lightning.utilities.cloud_io import load as pl_load


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


@config_add_options("val")
def config_export():
    return {
        "weights": ConfigEntry(),
        "output": ConfigEntry(),
    }


def main():
    config = Config()

    level = logging.ERROR
    if config.to_args().verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    model = dense_model.DenseModel(
        config.to_args(),
        flat_params=config.to_flat_args(),
        dictionary_path=config.to_args().val_dataloader.dictionary_path,
    )

    if torch.cuda.device_count() > 0:
        model = model.to("cuda:0")

    val_dataloader = build_val_dataloader(**{**config.to_args().val_dataloader, "batch_size": 1})
    # model_000500-val_loss=3.0454.ckpt

    # most basic trainer, uses good defaults
    # trainer = Trainer(
    #     callbacks=callbacks,
    #     gpus=-1,
    #     logger=logger,
    #     max_steps=config.to_args().trainer.max_steps,
    #     weights_save_path=config.to_args().trainer.output_path,
    #     checkpoint_callback=checkpoint_callback,
    #     val_check_interval=config.to_args().trainer.val_check_interval,
    #     distributed_backend="dp",
    #     log_save_interval=config.to_args().trainer.log_save_interval,
    #     gradient_clip_val=config.to_args().trainer.gradient_clip_val,
    #     precision=config.to_args().trainer.precision,
    # )
    # trainer.fit(model, train_dataloader, val_dataloader)
    checkpoints = []
    for model_file in os.listdir(config.to_args().trainer.output_path):
        match = re.match(r"model_(\d+)(-val_loss=([\d\.]+))?.ckpt", model_file)
        if not match:
            continue
        checkpoints.append(
            {
                "path": os.path.join(config.to_args().trainer.output_path, model_file),
                "iter": match.group(1),
                "loss": match.group(3),
            }
        )

    checkpoints = sorted(checkpoints, key=lambda x: x["loss"])
    # checkpoints = [os.listdir(config.to_args().trainer.output_path)]
    if config.to_args().val.weights is not None:
        checkpoint_data = pl_load(config.to_args().val.weights, map_location=lambda storage, loc: storage)
    else:
        checkpoint_data = pl_load(checkpoints[0]["path"], map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint_data["state_dict"])
    model.freeze()
    model.eval()

    with open(config.to_args().val.output, "w") as f:
        results = []
        count = 0
        print("Start Evaluation")
        for sample in val_dataloader:
            result = model.test_step(sample, 0)
            count += int(result["gt_str"] == result["pred_str"])
            results.append(
                {
                    "path": sample[b"path"][0].decode("utf-8"),
                    "id": sample[b"id"][0].decode("utf-8"),
                    "gt": result["gt_str"],
                    "pred": result["pred_str"],
                    "loss": result["loss"].item(),
                    "perplexity": result["perplexity"].item(),
                }
            )
        json.dump(
            {
                "checkpoint": checkpoints[0],
                "results": results,
                "count": count,
                "loss": np.mean([x["loss"] for x in results]).item(),
                "perplexity": np.mean([x["perplexity"] for x in results]).item(),
            },
            f,
        )


if __name__ == "__main__":
    sys.exit(main())
