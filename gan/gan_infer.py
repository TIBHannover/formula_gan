import sys
import os
import re
import json
import copy

import argparse
import logging

import torch
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load

from mlcore.config import Config, config_add_options, ConfigEntry, str2bool

from mlcore.callbacks import ProgressPrinter, ModelCheckpoint, LogModelWeightCallback
from callbacks import LogImageCallback, LogAttentionWeightCallback
from formula_pipeline import build_infer_dataloader

from gan_model import GANModel

import msgpack
import imageio


@config_add_options("infer")
def config_infer():
    return {
        "output_path": ConfigEntry(default=os.getcwd()),
        "create_msgpack": ConfigEntry(default=False, type=str2bool),
        "checkpoint_path": ConfigEntry(default=os.getcwd(), nargs="+"),
    }


class MsgPackCrawlerEntrySaver:
    def __init__(self, msg_path, chunck_size=1024, **kwargs):

        self.path = os.path.abspath(msg_path)
        self.chunck_size = chunck_size

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        shards_re = r"shard_(\d+).msg"

        self.shards_index = [
            int(re.match(shards_re, x).group(1)) for x in os.listdir(self.path) if re.match(shards_re, x)
        ]
        self.shard_open = None
        self.current_shard = 0

    def open_next(self):

        if len(self.shards_index) == 0:
            next_index = 0
        else:
            next_index = sorted(self.shards_index)[-1] + 1
        self.shards_index.append(next_index)
        self.current_shard = next_index

        if self.shard_open is not None and not self.shard_open.closed:
            self.shard_open.close()
        self.count = 0
        self.shard_open = open(os.path.join(self.path, f"shard_{next_index}.msg"), "wb")

    def __enter__(self):
        self.open_next()
        return self

    def __exit__(self, type, value, tb):
        self.shard_open.close()

    def handle(self, data_dict):
        if self.count >= self.chunck_size:
            self.open_next()

        self.shard_open.write(msgpack.packb(data_dict))
        self.count += 1


def main():

    config = Config()

    level = logging.ERROR
    if config.to_args().verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    os.makedirs(config.to_args().infer.output_path, exist_ok=True)
    checkpoints = []
    for path in config.to_args().infer.checkpoint_path:
        if os.path.isdir(path):

            checkpoints = [
                re.match(r"model_(\d+)\.ckpt", x).group(1)
                for x in os.listdir(path)
                if re.match(r"model_(\d+)\.ckpt", x)
            ]
            checkpoints.extend([{"path": os.path.join(path, f"model_{x}.ckpt"), "iter": int(x)} for x in checkpoints])

        else:
            match = re.match(r".*?/model_(\d+)\.ckpt", path)

            if not match:
                print(f"Could not find the checkpoint number in path {path}")
                return -1
            checkpoints.extend([{"path": path, "iter": int(match.group(1))}])

    infer_dataloader = build_infer_dataloader(**config.to_args().infer_dataloader)

    for checkpoint in checkpoints:
        pl_model = GANModel(params=config.to_args())
        pl_model.to(device)

        checkpoint_data = pl_load(checkpoint["path"], map_location=lambda storage, loc: storage)

        pl_model.load_state_dict(checkpoint_data["state_dict"])
        pl_model.freeze()
        if config.to_args().infer.create_msgpack:
            output = MsgPackCrawlerEntrySaver(os.path.join(config.to_args().infer.output_path, str(checkpoint["iter"])))
            with output as o:
                for i, data in enumerate(infer_dataloader):
                    image = data["image"].to(device)

                    z = torch.randn([image.shape[0], config.to_args().gan.z_dim]).type_as(image)
                    target_domain = torch.randint(size=[image.shape[0]], high=config.to_args().gan.num_classes).to(
                        z.device
                    )

                    model_output = pl_model(image, z, target_domain)  # .to("cuda:0")
                    for j in range(model_output.shape[0]):
                        generated_image = model_output[j, 0].detach().cpu()
                        # print(data[b"id"][j])

                        if config.to_args().infer.create_msgpack:
                            o.handle(
                                {
                                    b"image": imageio.imwrite(imageio.RETURN_BYTES, generated_image, format="jpg"),
                                    b"path": data[b"path"][j],
                                    b"equation": data[b"equation"][j],
                                    b"id": data[b"id"][j],
                                }
                            )
        elif config.to_args().infer.output_path is not None:
            os.makedirs(config.to_args().infer.output_path, exist_ok=True)
            for i, data in enumerate(infer_dataloader):
                image = data["image"].to(device)

                z = torch.zeros([image.shape[0], config.to_args().gan.z_dim]).type_as(image)
                target_domain = torch.ones([image.shape[0]], dtype=torch.long).to(z.device)

                model_output = pl_model(image, z, target_domain)  # .to("cuda:0")
                for j in range(model_output.shape[0]):
                    generated_image = model_output[j, 0].detach().cpu()
                    if np.sum(generated_image.numpy()) > 1:
                        output_path = os.path.join(
                            config.to_args().infer.output_path, os.path.basename(data["path"][j])
                        )
                        imageio.imwrite(output_path, generated_image)


if __name__ == "__main__":
    sys.exit(main())
