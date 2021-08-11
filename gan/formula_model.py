"""
This file defines the core research contribution   
"""
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from argparse import ArgumentParser

from mlcore.config import config_add_options, ConfigEntry, str2bool

import pytorch_lightning as pl

from crohme_dataset import CrohmeDataset, PadCollate
from msgpack_dataset import EqualShardSampler
import torchvision


class FormulaOCR(pl.LightningModule):
    def __init__(self, params, flat_params, **kwargs):
        super(FormulaOCR, self).__init__()
        # not the best model...
        # self.hparams = flat_params
        self.params = params

    # def training_step(self, batch, batch_idx):
    #     # REQUIRED
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     return {"loss": F.cross_entropy(y_hat, y)}

    # def validation_step(self, batch, batch_idx):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self.forward(x)
    # return {"val_loss": F.cross_entropy(y_hat, y)}

    # def validation_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     return {"avg_val_loss": avg_loss}

    def train_dataloader(self):
        collate_fn = PadCollate()
        shards_sampler = EqualShardSampler()
        return DataLoader(
            CrohmeDataset(
                self.params.train_dataloader.msgpack_path,
                self.params.train_dataloader.annotation_path,
                self.params.train_dataloader.dictionary_path,
                shards_sampler=shards_sampler,
                mean_height=self.params.train_dataloader.mean_height,
                max_height=self.params.train_dataloader.max_height,
                max_width=self.params.train_dataloader.max_width,
                max_image_area=self.params.train_dataloader.max_image_area,
                training=True,
            ),
            batch_size=self.params.train_dataloader.batch_size,
            collate_fn=collate_fn,
            num_workers=16,
            drop_last=True,
        )

    def val_dataloader(self):
        collate_fn = PadCollate()
        return DataLoader(
            CrohmeDataset(
                self.params.val_dataloader.msgpack_path,
                self.params.val_dataloader.annotation_path,
                self.params.val_dataloader.dictionary_path,
                mean_height=self.params.val_dataloader.mean_height,
                max_height=self.params.val_dataloader.max_height,
                max_width=self.params.val_dataloader.max_width,
                max_image_area=self.params.val_dataloader.max_image_area,
                training=False,
            ),
            batch_size=self.params.val_dataloader.batch_size,
            collate_fn=collate_fn,
            num_workers=16,
            drop_last=True,
        )

    def test_dataloader(self):
        collate_fn = PadCollate()
        return DataLoader(
            CrohmeDataset(
                self.params.test_dataloader.msgpack_path,
                self.params.test_dataloader.annotation_path,
                self.params.test_dataloader.dictionary_path,
                mean_height=self.params.test_dataloader.mean_height,
                max_height=self.params.test_dataloader.max_height,
                max_width=self.params.test_dataloader.max_width,
                max_image_area=self.params.test_dataloader.max_image_area,
                training=False,
            ),
            batch_size=self.params.test_dataloader.batch_size,
            collate_fn=collate_fn,
            num_workers=16,
            drop_last=True,
        )

    # def test_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(
    #         MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
    #         batch_size=self.params.test_dataloader.batch_size,
    # #     )

    # @config_add_options("train_dataloader")
    # def config_train_dataloader():
    #     return {
    #         "msgpack_path": ConfigEntry(
    #             default=[
    #                 "/data/va-formula-ocr/crohme/train_2014/msg/",
    #                 "/data/va-formula-ocr/ntcir_gan_paper_bmvc/gan_msg/",
    #             ],
    #             nargs="+",
    #         ),
    #         "annotation_path": ConfigEntry(default="/data/va-formula-ocr/crohme_2019/train_equations_2014.jsonl"),
    #         "dictionary_path": ConfigEntry(default="/data/va-formula-ocr/dictionary.txt"),
    #         "mean_height": ConfigEntry(default=128),
    #         "max_height": ConfigEntry(default=256),
    #         "max_width": ConfigEntry(default=1024),
    #         "max_image_area": ConfigEntry(default=200000),
    #         "batch_size": ConfigEntry(default=16),
    #     }

    # @config_add_options("val_dataloader")
    # def config_val_dataloader():
    #     return {
    #         "msgpack_path": ConfigEntry(default="/data/va-formula-ocr/crohme/test_2013/msg/"),
    #         "annotation_path": ConfigEntry(default="/data/va-formula-ocr/crohme_2019/test_equations_2013.jsonl"),
    #         "dictionary_path": ConfigEntry(default="/data/va-formula-ocr/dictionary.txt"),
    #         "mean_height": ConfigEntry(default=128),
    #         "max_height": ConfigEntry(default=256),
    #         "max_width": ConfigEntry(default=1024),
    #         "max_image_area": ConfigEntry(default=200000),
    #         "batch_size": ConfigEntry(default=16),
    #     }

    # @config_add_options("test_dataloader")
    # def config_val_dataloader():
    #     return {
    #         "msgpack_path": ConfigEntry(default="/data/va-formula-ocr/crohme/test_2013/msg/"),
    #         "annotation_path": ConfigEntry(default="/data/va-formula-ocr/crohme_2019/test_equations_2013.jsonl"),
    #         "dictionary_path": ConfigEntry(default="/data/va-formula-ocr/dictionary.txt"),
    #         "mean_height": ConfigEntry(default=128),
    #         "max_height": ConfigEntry(default=256),
    #         "max_width": ConfigEntry(default=1024),
    #         "max_image_area": ConfigEntry(default=200000),
    #         "batch_size": ConfigEntry(default=1),
    #     }
