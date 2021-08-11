import torch
import logging

import torch
import copy
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only

from models.attention_layer import SelfAttention2d


class LogImageCallback(pl.callbacks.Callback):
    def __init__(self, log_save_interval=None, nrow=2, **kwargs):
        super(LogImageCallback, self).__init__(**kwargs)
        self.log_save_interval = log_save_interval
        self.nrow = nrow

    @rank_zero_only
    def on_batch_end(self, trainer, pl_module):
        if trainer.logger is None:
            return

        if self.log_save_interval is None:
            log_interval = trainer.log_save_interval
        else:
            log_interval = self.log_save_interval

        if (trainer.global_step + 1) % log_interval == 0:

            if hasattr(pl_module, "source_image"):
                grid = torchvision.utils.make_grid(pl_module.source_image, normalize=True, nrow=self.nrow)
                trainer.logger.experiment.add_image(f"source/image", grid, trainer.global_step + 1)
                try:
                    trainer.logger.experiment.add_histogram(
                        f"source/dist", pl_module.source_image, trainer.global_step + 1
                    )
                except ValueError as e:
                    logging.info(f"LogImageCallback (source/dist): {e}")

            if hasattr(pl_module, "transfered_image"):
                grid = torchvision.utils.make_grid(pl_module.transfered_image, normalize=True, nrow=self.nrow)
                trainer.logger.experiment.add_image(f"transfered/image", grid, trainer.global_step + 1)

                try:
                    trainer.logger.experiment.add_histogram(
                        f"transfered/dist", pl_module.transfered_image, trainer.global_step + 1
                    )
                except ValueError as e:
                    logging.info(f"LogImageCallback (transfered/dist): {e}")

            if hasattr(pl_module, "target_image"):
                grid = torchvision.utils.make_grid(pl_module.target_image, normalize=True, nrow=self.nrow)
                trainer.logger.experiment.add_image(f"target/image", grid, trainer.global_step + 1)

                try:

                    trainer.logger.experiment.add_histogram(
                        f"target/dist", pl_module.target_image, trainer.global_step + 1
                    )
                except ValueError as e:
                    logging.info(f"LogImageCallback (target/dist): {e}")


class LogAttentionWeightCallback(pl.callbacks.Callback):
    def __init__(self, log_save_interval=None, **kwargs):
        super(LogAttentionWeightCallback, self).__init__(**kwargs)
        self.log_save_interval = log_save_interval

    @rank_zero_only
    def on_batch_end(self, trainer, pl_module):
        if trainer.logger is None:
            return

        if self.log_save_interval is None:
            log_interval = trainer.log_save_interval
        else:
            log_interval = self.log_save_interval

        if (trainer.global_step + 1) % log_interval == 0:

            if hasattr(pl_module, "generator"):
                for x in pl_module.generator.modules():
                    if isinstance(x, SelfAttention2d):
                        trainer.logger.experiment.add_scalar(
                            f"generator/attention_gamma", x.gamma, trainer.global_step + 1
                        )

            if hasattr(pl_module, "discriminator"):
                for x in pl_module.discriminator.modules():
                    if isinstance(x, SelfAttention2d):
                        trainer.logger.experiment.add_scalar(
                            f"discriminator/attention_gamma", x.gamma, trainer.global_step + 1
                        )

