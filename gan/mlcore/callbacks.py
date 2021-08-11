import torch
import logging

import torch
import copy
import pytorch_lightning as pl

from pytorch_lightning.utilities.distributed import rank_zero_only


class ProgressPrinter(pl.callbacks.ProgressBarBase):
    def __init__(self, refresh_rate: int = 100):
        super().__init__()
        self.refresh_rate = refresh_rate
        self.enabled = True

    @property
    def is_enabled(self):
        return self.enabled and self.refresh_rate > 0

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def on_batch_end(self, trainer, pl_module):
        super().on_batch_end(trainer, pl_module)
        if self.is_enabled and self.trainer.global_step % self.refresh_rate == 0:
            progress_bar_dict = copy.deepcopy(trainer.progress_bar_dict)

            progress_bar_dict.pop("v_num", None)
            logging.info(
                f"Train {self.trainer.global_step} " + " ".join([f"{k}:{v}" for k, v in progress_bar_dict.items()])
            )

    def on_validation_end(self, trainer, pl_module):
        super().on_batch_end(trainer, pl_module)
        progress_bar_dict = copy.deepcopy(trainer.progress_bar_dict)

        progress_bar_dict.pop("v_num", None)
        logging.info(
            f"Val {self.trainer.global_step+1} " + " ".join([f"{k}:{v}" for k, v in progress_bar_dict.items()])
        )


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, checkpoint_save_interval=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_save_interval = checkpoint_save_interval
        # self.filename = "checkpoint_{global_step}"

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):

        trainer.callback_metrics.update({"global_step": trainer.global_step + 1})
        super().on_validation_end(trainer, pl_module)

    def format_checkpoint_name(self, *args, **kwargs):
        return super().format_checkpoint_name(*args, **kwargs).replace("global_step=", "")

    @rank_zero_only
    def on_batch_end(self, trainer, pl_module):

        if self.checkpoint_save_interval is not None:

            if (trainer.global_step + 1) % self.checkpoint_save_interval == 0:
                self.on_validation_end(trainer, pl_module)


class LogModelWeightCallback(pl.callbacks.Callback):
    def __init__(self, log_save_interval=None, nrow=2, **kwargs):
        super(LogModelWeightCallback, self).__init__(**kwargs)
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
            for k, v in pl_module.state_dict().items():
                try:
                    trainer.logger.experiment.add_histogram(f"weights/{k}", v, trainer.global_step + 1)
                except ValueError as e:
                    logging.info(f"LogModelWeightCallback: {e}")
