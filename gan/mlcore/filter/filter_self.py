import os
import sys
import re
import argparse
import json
import tensorflow as tf
import numpy as np
import csv

import torch

import logging

from mlcore.callback import Callback


class FilterSELFManager(Callback):
    def __init__(
        self,
        path,
        cache_reader,
        cache_writer,
        filter_dataloader,
        alpha=0.5,
        patience=10,
        device="cpu",
        metric="loss",
        target="minimizing",
        index="id",
    ):

        self._alpha_filter = alpha
        self._iter = 0

        self.path = path
        self.cache_reader = cache_reader
        self.cache_writer = cache_writer
        self.filter_dataloader = filter_dataloader
        self.device = device
        self.index = index
        self.patience = patience
        self._patience_step = 0

        self._metric = metric
        self._target = target
        self._best_target = np.finfo(np.float32).max if target == "minimizing" else np.finfo(np.float32).min

    def on_test_end(self, **kwargs):
        if self._metric not in kwargs:
            print(kwargs)
            raise ValueError()

        alpha = min(1 - 1 / (self._iter + 1), self._alpha_filter)

        self._patience_step -= 1

        target = kwargs[self._metric]
        if self._target == "minimizing":
            if target < self._best_target:
                logging.info(
                    f"FilterSELFManager: skip filtering target:{target} < self._best_target:{self._best_target}"
                )
                self._best_target = target
                return {
                    "filter/iter": self._iter,
                    "filter/alpha": alpha,
                    "filter/best_target": self._best_target,
                    "filter/patience": self._patience_step,
                }
        else:
            if target > self._best_target:
                logging.info(
                    f"FilterSELFManager: skip filtering target:{target} > self._best_target:{self._best_target}"
                )
                self._best_target = target
                return {
                    "filter/iter": self._iter,
                    "filter/alpha": alpha,
                    "filter/best_target": self._best_target,
                    "filter/patience": self._patience_step,
                }

        if self._patience_step > 0:
            logging.info(f"FilterSELFManager: skip filtering patience:{self._patience_step}")
            return {
                "filter/iter": self._iter,
                "filter/alpha": alpha,
                "filter/best_target": self._best_target,
                "filter/patience": self._patience_step,
            }

        # reset alpha for mean trainer
        if hasattr(self.trainer, "_mean_teacher_step"):
            self.trainer._mean_teacher_step = 0

        logging.info(f"FilterSELFManager: start filtering with alpha:{alpha} for iter:{self._iter}")
        with self.cache_reader() as reader:
            with self.cache_writer() as writer:
                for batch_id, sample in enumerate(self.filter_dataloader):
                    result = self.trainer.val_step(sample, device=self.device)
                    prediction = result["model_output"]["prediction"].cpu().numpy()
                    for batch_element in range(prediction.shape[0]):
                        cache_id = sample[self.index][batch_element]
                        # Compute moving average of all predictions
                        old_prediction = reader.read(cache_id)

                        if old_prediction is None:
                            old_prediction = np.zeros_like(prediction[batch_element])
                        else:
                            old_prediction = old_prediction["prediction"]

                        average_prediction = old_prediction * alpha + (1 - alpha) * prediction[batch_element]

                        # print(average_prediction)
                        writer.write(cache_id, ["prediction"], [average_prediction])

        self._patience_step = self.patience
        self._iter += 1
        return {
            "filter/iter": self._iter,
            "filter/alpha": alpha,
            "filter/best_target": self._best_target,
            "filter/patience": self._patience_step,
        }

    def state_dict(self, **kwargs):
        return {
            "alpha_filter": self._alpha_filter,
            "iter": self._iter,
            "lowest_loss": self._best_target,
            "patience_step": self._patience_step,
        }

    def load_state_dict(self, data: dict):
        self._alpha_filter = data["alpha_filter"]
        self._iter = data["iter"]
        self._best_target = data["lowest_loss"]
        self._patience_step = data["patience_step"]


class FilterSELFIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        cache_reader,
        filter_method,
        filtered_delete=False,
        random_deletion=None,
        random_deletion_start=False,
        index="id",
    ):
        self.dataset = dataset
        self.cache_reader = cache_reader
        self.filter_method = filter_method
        self.filtered_delete = filtered_delete
        self.random_deletion = random_deletion
        self.random_deletion_start = random_deletion_start
        self.index = index

    def handle_sample(self, reader, sample):
        if "loss_weight" in sample:
            loss_weight = sample["loss_weight"]
        else:
            loss_weight = torch.tensor(1, dtype=torch.float32)

        if self.random_deletion is not None:
            loss_weight = (torch.rand(size=[]) > self.random_deletion).float()

        if sample[self.index] not in reader:
            if self.random_deletion_start:
                loss_weight = (torch.rand(size=[]) > self.random_deletion_start).float()
            return {**sample, "loss_weight": loss_weight}
        entry = reader.read(sample[self.index])
        if entry is None:
            return {**sample, "loss_weight": loss_weight}

        prediction = entry["prediction"]
        # TODO top_k or equal argmax
        # if prediction not in x['concept_ids'].tolist():
        #     continue
        decision = self.filter_method(sample, prediction)
        if decision:
            # print(f"{decision} {sample['id']} {sample['label']} {sample['concept_ids']} {prediction}")
            if self.filtered_delete:
                return None
            else:
                return {**sample, "loss_weight": torch.tensor(0, dtype=torch.float32)}

        return {**sample, "loss_weight": loss_weight}

    def __iter__(self):
        with self.cache_reader() as reader:
            for x in self.dataset:
                result = self.handle_sample(reader, x)
                if result is None:
                    continue
                yield result

    def __len__(self):
        return len(self.dataset)


class SplitIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, return_unlabeled=False):
        self.dataset = dataset
        self.return_unlabeled = return_unlabeled

    def __iter__(self):
        for x in self.dataset:

            if self.return_unlabeled:
                if x["loss_weight"] == 0:
                    yield x
                else:
                    continue
            else:
                if x["loss_weight"] == 1:
                    yield x
                else:
                    continue

    def __len__(self):
        return len(self.dataset)
