import os
import sys
import re
import argparse
import json
import tensorflow as tf
import numpy as np
import csv

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn.metrics

import torch

import logging

from mlio.pipeline import Pipeline, Dataset

from pytorch_lightning.callbacks import Callback

import torchvision


def get_model_args(sample, model_args):

    if callable(model_args):
        model_input = model_args(sample)
    if isinstance(model_args, str):
        model_input = sample[model_args]
    if isinstance(model_args, (list, set)):
        model_input = [get_model_args(sample, x) for x in model_args]
    return model_input


class FilterClusterManager(Callback):
    def __init__(
        self,
        path,
        cache_writer,
        filter_dataloader,
        min_elements=5,
        silhouette_score=0.1,
        element_cache=200,
        min_k=2,
        max_k=20,
        version=0,
        device="cpu",
        index="id",
        step=None,
        epoch=None,
        first_epoch=True,
        model_input="image",
        model_output="feature",
        concept_list="concept_ids",
        resnet_imagenet=True,
    ):
        self.feature_cache = {}

        if not os.path.exists(path):
            os.makedirs(path)

        self.path = os.path.join(path, f"{version}.json")
        self.meta_path = os.path.join(path, f"{version}_meta.json")

        self.element_cache = element_cache
        self.min_elements = min_elements
        self.silhouette_score = silhouette_score
        self.filter_dataloader = filter_dataloader
        self.device = device
        self.index = index
        self.concept_list = concept_list
        self.cache_writer = cache_writer
        self._iter = 0

        self.min_k = min_k
        self.max_k = max_k

        self.step = step
        self.epoch = epoch
        self.first_epoch = first_epoch

        self.model_input = model_input
        self.model_output = model_output

        self.resnet_imagenet = resnet_imagenet
        if resnet_imagenet:
            logging.info('Filter use only a pretrained model')
            resnet_model = torchvision.models.resnet.resnet50(pretrained=True)
            self.imagenet_features = torch.nn.Sequential(*list(resnet_model.children())[:-1]).to(torch.device("cuda:0"))

        self._epoch_counter = 0 if first_epoch else epoch  # We will start before the first sample arrives
        self._step_counter = step  # We will start before the first sample arrives

        if step is None and epoch is None:
            self.epoch = 1

        if step is not None and epoch is not None:
            print("I should learn assert")
            exit()

    def on_epoch_start(self, trainer, pl_module):

        if self.first_epoch and self._epoch_counter == 0:
            self._epoch_counter -= 1
            return self.clustering(trainer, pl_module)

        if self.epoch is None:
            return

        self._epoch_counter -= 1

        if self._epoch_counter > 0:
            return

        self._epoch_counter = self.epoch
        result = self.clustering(trainer, pl_module)

        return result

    def on_batch_start(self, trainer, pl_module):
        if self.step is None:
            return

        self._step_counter -= 1

        if self._step_counter > 0:
            return

        self._step_counter = self.step
        result = self.clustering(trainer, pl_module)

        return result

    def clustering(self, trainer, pl_module):
        rank = 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

        if rank == 0:
            logging.info(f"Cluster: start filtering")
            self._iter += 1
            result = {
                "filter/reject_count": 0,
                "filter/accept_count": 0,
                "filter/iter": self._iter,
                "filter/k_hist": np.zeros(shape=self.max_k),
                "filter/k_mean": 0,
                "filter/count": 0,
            }

            with self.cache_writer() as f:
                self.feature_cache = {}
                for batch_id, sample in enumerate(self.filter_dataloader):
                    if self.resnet_imagenet:
                        image = get_model_args(sample, self.model_input).to(torch.device("cuda:0"))
                        feature = self.imagenet_features(image)

                        feature = torch.flatten(feature, 1)
                    else:
                        prediction = pl_module(get_model_args(sample, self.model_input))

                        feature = get_model_args(prediction, self.model_output)

                    paths = sample[self.index]

                    # print(prediction)
                    concept_ids = sample[self.concept_list].numpy().tolist()
                    features = feature.detach().cpu().numpy()
                    self.add_batch(paths, concept_ids, features)
                    clustering_result = self.update_filter(f)
                    result["filter/reject_count"] += clustering_result["reject_count"]
                    result["filter/accept_count"] += clustering_result["accept_count"]
                    result["filter/k_hist"] += clustering_result["k_hist"]
                    result["filter/k_mean"] += clustering_result["k_mean"]
                    result["filter/count"] += clustering_result["count"]

            if result["filter/count"] > 0:
                result["filter/k_hist"] /= result["filter/count"]
                result["filter/k_mean"] /= result["filter/count"]

            result.update(
                {
                    "filter/d": result["filter/accept_count"]
                    / (result["filter/accept_count"] + result["filter/reject_count"])
                }
            )

            with open(self.meta_path, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "reject_count": result["filter/reject_count"],
                            "accept_count": result["filter/accept_count"],
                            "d": result["filter/d"],
                            "k_hist": result["filter/k_hist"].tolist(),
                        }
                    )
                )
            if trainer.logger:
                trainer.logger.log_metrics(
                    {
                        "filter/k_mean": result["filter/k_mean"],
                        "filter/count": result["filter/count"],
                        "filter/reject_count": result["filter/reject_count"],
                        "filter/accept_count": result["filter/accept_count"],
                        "filter/d": result["filter/d"],
                    },
                    step=trainer.global_step,
                )
        else:
            result = None

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return result

    def add_batch(self, paths, concept_ids, features):
        batch_size = len(paths)
        # print(paths)
        # print(concept_ids)
        # print(features)
        # print('###################################')
        # print('add_batch')
        # print(paths)

        # print(features.shape)
        for i in range(batch_size):
            for concept_id in concept_ids[i]:
                if concept_id != -1:
                    if concept_id not in self.feature_cache:

                        self.feature_cache[concept_id] = []
                    self.feature_cache[concept_id].append({self.index: paths[i], "feature": features[i]})
        # exit()

    def update_filter(self, cache_writer=None):
        result = {
            "reject_count": 0,
            "accept_count": 0,
            "k_mean": 0,
            "k_hist": np.zeros(shape=self.max_k),
            "count": 0,
        }
        # print(self.feature_cache)
        for concept, features in self.feature_cache.items():
            # print(f'{concept}: {len(features)}')
            if len(features) > self.element_cache:
                cluster_metrics = []
                logging.info(f"Cluster: compute new kmeans")
                path_list = np.asarray([x[self.index] for x in features[: self.element_cache]])
                feature_list = np.asarray([x["feature"] for x in features[: self.element_cache]])
                for cluster_size in range(self.min_k, self.max_k):
                    # print(f"cluster size {cluster_size}")
                    clusterer = KMeans(n_clusters=cluster_size, random_state=10,)
                    cluster_labels = clusterer.fit_predict(feature_list)

                    silhouette_avg = silhouette_score(feature_list, cluster_labels)
                    cluster_metrics.append(silhouette_avg)

                optimal_cluster_size = np.argmax(cluster_metrics) + self.min_k

                result["k_mean"] += optimal_cluster_size
                result["count"] += 1
                result["k_hist"][optimal_cluster_size] += 1
                clusterer = KMeans(n_clusters=optimal_cluster_size, random_state=10,)
                cluster_labels = clusterer.fit_predict(feature_list)

                sample_silhouette_values = silhouette_samples(feature_list, cluster_labels)
                # print("##############")
                for cluster_id in range(optimal_cluster_size):
                    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == cluster_id]
                    # print(
                    #     f"{cluster_id} {ith_cluster_silhouette_values.shape[0]} {np.mean(ith_cluster_silhouette_values)}"
                    # )
                    if ith_cluster_silhouette_values.shape[0] < self.min_elements:

                        for path in path_list[cluster_labels == cluster_id]:
                            cache_writer.write(path.encode("utf-8"), ["concept"], [-1])
                            result["reject_count"] += 1
                        continue

                    if np.mean(ith_cluster_silhouette_values) < self.silhouette_score:
                        for path in path_list[cluster_labels == cluster_id]:
                            cache_writer.write(path.encode("utf-8"), ["concept"], [-1])
                            result["reject_count"] += 1
                        continue

                    for path in path_list[cluster_labels == cluster_id]:
                        # print([path.encode('utf-8'), concept])
                        cache_writer.write(path.encode("utf-8"), ["concept"], [concept])
                        result["accept_count"] += 1

                del self.feature_cache[concept][: self.element_cache]
        return result

    def filter_file(self):
        patter_re = re.compile("(\d+)\.json")
        filter_files = [
            os.path.join(os.path.dirname(self.path), x)
            for x in os.listdir(os.path.dirname(self.path))
            if re.match(patter_re, x)
        ]
        patter_re = re.compile(".*?(\d+)\.json")
        filter_files = sorted(filter_files, key=lambda x: int(re.match(patter_re, x).group(1)))
        return filter_files[-1]


class FilterClusterDataset(Dataset):
    def __init__(
        self,
        dataset,
        cache_reader,
        filtered_delete=False,
        random_deletion_start=None,
        index="id",
        concept_list="concept_ids",
    ):
        self.dataset = dataset
        self.cache_reader = cache_reader
        self.filtered_delete = filtered_delete
        self.random_deletion_start = random_deletion_start
        self.index = index
        self.concept_list = concept_list

    def handle_sample(self, reader, sample):
        if "loss_weight" in sample:
            loss_weight = sample["loss_weight"]
        else:
            loss_weight = torch.tensor(1, dtype=torch.float32)

        if self.random_deletion_start is not None:
            loss_weight = (torch.rand(size=[]) > self.random_deletion_start).float()

        if sample[self.index] not in reader:
            return {**sample, "loss_weight": loss_weight}
        entry = reader.read(sample[self.index])
        if entry is None:
            return {**sample, "loss_weight": loss_weight}

        concept = entry["concept"]

        if concept == -1 or concept not in sample[self.concept_list].tolist():
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


class FilterClusterPipeline(Pipeline):
    def __init__(
        self,
        cache_reader,
        filtered_delete=False,
        random_deletion_start=None,
        index="id",
        concept_list="concept_ids",
        **kwargs,
    ):
        super(FilterClusterPipeline, self).__init__(**kwargs)
        self.cache_reader = cache_reader
        self.filtered_delete = filtered_delete
        self.random_deletion_start = random_deletion_start
        self.index = index
        self.concept_list = concept_list

    def call(self, datasets=None, **kwargs):
        return FilterClusterDataset(
            datasets,
            self.cache_reader,
            self.filtered_delete,
            self.random_deletion_start,
            self.index,
            self.concept_list,
        )

