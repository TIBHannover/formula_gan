import os
import sys
import re
import argparse
import logging
import random
import typing


import numpy as np
import torch
import torchvision

import multiprocessing as mp

import cv2
import imageio
import json
import time
import msgpack


from PIL import Image

Image.warnings.simplefilter("error", Image.DecompressionBombError)  # turn off Decompression bomb error
Image.warnings.simplefilter("error", Image.DecompressionBombWarning)  # turn off Decompression bomb warning
Image.MAX_IMAGE_PIXELS = 1000000000  # set max pixel up high


def list_files(path, patter=r".*?\.rec"):
    patter_re = re.compile(patter)
    return [os.path.join(path, x) for x in os.listdir(path) if re.match(patter_re, x)]


class Dataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        pass


class Pipeline:
    def __call__(self, datasets=None, **kwargs):
        datasets = self.call(datasets, **kwargs)
        return datasets


class RangePipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super(RangePipeline, self).__init__(**kwargs)
        self.args = args

    def call(self, datasets=None, **kwargs):
        args = self.args

        class Range(Dataset):
            def __iter__(self):
                for x in range(*args):
                    yield x

        return Range()


class ImageDataset(Dataset):
    def __init__(self, path, shuffle=True, sample_per_shard=1024, image_re=r".*?\.(jpg|jpeg|png|bmp)", return_rgb=True):
        super(ImageDataset, self).__init__()
        self.path = path
        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

        self.new_paths = []
        for path in self.path:
            if os.path.isdir(path):

                self.new_paths.extend(list_files(path, image_re))

            if os.path.isfile(path):
                self.new_paths.append(path)
        self.path = self.new_paths

        self.shuffle = shuffle
        self.return_rgb = return_rgb

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [alist[i * length // splits : (i + 1) * length // splits] for i in range(splits)]

            keys = split_list(self.path, worker_info.num_workers)[worker_info.id]

        else:
            keys = self.path

        if self.shuffle:
            random.shuffle(keys)

        cache = []
        for key in keys:
            with open(key, "rb") as f:
                image = imageio.imread(key)

                # TODO add mode option
                if self.return_rgb:
                    if len(image.shape) == 2:
                        image = np.stack([image] * 3, axis=-1)

                    if image.shape[-1] == 4:
                        image = image[:, :, 0:3]

                yield {"image": image, "path": key}

    def __len__(self):
        return self.length


class ImagePipeline(Pipeline):
    def __init__(self, path, shuffle=True, image_re=r".*?\.(jpg|jpeg|png|bmp)", return_rgb=True):
        super(ImagePipeline, self).__init__()
        self.path = path
        self.shuffle = shuffle
        self.image_re = image_re
        self.return_rgb = return_rgb

    def call(self, datasets=None, **kwargs):
        return ImageDataset(self.path, self.shuffle, image_re=self.image_re, return_rgb=self.return_rgb)


class MsgPackDataset(Dataset):
    def __init__(self, path, shuffle=True, sample_per_shard=1024, shard_re=r"shard_(\d+).msg"):
        super(MsgPackDataset, self).__init__()
        self.path = path

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

        self.length = 0

        self.shards_paths = []
        for p in self.path:
            self.shards_paths.extend([os.path.join(p, x) for x in os.listdir(p) if re.match(shard_re, x)])

        self.length = len(self.shards_paths) * sample_per_shard

        self.shuffle = shuffle

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [alist[i * length // splits : (i + 1) * length // splits] for i in range(splits)]

            keys = split_list(self.shards_paths, worker_info.num_workers)[worker_info.id]

        else:
            keys = self.shards_paths

        if self.shuffle:
            random.shuffle(keys)

        cache = []
        for key in keys:
            with open(key, "rb") as f:
                unpacker = msgpack.Unpacker(f, max_buffer_size=1024 * 1024 * 1024, raw=True)

                for x in unpacker:
                    yield x

    def __len__(self):
        return self.length


class MsgPackPipeline(Pipeline):
    def __init__(self, path, shuffle=True, sample_per_shard=1024, shard_re=r"shard_(\d+).msg"):
        super(MsgPackPipeline, self).__init__()
        self.path = path
        self.shuffle = shuffle
        self.sample_per_shard = sample_per_shard
        self.shard_re = shard_re

    def call(self, datasets=None, **kwargs):
        return MsgPackDataset(self.path, self.shuffle, self.sample_per_shard, shard_re=self.shard_re)


class CacheDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, cache_size=1024, shuffle=True):
        super(CacheDataset, self).__init__()
        self.cache_size = cache_size
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        cache = []
        for sample in self.dataset:
            if sample is None:
                continue

            cache.append(sample)
            if len(cache) < self.cache_size:
                continue

            if self.shuffle:
                random.shuffle(cache)

            while cache:
                y = cache.pop(0)
                if y is None:
                    continue
                yield y

        if self.shuffle:
            random.shuffle(cache)
        while cache:
            y = cache.pop(0)
            if y is None:
                continue
            yield y

    def __len__(self):
        return len(self.dataset)


class CachePipeline(Pipeline):
    def __init__(self, cache_size=1024, shuffle=True):
        super(CachePipeline, self).__init__()
        self.cache_size = cache_size
        self.shuffle = shuffle

    def call(self, datasets=None, **kwargs):
        return CacheDataset(datasets, cache_size=self.cache_size, shuffle=self.shuffle)


class MapDataset(Dataset):
    def __init__(self, dataset, map_fn):
        super(MapDataset, self).__init__()
        self.dataset = dataset
        self.map_fn = map_fn

    def __iter__(self):
        for sample in self.dataset:
            if sample is None:
                continue

            sample = self.map_fn(sample)

            if sample is None:
                continue
            yield sample

    def __len__(self):
        return len(self.dataset)


class MapPipeline(Pipeline):
    def __init__(self, map_fn):
        super(MapPipeline, self).__init__()
        self.map_fn = map_fn

    def call(self, datasets=None, **kwargs):
        return MapDataset(datasets, self.map_fn)


class SamplerDataset(Dataset):
    def __init__(
        self, dataset, length, target_dist=None, alpha=0.1, max_discard=0.9, update_dist_after=512, cache_size=128
    ):
        self.length = length
        self.sample_dist = mp.Array("f", [0.0] * length)
        self.lock = mp.Lock()
        self.dataset = dataset
        if target_dist:
            self.target_dist = target_dist
        else:
            self.target_dist = np.ones(shape=self.length) / self.length
        self.cache_size = cache_size
        self.alpha = alpha
        self.max_discard = max_discard

    def build_discard_prob(self):
        local_sample_dist = [self.sample_dist[i] for i in range(self.length)]
        local_sample_dist = np.asarray(local_sample_dist)
        local_sample_dist = self.target_dist / (local_sample_dist + 1e-5)
        return 1 - local_sample_dist / np.amax(local_sample_dist)

    def __iter__(self):
        dist = [0] * self.length
        count = 0

        discard_prob = self.build_discard_prob()

        for x in self.dataset:
            subgraph = x["subgraph"]["subgraph_label_list"][0]

            dist[subgraph] += 1
            count += 1
            if count % 128 == 0:
                self.update_dist(dist, count)

                discard_prob = self.build_discard_prob()

                # print(local_sample_dist[143])

            if random.random() < min(discard_prob[subgraph], self.max_discard):

                continue

            yield x

    def update_dist(self, dist, count):
        # logging.info('Update dist')
        # new_dist = [0] * self.length

        self.lock.acquire()

        try:
            for i in range(self.length):
                self.sample_dist[i] = (1 - self.alpha) * self.sample_dist[i] + self.alpha * float(dist[i]) / count
                # new_dist[i] = self.sample_dist[i]
        finally:
            self.lock.release()


class SamplerPipeline(Pipeline):
    def __init__(self, length, target_dist=None, alpha=0.1, max_discard=0.9, update_dist_after=512, cache_size=128):
        super(SamplerPipeline, self).__init__()
        self.length = length
        self.target_dist = target_dist
        self.alpha = alpha
        self.max_discard = max_discard
        self.update_dist_after = update_dist_after
        self.cache_size = cache_size

    def call(self, datasets=None, **kwargs):
        return SamplerDataset(
            datasets,
            length=self.length,
            target_dist=self.target_dist,
            alpha=self.alpha,
            max_discard=self.max_discard,
            update_dist_after=self.update_dist_after,
            cache_size=self.cache_size,
        )


class FilterPipeline(Pipeline):
    def __init__(self, filter_fn, **kwargs):
        super(FilterPipeline, self).__init__(**kwargs)
        self.filter_fn = filter_fn

    def call(self, datasets=None, **kwargs):
        filter_fn = self.filter_fn

        class Filter(Dataset):
            def __iter__(self):
                for x in datasets:
                    if filter_fn(x):
                        yield x

        return Filter()


class SequencePipeline(Pipeline):
    def __init__(self, pipelines, **kwargs):
        super(SequencePipeline, self).__init__(**kwargs)
        self.pipelines = pipelines

    def call(self, datasets=None, **kwargs):
        for i, pl in enumerate(self.pipelines):
            datasets = pl(datasets=datasets, **kwargs)

        return datasets


class MergePipeline(Pipeline):
    def __init__(self, pipelines, merge_fn=None, **kwargs):
        super(MergePipeline, self).__init__(**kwargs)
        self.merge_fn = merge_fn
        self.pipelines = pipelines

    def call(self, datasets=None, **kwargs):
        merge_fn = self.merge_fn

        pls = [pl(datasets) for pl in self.pipelines]

        class Merger(Dataset):
            def __iter__(self):
                for x in zip(*pls):
                    if merge_fn is None:
                        yield x
                    else:
                        x = merge_fn(x)

                        if x is None:
                            continue
                        yield x

        return Merger()


class ConcatShufflePipeline(Pipeline):
    def __init__(self, pipelines, target_dist=None):
        super(ConcatShufflePipeline, self).__init__()
        self.pipelines = pipelines
        self.target_dist = target_dist

    def call(self, datasets=None, **kwargs):

        pls = [pl(datasets) for pl in self.pipelines]

        class Concat(Dataset):
            def __iter__(self):
                datasets_iter = []

                for pl in pls:
                    datasets_iter.append(iter(pl))
                while True:
                    dataset_iter = random.choice(datasets_iter)
                    try:
                        s = next(dataset_iter)
                        yield s
                    except StopIteration as e:
                        return

            def __len__(self):
                return sum([len(d) for d in pls])

        return Concat()


class RepeatPipeline(Pipeline):
    def __init__(self, times=-1):
        super(RepeatPipeline, self).__init__()

    def call(self, datasets=None, **kwargs):
        class Repeat(Dataset):
            def __iter__(self):
                dataset_iter = iter(datasets)
                while True:
                    try:
                        s = next(dataset_iter)
                        yield s
                    except StopIteration as e:
                        dataset_iter = iter(datasets)

        return Repeat()


if __name__ == "__main__":
    a = SequencePipeline(
        [
            RangePipeline(100),
            FilterPipeline(lambda x: x % 2 == 0),
            # MapPipeline(lambda x: x ** 2),
            # CachePipeline(cache_size=64, shuffle=True),
        ]
    )

    b = SequencePipeline(
        [
            RangePipeline(100),
            FilterPipeline(lambda x: x % 2 == 1),
            # MapPipeline(lambda x: x ** 2),
            # CachePipeline(cache_size=3, shuffle=False),
        ]
    )

    c = ConcatShufflePipeline([a, b])
    for x in c():
        print(x)
# for x in a():
#     print(x)
