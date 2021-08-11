import os
import re

import torch
import time
import msgpack
import random


class EqualShardSampler:
    def __init__(self, target_dist=None):
        self.target_dist = target_dist

    def __call__(self, shards):
        counts = {}
        for i, x in enumerate(shards):
            if x["path_index"] not in counts:
                counts[x["path_index"]] = []
            counts[x["path_index"]].append(i)

        if self.target_dist is None:
            target_dist = [1] * len(counts.keys())
        else:
            target_dist = self.target_dist
        subset_shards = random.choices(range(len(counts.keys())), k=len(shards))
        sample = [random.sample(counts[x], k=1)[0] for x in subset_shards]
        return sample


class MsgPackIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, transformation=None, shuffle=True, element_sampler=None, shards_sampler=None):
        super(MsgPackIterableDataset, self).__init__()
        self.path = path
        self.length = 0

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

        self.shards = []
        for i, p in enumerate(self.path):
            shards_re = r"shard_(\d+).msg"
            shards_index = [int(re.match(shards_re, x).group(1)) for x in os.listdir(p) if re.match(shards_re, x)]
            self.shards.extend(
                [
                    {"path_index": i, "path": p, "shard_index": s, "shard_path": os.path.join(p, f"shard_{s}.msg")}
                    for s in shards_index
                ]
            )

        self.length = len(self.shards) * 1024

        self.transformation = transformation
        self.shuffle = shuffle
        self.element_sampler = element_sampler
        self.shards_sampler = shards_sampler

        # print(f'Dataset: {self.length} {self.path}')

    def __iter__(self):
        if self.shards_sampler is not None:
            shard_indices = self.shards_sampler(self.shards)
        else:
            shard_indices = list(range(len(self.shards)))

        if self.shuffle:
            random.shuffle(shard_indices)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [alist[i * length // splits : (i + 1) * length // splits] for i in range(splits)]

            shard_indices_split = split_list(shard_indices, worker_info.num_workers)[worker_info.id]

        else:
            shard_indices_split = shard_indices

        cache = []
        for shard_index in shard_indices_split:
            shard = self.shards[shard_index]
            # print(shard)
            with open(os.path.join(shard["path"], f"shard_{shard['shard_index']}.msg"), "rb") as f:
                unpacker = msgpack.Unpacker(f, max_buffer_size=1024 * 1024 * 1024, raw=True)
                # for x in unpacker:
                #     if self.transformation is not None:
                #         x = self.transformation(x)
                #     if x is None:
                #         continue
                #     yield x

                for x in unpacker:
                    if x is None:
                        continue
                    if len(cache) < 8000:
                        cache.append(x)
                        continue
                    if self.shuffle:
                        random.shuffle(cache)

                    while cache:
                        y = cache.pop()
                        if y is None:
                            continue

                        if self.transformation is not None:
                            y = self.transformation(y)
                        if y is None:
                            continue
                        yield y

        while cache:
            y = cache.pop()
            if y is None:
                continue

            if self.transformation is not None:
                y = self.transformation(y)

            if y is None:
                continue
            yield y

    def __len__(self):
        return self.length
