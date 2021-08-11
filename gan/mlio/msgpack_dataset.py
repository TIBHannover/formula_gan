import os
import sys
import re
import argparse

import time
import imageio

import torch
import time
import msgpack
import random


class MsgPackIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, random=True, transformation=None):
        super(MsgPackIterableDataset, self).__init__()
        self.path = path
        self.length = 0

        shards_re = r"shard_(\d+).msg"
        self.shards_index = [
            int(re.match(shards_re, x).group(1)) for x in os.listdir(self.path) if re.match(shards_re, x)
        ]

        self.length = len(self.shards_index) * 1024

        self.transformation = transformation
        self.random = random

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        shards_index = self.shards_index
        if self.random:
            random.shuffle(shards_index)

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [alist[i * length // splits : (i + 1) * length // splits] for i in range(splits)]

            keys = split_list(self.shards_index, worker_info.num_workers)[worker_info.id]

        else:
            keys = self.shards_index
        cache = []
        for key in keys:
            with open(os.path.join(self.path, f"shard_{key}.msg"), "rb") as f:
                unpacker = msgpack.Unpacker(f, max_buffer_size=1024 * 1024 * 1024)

                for x in unpacker:
                    if x is None:
                        continue
                    if len(cache) < 8000:
                        cache.append(x)
                        continue
                    if self.random:
                        random.shuffle(cache)

                    while cache:
                        y = cache.pop()
                        if y is None:
                            continue

                        if self.transformation is not None:
                            y = self.transformation(y)
                        yield y

        while cache:
            y = cache.pop()
            if y is None:
                continue

            if self.transformation is not None:
                y = self.transformation(y)
            yield y

    def __len__(self):
        return self.length


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-p", "--path", help="verbose output")
    parser.add_argument("-o", "--output", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dataset = MsgPackIterableDataset(path=args.path)

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=8)

    paths = []
    start = time.time()
    for i, x in enumerate(dataloader):
        #        print(x.keys())
        path = x["path"][0]
        output_path = os.path.join(args.output, path)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        imageio.imsave(output_path, x["image"][0, :, :, :].numpy())
        if i % 1000 == 0:
            end = time.time()
            print(f"{1000/(end - start):.2f} image/s")
            start = end

    # start = time.time()
    # for i, x in enumerate(loader):
    #     path = np.asarray(x['path'])
    #     image = np.asarray(x['image'])
    #     # print(y)
    #
    #     if i % 1000 == 0:
    #         end = time.time()
    #         print(f'{1000/(end - start):.2f} image/s')
    #         start = end
    #     # print(i)
    #
    # # print(next(loader))

    return 0


if __name__ == "__main__":
    sys.exit(main())
