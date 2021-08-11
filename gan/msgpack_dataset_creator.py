import msgpack

import imageio

import os
import sys
import re
import argparse

import numpy as np
import cv2
import time

import json

import multiprocessing as mp
import random

import logging


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--image_path", help="verbose output")
    parser.add_argument("-a", "--annotation", help="verbose output")
    parser.add_argument("-m", "--reject_min_dim", type=int, help="path to dir of images")
    parser.add_argument("-o", "--output", help="path to dir of images")
    parser.add_argument("-d", "--max_dim", type=int, default=512, help="path to dir of images")
    parser.add_argument("-p", "--process", type=int, default=1, help="path to dir of images")
    parser.add_argument("-c", "--chunck", type=int, default=1024, help="Images per file")

    parser.add_argument("-s", "--shuffle", action="store_true", default=True, help="verbose output")

    parser.add_argument("--skip_scan", action="store_true", help="verbose output")

    args = parser.parse_args()
    return args


class MsgPackWriter:
    # TODO multigpu
    def __init__(self, path, chunck_size=1024):
        self.path = os.path.abspath(path)
        self.chunck_size = chunck_size

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        shards_re = r"shard_(\d+).msg"

        self.shards_index = [
            int(re.match(shards_re, x).group(1)) for x in os.listdir(self.path) if re.match(shards_re, x)
        ]
        self.shard_open = None

    def open_next(self):

        if len(self.shards_index) == 0:
            next_index = 0
        else:
            next_index = sorted(self.shards_index)[-1] + 1
        self.shards_index.append(next_index)

        if self.shard_open is not None and not self.shard_open.closed:
            self.shard_open.close()
        self.count = 0
        self.shard_open = open(os.path.join(self.path, f"shard_{next_index}.msg"), "wb")

    def __enter__(self):
        self.open_next()
        return self

    def __exit__(self, type, value, tb):
        self.shard_open.close()

    def write(self, data):
        if self.count >= self.chunck_size:
            self.open_next()

        self.shard_open.write(msgpack.packb(data))
        self.count += 1
        # print(self.count)


class ImageDataloader:
    def __init__(self, path, filter=None, shuffle=True):
        image_re = re.compile(r".*?\.(png|tif|tiff|jpg|jpeg|gif)", re.IGNORECASE)
        self.path = path
        if os.path.isdir(path):
            self.paths = [os.path.join(p, x) for p, _, f in os.walk(path) for x in f if image_re.match(x)]
            if filter is not None:
                filter_re = re.compile(filter)
                self.paths = [x for x in self.paths if filter_re.match(x)]
        else:
            self.paths = [path]
        if shuffle:
            print("Shuffle paths")
            random.shuffle(self.paths)
            print(self.paths[:10])

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        for path in self.paths:
            yield {"path": path, "rel_path": os.path.relpath(path, os.path.commonpath([self.path, path]))}


class FakeDataloader:
    def __init__(self, path, entries, splits=r"^(\w{3})(\w{3})\w*$", shuffle=True):
        self.splits_re = re.compile(splits, re.IGNORECASE)
        self.path = path
        self.entries = entries

        if shuffle:
            print("Shuffle paths")
            random.shuffle(self.entries)
            print(self.entries[:10])

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        for entry in self.entries:
            match = re.match(self.splits_re, entry["image_hash"])
            path = os.path.join(match.group(1), match.group(2), match.group(0) + ".jpg")
            yield {"path": os.path.join(self.path, path), "rel_path": path, "entry": entry}


def read_image(args):
    path = args["path"]
    rel_path = args["rel_path"]
    reject_min_dim = args["reject_min_dim"]
    max_dim = args["max_dim"]
    entry = args["entry"]
    print("############ ")

    try:
        image = imageio.imread(path)
    except:
        print(path)
        return None

    if reject_min_dim:
        if image.shape[0] < reject_min_dim or image.shape[1] < reject_min_dim:
            return None
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.shape[-1] == 4:
        image = image[:, :, 0:3]

    # print(image.shape)

    shape = np.asarray(image.shape[:-1], dtype=np.float32)
    long_dim = max(shape)
    scale = min(1, max_dim / long_dim)

    new_shape = np.asarray(shape * scale, dtype=np.int32)
    image = cv2.resize(image, tuple(new_shape[::-1].tolist()))

    # print(image.shape)<
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return {"image": imageio.imwrite(imageio.RETURN_BYTES, image, format="jpg"), "path": rel_path, "entry": entry}


def read_annotations(annotation_path):
    data = {}
    with open(annotation_path, "r") as f:
        for line in f:
            d = json.loads(line)
            data[d["image_hash"]] = d
    return data


def main():
    args = parse_args()

    logging.info("Reading image folder")
    if args.skip_scan:
        annotation = read_annotations(args.annotation)
        image_loader = FakeDataloader(args.image_path, entries=[y for x, y in annotation.items()], shuffle=args.shuffle)
    else:
        image_loader = ImageDataloader(args.image_path, shuffle=args.shuffle)

        whitelist = None
        if args.annotation:
            whitelist = read_annotations(args.annotation)

            def filter_white(a):
                match = re.match(r"^(.*?/)*([^_]*)(.*)[\.]*(\..*?)$", a["image_hash"])
                if not match:
                    return False
                if match[2] in whitelist:
                    return True
                return False

            image_loader = list(filter(filter_white, image_loader))

    logging.info(f"Found {len(image_loader)} images")
    with mp.Pool(args.process) as pool:
        with MsgPackWriter(args.output) as f:
            start = time.time()
            image_loader = [{**x, "reject_min_dim": args.reject_min_dim, "max_dim": args.max_dim} for x in image_loader]
            for i, x in enumerate(pool.imap(read_image, image_loader)):
                if x is None:
                    continue
                # print(i)
                f.write(x)

                # f.write(json.dumps({'key': x['path'], 'path': x['path'], 'index': i}) + '\n')
                if i % 1000 == 0:
                    # txn.commit()
                    end = time.time()
                    logging.info(f"{i}: {1000/(end - start):.2f} image/s")
                    start = end

    return 0


if __name__ == "__main__":
    sys.exit(main())
