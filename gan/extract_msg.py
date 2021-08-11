import os
import sys
import re
import argparse
import msgpack
import imageio
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-m", "--msgpack_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    shard_re = r"shard_(\d+).msg"

    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)

    shards_paths = []
    if os.path.isfile(args.msgpack_path):
        shards_paths.append(args.msgpack_path)
    elif os.path.isdir(args.msgpack_path):
        for p in [args.msgpack_path]:
            shards_paths.extend([os.path.join(p, x) for x in os.listdir(p) if re.match(shard_re, x)])

    count = 0
    for path in shards_paths:
        with open(path, "rb") as f:
            packer = msgpack.Unpacker(f, max_buffer_size=1024 * 1024 * 1024, raw=True)
            for p in packer:
                ### All information you want
                id = p[b"id"].decode()
                equation = p[b"equation"].decode()
                image = imageio.imread(p[b"image"])
                image = 255 - image

                ### If you want to extract image and informations
                if args.output_path is not None:
                    imageio.imwrite(os.path.join(args.output_path, f"{id}.jpg"), image)

                count += 1
    print(count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
