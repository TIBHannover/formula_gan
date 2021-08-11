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

import tensorflow as tf

from msgpack_dataset_creator import MsgPackWriter

from google.protobuf.json_format import MessageToJson


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-p', '--path', help='verbose output')
    parser.add_argument('-o', '--output', help='path to dir of images')
    parser.add_argument('-m', '--mapping')
    parser.add_argument('--process', default=8, type=int)

    parser.add_argument('-s', '--shuffle', action='store_true', default=True, help='verbose output')

    args = parser.parse_args()
    return args


def read_records(args):
    record = args['record']
    mapping = args['mapping']
    data_dict_list = []
    for example in tf.data.TFRecordDataset(record):
        data_dict = {}
        for key, value in tf.train.Example.FromString(example.numpy()).features.feature.items():
            values = []
            for v in value.float_list.value:
                values.append(v)
            for v in value.bytes_list.value:
                values.append(v)
            for v in value.int64_list.value:
                values.append(v)

            if len(values) == 1:
                values = values[0]
            if key in mapping:
                data_dict[mapping[key]] = values
            else:
                data_dict[key] = values
        data_dict_list.append(data_dict)
    return data_dict_list


def main():
    args = parse_args()

    if args.mapping is not None:
        mapping = json.loads(args.mapping)
    else:
        mapping = {}
    #record_re = re.compile()
    records = [os.path.join(args.path, x) for x in os.listdir(args.path)]

    with mp.Pool(args.process) as pool:
        with MsgPackWriter(args.output) as f:
            start = time.time()
            records = [{'record': x, 'mapping': mapping} for x in records]
            count = 0
            for i, x in enumerate(pool.imap(read_records, records)):
                for example in x:
                    if example is None:
                        continue
                    #print(i)
                    f.write(example)

                    # f.write(json.dumps({'key': x['path'], 'path': x['path'], 'index': i}) + '\n')
                    if count % 1000 == 0:
                        # txn.commit()
                        end = time.time()
                        print(f'{count}: {1000/(end - start):.2f} image/s')
                        start = end
                    count += 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
