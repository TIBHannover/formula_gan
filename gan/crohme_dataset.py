import argparse
import re
import sys
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import pad
import numpy as np
import imageio
import torchvision
import cv2
import random

from PIL import Image

from multiprocessing import Pool

from msgpack_dataset import MsgPackIterableDataset

# import matplotlib.pyplot as plt


def pad_tensor(input, max_shape, value=0):
    if len(max_shape) == 0:
        return input

    padding_size = torch.LongTensor(max_shape - input.shape)
    padding_size = torch.LongTensor([[0, x] for x in padding_size.tolist()[::-1]]).view(2 * max_shape.shape[0]).tolist()
    result = pad(input, padding_size, value=value)

    return result


def pad_dict(input, max_shapes, padding_values):
    results = {}
    for key, value in input.items():
        results[key] = pad_tensor(value, max_shapes[key], 0)
    return results


def stack_dict(input, keys):
    results = {}
    for key in keys:
        try:
            results[key] = torch.stack(list(map(lambda x: x[key], input)), dim=0)
        except:
            results[key] = list(map(lambda x: x[key], input))
    return results


class PadCollate:
    def init(self):
        pass

    def pad_collate(self, batch):
        if isinstance(batch[0], dict):
            keys = set([a for i in map(lambda x: list(x.keys()), batch) for a in i])
            max_shapes = {}
            for key in keys:
                max_shapes[key] = np.amax(
                    list(map(lambda x: list(x[key].shape) if getattr(x[key], "shape", None) else [], batch)), axis=0
                )

            batch = list(map(lambda x: pad_dict(x, max_shapes, 0), batch))
            return stack_dict(batch, keys)
        elif isinstance(batch[0], (list, set)):
            raise ValueError("Not implemented yet")
        else:
            raise ValueError("What is this?")

    def __call__(self, batch):
        return self.pad_collate(batch)


class Vocabulary:
    def __init__(self):
        default_order = ["#PAD", "#EOS", "#UNK"]
        # Not necessary
        default_order += [chr(x) for x in range(48, 58)]  # 0-9
        default_order += [chr(x) for x in range(65, 91)]  # A-Z
        default_order += [chr(x) for x in range(97, 123)]  # a-z
        default_order += [x for x in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"]
        self._default_order = default_order

    def to_sequence(self, string):
        return [
            self._default_order.index(c) if c in self._default_order else self._default_order.index("#UNK")
            for c in string
        ]

    def __len__(self):
        return len(self._default_order)

    def to_string(self, sequence):
        return [self._default_order[c] for c in sequence]

    def start_id(self):
        return self._default_order["#PAD"]

    def unknown_id(self):
        return self._default_order["#UNK"]

    def eos_id(self):
        return self._default_order["#EOS"]


# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./2194/2/334_EFFLORESCENT_24742.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./2128/2/369_REDACTED_63458.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./2069/4/192_whittier_86389.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./2025/2/364_SNORTERS_72304.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./2013/2/370_refract_63890.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./1881/4/225_Marbling_46673.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./1863/4/223_Diligently_21672.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./1817/2/363_actuating_904.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./1730/2/361_HEREON_35880.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./1696/4/211_Queened_61779.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./1650/2/355_stony_74902.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./1332/4/224_TETHERED_78397.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./936/2/375_LOCALITIES_44992.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./913/4/231_randoms_62372.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./905/4/234_Postscripts_59142.jpg
# /data/mjsynth/mnt/ramdisk/max/90kDICT32px/./869/4/234_TRIASSIC_80582.jpg
# ^C/data/mjsynth/mnt/ramdisk/max/90kDICT32px/./699/5/332_PUDGIER_61116.jpg


def pre_process(args):
    line, lexicon, vocabulary = args
    filename, lexicon_ind = line.split()
    lexicon_ind = int(lexicon_ind)
    string = lexicon[lexicon_ind]
    sequence = vocabulary.to_sequence(string)
    # print(self._vocabulary.to_string(sequence))
    # print(filename)
    # print(self._lexicon[int(lexicon_ind)])
    return [filename, lexicon_ind, string, sequence]


def read_dictonary(dictonary_path):

    # print("Read dictonary")
    dictonary = {}
    with open(dictonary_path, "r") as fr:
        for line in fr.readlines():
            word, index = line.split("\t")
            dictonary[word] = int(index) + 2
    dictonary["<start>"] = 1
    dictonary["<end>"] = 2
    dictonary["<pad>"] = 0
    return dictonary


class CrohmeDataset(MsgPackIterableDataset):
    def __init__(
        self,
        msgpack_path,
        annotation_path,
        dictonary_path,
        training,
        max_image_area=30000,
        mean_height=128,
        max_height=256,
        max_width=1024,
        transform=None,
        **kwargs
    ):
        super(CrohmeDataset, self).__init__(msgpack_path, **kwargs)
        self._dictonary = read_dictonary(dictonary_path)
        self._transform = transform
        self._mean_height = mean_height
        self._training = training
        self._max_image_area = max_image_area
        self._max_height = max_height
        self._max_width = max_width

        if self._transform is None:
            if self._training:

                self._transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToPILImage(),
                        # torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomRotation(5, fill=(0,)),
                        # torchvision.transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
                        # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                        torchvision.transforms.ToTensor(),
                    ]
                )
            else:

                self._transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToPILImage(),
                        # torchvision.transforms.RandomHorizontalFlip(),
                        # torchvision.transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
                        # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                        torchvision.transforms.ToTensor(),
                        # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

        # self._
        # with open(self._lexicon_path) as f:
        #     self._lexicon = []
        #     for line in f:
        #         self._lexicon.append(line.strip())

        # with open(self._annotations_path) as f:
        #     lines = list(f)

        # with Pool(8) as pool:
        #     self._annotations = pool.map(pre_process, [(line, self._lexicon, self._vocabulary) for line in lines])

    # Override to give PyTorch access to any image on the dataset
    def __iter__(self):
        for x in super(CrohmeDataset, self).__iter__():
            sequence = (
                [self._dictonary["<start>"]]
                + [
                    self._dictonary[c] if c in self._dictonary else 400
                    for c in x[b"equation"].decode("utf-8").split(" ")
                ]
                + [self._dictonary["<end>"]]
            )
            if 400 in sequence:
                exit()

            # print(x["equation"])
            # print(sequence)

            image = torch.from_numpy(imageio.imread(x[b"image"]))
            image = image.unsqueeze(0)
            # image = image.repeat(3, 1, 1)  # TODO remove me

            if self._transform is not None:
                image = self._transform(image)

            shape = np.asarray(image.shape[1:], dtype=np.float32)

            scale = self._mean_height / shape[0]

            if self._training:
                scale *= random.random() + 0.5

            new_shape = np.asarray(shape * scale, dtype=np.int32)
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=[new_shape[0], new_shape[1]], mode="bilinear", align_corners=True
            ).squeeze(0)

            if image.shape[1] * image.shape[2] > self._max_image_area:
                # print(x[b'id'])
                continue

            if image.shape[1] > self._max_height:
                # print(x[b'id'])
                continue

            if image.shape[2] > self._max_width:
                # print(x[b'id'])
                continue
            sequence = torch.from_numpy(np.asarray(sequence))
            # print()
            yield {
                "path": x[b"path"].decode("utf-8"),
                "image": image,
                "sequence": sequence,
                "sequence_mask": torch.ones_like(sequence, dtype=torch.int8),
            }


# # import matplotlib.pyplot as plt


# def parse_args():
#     parser = argparse.ArgumentParser(description="")

#     parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
#     parser.add_argument("-p", "--path", help="verbose output")
#     args = parser.parse_args()
#     return args


# def main():
#     args = parse_args()

#     dataset = MjSynthDataset(data_path=args.path, split="train")
#     collate_fn = PadCollate()
#     print(len(dataset))
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn)
#     for i_batch, sample_batched in enumerate(dataloader):
#         print(i_batch)
#         print(sample_batched)
#         print(sample_batched["sequence"][0])
#         plt.imshow(sample_batched["image"][0])
#         plt.show()
#         # return
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())
