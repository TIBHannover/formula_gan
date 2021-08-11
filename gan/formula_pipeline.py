import numpy as np
import imageio
import os

import torch

import random

from torch.utils.data import DataLoader
from mlio.pipeline import (
    Dataset,
    Pipeline,
    MapDataset,
    MsgPackPipeline,
    SequencePipeline,
    ConcatShufflePipeline,
    MergePipeline,
    ImagePipeline,
)


import torchvision


from mlcore.config import Config, config_add_options, ConfigEntry, str2bool
from mlio.pad_collate import PadCollate


class Vocabulary:
    def __init__(self):
        default_order = ["#PAD", "#EOS", "#UNK"]
        # Not necessary
        default_order += [chr(x) for x in range(48, 58)]  # 0-9
        default_order += [chr(x) for x in range(65, 91)]  # A-Z
        default_order += [chr(x) for x in range(97, 123)]  # a-z
        default_order += [x for x in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"]
        self.default_order = default_order

    def to_sequence(self, string):
        return [
            self.default_order.index(c) if c in self.default_order else self.default_order.index("#UNK") for c in string
        ]

    def __len__(self):
        return len(self.default_order)

    def to_string(self, sequence):
        return [self.default_order[c] for c in sequence]

    def start_id(self):
        return self.default_order["#PAD"]

    def unknown_id(self):
        return self.default_order["#UNK"]

    def eos_id(self):
        return self.default_order["#EOS"]


def pre_process(args):
    line, lexicon, vocabulary = args
    filename, lexicon_ind = line.split()
    lexicon_ind = int(lexicon_ind)
    string = lexicon[lexicon_ind]
    sequence = vocabulary.to_sequence(string)
    # print(self.vocabulary.to_string(sequence))
    # print(filename)
    # print(self.lexicon[int(lexicon_ind)])
    return [filename, lexicon_ind, string, sequence]


def read_dictionary(dictonary_path):

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


class ChromeDecoderPipeline(Pipeline):
    def __init__(self, dictonary_path):
        super(ChromeDecoderPipeline, self).__init__()

        self.dictonary = read_dictionary(dictonary_path)

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            sequence = (
                [self.dictonary["<start>"]]
                + [
                    self.dictonary[c] if c in self.dictonary else 400
                    for c in sample[b"equation"].decode("utf-8").split(" ")
                ]
                + [self.dictonary["<end>"]]
            )

            sequence = torch.from_numpy(np.asarray(sequence))

            if 400 in sequence:
                exit()

            return {
                **sample,
                "sequence": sequence,
                "sequence_mask": torch.ones_like(sequence, dtype=torch.int8),
            }

        return MapDataset(datasets, map_fn=decode)


class ChromeImagePreprocessingPipeline(Pipeline):
    def __init__(
        self, transformation, mean_height, max_height, max_width, max_image_area, training=True, image_dim_multiple=16
    ):
        super(ChromeImagePreprocessingPipeline, self).__init__()

        self.transformation = transformation
        self.mean_height = mean_height
        self.max_height = max_height
        self.max_width = max_width
        self.max_image_area = max_image_area
        self.training = training
        self.image_dim_multiple = image_dim_multiple

    def call(self, datasets=None, **kwargs):
        def decode(sample):

            image = torch.from_numpy(imageio.imread(sample[b"image"]))
            image = image.unsqueeze(0)
            # image = image.repeat(3, 1, 1)  # TODO remove me

            if self.transformation is not None:
                image = self.transformation(image)

            shape = np.asarray(image.shape[1:], dtype=np.float32)

            scale = self.mean_height / shape[0]

            if self.training:
                scale *= random.random() + 0.5

            new_shape = np.asarray(shape * scale, dtype=np.int32)
            new_shape = np.asarray(
                np.round(new_shape / self.image_dim_multiple) * self.image_dim_multiple, dtype=np.int32
            )

            if new_shape[1] < 32:
                return None

            if new_shape[0] < 32:
                return None

            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=[new_shape[0], new_shape[1]], mode="bilinear", align_corners=True
            ).squeeze(0)

            if image.shape[1] * image.shape[2] > self.max_image_area:
                # print(sample[b'id'])
                return None

            if image.shape[1] > self.max_height:
                # print(sample[b'id'])
                return None

            if image.shape[2] > self.max_width:
                # print(sample[b'id'])
                return None
            return {
                **sample,
                "image": image,
            }

        return MapDataset(datasets, map_fn=decode)


class ChromeInferenceImagePreprocessingPipeline(Pipeline):
    def __init__(self, transformation, mean_height, max_height, max_width, max_image_area, image_dim_multiple=16):
        super(ChromeInferenceImagePreprocessingPipeline, self).__init__()

        self.transformation = transformation
        self.mean_height = mean_height
        self.max_height = max_height
        self.max_width = max_width
        self.max_image_area = max_image_area

        self.image_dim_multiple = image_dim_multiple

    def call(self, datasets=None, **kwargs):
        def decode(sample):

            image = torch.from_numpy(sample["image"])
            image = 255 - image.unsqueeze(0)
            # image = image.repeat(3, 1, 1)  # TODO remove me

            if self.transformation is not None:
                image = self.transformation(image)

            shape = np.asarray(image.shape[1:], dtype=np.float32)

            scale = self.mean_height / shape[0]

            new_shape = np.asarray(shape * scale, dtype=np.int32)
            new_shape = np.asarray(
                np.round(new_shape / self.image_dim_multiple) * self.image_dim_multiple, dtype=np.int32
            )

            if new_shape[1] < 32:
                return None

            if new_shape[0] < 32:
                return None

            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=[new_shape[0], new_shape[1]], mode="bilinear", align_corners=True
            ).squeeze(0)

            if image.shape[1] * image.shape[2] > self.max_image_area:
                # print(sample[b'id'])
                return None

            if image.shape[1] > self.max_height:
                # print(sample[b'id'])
                return None

            if image.shape[2] > self.max_width:
                # print(sample[b'id'])
                return None
            return {
                **sample,
                "image": image,
            }

        return MapDataset(datasets, map_fn=decode)


def build_train_dataloader(
    msgpack_path, annotation_path, dictionary_path, batch_size, mean_height, max_height, max_width, max_image_area
):
    transformation = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            # torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(5, fill=(0,)),
            # torchvision.transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
            # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            torchvision.transforms.ToTensor(),
        ]
    )
    pipeline = SequencePipeline(
        [
            MsgPackPipeline(msgpack_path),
            ChromeDecoderPipeline(dictionary_path),
            ChromeImagePreprocessingPipeline(
                transformation=transformation,
                mean_height=mean_height,
                max_height=max_height,
                max_width=max_width,
                max_image_area=max_image_area,
                training=True,
            ),
        ]
    )
    collate_fn = PadCollate()
    # shards_sampler = EqualShardSampler()
    return DataLoader(
        pipeline(),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=16,
        drop_last=True,
    )


def build_val_dataloader(
    msgpack_path, annotation_path, dictionary_path, batch_size, mean_height, max_height, max_width, max_image_area
):
    transformation = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomRotation(5, fill=(0,)),
            # torchvision.transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
            # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            torchvision.transforms.ToTensor(),
        ]
    )
    pipeline = SequencePipeline(
        [
            MsgPackPipeline(msgpack_path, shuffle=False),
            ChromeDecoderPipeline(dictionary_path),
            ChromeImagePreprocessingPipeline(
                transformation=transformation,
                mean_height=mean_height,
                max_height=max_height,
                max_width=max_width,
                max_image_area=max_image_area,
                training=False,
            ),
        ]
    )
    collate_fn = PadCollate()
    # shards_sampler = EqualShardSampler()
    return DataLoader(
        pipeline(),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=16,
        drop_last=True,
    )


class DomainDataset(Dataset):
    def __init__(self, dataset, domain_flag):
        super(DomainDataset, self).__init__()
        self.dataset = dataset
        self.domain_flag = domain_flag

    def __iter__(self):
        for sample in self.dataset:
            if sample is None:
                continue

            sample = {**sample, "domain": self.domain_flag}

            if sample is None:
                continue
            yield sample

    def __len__(self):
        return len(self.dataset)


class DomainPipeline(Pipeline):
    def __init__(self, domain_flag):
        super(DomainPipeline, self).__init__()
        self.domain_flag = domain_flag

    def call(self, datasets=None, **kwargs):
        return DomainDataset(datasets, self.domain_flag)


def build_gan_dataloader(
    target_msgpack_path,
    source_msgpack_path,
    annotation_path,
    dictionary_path,
    batch_size,
    mean_height,
    max_height,
    max_width,
    max_image_area,
    target_domain_flag=None,
    source_domain_flag=None,
):
    transformation = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            # torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(5, fill=(0,)),
            # torchvision.transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
            # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            torchvision.transforms.ToTensor(),
        ]
    )

    def build_pipeline(msgpack_path, domain_flag, transformation):

        # Image from the source distribution
        if not isinstance(msgpack_path, (list, set)):
            msgpack_path = [msgpack_path]

        pipelines = []

        for path, domain in zip(msgpack_path, domain_flag):
            pipelines.append(
                SequencePipeline(
                    [
                        MsgPackPipeline(path),
                        ChromeDecoderPipeline(dictionary_path),
                        ChromeImagePreprocessingPipeline(
                            transformation=transformation,
                            mean_height=mean_height,
                            max_height=max_height,
                            max_width=max_width,
                            max_image_area=max_image_area,
                        ),
                        DomainPipeline(domain),
                    ]
                )
            )
        return ConcatShufflePipeline(pipelines)

    source_pipeline = build_pipeline(source_msgpack_path, source_domain_flag, transformation)

    # Image from the target distirbution
    target_pipeline = build_pipeline(target_msgpack_path, target_domain_flag, transformation)

    def merge_fn(sample):
        return {
            "source_image": sample[0]["image"],
            "source_domain": sample[0]["domain"],
            "source_sequence": sample[0]["sequence"],
            "source_sequence_mask": sample[0]["sequence_mask"],
            "target_image": sample[1]["image"],
            "target_domain": sample[1]["domain"],
            "target_sequence": sample[1]["sequence"],
            "target_sequence_mask": sample[1]["sequence_mask"],
        }

    pipeline = MergePipeline([source_pipeline, target_pipeline], merge_fn=merge_fn)

    collate_fn = PadCollate()
    # shards_sampler = EqualShardSampler()
    return DataLoader(
        pipeline(),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=16,
        drop_last=True,
    )


@config_add_options("gan_dataloader")
def config_gan_dataloader():
    return {
        "source_msgpack_path": ConfigEntry(
            default=[
                "/data/va-formula-ocr/crohme/train_2014/msg/",
                "/data/va-formula-ocr/ntcir_gan_paper_bmvc/gan_msg/",
            ],
            nargs="+",
        ),
        "target_msgpack_path": ConfigEntry(
            default=[
                "/data/va-formula-ocr/crohme/train_2014/msg/",
                "/data/va-formula-ocr/ntcir_gan_paper_bmvc/gan_msg/",
            ],
            nargs="+",
        ),
        "annotation_path": ConfigEntry(default="/data/va-formula-ocr/crohme_2019/train_equations_2014.jsonl"),
        "dictionary_path": ConfigEntry(default="/data/va-formula-ocr/dictionary.txt"),
        "mean_height": ConfigEntry(default=120, type=int),
        "max_height": ConfigEntry(default=128, type=int),
        "max_width": ConfigEntry(default=512, type=int),
        "max_image_area": ConfigEntry(default=200000, type=int),
        "batch_size": ConfigEntry(default=4, type=int),
        "source_domain_flag": ConfigEntry(default=[], type=int, nargs="+"),
        "target_domain_flag": ConfigEntry(default=[], type=int, nargs="+"),
    }


def test_dataloader(
    msgpack_path, annotation_path, dictionary_path, batch_size, mean_height, max_height, max_width, max_image_area
):
    collate_fn = PadCollate()
    return DataLoader(
        CrohmeDataset(
            msgpack_path,
            annotation_path,
            dictionary_path,
            mean_height=mean_height,
            max_height=max_height,
            max_width=max_width,
            max_image_area=max_image_area,
            training=False,
        ),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=16,
        drop_last=True,
    )


@config_add_options("train_dataloader")
def config_val_dataloader():
    return {
        "msgpack_path": ConfigEntry(default="/data/va-formula-ocr/crohme/test_2013/msg/"),
        "annotation_path": ConfigEntry(default="/data/va-formula-ocr/crohme_2019/test_equations_2013.jsonl"),
        "dictionary_path": ConfigEntry(default="/data/va-formula-ocr/dictionary.txt"),
        "mean_height": ConfigEntry(default=128, type=int),
        "max_height": ConfigEntry(default=128, type=int),
        "max_width": ConfigEntry(default=1024, type=int),
        "max_image_area": ConfigEntry(default=200000, type=int),
        "batch_size": ConfigEntry(default=16, type=int),
    }


@config_add_options("val_dataloader")
def config_val_dataloader():
    return {
        "msgpack_path": ConfigEntry(default="/data/va-formula-ocr/crohme/test_2013/msg/"),
        "annotation_path": ConfigEntry(default="/data/va-formula-ocr/crohme_2019/test_equations_2013.jsonl"),
        "dictionary_path": ConfigEntry(default="/data/va-formula-ocr/dictionary.txt"),
        "mean_height": ConfigEntry(default=128, type=int),
        "max_height": ConfigEntry(default=128, type=int),
        "max_width": ConfigEntry(default=1024, type=int),
        "max_image_area": ConfigEntry(default=200000, type=int),
        "batch_size": ConfigEntry(default=16, type=int),
    }


@config_add_options("test_dataloader")
def config_val_dataloader():
    return {
        "msgpack_path": ConfigEntry(default="/data/va-formula-ocr/crohme/test_2013/msg/"),
        "annotation_path": ConfigEntry(default="/data/va-formula-ocr/crohme_2019/test_equations_2013.jsonl"),
        "dictionary_path": ConfigEntry(default="/data/va-formula-ocr/dictionary.txt"),
        "mean_height": ConfigEntry(default=128, type=int),
        "max_height": ConfigEntry(default=256, type=int),
        "max_width": ConfigEntry(default=1024, type=int),
        "max_image_area": ConfigEntry(default=200000, type=int),
        "batch_size": ConfigEntry(default=1, type=int),
    }


def build_infer_dataloader(
    path,
    msgpack_path,
    dictionary_path,
    batch_size,
    mean_height,
    max_height,
    max_width,
    max_image_area,
):
    transformation = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            # torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(5, fill=(0,)),
            # torchvision.transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
            # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            torchvision.transforms.ToTensor(),
        ]
    )

    if path is not None:

        def build_pipeline(path, transformation):

            return SequencePipeline(
                [
                    ImagePipeline(path, shuffle=False, return_rgb=False),
                    ChromeInferenceImagePreprocessingPipeline(
                        transformation=transformation,
                        mean_height=mean_height,
                        max_height=max_height,
                        max_width=max_width,
                        max_image_area=max_image_area,
                    ),
                ]
            )

        pipeline = build_pipeline(path, transformation)
    else:

        def build_pipeline(msgpack_path, transformation):

            # Image from the source distribution
            if not isinstance(msgpack_path, (list, set)):
                msgpack_path = [msgpack_path]

            pipelines = []

            for path in msgpack_path:

                pipelines.append(
                    SequencePipeline(
                        [
                            MsgPackPipeline(path),
                            ChromeDecoderPipeline(dictionary_path),
                            ChromeImagePreprocessingPipeline(
                                transformation=transformation,
                                mean_height=mean_height,
                                max_height=max_height,
                                max_width=max_width,
                                max_image_area=max_image_area,
                            ),
                        ]
                    )
                )
            return ConcatShufflePipeline(pipelines)

        pipeline = build_pipeline(msgpack_path, transformation)

    # def merge_fn(sample):
    #     return {
    #         "source_image": sample[0]["image"],
    #         "source_domain": sample[0]["domain"],
    #         "source_sequence": sample[0]["sequence"],
    #         "source_sequence_mask": sample[0]["sequence_mask"],
    #     }

    # pipeline = MergePipeline([source_pipeline], merge_fn=merge_fn)

    collate_fn = PadCollate()
    # shards_sampler = EqualShardSampler()
    return DataLoader(
        pipeline(),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=16,
        drop_last=True,
    )


@config_add_options("infer_dataloader")
def config_infer_dataloader():
    return {
        "path": ConfigEntry(),
        "msgpack_path": ConfigEntry(
            nargs="+",
        ),
        "dictionary_path": ConfigEntry(default="/app/data/dictionary.txt"),
        "mean_height": ConfigEntry(default=120, type=int),
        "max_height": ConfigEntry(default=128, type=int),
        "max_width": ConfigEntry(default=512, type=int),
        "max_image_area": ConfigEntry(default=200000, type=int),
        "batch_size": ConfigEntry(default=1, type=int),
    }
