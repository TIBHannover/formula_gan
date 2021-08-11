import sys
import argparse
import re
import os
import json
import glob
from multiprocessing import Pool
from functools import partial

import numpy as np

# from scipy.misc import imread, imsave
import imageio

from msgpack_dataset_creator import MsgPackWriter


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--image_path", nargs="+", help="verbose output")
    parser.add_argument("-a", "--annotation", nargs="+", help="verbose output")
    parser.add_argument("-d", "--dictonary", help="path to dir of images")
    parser.add_argument("-m", "--msg_output", help="path to dir of images")
    parser.add_argument("-o", "--image_output", help="path to dir of images")
    parser.add_argument("-p", "--process", type=int, default=1, help="path to dir of images")
    parser.add_argument("-c", "--chunck", type=int, default=1024, help="Images per file")
    parser.add_argument("--invert", action="store_true", help="verbose output")

    parser.add_argument("-s", "--shuffle", action="store_true", default=True, help="verbose output")

    args = parser.parse_args()
    return args


# nthreads = 8

# dataset_source_dir = '/data/va-formula-ocr/ntcir_12_v001'
# vocabulary_file = 'data/dictionary.txt'

# images_dir = os.path.join(dataset_source_dir, 'katex')
# invert_image = True
# equations_dir = os.path.join(dataset_source_dir, 'equations')

# caption_file = 'tmp_caption.txt'
# image_out_dir = 'tmp_images'
# output =


def get_equations(equations_dir, id_key="file", eq_id="norm"):

    samples = []
    if os.path.isfile(equations_dir):

        with open(equations_dir, "r") as fr:
            for line in fr.readlines():
                samples.append(json.loads(line))
    else:
        equation_shards = [os.path.join(equations_dir, p) for p in os.listdir(equations_dir) if p.endswith(".txt")]

        for equation_shard in equation_shards:

            with open(equation_shard, "r") as fr:
                for line in fr.readlines():
                    samples.append(json.loads(line))

    key_equation_mapping = {}
    for sample in samples:
        if sample[id_key] not in key_equation_mapping:
            key_equation_mapping[sample[id_key]] = sample[eq_id]
        else:
            raise KeyError(f"{id_key} already in dict!")
    print(len(key_equation_mapping))
    return key_equation_mapping


def get_vocabulary(vocabulary_file):

    print("Read vocabulary")
    vocabulary = []
    with open(vocabulary_file, "r") as fr:
        for line in fr.readlines():
            word = line.split("\t")[0]
            vocabulary.append(word)
    vocabulary = set(vocabulary)
    return vocabulary


def is_valid(equation, vocabulary):
    for v in equation.split(" "):
        if v not in vocabulary:
            return False
    return True


def crop_image(image):
    print(image.shape)
    mask = np.abs(image) < 1

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    return image[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]]


def _process_image(image_tuple, invert_image, image_out_dir, crop_boarder=True):
    image_file, basename, equation = image_tuple

    image = imageio.imread(image_file, as_gray=True)
    image = image.astype(np.uint8)

    if invert_image:
        image = np.invert(image)

    if crop_boarder:
        image = crop_image(image)
        image = np.pad(image, [[5, 5], [5, 5]], "constant", constant_values=0)

    if image_out_dir is not None:
        fname_out = basename + "_0.bmp"
        imageio.imsave(os.path.join(image_out_dir, fname_out), image)
    return {
        "image": imageio.imwrite(imageio.RETURN_BYTES, image, format="jpg"),
        "path": image_file,
        "id": basename,
        "equation": equation,
    }


def main():

    args = parse_args()
    key_equation_mapping = {}
    for x in args.annotation:
        sub_key_equation_mapping = get_equations(x,id_key='uuid')
        for k, v in sub_key_equation_mapping.items():
            if k in key_equation_mapping:
                raise KeyError(f"{k} already in dict!")
            key_equation_mapping[k] = v

    vocabulary = get_vocabulary(args.dictonary)

    images_ctr = 0
    invalid_ctr = 0
    valid_samples = []  # (image file path, filename without extension, equation)
    for x in args.image_path:
        for image_file in glob.glob(os.path.join(x, "*")):
            basename = os.path.basename(os.path.splitext(image_file)[0])
            image_id = basename

            m = re.match(r'([a-f0-9]+).*?', image_id)
            image_id = m.group(1)
            # invalid image: no corresponding equation found
            if image_id not in key_equation_mapping:
                print(f"Image {image_file} with id {image_id} has no equation")
                continue

            images_ctr += 1

            # normalize equation
            equation = key_equation_mapping[image_id]
            equation = (
                equation.replace("\\lparen", "(")
                .replace("\\rparen", ")")
                .replace("\\lbrack", "[")
                .replace("\\rbrack", "]")
                .replace("\\lbrace", "\\{")
                .replace("\\rbrace", "\\}")
                .replace("\\vert", "|")
                .replace("\\vert", "|")
                .replace("\lt", "<")
                .replace("\gt", ">")
                .replace("\\to", "\\rightarrow")
                .replace("\\@cdots", "\\cdots")
            )

            # invalid image: invalid word in equation for given vocabulary
            if not is_valid(equation, vocabulary):

                print(basename)
                print(equation)
                print(vocabulary)
                exit()
                invalid_ctr += 1
                continue

            valid_samples.append((image_file, basename, equation))

    print(f"Invalid samples: {invalid_ctr}/{images_ctr}")
    print(f"Valid samples: {len(valid_samples)}")

    # print("Store captions to file")
    # with open(caption_file, "w") as fw:
    #     for _, basename, equation in valid_samples:
    #         fw.write(f"{basename} {equation}\n")  # todo: perhaps \n\r

    print("Process all relevant images")
    if args.msg_output:
        os.makedirs(args.msg_output, exist_ok=True)
    if args.image_output:
        os.makedirs(args.image_output, exist_ok=True)

    with Pool(args.process) as p:
        processed_ctr = 0
        with MsgPackWriter(args.msg_output) as f:
            for image_result in p.imap_unordered(
                partial(_process_image, invert_image=args.invert, image_out_dir=args.image_output), valid_samples
            ):

                f.write(image_result)
                processed_ctr += 1
                if processed_ctr % 1000 == 0:
                    print(f"Processed {processed_ctr}/{len(valid_samples)}")


if __name__ == "__main__":
    sys.exit(main())
