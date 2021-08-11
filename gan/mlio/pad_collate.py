import torch

import numpy as np


import torch.nn.functional as F


def pad_tensor(input, max_shape, value=0):
    if len(max_shape) == 0:
        return input

    padding_size = torch.LongTensor(max_shape - input.shape)
    padding_size = torch.LongTensor([[0, x] for x in padding_size.tolist()[::-1]]).view(2 * max_shape.shape[0]).tolist()
    result = F.pad(input, padding_size, value=value)

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
