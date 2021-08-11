import torch


class TransformIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, transformation=None):
        super(TransformIterableDataset, self).__init__()
        self.dataset = dataset
        self.transformation = transformation

    def __iter__(self):
        for sample in self.dataset:
            if self.transformation is not None:
                x = self.transformation(sample)
            yield x

    def __len__(self):
        return self.length
