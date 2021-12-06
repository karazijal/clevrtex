from pathlib import Path

import torch
import numpy as np

from ool.data.utils import DatasetReadError, index_with_bias_and_limit


class FlatMnist:
    NAME = 'multi_mnist.npz'
    shape = (1, 50, 50)
    splits = {
        'test': (0, 5000),
        'val': (5000, 10000),
        'train': (10000, None)
    }
    def __init__(self, path: Path, device=None, split=None):
        if split is None:
            split = 'train'
            print("None split -- assuming train version")

        try:
            data = np.load(str(path / self.NAME), allow_pickle=True)
        except FileNotFoundError:
            raise DatasetReadError()
        self.imgs = torch.from_numpy(data['imgs'])
        self.lbls = data['labels']
        if device:
            self.imgs = self.imgs.to(device)
        self.bias, limit = self.splits[split]
        self.limit = limit or len(self.img)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, itm):
        itm = index_with_bias_and_limit(itm, self.bias, self.limit)
        img = self.imgs[itm]
        lbl = self.lbls[itm]
        img = img.to(torch.float).view(1, 50, 50) / 255.
        lbl = torch.tensor(len(lbl)).to(torch.int)
        x = (img, lbl)
        return x
