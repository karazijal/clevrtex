import numpy as np
import torch
import torchvision
from torchvision import datasets


class PascalVOC(torch.utils.data.Dataset):
    shape = (3, 500, 375)

    def __init__(self, path, split="train"):
        self.d = datasets.voc.VOCSegmentation(path, download=True, image_set=split)
        self.t = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.d)

    def __getitem__(self, index):
        img, mask = self.d[index]
        img = self.t(img)
        mask = np.array(mask)
        v = np.unique(mask)
        v.sort()
        v = v[1:-1]
        m = np.zeros((11, *mask.shape))
        for i, k in enumerate(v):
            m[i + 1] = (mask == k).astype(float)
        m[0] = 1 - (m[1:] == 1).any(axis=0).astype(float)
        vis = torch.from_numpy((m == 1).any(axis=1).any(axis=1))
        m = torch.from_numpy(m).unsqueeze(1)
        return img, m, vis
