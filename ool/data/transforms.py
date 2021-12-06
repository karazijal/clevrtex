import functools

import torch
import torchvision
import torchvision.transforms.functional_tensor as Ftv

class ChainTransforms:
    def __init__(self, transforms, *args):
        if len(args) > 0:
            transforms = [transforms, *args]
        self.transforms = transforms

    def __call__(self, itm):
        for t in self.transforms:
            itm = t(itm)
        return itm

class Resize:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, itm, *args):
        if len(args) > 0:
            img, rest = itm, args
        else:
            img, *rest = itm
        nimg = Ftv.resize(img, size=self.target_size)
        if len(rest) > 0 and rest[0].shape[-2:] == img.shape[-2:]:
            masks = rest[0]
            if torchvision.__version__.startswith('0.8'):
                # major compat change
                masks = Ftv.resize(masks, size=self.target_size, interpolation=0)
            else:
                masks = Ftv.resize(masks, size=self.target_size, interpolation='nearest')
            rest = (masks, *rest[1:])
        return (nimg, *rest)


class MaxObjectsFilter:
    def __init__(self, n):
        self.n = n

    def __call__(self, itm, *args):
        if len(args) > 0:
            img, rest = itm, args
        else:
            img, *rest = itm
        masks, *rest = rest
        N, C, H, W = masks.shape
        masks = torch.cat(
            [masks[:self.n + 1], torch.zeros(N - self.n - 1, C, H, W, dtype=masks.dtype, device=masks.device)], dim=0)
        masks[0] = 1. - torch.any(masks[1:].to(bool), dim=0).to(torch.float)
        if len(rest) and rest[0].shape[0] == N:
            vis, *rest = rest
            vis = torch.cat(
            [vis[:self.n + 1], torch.zeros(N - self.n - 1, *vis.shape[1:], dtype=vis.dtype, device=vis.device)], dim=0)
            rest = (vis, *rest)
        return (img, masks, *rest)

class CentreCrop:
    """Centre-crops the image to a square shape calculating the new size based on smaller dimension"""
    def __init__(self, crop_fraction):
        self.cf = crop_fraction

    # @functools.lru_cache(None)
    def croping_bounds(self, input_size):
        h,w =input_size[-2:]
        dim = min(h,w)
        crop_size = int(self.cf * float(dim))
        h_start = (h-crop_size) // 2
        w_start = (w-crop_size) // 2
        h_slice = slice(h_start, h_start+crop_size)
        w_slice = slice(w_start, w_start+crop_size)
        return h_slice, w_slice

    def __call__(self, itm, *args):
        if len(args) > 0:
            img, rest = itm, args
        else:
            img, *rest = itm
        hs,ws = self.croping_bounds(img.shape)
        img = img[..., hs, ws]
        if len(rest) > 0:
            mask = rest[0]
            mask =  mask[..., hs, ws]
            return (img, mask, *rest[1:])
        return (img, *rest)

class HFlip:
    """Horizontally flip the image to a square shape calculating the new size based on smaller dimension"""
    def __call__(self, itm, *args):
        if len(args) > 0:
            img, rest = itm, args
        else:
            img, *rest = itm
        img = torch.flip(img, dims=(-2,))
        if len(rest) > 0:
            mask = rest[0]
            mask =  torch.flip(mask, dims=(-2,))
            return (img, mask, *rest[1:])
        return (img, *rest)


class ObjectOnly:
    def __call__(self, itm, *args):
        if len(args) > 0:
            img, rest = itm, args
        else:
            img, *rest = itm
        ori_img = img
        masks, *rest = rest
        for i in range(1, masks.shape[0]):
            mask = masks[i]
            _, h, w = torch.where(mask > 0)

            if h.min() == h.max() or w.min() == w.max():
                continue
            hs = slice(h.min().item(), h.max().item() + 1)
            ws = slice(w.min().item(), w.max().item() + 1)
            mask = mask[..., hs, ws]
            img = img[..., hs, ws]
            break
        else:
            return (ori_img, masks[0], *rest)
        return (img, mask, *rest)
