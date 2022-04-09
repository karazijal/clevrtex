import itertools

import torch
import torchvision as tv

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap

from PIL import ImageFont

FNT = ImageFont.truetype("dejavu/DejaVuSansMono.ttf", 7)
CMAPSPEC = {
    "cmap": ListedColormap(
        ["black", "red", "green", "blue", "yellow"]
        + list(
            itertools.chain.from_iterable(
                itertools.chain.from_iterable(
                    zip(
                        [get_cmap("tab20b")(i) for i in range(i, 20, 4)],
                        [get_cmap("tab20c")(i) for i in range(i, 20, 4)],
                    )
                )
                for i in range(4)
            )
        )
        + ["cyan", "magenta"]
        + [get_cmap("Set3")(i) for i in range(12)]
        + ["white"],
        name="SemSegMap",
    ),
    "vmin": 0,
    "vmax": 13,
}


@torch.no_grad()
def _to_img(img, lim=None, dim=-3):
    if lim:
        img = img[:lim]
    img = (img.clamp(0, 1) * 255).to(torch.uint8).cpu().detach()
    if img.shape[dim] < 3:
        epd_dims = [-1 for _ in range(len(img.shape))]
        epd_dims[dim] = 3
        img = img.expand(*epd_dims)
    return img


@torch.no_grad()
def log_semantic_images(input, output, true_masks, pred_masks, prefix=""):
    img = _to_img(input)
    omg = _to_img(output, lim=len(img))
    true_masks = true_masks[: len(img)].to(torch.float).argmax(1).squeeze(1)
    pred_masks = pred_masks[: len(img)].to(torch.float).argmax(1).squeeze(1)
    tms = (_cmap_tensor(true_masks) * 255.0).to(torch.uint8)
    pms = (_cmap_tensor(pred_masks) * 255.0).to(torch.uint8)
    vis_imgs = list(itertools.chain.from_iterable(zip(img, omg, tms, pms)))
    grid = tv.utils.make_grid(vis_imgs, pad_value=128, nrow=16).detach().cpu()
    self.logger.experiment[0].add_image(prefix + "segmentation", grid, self.current_epoch)


@torch.no_grad()
def _cmap_tensor(self, t):
    t_hw = t.cpu().detach().numpy()
    o_hwc = CMAPSPEC["cmap"](t_hw)[..., :3]  # drop alpha
    o = torch.from_numpy(o_hwc).transpose(-1, -2).transpose(-2, -3)
    return o
