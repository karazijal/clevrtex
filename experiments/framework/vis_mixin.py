import itertools
from pathlib import Path

import torch
import torchvision as tv

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap

import numpy as np

from PIL import Image, ImageFont, ImageDraw

FNT = ImageFont.truetype("dejavu/DejaVuSansMono.ttf", 7)
CMAPSPEC = {
    'cmap': ListedColormap(
        ['black',
         'red',
         'green',
         'blue',
         'yellow'] +
        list(itertools.chain.from_iterable(itertools.chain.from_iterable(zip(
            [get_cmap('tab20b')(i) for i in range(i, 20, 4)],
            [get_cmap('tab20c')(i) for i in range(i, 20, 4)]
        )) for i in range(4))) +
        ['cyan',
         'magenta'] +
        [get_cmap('Set3')(i) for i in range(12)] +
        ['white'],
        name='SemSegMap'),
    'vmin': 0,
    'vmax': 13
}

class SpatialVisMixin:

    def log_as_fig(self, img, name, zeroone=True, colorbar=False, **kwargs):
        fig = plt.figure()
        img = img[0].detach().cpu()
        if zeroone:
            img[0, 0] = 1.0
            img[-1, -1] = 0.0
        plt.imshow(img, **kwargs)
        plt.axis('off')
        if colorbar:
            plt.colorbar()
        self.logger.experiment.add_figure(name, fig, self.current_epoch)

    @torch.no_grad()
    def _to_img(self, img, lim=None, dim=-3):
        if lim:
            img = img[:lim]
        img = (img.clamp(0, 1) * 255).to(torch.uint8).cpu().detach()
        if img.shape[dim] < 3:
            epd_dims = [-1 for _ in range(len(img.shape))]
            epd_dims[dim] = 3
            img = img.expand(*epd_dims)
        return img

    @torch.no_grad()
    def log_recons(self, input, output, patches, masks, background=None, where=None, pres=None, depth=None, prefix=''):
        if not self.trainer.is_global_zero: return
        vis_imgs = []
        img = self._to_img(input)
        vis_imgs.extend(img)
        omg = self._to_img(output, lim=len(img))
        vis_imgs.extend(omg)

        if background is not None and not torch.all(background == 0.):
            bg = self._to_img(background, lim=len(img))
            vis_imgs.extend(bg)

        if where is not None and depth is not None and pres is not None:
            where = where.detach().cpu().numpy()
            depth = depth.detach().cpu().numpy()
            pres = pres.detach().cpu().to(torch.uint8).numpy()

        masks = masks[:len(img)]
        patches = patches[:len(img)]
        ms = masks * patches
        for sid in range(patches.size(1)):
            # p = (patches[:len(img), sid].clamp(0., 1.) * 255.).to(torch.uint8).detach().cpu()
            # if p.shape[1] == 3:
            #     vis_imgs.extend(p)
            m = self._to_img(ms[:, sid])
            m_hat = []
            if pres is not None:
                for i in range(0, len(img)):
                    if pres[i, sid][0] == 1:
                        if where is not None and depth is not None and pres is not None:
                            img_to_draw = Image.fromarray(m[i].permute(1, 2, 0).numpy())
                            draw = ImageDraw.Draw(img_to_draw)
                            text = f"{where[i, sid][0]:.2f} {where[i, sid][1]:.2f}\n{where[i, sid][2]:.2f} {where[i, sid][3]:.2f}\n{depth[i, sid][0]:.2f}"
                            draw.multiline_text((2, 2), text, font=FNT, fill=(0, 255, 0, 128), spacing=2)
                            m_hat.append(torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1))
                        else:
                            m[i, 0, :2, :2] = 0
                            m[i, 1, :2, :2] = 255
                            m[i, 2, :2, :2] = 0
                            m_hat.append(m[i])
                    else:
                        m_hat.append(m[i])
            else:
                m_hat.extend(m)
            vis_imgs.extend(m_hat)
        grid = tv.utils.make_grid(vis_imgs, pad_value=128, nrow=len(img), padding=1).detach().cpu()
        self.logger.experiment.add_image(prefix + 'recon', grid, self.current_epoch)

    @torch.no_grad()
    def log_slots(self, slots, name='slots'):
        if not self.trainer.is_global_zero: return
        m = self._to_img(slots)
        m = m.view(-1, 3, *slots.shape[-2:])
        grid = tv.utils.make_grid(m, pad_value=128, nrow=len(slots), padding=1).detach().cpu()
        self.logger.experiment.add_image(name, grid, self.current_epoch)

    @torch.no_grad()
    def log_imgs_grid(self, input, *imgs, prefix=''):
        if not self.trainer.is_global_zero: return
        viss = []
        img = self._to_img(input)
        viss.append(img)
        for i in imgs:
            viss.append(self._to_img(i, lim=len(img)))
        vis = []
        for imgs in zip(*viss):
            for i in imgs:
                vis.append(i)
        nrow = len(viss)
        nrow = (16 // nrow) * nrow
        grid = tv.utils.make_grid(vis, pad_value=128, nrow=nrow).detach().cpu()
        self.logger.experiment.add_image(prefix + 'outputs', grid, self.current_epoch)

    @torch.no_grad()
    def log_images(self, input, output, bboxes=None, pres=None, prefix=''):
        if not self.trainer.is_global_zero: return
        vis_imgs = []
        img = self._to_img(input)
        omg = self._to_img(output, lim=len(img))

        if bboxes is not None:
            bboxes = bboxes.detach().cpu()
            pres = pres.detach().cpu()
            for i, (i_img, o_img) in enumerate(zip(img, omg)):
                vis_imgs.append(i_img)
                img_bbox = []
                for si in range(len(bboxes[i])):
                    if pres[i, si] == 1:
                        img_bbox.append(bboxes[i, si])
                if img_bbox:
                    img_to_draw = Image.fromarray(o_img.permute(1, 2, 0).numpy())
                    draw = ImageDraw.Draw(img_to_draw)

                    for bi, bbox in enumerate(img_bbox):
                        draw.rectangle(bbox.to(torch.int64).tolist(), width=1, outline='green')
                    o_img = torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)

                vis_imgs.append(o_img)
        else:
            for i, (i_img, o_img) in enumerate(zip(img, omg)):
                vis_imgs.append(i_img)
                vis_imgs.append(o_img)
        grid = tv.utils.make_grid(vis_imgs, pad_value=128, nrow=16).detach().cpu()
        self.logger.experiment.add_image(prefix + 'output', grid, self.current_epoch)

    @torch.no_grad()
    def log_semantic_images(self, input, output, true_masks, pred_masks, prefix=''):
        if not self.trainer.is_global_zero: return
        assert len(true_masks.shape) == 5 and len(pred_masks.shape) == 5
        img = self._to_img(input)
        omg = self._to_img(output, lim=len(img))
        true_masks = true_masks[:len(img)].to(torch.float).argmax(1).squeeze(1)
        pred_masks = pred_masks[:len(img)].to(torch.float).argmax(1).squeeze(1)
        tms = (self._cmap_tensor(true_masks) * 255.).to(torch.uint8)
        pms = (self._cmap_tensor(pred_masks) * 255.).to(torch.uint8)
        vis_imgs = list(itertools.chain.from_iterable(zip(img, omg, tms, pms)))
        grid = tv.utils.make_grid(vis_imgs, pad_value=128, nrow=16).detach().cpu()

        if hasattr(self, 'special_output_vis_img_path') and self.special_output_vis_img_path is not None:
            out_p = Path(self.special_output_vis_img_path)
            if hasattr(self, 'special_output_batch_indx'):
                batch_idxs = self.special_output_batch_indx
            else:
                batch_idxs = [0, 1, 2, 3, 4, 5, 6, 7] # Take first 8
            for idx in batch_idxs:
                self._save_img(img[idx], out_p, f'{self.data}_inp_{idx}.png')
                self._save_img(omg[idx], out_p, f'{self.data}_out_{idx}.png')
                self._save_img(tms[idx], out_p, f'{self.data}_tru_{idx}.png')
                self._save_img(pms[idx], out_p, f'{self.data}_pre_{idx}.png')
            self._save_img(grid, out_p, f'{self.data}_grid.png')
        self.logger.experiment.add_image(prefix + 'segmentation', grid, self.current_epoch)

    @torch.no_grad()
    def log_grid(self, input, name='grid'):
        if not self.trainer.is_global_zero: return
        img = self._to_img(input)
        nrow = int(np.sqrt(len(input)))
        grid = tv.utils.make_grid(img, pad_value=128, nrow=16).detach().cpu()
        self.logger.experiment.add_image(name, grid, self.current_epoch)

    @torch.no_grad()
    def _cmap_tensor(self, t):
        t_hw = t.cpu().detach().numpy()
        o_hwc = CMAPSPEC['cmap'](t_hw)[...,:3] # drop alpha
        o = torch.from_numpy(o_hwc).transpose(-1,-2).transpose(-2, -3)
        return o

    def _save_img(self, tensor, outp, name):
        o = tensor
        if o.shape[0] <= 4:
            o = o.permute(1,2,0)
        i = Image.fromarray(o.detach().cpu().numpy())
        i.save(str(Path(outp) / name))
