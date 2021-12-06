"""Model training interface."""
import torch
import torch as th
from torch.distributions import Beta
from torch.nn import functional as F
import ttools
from ttools.modules import image_operators as imops

from ool.metrics import align_masks_iou, dices, ari, ari2
from experiments.framework.vis_mixin import SpatialVisMixin

LOG = ttools.get_logger(__name__)

class Redirect:
    def __init__(self, ref):
        self.ref = ref
        self.is_global_zero = True

    @property
    def experiment(self):
        return self.ref.writer

class Interface(ttools.ModelInterface, SpatialVisMixin):
    def __init__(self, model, device="cpu", lr=1e-4, w_beta=0, w_probs=0,
                 lr_bg=None, background=None):
        self.model = model

        #some random bits to make reuing the code a bit easier
        self.logger = Redirect(self)
        self.trainer = Redirect(self)
        self.ddp = False
        self._prints = set()

        if lr_bg is None:
            lr_bg = lr

        self.opt = th.optim.AdamW(model.parameters(), lr=lr)

        self.device = device
        self.model.to(device)
        self.background = background
        if background is not None:
            self.background.to(device)
            self.opt_bg = th.optim.AdamW(
                self.background.parameters(), lr=lr_bg)
        else:
            self.opt_bg = None

        self.w_beta = w_beta
        self.w_probs = w_probs

        self.beta = Beta(th.tensor(2.).to(device), th.tensor(2.).to(device))

        self.loss = th.nn.MSELoss()
        self.loss.to(device)

    def forward(self, im, hard=False):
        if self.background is not None:
            bg = self.background()
        else:
            bg = None
        return self.model(im, bg, hard=hard)

    def onceprint(self, *args, **kwargs):
        """Just a useful debug function to see shapes when fisrt running"""
        k = '_'.join(str(a) for a in args)
        if k not in self._prints:
            print(*args, **kwargs)
            self._prints.add(k)

    def should_log_pictures(self):
        self.onceprint(f"WARNING: will downsample the picture logging to concerve storage")
        if self.current_epoch < 10: # log early stages
            return True
        if self.current_epoch < 50:
            return self.current_epoch % 3 == 0 # log every 4
        if self.current_epoch < 100:
            return self.current_epoch % 5 == 0 # log every 5
        return self.current_epoch % 10 == 9 # log every 10

    def training_step(self, batch):
        im = batch[0].to(self.device)
        # im = batch["im"].to(self.device)
        fwd_data = self.forward(im)
        fwd_data_hard = self.forward(im, hard=True)

        out = fwd_data["reconstruction"]
        layers = fwd_data["layers"]
        out_hard = fwd_data_hard["reconstruction"]
        layers_hard = fwd_data_hard["layers"]
        im = imops.crop_like(im, out)

        l = fwd_data['l']
        learned_dict = fwd_data["dict"]
        dict_codes = fwd_data["dict_codes"]
        im_codes = fwd_data["im_codes"]
        weights = fwd_data["weights"]
        probs = fwd_data["probs"]

        rec_loss = self.loss(out, im)
        beta_loss = (self.beta.log_prob(
            weights.clamp(1e-5, 1-1e-5)).exp().mean() + self.beta.log_prob(
                probs.clamp(1e-5, 1-1e-5)).exp().mean()) / 2

        probs_loss = probs.abs()

        self.opt.zero_grad()
        if self.opt_bg is not None:
            self.opt_bg.zero_grad()

        w_probs = th.tensor(self.w_probs).to(probs_loss)[None, :, None] \
            .expand_as(probs_loss)
        loss = rec_loss + self.w_beta * beta_loss + \
            (w_probs * probs_loss).mean()

        loss.backward()

        self.opt.step()
        if self.opt_bg is not None:
            self.opt_bg.step()

        with th.no_grad():
            psnr = -10*th.log10(F.mse_loss(out, im))
            psnr_hard = -10*th.log10(F.mse_loss(out_hard, im))

        self.writer.add_scalar('loss', loss.detach().item(), global_step=self.global_step)

        return {
            "rec_loss": rec_loss.item(),
            "beta_loss": beta_loss.item(),
            "psnr": psnr.item(),
            "psnr_hard": psnr_hard.item(),
            "out": out.detach(),
            "layers": layers.detach(),
            "out_hard": out_hard.detach(),
            "layers_hard": layers_hard.detach(),
            "dict": learned_dict.detach(),
            "probs_loss": probs_loss.mean().item(),
            "im_codes": im_codes.detach(),
            "dict_codes": dict_codes.detach(),
            "background": fwd_data["background"].detach(),
            "loss": loss.detach(),
            "l": l,
        }

    def validation_step(self, batch, running_val_data):
        batch = [b.to(self.device) for b in batch]
        im = batch[0].to(self.device)
        fwd_data = self.forward(im)
        fwd_data_hard = self.forward(im, hard=True)

        out = fwd_data["reconstruction"]
        layers = fwd_data["layers"]
        out_hard = fwd_data_hard["reconstruction"]
        layers_hard = fwd_data_hard["layers"]
        im = imops.crop_like(im, out)

        l = fwd_data['l']
        learned_dict = fwd_data["dict"]
        dict_codes = fwd_data["dict_codes"]
        im_codes = fwd_data["im_codes"]
        weights = fwd_data["weights"]
        probs = fwd_data["probs"]

        rec_loss = self.loss(out, im)
        beta_loss = (self.beta.log_prob(
            weights.clamp(1e-5, 1 - 1e-5)).exp().mean() + self.beta.log_prob(
            probs.clamp(1e-5, 1 - 1e-5)).exp().mean()) / 2

        probs_loss = probs.abs()


        w_probs = th.tensor(self.w_probs).to(probs_loss)[None, :, None] \
            .expand_as(probs_loss)
        loss = rec_loss + self.w_beta * beta_loss + \
               (w_probs * probs_loss).mean()

        with th.no_grad():
            psnr = -10 * th.log10(F.mse_loss(out, im))
            psnr_hard = -10 * th.log10(F.mse_loss(out_hard, im))

        output = {
            'canvas': fwd_data['reconstruction'],
            'other_canvas': fwd_data_hard['reconstruction'],
            'background': fwd_data['bg'],
            'other_background': fwd_data_hard['bg'],
            'layers': fwd_data['l']
        }
        for k, v in output.items():
            if isinstance(v, dict):
                for kk,vv in v.items():
                    self.onceprint(f'{k}.{kk}: {vv.shape}')
            else:
                self.onceprint(f'{k}: {v.shape}')
        self.maybe_log_validation_outputs(batch, self.batch_idx, output, '')
        self.writer.add_scalar('loss', loss.detach().item(), global_step=self.global_step)

        return {
            "rec_loss": rec_loss.item(),
            "beta_loss": beta_loss.item(),
            "psnr": psnr.item(),
            "psnr_hard": psnr_hard.item(),
            "out": out.detach(),
            "layers": layers.detach(),
            "out_hard": out_hard.detach(),
            "layers_hard": layers_hard.detach(),
            "dict": learned_dict.detach(),
            "probs_loss": probs_loss.mean().item(),
            "im_codes": im_codes.detach(),
            "dict_codes": dict_codes.detach(),
            "background": fwd_data["background"].detach(),
            "loss": loss.detach(),
            "l": l,
        }

    def log(self, name, value, on_step=None, on_epoch=None, sync_dist=None, **kwargs):
        # on_step = self._LightningModule__auto_choose_log_on_step(on_step)  # Get around name mangling
        # on_epoch = self._LightningModule__auto_choose_log_on_epoch(on_epoch)  # Get around name mangling
        # if sync_dist is None:
        #     sync_dist = self.ddp
        # if self.cpu_metrics:
        n = 1
        if isinstance(value, torch.Tensor):
            if len(value.shape):
                n = value.shape[0]
            value = value.mean().detach().cpu().item()
        self.writer.add_scalar(name, value, global_step=self.global_step)

    def maybe_log_validation_outputs(self, batch, batch_idx, output, prefix=''):
        img, *other = batch
        mse = F.mse_loss(output['canvas'], img, reduction='none').sum((1, 2, 3))
        self.log(prefix + 'mse', mse, prog_bar=True, on_epoch=True)

        ali_pmasks = None
        ali_tmasks = None

        if len(other) == 1:
            cnts = other[0]
        elif len(other) == 2:
            masks, vis = other
            # Transforms might have changed this.
            # cnts = torch.sum(vis, dim=-1) - 1  # -1 for discounting the background from visibility
            # estimate from masks
            cnts = torch.round(masks.to(torch.float)).flatten(2).any(-1).to(torch.float).sum(-1) - 1

            if 'steps' in output or 'layers' in output:
                if 'steps' in output:
                    pred_masks = output['steps']['mask']
                    pred_vis = output['steps']['z_pres'].squeeze(-1)
                    has_bg = False
                elif 'layers' in output:
                    pred_masks = output['layers']['mask']
                    pred_vis = pred_masks.new_ones(pred_masks.shape[:2])
                    has_bg = True

                ali_pmasks, ali_tmasks, ious, ali_pvis, ali_tvis = align_masks_iou(pred_masks,
                                                                                   masks,
                                                                                   pred_mask_vis=pred_vis,
                                                                                   true_mask_vis=vis,
                                                                                   has_bg=has_bg)

                ali_cvis = ali_pvis | ali_tvis
                num_paired_slots = ali_cvis.sum(-1) - 1
                mses = F.mse_loss(output['canvas'][:, None] * ali_pmasks,
                                  img[:, None] * ali_tmasks, reduction='none').sum((-1, -2, -3))

                bg_mse = mses[:, 0]
                self.log(prefix + 'bg_mse', bg_mse, prog_bar=False, on_epoch=True)

                slot_mse = mses[:, 1:].sum(-1) / num_paired_slots
                self.log(prefix + 'slot_mse', slot_mse, prog_bar=False, on_epoch=True)

                mious = ious[:, 1:].sum(-1) / num_paired_slots
                # mious = torch.where(zero_mask, 0., mious)
                self.log(prefix + 'miou', mious, prog_bar=False, on_epoch=True)

                dice = dices(ali_pmasks, ali_tmasks)
                mdice = dice[:, 1:].sum(-1) / num_paired_slots
                # mdice = torch.where(zero_mask, 0., mdice)
                self.log(prefix + 'dice', mdice, prog_bar=True, on_epoch=True)

                aris = ari(ali_pmasks, ali_tmasks)
                self.log(prefix + 'ari', aris, prog_bar=False, on_epoch=True)

                aris_fg = ari(ali_pmasks, ali_tmasks, True).mean().detach()
                self.log(prefix + 'ari_fg', aris_fg, prog_bar=False, on_epoch=True, sync_dist=self.ddp)

                if not has_bg:
                    pred_masks_wbg = torch.cat([1 - pred_masks.sum(1, keepdim=True), pred_masks], 1)
                else:
                    pred_masks_wbg = pred_masks
                ari_2 = ari2(pred_masks_wbg, masks.to(torch.float))
                ari_2fg = ari2(pred_masks_wbg, masks.to(torch.float), True)
                self.log(prefix + 'ari2', ari_2, prog_bar=False, on_epoch=True, sync_dist=self.ddp)
                self.log(prefix + 'ari2_fg', ari_2fg, prog_bar=False, on_epoch=True, sync_dist=self.ddp)

            else:
                self.onceprint("WARNING: Model returned neither 'steps' neither 'layers' in output")

        pred_counts = None
        if 'counts' in output:
            pred_counts = output['counts'].detach().to(torch.int)
        elif 'layers' in output:
            self.onceprint('WARNING: Estimating counts from layer masks')
            pred_counts = (output['layers']['mask'] > .5).flatten(2).any(-1).detach().to(torch.int).sum(-1) - 1
        if pred_counts is not None:
            self.log(prefix + 'acc', (pred_counts == cnts).to(float), prog_bar=True, on_epoch=True)
            self.log(prefix + 'cnt', pred_counts.to(float).mean(), prog_bar=True, on_epoch=True)
        else:
            self.onceprint('WARNING: No counts or layer masks in output; cannot track ACCURACY')

        if 'mc_expected_canvas' in output:
            mse = F.mse_loss(output['mc_expected_canvas'], img, reduction='none').sum((1, 2, 3))
            self.log(prefix + 'mc_mse', mse, prog_bar=True, on_epoch=True)

        if batch_idx == 0 and self.should_log_pictures():
            # if 'heatmap' in output:
            #     probs = tv.utils.make_grid(output['heatmap'][:32])
            #     self.log_as_fig(probs, prefix + 'heatmap')
            # Log input - output w/ bbox pairs
            if 'steps' in output and 'bbox' in output['steps']:
                self.log_images(img,
                                output['canvas'],
                                bboxes=output['steps']['bbox'],
                                pres=output['steps']['z_pres'],
                                prefix=prefix)
            elif 'canvas_with_bbox' in output:
                self.log_images(img, output['canvas_with_bbox'], prefix=prefix)
            else:
                self.onceprint("WARNING: Neither steps with bbox or canvas_with_bbox found in output")
            # Log input - output with background/slot breakdown underneath
            # self.log_recons(img[:32], output['canvas'],
            #                 output['steps']['patch'], output['steps']['mask'],
            #                 output.get('background', None),
            #                 output['steps']['where'], output['steps']['z_pres'], output['steps']['z_depth'], prefix=prefix)
            if 'steps' in output:
                self.log_recons(img[:32],
                                output['canvas'],
                                output['steps']['patch'],
                                output['steps']['mask'],
                                output.get('background', None),
                                pres=output['steps']['z_pres'],
                                prefix=prefix + 's_' if 'layers' in output else '')
                # If possible log what is seen in each of the slots.
                if 'robj' in output['steps'] and 'rmsk' in output['steps']:
                    self.log_slots(output['steps']['robj'][:32] * output['steps']['rmsk'][:32], prefix + 'slots')

            if 'layers' in output:
                self.log_recons(img[:32],
                                output['canvas'],
                                output['layers']['patch'],
                                output['layers']['mask'],
                                output.get('background', None),
                                prefix=prefix + 'l_' if 'steps' in output else '')

            # If masks have been aligned; log semantic map
            if ali_pmasks is not None and ali_tmasks is not None:
                self.log_semantic_images(img[:32], output['canvas'], ali_tmasks, ali_pmasks, prefix=prefix)

            if 'other_canvas' in output:
                self.log_images(img, output['other_canvas'], prefix=prefix + 'other_')
            if 'mc_expected_canvas' in output:
                self.log_images(img, output['mc_expected_canvas'], prefix=prefix + 'mc_')
            if 'output_samples' in output:
                self.log_imgs_grid(img, *output['output_samples'], prefix=prefix)
