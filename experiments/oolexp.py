import torch
import torch.nn.functional as F

from framework.base_experiment import BaseExperiment
from framework.vis_mixin import SpatialVisMixin

from ool.metrics import align_masks_iou, dices, ari, ari2

class OOLBoxExp(BaseExperiment, SpatialVisMixin):
    """For models that predict bounding boxes using some form of STN"""
    def __init__(self, seed, monitor, mode):
        super(OOLBoxExp, self).__init__(seed, monitor, mode)

    # def training_step(self, batch, batch_idx):
    #     batch = self.accelated_batch_postprocessing(batch)
    #     img, *other = batch
    #     output = self.model(img, return_state=False)
    #
    #     # self.log('loss', output['loss'].mean().detach().cpu().item(), prog_bar=False, on_step=True, on_epoch=False, reduce_fx=None)
    #     # self.log('elbo', output['elbo'].mean().detach().cpu().item(), prog_bar=False, on_step=True, on_epoch=False, reduce_fx=None)
    #     # self.log('kl',   output['kl'].mean().detach().cpu().item(), prog_bar=False, on_step=True, on_epoch=False, reduce_fx=None)s
    #     # self.add_scalar_step('loss', output['loss'].mean().detach().cpu())
    #     # self.add_scalar_step('elbo', output['elbo'].mean().detach().cpu())
    #     # self.add_scalar_step('kl', output['elbo'].mean().detach().cpu())
    #
    #
    #     # with torch.no_grad():
    #     #     self.add_scalar_step('loss_comp_ratio', (output['rec_loss']/output['kl']).mean().detach().cpu())
    #     #     for k in output:
    #     #         if k.startswith('kl_') or k.endswith('_loss'):
    #     #             val = output[k]
    #     #             if isinstance(val, torch.Tensor):
    #     #                 val = val.mean().detach().cpu()
    #     #             self.add_scalar_step(k, val)
    #
    #     return output['loss']

    def training_step(self, batch, batch_idx):
        batch = self.accelated_batch_postprocessing(batch)
        img, *other = batch
        output = self.model(img, return_state=False)
        self.log('loss', output['loss'], prog_bar=False, on_epoch=False, sync_dist=self.ddp)
        self.log('elbo', output['elbo'], on_epoch=False, sync_dist=self.ddp)
        self.log('kl', output['kl'], on_epoch=False, sync_dist=self.ddp)
        self.log('slot_cost', output['slot_cost'], on_epoch=False, sync_dist=self.ddp)

        self.log('loss_comp_ratio', output['rec_loss']/output['kl'], prog_bar=False, on_epoch=False)
        for k in output:
            if k.startswith('kl_') or k.endswith('_loss'):
                val = output[k]
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                self.log(k, val, on_epoch=False, sync_dist=self.ddp)

        return output['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        batch = self.accelated_batch_postprocessing(batch)
        prefix = '' if dataloader_idx is None else f"v{dataloader_idx}/"
        img, *other = batch
        self.onceprint(f'input {img.shape}')
        output = self.model(img, return_state=True)

        mse = F.mse_loss(output['canvas'], img, reduction='none').sum((1, 2, 3)).mean().detach()
        self.log(prefix + 'mse', mse, prog_bar=True, on_epoch=True,  sync_dist=self.ddp)

        # Check how much the depth sorting is changing things
        if 'wmask' in output['steps']:
            w_mse = F.mse_loss(output['steps']['mask'], output['steps']['wmask'], reduction='none').sum((-1, -2, -3)).mean(-1).mean().detach()
            self.log(prefix + 'w_mse', w_mse, prog_bar=False, on_epoch=True, sync_dist=self.ddp)

        ali_pmasks = None
        ali_tmasks = None

        if len(other) == 1:
            cnts = other[0]
        elif len(other) >= 2:
            masks, vis, *_ = other
            # Cropping might have moved object out of view
            # cnts = torch.sum(vis, dim=-1) - 1  # -1 for discounting the background from visibility
            cnts = torch.round(masks.to(torch.float)).flatten(2).any(-1).to(torch.float).sum(-1) -1
            pred_masks = output['steps']['mask']
            pred_vis = output['steps']['z_pres'].squeeze(-1)

            ali_pmasks, ali_tmasks, ious, ali_pvis, ali_tvis = align_masks_iou(pred_masks,
                                                                               masks,
                                                                               pred_mask_vis=pred_vis,
                                                                               true_mask_vis=vis,
                                                                               has_bg=False)

            ali_cvis = ali_pvis | ali_tvis
            num_paired_slots = ali_cvis.sum(-1) - 1
            mses = F.mse_loss(output['canvas'][:, None] * ali_pmasks,
                              img[:, None] * ali_tmasks, reduction='none').sum((-1, -2, -3))

            bg_mse = mses[:, 0].mean().detach()
            self.log(prefix + 'bg_mse', bg_mse, prog_bar=False, on_epoch=True, sync_dist=self.ddp)

            slot_mse = (mses[:, 1:].sum(-1) / num_paired_slots).mean().detach()
            self.log(prefix + 'slot_mse', slot_mse, prog_bar=False, on_epoch=True, sync_dist=self.ddp)

            mious = (ious[:, 1:].sum(-1) / num_paired_slots).mean().detach()
            # mious = torch.where(zero_mask, 0., mious)
            self.log(prefix + 'miou', mious, prog_bar=False, on_epoch=True, sync_dist=self.ddp)

            dice = dices(ali_pmasks, ali_tmasks)
            mdice = (dice[:, 1:].sum(-1) / num_paired_slots).mean().detach()
            # mdice = torch.where(zero_mask, 0., mdice)
            self.log(prefix + 'dice', mdice, prog_bar=True, on_epoch=True, sync_dist=self.ddp)

            aris = ari(ali_pmasks, ali_tmasks).mean().detach()
            self.log(prefix + 'ari', aris, prog_bar=False, on_epoch=True, sync_dist=self.ddp)

            aris_fg = ari(ali_pmasks, ali_tmasks, True).mean().detach()
            self.log(prefix + 'ari_fg', aris_fg, prog_bar=False, on_epoch=True, sync_dist=self.ddp)

            # if 'wmask' in output['steps']:
            #     wpred_masks_wbg = torch.cat([output['background_mask'][:, None], output['steps']['wmask']], 1)
            #     ari_2 = ari2(wpred_masks_wbg, masks.to(torch.float))
            #     ari_2fg = ari2(wpred_masks_wbg, masks.to(torch.float), True)
            #     self.log(prefix + 'ari2', ari_2, prog_bar=False, on_epoch=True, sync_dist=self.ddp)
            #     self.log(prefix + 'ari2_fg', ari_2fg, prog_bar=False, on_epoch=True, sync_dist=self.ddp)

        pred_counts = output['counts'].detach().to(torch.int)
        acc = (pred_counts == cnts).to(float).mean().detach()
        self.log(prefix + 'acc', acc, prog_bar=True, on_epoch=True,  sync_dist=self.ddp)
        self.log(prefix + 'cnt', output['counts'].detach().mean(), prog_bar=True, on_epoch=True,  sync_dist=self.ddp)

        if 'mc_expected_canvas' in output:
            mse = F.mse_loss(output['mc_expected_canvas'], img, reduction='none').sum((1, 2, 3)).mean().detach()
            self.log(prefix + 'mc_mse', mse, prog_bar=True, on_epoch=True,  sync_dist=self.ddp)

        if batch_idx == 0 and self.should_log_pictures():
            # if 'heatmap' in output:
            #     probs = tv.utils.make_grid(output['heatmap'][:32])
            #     self.log_as_fig(probs, prefix + 'heatmap')
            # Log input - output w/ bbox pairs
            self.log_images(img,
                            output['canvas'],
                            bboxes=output['steps']['bbox'],
                            pres=output['steps']['z_pres'],
                            prefix=prefix)
            # Log input - output with background/slot breakdown underneath
            # self.log_recons(img[:32], output['canvas'],
            #                 output['steps']['patch'], output['steps']['mask'],
            #                 output.get('background', None),
            #                 output['steps']['where'], output['steps']['z_pres'], output['steps']['z_depth'], prefix=prefix)
            self.log_recons(img[:32],
                            output['canvas'],
                            output['steps']['patch'],
                            output['steps']['mask'],
                            output.get('background', None),
                            pres=output['steps']['z_pres'],
                            prefix=prefix)
            # if 'wmask' in output['steps']:
            #     self.log_recons(img[:32],
            #                     output['canvas'],
            #                     output['steps']['wpatch'],
            #                     output['steps']['wmask'],
            #                     output.get('background', None),
            #                     pres=output['steps']['z_pres'],
            #                     prefix=prefix+'w')
            # If possible log what is seen in each of the slots.
            if 'robj' in output['steps'] and 'rmsk' in output['steps']:
                self.log_slots(output['steps']['robj'][:32] * output['steps']['rmsk'][:32], prefix + 'slots')

            # If masks have been aligned; log semantic map
            if ali_pmasks is not None and ali_tmasks is not None:
                self.log_semantic_images(img[:32], output['canvas'], ali_tmasks, ali_pmasks, prefix=prefix)

            if 'other_canvas' in output:
                self.log_images(img, output['other_canvas'], prefix=prefix + 'other_')
            # if 'mc_expected_canvas' in output:
            #     self.log_images(img, output['mc_expected_canvas'], prefix=prefix + 'mc_')
            # if 'output_samples' in output:
            #     self.log_imgs_grid(img, *output['output_samples'], prefix=prefix)

class OOLLayeredBoxExp(BaseExperiment, SpatialVisMixin):
    def __init__(self, seed, monitor, mode):
        super(OOLLayeredBoxExp, self).__init__(seed, monitor, mode)

    @torch.no_grad()
    def maybe_log_training_outputs(self, output):
        self.log('loss', output['loss'], on_epoch=False)
        if 'elbo' in output:
            self.log('elbo', output['elbo'], on_epoch=False)
        else:
            self.onceprint('WARNING: Output is missing ELBO')
        if 'kl' in output:
            self.log('kl', output['kl'], on_epoch=False)
            if 'rec_loss' in output:
                self.log('loss_comp_ratio', (output['rec_loss'] / output['kl']), on_epoch=False)
        else:
            self.onceprint('WARNING: Output is missing KL')

        for k in output:
            if k.startswith('kl_') or k.endswith('_loss'):
                val = output[k]
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                self.log(k, val, on_epoch=False)

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
            cnts = torch.round(masks.to(torch.float)).flatten(2).any(-1).to(torch.float).sum(-1) -1

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
                    pred_masks_wbg = torch.cat([1-pred_masks.sum(1, keepdim=True), pred_masks], 1)
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
