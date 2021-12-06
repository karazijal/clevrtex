import sys
import torch
import bisect

import numpy as np

from oolexp import OOLLayeredBoxExp

from ool.picture.models.thirdparty.dti_sprites.model import DTISprites, Metrics


class LitDTI(OOLLayeredBoxExp):
    def __init__(self,
                 tag='test',
                 seed=None,
                 data='clevr-crop-(128,128)',
                 batch_size=32,
                 grad_clip=None,
                 optim='adam',
                 lr=None,
                 epochs=450,
                 learning_rate=1e-4,
                 slots=10,
                 n_bg=1,
                 complex=False
                 ):
        super(LitDTI, self).__init__(seed, 'mse', 'min')

        self.save_hyperparameters()
        self.check_cluster_interval = 250
        kwargs = {
            'name': 'dti_sprites',
            'n_sprites': slots,
            'n_backgrounds': n_bg,
            'n_objects': slots, # TODO: how does sprite-to-slot mapping work?
            'freeze_sprite': 40,
            'inject_noise': 0.4,
            'encoder_name': 'resnet18',
            'with_pool': [2, 2],
            'transformation_sequence': 'identity_projective' if not complex else 'color_projective',
            'transformation_sequence_bkg': 'color' if not complex else 'color_projective',
            'transformation_sequence_layer': 'color_position',
            'curriculum_learning': [150],
            'curriculum_learning_bkg': False if not complex else [150],
            'curriculum_learning_layer': False,
            'proto_init': 'constant',
            'mask_init': 'gaussian',
            'bkg_init': 'mean',
            'sprite_size': [40, 40],
            'gaussian_weights_std': 10,
            'pred_occlusion': True,
            'estimate_minimum': True,
            'greedy_algo_iter': 3,
            'add_empty_sprite': True,
            'lambda_empty_sprite': 1.0e-4
        }

        self.milestones = [250, 400]
        self.gamma = [1, 0.1]
        dl = self.train_dataloader()
        bg_mean = []
        have = 0
        for img, *rest in dl:
            bg_mean.append(img)
            have += img.shape[0]
            if have >= 100*n_bg:
                break
        bg_mean = torch.cat(bg_mean, dim=0)[:100*n_bg]
        bg_mean = bg_mean.view(n_bg, 100, *bg_mean.shape[-3:]).mean(1)
        print(bg_mean.shape)
        self.model = DTISprites(img_size=self.input_shape[-2:], n_ch=self.input_shape[0], mean_bg=bg_mean, **kwargs)
        self.pred_class = getattr(self.model, 'pred_class', False) or getattr(self.model, 'estimate_minimum', False)
        self.n_prototypes = self.model.n_prototypes
        self.n_backgrounds = getattr(self.model, 'n_backgrounds', 0)
        self.n_objects = max(self.model.n_objects, 1)
        if self.pred_class:
            self.n_clusters = self.n_prototypes * self.n_objects
        else:
            self.n_clusters = self.n_prototypes ** self.n_objects * max(self.n_backgrounds, 1)
        self.learn_masks = getattr(self.model, 'learn_masks', False)
        self.learn_backgrounds = getattr(self.model, 'learn_backgrounds', False)

        metric_names = ['time/img', 'loss']
        metric_names += [f'prop_clus{i}' for i in range(self.n_clusters)]
        # train_iter_interval = cfg["training"]["train_stat_interval"]
        # self.train_stat_interval = train_iter_interval
        self.train_metrics = Metrics(*metric_names)

    def on_load_checkpoint(self, checkpoint) -> None:
        # print(checkpoint.keys())
        # print(checkpoint['state_dict'].keys())
        # model_state_dict = {}
        for k,v in checkpoint['state_dict'].items():
            if 'activations' in k:
                v = torch.ones_like(v).to(v)  # force true
                checkpoint[k] = v
            if 'mask_params' in k:
                v = v.clone()
                checkpoint[k] = v

        # self.model.mask_params = torch.nn.Parameter(self.model.mask_params.clone())

        if 'dti_train_metrics' in checkpoint:
            self.train_metrics = self.train_metrics.load_from_state_dict(checkpoint['dti_train_metrics'])
        else:
            print("Missing dti_train_metrics")

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['dti_train_metrics'] = self.train_metrics.state_dict()


    # @classmethod
    # def load_from_checkpoint(
    #     cls,
    #     checkpoint_path: Union[str, IO],
    #     map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
    #     hparams_file: Optional[str] = None,
    #     strict: bool = True,
    #     **kwargs,
    # ):
    def training_step(self, batch, batch_idx):
        # if self.current_epoch in self.milestones:
        #     opt = self.trainer.optimizers[0]
        #     for i,g in enumerate(opt.param_groups):
        #         g['lr'] *= self.gamma[i]
        img, *other = batch
        output = self.model(img)
        distances = output['distance']
        B = img.shape[0]
        with torch.no_grad():
            if self.pred_class:
                proportions = (1 - distances).mean(0)
            else:
                argmin_idx = distances.min(1)[1]
                one_hot = torch.zeros(B, distances.size(1), device=self.device).scatter(1, argmin_idx[:, None], 1)
                proportions = one_hot.sum(0) / B
        self.train_metrics.update({f'prop_clus{i}': p.item() for i, p in enumerate(proportions)})

        if self.global_step % self.check_cluster_interval == 0:
            self.check_cluster(self.global_step, self.current_epoch, None)

        self.maybe_log_training_outputs(output)

        return output['loss']

    def lr_scheduler(self, opt):
        def lr_lambda_0(step):
            if step < 1000:
                return float(step) / float(1000)
            pos = bisect.bisect(self.milestones, step //  self.data_len)
            return self.gamma[0] ** pos
        def lr_lambda_1(step):
            if step < 1000:
                return float(step) / float(1000)
            pos = bisect.bisect(self.milestones, step //  self.data_len)
            return self.gamma[1] ** pos
        return torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda_0, lr_lambda_1]), 'step'

    def configure_optimizers(self):
        r = super(LitDTI, self).configure_optimizers()
        opt = r['optimizer']
        print(type(opt), opt)
        self.model.set_optimizer(opt)
        return r

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        prefix = '' if dataloader_idx is None else f"v{dataloader_idx}/"
        img, *other = batch
        output = self.model(img)
        self.maybe_log_validation_outputs(batch, batch_idx, output, prefix)

    def on_train_epoch_end(self, outputs) -> None:
        self.print("Model step")
        self.model.step()

    def check_cluster(self, cur_iter, epoch, batch):
        # if hasattr(self.model, '_diff_selections') and self.visualizer is not None:
        #     diff = self.model._diff_selections
        #     x, y = [[cur_iter] * len(diff[0])], [diff[1]]
        #     self.visualizer.line(y, x, win='diff selection', update='append', opts=dict(title='diff selection',
        #                                                                                 legend=diff[0], width=VIZ_WIDTH,
        #                                                                                 height=VIZ_HEIGHT))
        self.print('Checking clusters')
        proportions = torch.Tensor([self.train_metrics[f'prop_clus{i}'].avg for i in range(self.n_clusters)])
        if self.n_backgrounds > 1:
            proportions = proportions.view(self.n_prototypes, self.n_backgrounds)
            for axis, is_bkg in zip([1, 0], [False, True]):
                prop = proportions.sum(axis)
                reassigned, idx = self.model.reassign_empty_clusters(prop, is_background=is_bkg)
                # msg = PRINT_CHECK_CLUSTERS_FMT(epoch, self.n_epoches, batch, self.n_batches, reassigned, idx)
                # if is_bkg:
                #     msg += ' for backgrounds'
                # self.print_and_log_info(msg)
                # self.print_and_log_info(', '.join(['prop_{}={:.4f}'.format(k, prop[k]) for k in range(len(prop))]))
        elif self.n_objects > 1:
            k = np.random.randint(0, self.n_objects)
            if self.n_clusters == self.n_prototypes ** self.n_objects:
                prop = proportions.view((self.n_prototypes,) * self.n_objects).transpose(0, k).flatten(1).sum(1)
            else:
                prop = proportions.view(self.n_objects, self.n_prototypes)[k]
            reassigned, idx = self.model.reassign_empty_clusters(prop)
            # msg = PRINT_CHECK_CLUSTERS_FMT(epoch, self.n_epoches, batch, self.n_batches, reassigned, idx)
            # msg += f' for object layer {k}'
            # self.print_and_log_info(msg)
            # self.print_and_log_info(', '.join(['prop_{}={:.4f}'.format(k, prop[k]) for k in range(len(prop))]))
        else:
            reassigned, idx = self.model.reassign_empty_clusters(proportions)
            # msg = PRINT_CHECK_CLUSTERS_FMT(epoch, self.n_epoches, batch, self.n_batches, reassigned, idx)
            # self.print_and_log_info(msg)
        self.train_metrics.reset(*[f'prop_clus{i}' for i in range(self.n_clusters)])


if __name__ == '__main__':
    print(' '.join(sys.argv))
    LitDTI.parse_args_and_execute()
