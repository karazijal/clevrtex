import sys

import torch

import ool.picture.models.thirdparty.space.model as spc
from ool.picture.models.thirdparty.space.model import Space

from oolexp import OOLLayeredBoxExp

class MultipleOptimizer(torch.optim.Optimizer):
    def __init__(self, *optimisers):
        self.opts = optimisers
        self.defaults = self.opts[0].defaults
        self.state = self.opts[0].state
        self.param_groups = []
        for opt in self.opts:
            self.param_groups.extend(opt.param_groups)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        return f"Multi:{' '.join(str(opt) for opt in self.opts)}"

    def state_dict(self):
        return {
            'opts': [
                opt.state_dict() for opt in self.opts
            ]
        }

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.opts, state_dict['opt']):
            opt.load_state_dict(sd)

    def zero_grad(self, set_to_none: bool = False):
        for opt in self.opts:
            opt.zero_grad(set_to_none)

    def step(self, closure):
        for opt in self.opts:
            opt.step(closure)

    def add_param_group(self, param_group):
        raise NotImplementedError()


class LitSPACE(OOLLayeredBoxExp):
    def __init__(self,
                 tag='test',
                 seed=None,
                 data='clevr-crop-(128, 128)',
                 batch_size=16,
                 grad_clip=1.0,
                 # learning_rate=1e-4,
                 max_steps=160000,
                 fg_std = 0.15,
                 bg_std = 0.15,
                 ):
        super(LitSPACE, self).__init__(seed, 'mse', 'min')
        self.save_hyperparameters()
        spc.arch.fg_sigma = fg_std
        spc.arch.bg_sigma = bg_std
        self.model = Space()

    def training_step(self, batch, batch_idx):
        batch = self.accelated_batch_postprocessing(batch)
        img, *other = batch
        output = self.model(img, self.trainer.global_step)
        self.maybe_log_training_outputs(output)
        return output['loss']

    def configure_optimizers(self):
        adam = torch.optim.Adam(list(self.model.bg_module.parameters()), lr=1e-3)
        rms = torch.optim.RMSprop(list(self.model.fg_module.parameters()), lr=1e-5)
        return MultipleOptimizer(rms, adam)

    # def trainer_kwargs(self):
    #     return dict(accumulate_grad_batches=3)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        prefix = '' if dataloader_idx is None else f"v{dataloader_idx}/"
        batch = self.accelated_batch_postprocessing(batch)
        img, *other = batch
        output = self.model(img, self.trainer.global_step)
        self.maybe_log_validation_outputs(batch, batch_idx, output, prefix)


if __name__ == '__main__':
    print(' '.join(sys.argv))
    LitSPACE.parse_args_and_execute()
