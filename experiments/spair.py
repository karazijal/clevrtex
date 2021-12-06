import sys

from ool.picture.models.spair import SPAIR, SimpleSPAIR, BroadCastSPAIR
from ool.schedules import Schedule

from oolexp import OOLLayeredBoxExp


class LitSpair(OOLLayeredBoxExp):
    def __init__(self,
                 tag='test',
                 seed=None,
                 data='clevr-crop-(128,128)',
                 batch_size=128,  # Original is 32
                 grad_clip=1.0,
                 optim='adam',
                 lr='plateau',
                 max_steps=250000,
                 learning_rate=1e-4,
                 # Also Pixel per cell is set to 12

                 decoder_bias=0.,
                 what_dim=64,
                 bg_dim=1,

                 prior='itr_lin(10000,1e-4,.99)',
                 temp=1.0,
                 step='set(relaxedbernoulli-bern_kl)',

                 patch_shape=24,
                 anchor=48,

                 # beta_pres='itr_lin(10000,1.0,.5)',
                 # beta_what='itr_lin(10000,1.0,.5)',
                 # beta_where='itr_lin(10000,1.0,.5)',
                 # beta_depth='itr_lin(10000,1.0,.5)',

                 beta_pres=2.7,
                 beta_what=2.7,
                 beta_where=2.7,
                 beta_depth=2.7,

                 pres_kl='original',
                 factor_loss=False,
                 mask=None,
                 rein=False,
                 arch='original',
                 context=1,
                 passf=100,

                 large=True,
                 std=0.15
                 ):
        super(LitSpair, self).__init__(seed, 'mse', 'min')

        self.save_hyperparameters()
        patch_shape = (self.input_shape[0], patch_shape, patch_shape)
        std = Schedule.build(std)
        model_kwargs = dict(
            patch_size=patch_shape,
            anchor=anchor,
            pres_kl=pres_kl,
            passf=passf,
            what_dim=what_dim,
            bg_dim=bg_dim,
            rein=rein,
            factor_loss=factor_loss,
            context_grid_size=context,
            large=large,
            std=std.init
        )

        if arch == 'original':
            self.model = SPAIR(**model_kwargs)
        elif arch == 'simple':
            self.model = SimpleSPAIR(**model_kwargs)
        elif arch == 'broadcast':
            self.model = BroadCastSPAIR(**model_kwargs)

        self.add_scheduled_value(self.model, 'z_pres_prior_p', prior)
        self.add_scheduled_value(self.model, 'z_pres_temperature', temp)
        self.add_scheduled_value(self.model, 'pres_dist_name', step)
        self.add_scheduled_value(self.model, 'output_hparam', std)

        self.add_scheduled_value(self.model, 'beta_pres', beta_pres)
        self.add_scheduled_value(self.model, 'beta_where', beta_where)
        self.add_scheduled_value(self.model, 'beta_depth', beta_depth)
        self.add_scheduled_value(self.model, 'beta_what', beta_what)

        if mask:
            if isinstance(mask, bool):
                mask = int(mask)
            self.add_scheduled_value(self.model, 'mask', mask)

    def training_step(self, batch, batch_idx):
        batch = self.accelated_batch_postprocessing(batch)
        img, *other = batch
        output = self.model(img)
        self.maybe_log_training_outputs(output)
        return output['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        prefix = '' if dataloader_idx is None else f"v{dataloader_idx}/"
        batch = self.accelated_batch_postprocessing(batch)
        img, *other = batch
        output = self.model(img)
        self.maybe_log_validation_outputs(batch, batch_idx, output, prefix)

if __name__ == '__main__':
    print(' '.join(sys.argv))
    LitSpair.parse_args_and_execute()
