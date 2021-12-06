import sys

from ool.picture.models.thirdparty.genesis2.model import GenesisV2

from oolexp import OOLLayeredBoxExp

import warnings

class LitGenesis(OOLLayeredBoxExp):
    def __init__(self,
                 tag='test',
                 seed=None,
                 data='clevr-crop-(128,128)',
                 batch_size=32,
                 grad_clip=None,
                 optim='adam',
                 lr=None,
                 max_steps=500000,
                 learning_rate=1e-4,
                 g_goal=0.5655,  # 0.5645 for Sketchy and 0.5655 for others
                 pixel_std=0.7,  # For large dataset
                 ):
        super(LitGenesis, self).__init__(seed, 'mse', 'min')
        self.save_hyperparameters()
        self.model = GenesisV2(g_goal=g_goal, pixel_std=pixel_std)
        print(self.name)

    def training_step(self, batch, batch_idx):
        img, *other = batch
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, instead of relying on the computed output size. If you wish to restore the old behavior, please set recompute_scale_factor=True. See the documentation of nn.Upsample for details.')
            output = self.model(img)
        self.maybe_log_training_outputs(output)
        return output['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        prefix = '' if dataloader_idx is None else f"v{dataloader_idx}/"
        img, *other = batch
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, instead of relying on the computed output size. If you wish to restore the old behavior, please set recompute_scale_factor=True. See the documentation of nn.Upsample for details.')
            output = self.model(img)
        self.maybe_log_validation_outputs(batch, batch_idx, output, prefix)


if __name__ == '__main__':
    print(' '.join(sys.argv))
    LitGenesis.parse_args_and_execute()
