import math
from ool.picture.models.iodine import IODINE

from oolexp import OOLLayeredBoxExp


class LitIODINE(OOLLayeredBoxExp):
    def __init__(self,
                 tag='test',
                 seed=None,
                 data='clevr-crop-(128,128)',
                 batch_size=32,
                 grad_clip=5.0,
                 optim='adam',
                 lr='',
                 max_steps=1000000,
                 learning_rate=3e-4,

                 z_size=128,
                 K=11,
                 inference_iters=5,
                 # refinenet_channels_in=16,
                 lstm_dim=256,
                 conv_channels=64,
                 kl_beta=1,
                 geco_warm_start=1000,
                 std=0.1
                 ):
        super(LitIODINE, self).__init__(seed, 'mse', 'min')

        self.save_hyperparameters()
        self.model = IODINE(
            input_size=self.input_shape,
            batch_size=batch_size,
            z_size=z_size,
            K=K,
            inference_iters=inference_iters,
            # refinenet_channels_in=refinenet_channels_in,
            lstm_dim=lstm_dim,
            conv_channels=conv_channels,
            kl_beta=kl_beta,
            log_scale=math.log(std)
            # geco_warm_start=geco_warm_start
        )

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
    LitIODINE.parse_args_and_execute()
