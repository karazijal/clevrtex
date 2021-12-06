import sys

from ool.picture.models.thirdparty.gnm.model import GNM, arrow_args, hyperparam_anneal

from oolexp import OOLLayeredBoxExp


class LitGNM(OOLLayeredBoxExp):
    def __init__(self,
                 tag='test',
                 seed=None,
                 data='clevr-crop-(128,128)',
                 batch_size=32,
                 grad_clip=1.0,
                 optim='rmsprop',
                 lr=None,
                 # epochs=600,
                 max_steps=300000,
                 learning_rate=1e-4,
                 std=0.2,
                 z_what_dim =64,
                 z_bg_dim = 10
                 ):
        super(LitGNM, self).__init__(seed, 'mse', 'min')

        self.save_hyperparameters()
        args = hyperparam_anneal(arrow_args, 0)
        args.const.likelihood_sigma = std
        args.z.z_what_dim = z_what_dim
        args.z.z_bg_dim = z_bg_dim
        self.model = GNM(args)

    def training_step(self, batch, batch_idx):
        batch = self.accelated_batch_postprocessing(batch)
        img, *other = batch
        output = self.model(img, self.trainer.global_step)
        self.maybe_log_training_outputs(output)
        return output['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        prefix = '' if dataloader_idx is None else f"v{dataloader_idx}/"
        batch = self.accelated_batch_postprocessing(batch)
        img, *other = batch
        output = self.model(img, self.trainer.global_step)
        self.maybe_log_validation_outputs(batch, batch_idx, output, prefix)


if __name__ == '__main__':
    print(' '.join(sys.argv))
    LitGNM.parse_args_and_execute()
