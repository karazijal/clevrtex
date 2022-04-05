import sys
import math
import torch

from ool.picture.models.thirdparty.emorl.model import EfficientMORL

from oolexp import OOLLayeredBoxExp


class LitEffMorl(OOLLayeredBoxExp):
    def __init__(
        self,
        tag="test",
        seed=1200,
        data="clevr-crop-(96,96)",
        # Many of these are pased as hparams
        batch_size=32,
        grad_clip=5.0,
        optim="adam",
        lr="decay(10000,100000,0.5)",
        max_steps=500000,
        learning_rate=4e-4,
        use_geco=True,
        num_slots=11,
        z_dim=64,
        refinement_curriculum=[(-1, 3), (100000, 1), (200000, 1)],
        geco_ema_alpha=0.99,  # GECO EMA step parameter
        geco_beta_stepsize=1e-6,  # GECO Lagrange parameter beta,
        kl_beta_init=1,  # kl_beta from beta-VAE,
        geco_reconstruction_target=-108000,
        std=0.1,
    ):
        super(LitEffMorl, self).__init__(seed, "mse", "min")

        self.save_hyperparameters()
        self.model = EfficientMORL(
            K=num_slots,
            z_size=z_dim,
            input_size=self.input_shape[-3:],
            batch_size=batch_size,
            stochastic_layers=3,
            log_scale=math.log(std),
            image_likelihood="Gaussian",
            geco_warm_start=1000,
            refinement_iters=3,
            bottom_up_prior=False,
            reverse_prior_plusplus=True,
            use_geco=True,
            training=self.hparams,
        )
        # self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        img, *other = batch
        # opt = self.optimizers()
        # opt.zero_grad()
        output = self.model(img, self.global_step)
        # self.manual_backward(output['loss'])
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.hparams.grad_clip)
        # opt.step()
        self.model.update_geco(self.global_step, output)
        self.maybe_log_training_outputs(output)

        # This will handle zero_grad/backwards/ddp and the rest of training loop builerplate
        return output["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        prefix = "" if dataloader_idx is None else f"v{dataloader_idx}/"
        img, *other = batch
        # Force graph creation
        with torch.set_grad_enabled(True):
            # self.model.train()
            output = self.model(img, self.global_step)
        self.maybe_log_validation_outputs(batch, batch_idx, output, prefix)


if __name__ == "__main__":
    print(" ".join(sys.argv))
    LitEffMorl.parse_args_and_execute()
