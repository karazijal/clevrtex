import sys

import torch
import torch.nn.functional as F
import torchvision as tv

import fire

from ool.picture.models.monet import MONet

from oolexp import OOLLayeredBoxExp


class LitMONet(OOLLayeredBoxExp):
    def __init__(
        self,
        tag="test",
        seed=None,
        data="clevr-crop-(128,128)",
        batch_size=64,
        grad_clip=None,
        optim="rmsprop",
        lr="",
        max_steps=1000000,
        learning_rate=1e-4,
        z_dim=16,
        n_slots=11,  # Other datasets used 7
        n_blocks=6,  # Other datasets used 5
        bg_std=0.09,
        fg_std=0.15,
    ):
        super(LitMONet, self).__init__(seed, "mse", "min")

        self.save_hyperparameters()
        self.model = MONet(
            n_slots=n_slots,
            numb=n_blocks,
            shape=self.input_shape,
            z_dim=z_dim,
            bg_scl=bg_std,
            fg_scl=fg_std,
        )

    def training_step(self, batch, batch_idx):
        batch = self.accelated_batch_postprocessing(batch)
        img, *other = batch
        output = self.model(img)
        self.maybe_log_training_outputs(output)
        return output["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        prefix = "" if dataloader_idx is None else f"v{dataloader_idx}/"
        batch = self.accelated_batch_postprocessing(batch)
        img, *other = batch
        output = self.model(img)
        self.maybe_log_validation_outputs(batch, batch_idx, output, prefix)


if __name__ == "__main__":
    LitMONet.parse_args_and_execute()
