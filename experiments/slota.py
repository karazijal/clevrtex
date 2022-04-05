import sys

from ool.picture.models.thirdparty.slot_attention.model import SlotAttentionAutoEncoder

from oolexp import OOLLayeredBoxExp


class LitSlot(OOLLayeredBoxExp):
    def __init__(
        self,
        tag="test",
        seed=None,
        data="clevr-custom",
        batch_size=32,
        grad_clip=None,
        optim="adam",
        lr="decay(10000,100000,0.5)",
        max_steps=500000,
        learning_rate=4e-4,
        slots=11,
        num_iter=3,
    ):
        super(LitSlot, self).__init__(seed, "mse", "min")

        self.save_hyperparameters()
        self.model = SlotAttentionAutoEncoder(
            self.input_shape, num_slots=slots, num_iterations=num_iter
        )

    def training_step(self, batch, batch_idx):
        img, *other = batch
        output = self.model(img)
        self.maybe_log_training_outputs(output)
        return output["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        prefix = "" if dataloader_idx is None else f"v{dataloader_idx}/"
        img, *other = batch
        output = self.model(img)
        self.maybe_log_validation_outputs(batch, batch_idx, output, prefix)


if __name__ == "__main__":
    print(" ".join(sys.argv))
    LitSlot.parse_args_and_execute()
