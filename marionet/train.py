"""Train script."""
import os
import logging
from pathlib import Path
import torch as th
import numpy as np
import torch.cuda
from torch.utils.tensorboard import SummaryWriter

import ttools

from marionet import callbacks
from marionet import models
from marionet.interfaces import Interface

from ool.data.dataspec import DataSpec
from ool.utils import exp_log_dir

LOG = logging.getLogger(__name__)

th.backends.cudnn.deterministic = True


def _worker_init_fn(_):
    np.random.seed()


def _set_seed(seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def main(args):
    """Training entrypoint."""
    LOG.info(f"Using seed {args.seed}.")

    device = "cuda" if th.cuda.is_available() and args.cuda else "cpu"
    if th.cuda.is_available():
        torch.cuda.set_device(int(args.gpus))
        device += f":{args.gpus}"
    device = torch.device(device)
    # if args.crop:
    #     Dataset = datasets.RandomCropDataset
    # elif args.sprites:
    #     Dataset = datasets.SpriteDataset
    # else:
    #     Dataset = datasets.AnimationDataset
    # data = Dataset(args.data, args.canvas_size)

    spec = DataSpec(args.data)
    data = spec.get_dataloader(1, "no", subset="train", drop_last=True).dataset
    val_data = spec.get_dataloader(
        1, "no", subset="val", drop_last=True, shuffle=False
    ).dataset

    # data = torch.utils.data.Subset(data, list(range(10)))
    # val_data = torch.utils.data.Subset(val_data, list(range(100)))

    args.checkpoint_dir = Path(exp_log_dir(f"{spec}/marionet/{args.tag}"))
    print(args.checkpoint_dir)
    if not args.background:
        bg_color = data.bg
        background = None
    else:
        bg_color = None
        background = models.Dictionary(
            int(args.background),
            (args.canvas_size, args.canvas_size),
            3,
            bottleneck_size=args.dim_bg,
            no_layernorm=args.no_layernorm,
        )
        background.to(device)
    print(background.latent.shape)
    dataloader = th.utils.data.DataLoader(
        data,
        batch_size=args.bs,
        num_workers=args.num_worker_threads,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = th.utils.data.DataLoader(
        val_data,
        batch_size=args.bs,
        num_workers=args.num_worker_threads,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    _set_seed(args.seed)

    learned_dict = models.Dictionary(
        args.num_classes,
        (
            args.canvas_size // args.layer_size * 2,
            args.canvas_size // args.layer_size * 2,
        ),
        4,
        bottleneck_size=args.dim_z,
    )
    learned_dict.to(device)

    model = models.Model(
        learned_dict,
        args.layer_size,
        args.num_layers,
        canvas_size=args.canvas_size,
        dim_z=args.dim_z,
        dim_bg=args.dim_bg,
        num_bg=int(args.background),
        bg_color=bg_color,
        shuffle_all=args.shuffle_all,
        no_layernorm=args.no_layernorm,
        no_spatial_transformer=args.no_spatial_transformer,
        spatial_transformer_bg=args.spatial_transformer_bg,
        straight_through_probs=args.straight_through_probs,
    )

    interface = Interface(
        model,
        device=device,
        lr=args.lr,
        w_beta=args.w_beta,
        w_probs=args.w_probs,
        lr_bg=args.lr_bg,
        background=background,
    )

    model_checkpointer = ttools.Checkpointer(
        os.path.join(args.checkpoint_dir, "model"), model, optimizers=interface.opt
    )
    if background is not None:
        bg_checkpointer = ttools.Checkpointer(
            os.path.join(args.checkpoint_dir, "bg"),
            background,
            optimizers=interface.opt_bg,
        )

    starting_epoch = None

    if args.load_model is not None:
        model_checkpointer.try_and_init_from(args.load_model)
        if args.load_bg is not None:
            bg_checkpointer.try_and_init_from(args.load_bg)
    else:
        extras, _ = model_checkpointer.load_latest()
        if extras is not None:
            starting_epoch = extras["epoch"]
        if background is not None:
            bg_checkpointer.load_latest()

    trainer = ttools.Trainer(interface)
    writer = SummaryWriter(args.checkpoint_dir)
    interface.writer = writer
    interface.data_len = len(dataloader)

    keys = ["rec_loss", "psnr", "beta_loss", "probs_loss", "psnr_hard"]
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(keys=keys))
    trainer.add_callback(
        ttools.callbacks.TensorBoardLoggingCallback(
            keys=keys, writer=writer, val_writer=writer, frequency=100
        )
    )
    trainer.add_callback(
        ttools.callbacks.CheckpointingCallback(
            model_checkpointer, interval=None, max_epochs=1
        )
    )
    if background is not None:
        trainer.add_callback(
            ttools.callbacks.CheckpointingCallback(
                bg_checkpointer, interval=None, max_epochs=1
            )
        )

    trainer.add_callback(
        callbacks.VizCallback(
            suffix="", writer=writer, val_writer=writer, frequency=2000
        )
    )
    trainer.add_callback(
        callbacks.VizCallback(
            suffix="_hard", writer=writer, val_writer=writer, frequency=2000
        )
    )
    trainer.add_callback(
        callbacks.DictCallback(writer=writer, val_writer=writer, frequency=2000)
    )
    trainer.add_callback(
        callbacks.BackgroundCallback(writer=writer, val_writer=writer, frequency=2000)
    )
    trainer.add_callback(callbacks.InterfaceProgCallback(interface))

    trainer.train(
        dataloader,
        num_epochs=max(1, args.num_steps * args.bs // len(data)),
        val_dataloader=val_dataloader,
        starting_epoch=starting_epoch,
    )


def foo_type_fn(p):
    print(p)
    return int(p)


if __name__ == "__main__":
    parser = ttools.BasicArgumentParser()

    # Representation
    parser.add_argument("--layer_size", type=int, default=8, help="size of anchor grid")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument(
        "--num_classes", type=int, default=150, help="size of dictioanry"
    )
    parser.add_argument(
        "--canvas_size", type=int, default=128, help="spatial size of the canvas"
    )

    # Model
    parser.add_argument("--dim_z", type=int, default=128)
    parser.add_argument("--dim_bg", type=int, default=4)
    parser.add_argument("--load_model", type=str)
    parser.add_argument("--load_bg", type=str)
    parser.add_argument("--no_layernorm", action="store_true", default=False)

    # parser.add_argument("--data", type=str, default='clevr-crop-(128,128)')
    # Training options
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--w_beta", type=float, default=0.002)
    parser.add_argument("--w_probs", type=float, nargs="+", default=[5e-3])
    parser.add_argument("--lr_bg", type=float, default=1e-3)
    parser.add_argument("--shuffle_all", action="store_true", default=False)
    parser.add_argument("--crop", action="store_true", default=False)
    parser.add_argument("--background", type=int, default=1)
    parser.add_argument("--sprites", action="store_true", default=False)
    parser.add_argument("--no_spatial_transformer", action="store_true", default=False)
    parser.add_argument("--spatial_transformer_bg", action="store_true", default=False)
    parser.add_argument("--straight_through_probs", action="store_true", default=False)

    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--gpus", type=foo_type_fn, default=0)

    parser.set_defaults(
        num_worker_threads=4, bs=8, lr=1e-4, data="clevr-crop-(128,128)"
    )

    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)
