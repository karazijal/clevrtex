import itertools
import functools
import random
from datetime import datetime
from pathlib import Path
import shutil
import warnings
import gc
import datetime

import os
import pytorch_lightning.plugins.environments
from matplotlib import pyplot as plt

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler
from pprint import pformat

from . import strictfire
from ool.data import DataSpec
from ool.utils import exp_log_dir, watermark_source
from ool.schedules import ScheduledValue
import ool.env

# Monkeypatch fixes for memory consumption
import pytorch_lightning.trainer.training_loop as loop


def fixed_update_running_loss(self):
    accumulated_loss = self.accumulated_loss.mean()

    if accumulated_loss is not None:
        next_loss = self.accumulated_loss.mean() * self.trainer.accumulate_grad_batches
        if self.trainer.move_metrics_to_cpu:
            next_loss = next_loss.cpu()
        # calculate running loss for display
        self.running_loss.append(next_loss)

    # reset for next set of accumulated grads
    self.accumulated_loss.reset()


loop.TrainLoop.update_running_loss = fixed_update_running_loss

import pytorch_lightning.trainer.connectors.logger_connector as logcon
from pytorch_lightning.core.step_result import Result


def fixed_cache_training_step_metrics(self, opt_closure_result):
    """
    This function is responsible to update
    logger_connector internals metrics holder based for depreceated logging
    """
    using_results_obj = isinstance(opt_closure_result.training_step_output, Result)

    # temporary dict to collect metrics
    logged_metrics_tmp = {}
    pbar_metrics_tmp = {}
    callback_metrics_tmp = {}

    if using_results_obj:
        batch_log_metrics = opt_closure_result.training_step_output_for_epoch_end.get_batch_log_metrics(
            include_forked_originals=False
        )
        logged_metrics_tmp.update(batch_log_metrics)

        batch_pbar_metrics = opt_closure_result.training_step_output_for_epoch_end.get_batch_pbar_metrics(
            include_forked_originals=False
        )
        pbar_metrics_tmp.update(batch_pbar_metrics)

        forked_metrics = (
            opt_closure_result.training_step_output_for_epoch_end.get_forked_metrics()
        )
        callback_metrics_tmp.update(forked_metrics)
        callback_metrics_tmp.update(logged_metrics_tmp)

    else:
        batch_log_metrics = (
            opt_closure_result.training_step_output_for_epoch_end.log_metrics
        )
        logged_metrics_tmp.update(batch_log_metrics)

        batch_pbar_metrics = (
            opt_closure_result.training_step_output_for_epoch_end.pbar_on_batch_end
        )
        pbar_metrics_tmp.update(batch_pbar_metrics)

    # track progress bar metrics
    if len(pbar_metrics_tmp) > 0:
        self.add_progress_bar_metrics(pbar_metrics_tmp)

    self._callback_metrics.update(callback_metrics_tmp)

    # save legacy log metrics
    self._logged_metrics.update(logged_metrics_tmp)
    self.cached_results.legacy_batch_log_metrics.update(logged_metrics_tmp)


logcon.LoggerConnector.cache_training_step_metrics = fixed_cache_training_step_metrics


class BaseExperiment(pl.LightningModule):
    """Base experiment module that handles running, training, checkpointing etc."""

    def log(self, name, value, on_step=None, on_epoch=None, sync_dist=None, **kwargs):
        on_step = self._LightningModule__auto_choose_log_on_step(
            on_step
        )  # Get around name mangling
        on_epoch = self._LightningModule__auto_choose_log_on_epoch(
            on_epoch
        )  # Get around name mangling
        if sync_dist is None:
            sync_dist = self.ddp
        if self.cpu_metrics:
            if isinstance(value, torch.Tensor):
                value = value.mean().detach().cpu()
        super(BaseExperiment, self).log(
            name,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
            **kwargs,
        )
        if self.cpu_metrics and self._results is not None:
            self._results.cpu()

    @staticmethod
    def handle_experiment_seed(seed):
        if seed is None:
            print("No seed passed setting 0; use 'random' for a random", 0)
            pl.seed_everything(0)
        elif seed == "random":
            seed = np.random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
            pl.seed_everything(seed)
            print(f"Using random seed ", seed)
        else:
            print(f"Using specified seed value ", seed)
            pl.seed_everything(seed)
        return seed

    @property
    def name(self):
        data_str = ""
        if hasattr(self, "data"):
            data_str = str(self.data) + "/"
        model_arch_str = ""
        if hasattr(self.model, "archid"):
            model_arch_str = self.model.archid + "-"
        return f"{data_str}{self.model.shortname}/{model_arch_str}{self.hparams.tag}"

    def __init__(self, seed, monitor, mode):
        super(BaseExperiment, self).__init__()
        self.seed = self.handle_experiment_seed(seed)
        ool.env.env_report()
        self.monitor = monitor
        self.mode = mode
        self.workers = 0
        self.drop_last_batch = False
        self._data_len = None
        self.__epo_scheduledvs = []
        self.__itr_scheduledvs = []
        self._prints = set()
        self.nowatermark = False

    @property
    def data_len(self):
        if (
            hasattr(self, "trainer")
            and isinstance(self.trainer, pl.Trainer)
            and hasattr(self.trainer, "num_training_batches")
        ):
            try:
                data_len = int(self.trainer.num_training_batches)
                if data_len > 0:
                    return data_len
            except ValueError:
                pass
        if self._data_len is None:
            warnings.warn(
                "Check for data_len requested but data has not been loaded yet -- forcing. Avoid this by delaying check logic"
            )
            self._data_len = len(self.train_dataloader())
        return self._data_len

    def onceprint(self, *args, **kwargs):
        """Just a useful debug function to see shapes when fisrt running"""
        k = "_".join(str(a) for a in args)
        if k not in self._prints:
            self.print(*args, **kwargs)
            self._prints.add(k)

    def add_scheduled_value(self, object, name, schedule):
        v = ScheduledValue(object, name, schedule)
        if v.is_itr:
            self.__itr_scheduledvs.append(v)
        else:
            self.__epo_scheduledvs.append(v)

    def update_epoch_scheduled_values(self, epoch, total_epoch, should_log=True):
        for v in self.__epo_scheduledvs:
            x = v.update(epoch, total_epoch)
            if x is not None and should_log:
                self.logger.experiment[0].add_scalar(v.name, x, self.current_epoch)
                # self.log(v.name, x, on_step=False, on_epoch=True)

    def update_itera_scheduled_values(self, step, total_steps, should_log=True):
        for v in self.__itr_scheduledvs:
            x = v.update(step, total_steps)
            if x is not None and should_log:
                self.logger.experiment[0].add_scalar(v.name, x, self.global_step)
                # self.log(v.name, x, on_step=True, on_epoch=False)

    def should_log_pictures(self):
        self.onceprint(
            f"WARNING: will downsample the picture logging to concerve storage"
        )
        if self.current_epoch < 10:  # log early stages
            return True
        if self.current_epoch < 50:
            return self.current_epoch % 3 == 0  # log every 4
        if self.current_epoch < 100:
            return self.current_epoch % 5 == 0  # log every 5
        return self.current_epoch % 10 == 9  # log every 10

    def scheduled_values_summary(self):
        summary = []
        for v in self.__itr_scheduledvs:
            summary.append(str(v))
        for v in self.__epo_scheduledvs:
            summary.append(str(v))
        return "\n".join(summary)

    # def add_scalar_step(self, key, value):
    #     """Logging from training loop causes GPU memory to skyrocket for some reason"""
    #     self.logger.experiment[0].add_scalar(key, value, self.global_step)

    def on_fit_start(self) -> None:
        print(
            f"{ool.env.print_prefix()} - FIT START "
            f"global_rank={self.trainer.global_rank} "
            f"node_rank={self.trainer.node_rank} "
            f"local_rank={self.trainer.local_rank}"
        )
        self.print(pformat(self.hparams))
        self.print(self.scheduled_values_summary())
        if self.trainer.is_global_zero:
            if not self.nowatermark:
                watermark_source(self.path)
        # Guestimate the total length for schedule.. This is for initialisation and will be refined on epoch/step start.
        with torch.no_grad():
            me = self.hparams.get("epochs", None)
            me = me or self.hparams.get("max_steps", None) // 1000
            self.update_epoch_scheduled_values(0, me, should_log=False)
            self.update_itera_scheduled_values(0, me * 1000, should_log=False)

    def on_train_start(self) -> None:
        self.print(f"Starting from {self.global_step} {self.current_epoch}")

    def on_train_epoch_start(self) -> None:
        with torch.no_grad():
            step = self.current_epoch
            total_steps = self.__max_epochs
            self.update_epoch_scheduled_values(step, total_steps)

    @functools.cached_property
    def data(self):
        return DataSpec(self.hparams.data)

    @property
    def input_shape(self):
        return self.data.shape

    @property
    def __max_epochs(self):
        if hasattr(self.hparams, "epochs"):
            return self.hparams.epochs
        return int(self.hparams.max_steps / self.data_len + 0.5)

    def get_max_epochs(self):
        return self.__max_epochs

    @property
    def __max_steps(self):
        if hasattr(self.hparams, "max_steps"):
            return self.hparams.max_steps
        return self.hparams.epochs * self.data_len

    def get_max_steps(self):
        return self.__max_steps

    def prepare_data(self):
        print(
            f"{ool.env.print_prefix()} - PREP DATA "
            f"global_rank={self.trainer.global_rank} "
            f"node_rank={self.trainer.node_rank} "
            f"local_rank={self.trainer.local_rank} "
            f"{self.trainer.accelerator_connector.is_slurm_managing_tasks} {self.trainer.accelerator_connector.cluster_environment}"
        )
        self.data.prepare()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """This method will overwrite the parent class method"""
        data = self.data.get_dataloader(
            self.hparams.batch_size,
            workers=self.workers,
            shuffle=True,
            subset="train",
            device=self.device,
            drop_last=self.drop_last_batch,
        )
        return data

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """This method will overwrite the parent class method"""
        return self.data.get_dataloader(
            self.hparams.batch_size,
            workers=self.workers,
            shuffle=False,
            subset="val",
            device=self.device,
            drop_last=self.drop_last_batch,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """This method will overwrite the parent class method"""
        return self.data.get_dataloader(
            self.hparams.batch_size,
            workers=self.workers,
            shuffle=False,
            subset="test",
            device=self.device,
        )

    def test_step(self, *args, **kwargs):
        self.onceprint(f"WARNING: Using validation loop for TESTING")
        return self.validation_step(*args, **kwargs)

    @torch.no_grad()
    def accelated_batch_postprocessing(self, batch):
        """Incase there's some batch post-processing that needs to happen on the accelarator for speed"""
        return self.data.post_processing(batch)

    def get_progress_bar_dict(self):
        """Remove v_num from progbar"""
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        # tqdm_dict.pop("loss_step", None)
        # tqdm_dict.pop("loss_epoch", None)
        tqdm_dict["i"] = self.global_step
        return tqdm_dict

    def configure_optimizers(self):
        osel, *oopts = self.hparams.optim.split("-")
        opt = {
            "rmsprop": torch.optim.RMSprop,
            "adam": torch.optim.Adam,
            "adamax": torch.optim.Adamax,
            "sgd": torch.optim.SGD,
        }[osel]
        for o in oopts:
            if "nest" in o:
                opt = functools.partial(opt, nesterov=True)
            if "m" in o:
                momentum = float(o[1:])
                opt = functools.partial(opt, momentum=momentum)

        if hasattr(self.model, "param_groups"):
            opt_params = [
                {
                    "params": list(pg["params"]),
                    "lr": self.hparams.learning_rate * pg["lr"],
                    **pg.get("other", {}),
                }
                for pg in self.model.param_groups()
            ]
        elif hasattr(self.model, "model_parameters"):
            opt_params = [
                {
                    "params": list(self.model.model_parameters()),
                    "lr": self.hparams.learning_rate,
                }
            ]
            if hasattr(self.model, "baseline_parameters"):
                opt_params.append(
                    {
                        "params": list(self.model.baseline_parameters()),
                        "lr": 10 * self.hparams.learning_rate,
                    }
                )
        else:
            opt_params = [
                {
                    "params": list(self.model.parameters()),
                    "lr": self.hparams.learning_rate,
                }
            ]

        params = set(itertools.chain.from_iterable(pg["params"] for pg in opt_params))
        missing_params = []
        for name, param in self.model.named_parameters():
            if param not in params:
                print(f"{name} is missing from param_groups def")
                missing_params.append(param)
        if missing_params:
            print(
                f"{len(missing_params)} param groups missing from param_groups definition adding with lr={self.hparams.learning_rate}"
            )
            opt_params.append(
                {"params": missing_params, "lr": self.hparams.learning_rate}
            )

        opt = opt(opt_params)
        r = {"optimizer": opt}
        if self.hparams.lr is not None:
            if self.hparams.lr == "plateau":
                r["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt, self.mode, 0.1, 3, True, min_lr=5e-5
                    ),
                    "monitor": self.monitor,
                    "reduce_on_plateau": True,
                }
            elif self.hparams.lr.startswith("decay("):
                s = [n.strip() for n in self.hparams.lr[6:-1].strip().split(",")]
                warmup_steps = int(s[0])
                decay_steps = int(s[1])
                decay_rate = 0.5
                if len(s) >= 3:
                    decay_rate = float(s[2])

                def lr_lambda(step):
                    if step < warmup_steps:
                        lr = float(step) / float(warmup_steps)
                    else:
                        lr = 1
                    return lr * decay_rate ** (float(step) / float(decay_steps))

                r["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda),
                    "interval": "step",
                }
            elif self.hparams.lr.startswith("warm("):
                s = [n.strip() for n in self.hparams.lr[5:-1].strip().split(",")]
                warmup_steps = int(s[0])

                def lr_lambda(step):
                    if step < warmup_steps:
                        lr = float(step) / float(warmup_steps)
                    else:
                        lr = 1
                    return lr

                r["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda),
                    "interval": "step",
                }
        else:
            lrs, interval = self.lr_scheduler(opt)
            if lrs is not None:
                r["lr_scheduler"] = {"scheduler": lrs, "interval": interval}
        return r

    def trainer_kwargs(self):
        return {}

    def lr_scheduler(self, opt):
        return None, None

    # @property
    # def ddp(self):
    #     print(self.trainer.accelerator.distributed_backend)
    #
    #     return self.trainer.accelerator.distributed_backend and \
    #            self.trainer.accelerator.distributed_backend.startwith('ddp')

    @property
    def path(self):
        if hasattr(self, "trainer") and self.trainer is not None:
            return self.trainer.default_root_dir
        return self._path

    def run(
        self,
        dir=None,
        resume=None,
        with_hparams=True,
        limit=None,
        benchmark=True,
        weights=None,
        profile=None,
        tune=None,
        detect_anomaly=None,
        no_stop=True,
        gpus=None,
        track_grads=False,
        quiet=False,
        save_top=1,
        workers=-1,
        dp=False,
        ddp=False,
        cpu_metrics=False,
        amp=None,
        acc=None,
        drop_last_batch=False,
        gc=0,
        nowatermark=False,
        resckpt=None,
        nodes=1,
    ):
        if nodes < 1:
            nodes = 1
        elif nodes > 1:
            ddp = True
        if gpus is None:
            if ddp:
                gpus = -1
            else:
                gpus = 1 if torch.cuda.is_available() else 0
        resume_dict = {}
        if resume:
            if isinstance(resume, str):
                if resume.endswith("ckpt"):
                    model_path = resume
                    p = Path(resume).parent
                else:
                    p = Path(resume)
                    model_path = sorted(
                        (a for a in p.glob("*.ckpt")), key=lambda a: a.stat().st_mtime
                    )[-1]
            elif isinstance(resume, bool):
                p = Path(exp_log_dir(self.name, no_unique=True, dir=dir))
                experiments = itertools.chain(
                    p.parent.glob(p.name),
                    p.parent.glob(p.name + "_?"),
                    p.parent.glob(p.name + "_??"),
                    p.parent.glob(p.name + "__id*"),
                )
                checkpoints = itertools.chain.from_iterable(
                    itertools.chain(p.glob("*.ckpt"), p.glob("checkpoints/*.ckpt"))
                    for p in experiments
                )
                model_path = sorted(
                    (a for a in checkpoints), key=lambda a: a.stat().st_mtime
                )[-1]
                p = model_path.parent
            if p.name == "checkpoints":
                p = p.parent
            print(f"Resuming model from {model_path} ({p})")
            if with_hparams:
                model = self.__class__.load_from_checkpoint(
                    model_path, hparams_file=str(p / "hparams.yaml")
                )
                loaded_hparams = model.hparams
                overwrites = {}
                for k, v in self.hparams.items():
                    if k in loaded_hparams and v != loaded_hparams[k]:
                        overwrites[k] = v
                if overwrites:
                    print(f"Overwriting: {pformat(overwrites)}")
                    model.hparams.update(overwrites)
            resume_dict["resume_from_checkpoint"] = model_path
            if with_hparams:
                del self.model
                self = model
        else:
            # This only prevents creation of the unique experiment directory on ddp spawn where we know the rank
            p = Path(
                exp_log_dir(self.name, dir=dir, dist_safe=ddp or ool.env.is_slurm())
            )
            print(f"{ool.env.print_prefix()} - Experiment path {p}")
            os.environ["OOL_EXPERIMENT_PATH"] = str(p)
        self.nowatermark = nowatermark
        self._path = p
        self.ddp = ddp
        self.cpu_metrics = cpu_metrics
        self.gc = gc
        self.workers = workers
        self.drop_last_batch = drop_last_batch
        if weights:
            print(f"Loading weights from model {weights}")
            pl_model = self.__class__.load_from_checkpoint(weights)
            self.model.load_state_dict(
                pl_model.model.state_dict()
            )  # Will fail if not compat, ie do not overwrite
        logger = pl.loggers.TensorBoardLogger(str(self.path), name="", version="")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        wandb_logger = pl.loggers.WandbLogger(
            project="clevrtex",
            name=f"{self.hparams.tag}-{timestamp}",
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=self.monitor,
            filename="best",
            save_top_k=save_top,
            mode=self.mode,
            save_last=True,
        )
        clbs = [
            checkpoint_callback,
            CheckPointPNRG(),
            CheckPointMetricInjector(),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
            TrainingStatusCallback(),
        ]
        if not no_stop:
            clbs.append(
                pl.callbacks.EarlyStopping(
                    monitor=self.monitor,
                    min_delta=1e-4,
                    patience=5,
                    verbose=True,
                    mode=self.mode,
                    strict=True,
                )
            )
        epochs = self.hparams.get("epochs", None)
        steps = self.hparams.get("max_steps", None)
        trainer_kwargs = {
            "default_root_dir": str(self.path),
            "gpus": gpus,
            "logger": [logger, wandb_logger],
            "callbacks": clbs,
            "max_epochs": epochs,
            "min_epochs": int(0.2 * epochs)
            if epochs
            else epochs,  # run at least for 20% of the epochs
            "max_steps": steps,
            "min_steps": int(0.2 * steps)
            if steps
            else steps,  # run at least for 20% of the epochs,
            "move_metrics_to_cpu": cpu_metrics,
            "num_nodes": nodes,
            **resume_dict,
        }
        if dp:
            trainer_kwargs["accelerator"] = "dp"
        if ddp:
            trainer_kwargs["accelerator"] = "ddp"
            trainer_kwargs["sync_batchnorm"] = True
            trainer_kwargs["prepare_data_per_node"] = True
        if ool.env.is_slurm():
            quiet = True
            trainer_kwargs["log_gpu_memory"] = "min_max"
            trainer_kwargs["precision"] = 32
        if track_grads:
            trainer_kwargs["track_grad_norm"] = 2
        if quiet:
            trainer_kwargs["progress_bar_refresh_rate"] = 0
            trainer_kwargs["callbacks"].append(StdProgressLogger())
        if limit:
            trainer_kwargs["limit_train_batches"] = limit
            trainer_kwargs["limit_val_batches"] = limit
        if getattr(self.hparams, "grad_clip", None):
            trainer_kwargs["gradient_clip_val"] = self.hparams.grad_clip
        if benchmark:
            trainer_kwargs["benchmark"] = True
        if profile:
            trainer_kwargs["profiler"] = AdvancedProfiler(
                output_filename=str(p / "profile.out")
            )
        if tune:
            trainer_kwargs["auto_lr_find"] = True

        if amp:
            trainer_kwargs["precision"] = 16
        if acc:
            trainer_kwargs["accumulate_grad_batches"] = int(acc)

        trainer_kwargs.update(self.trainer_kwargs())
        trainer = pl.Trainer(**trainer_kwargs)
        print(
            f"{ool.env.print_prefix()} - POST TRAINER INIT "
            f"Trainer properties: "
            f"global_rank={trainer.global_rank} "
            f"node_rank={trainer.node_rank} "
            f"local_rank={trainer.local_rank} "
            f"{trainer.accelerator_connector.is_slurm_managing_tasks} {trainer.accelerator_connector.cluster_environment}"
        )
        if tune:
            print(f"{ool.env.print_prefix()} - Tunning {self.name}")
            lr_finder = trainer.tuner.lr_find(self)
            new_lr = lr_finder.suggestion()
            self.print("LR resutls", lr_finder.results, "\nSuggestion", new_lr)
            self.hparams.learning_rate = new_lr
        else:
            with torch.autograd.set_detect_anomaly(bool(detect_anomaly)):
                try:
                    print(f"{ool.env.print_prefix()} - Fitting {self.name}")
                    trainer.fit(self)
                except:
                    if self.current_epoch <= 0 and not resume:
                        print(
                            f"{ool.env.print_prefix()} - Cleaning directory {self.path}"
                        )
                        shutil.rmtree(self.path)
                    raise

    def just_return_the_name(self, *args, **kwargs):
        return self.name

    @classmethod
    def parse_args_and_execute(cls):
        print(cls.__name__)
        strictfire.StrictFire(cls)

    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int) -> None:
        with torch.no_grad():
            step = self.global_step
            total_steps = self.__max_steps
            self.update_itera_scheduled_values(step, total_steps)

        if self.gc > 0 and batch_idx % self.gc == 0:
            if not hasattr(self, "_gc_stat"):
                self._gc_stat = 0
                self._gc_cont = 0
            if batch_idx == 0 and self._gc_cont > 0:
                print(
                    f"Extra GC collected {float(self._gc_stat) / self._gc_cont} on average"
                )
                self._gc_stat = 0
                self._gc_cont = 0
            self._gc_stat += gc.collect()
            self._gc_cont += 1

        return self.accelated_batch_postprocessing(batch)

    def on_validation_batch_start(
        self, batch, batch_idx: int, dataloader_idx: int
    ) -> None:
        return self.accelated_batch_postprocessing(batch)

    def on_test_batch_start(self, batch, batch_idx: int, dataloader_idx: int) -> None:
        return self.accelated_batch_postprocessing(batch)


class TrainingStatusCallback(pl.callbacks.Callback):
    def __init__(self):
        self.status = None

    def on_fit_start(self, trainer, pl_module):
        self.status = None

    def on_keyboard_interrupt(self, trainer, pl_module):
        if self.status is None:
            self.status = "INTERRUPT"

    def on_fit_end(self, trainer, pl_module):
        if self.status is None:
            self.status = "SUCCESS"

        pl_module.print(
            f"Stopping due to {self.status} at {trainer.global_step} step, {trainer.current_epoch} epoch"
        )
        with (Path(pl_module.path) / f"{self.status}").open("w") as outf:
            pass


class StdProgressLogger(pl.callbacks.Callback):
    def __init__(self):
        self.start = datetime.now()

    def on_train_epoch_start(self, trainer, pl_module):
        self.start = datetime.now()

    def on_validation_epoch_end(self, trainer, pl_module):
        secs = (datetime.now() - self.start).total_seconds()
        metrics = " ".join(
            f"{k}= {float(v):.3f}" for k, v in trainer.progress_bar_metrics.items()
        )
        pl_module.print(
            f"{trainer.current_epoch}/{pl_module.get_max_epochs()} [{pl_module.global_step}/{pl_module.get_max_steps()}] epoch in {secs:.2f}s: {metrics}"
        )


class CheckPointMetricInjector(pl.callbacks.Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        metrics = trainer.callback_metrics
        metrics.update(trainer.logged_metrics)

        key = "metrics"
        if key in checkpoint:
            key = str(self.__class__.__name__).lower() + "_" + key
        checkpoint[key] = metrics

        key = "monitor"
        if key in checkpoint:
            key = str(self.__class__.__name__).lower() + "_" + key
        checkpoint[key] = {pl_module.monitor: metrics[pl_module.monitor]}

        key = "name"
        if key in checkpoint:
            key = str(self.__class__.__name__).lower() + "_" + key
        checkpoint[key] = pl_module.name

        key = "path"
        if key in checkpoint:
            key = str(self.__class__.__name__).lower() + "_" + key
        checkpoint[key] = pl_module.path


class CheckPointPNRG(pl.callbacks.Callback):
    """Preserve and restore PRNGs when checkpointing"""

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["prng_states"] = {
            "python.random": random.getstate(),
            "numpy": np.random.get_state(),
        }
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            # TODO: how should this handle DP training?
            checkpoint["prng_states"]["cuda"] = torch.cuda.get_rng_state_all()

        return checkpoint["prng_states"]

    def on_load_checkpoint(self, callback_state):
        random.setstate(callback_state["python.random"])
        np.random.set_state(callback_state["numpy"])
        print("Restored python.random and numpy default PRNGs state")
        if torch.cuda.is_available() and "cuda" in callback_state:
            torch.cuda.set_rng_state_all(callback_state["cuda"])
            print(f"Restored CUDA PRNGs state to {len(callback_state['cuda'])} devices")
