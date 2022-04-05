import warnings
from pathlib import Path

import numpy as np
import torch

import ool.data.utils
import ool.env
from ool.data.mmnist import FlatMnist
from ool.data.multi import ModDataset, FilteredDataset, WrappedCLEVRTEX
from ool.data.pascal import PascalVOC
from ool.data.transforms import (
    Resize,
    MaxObjectsFilter,
    ChainTransforms,
    CentreCrop,
    ObjectOnly,
    HFlip,
)
from ool.data.utils import DatasetReadError

MAX_OBGJECTS = 20


def shape_from_name(name, var=None):
    if "mnist" in name:
        return FlatMnist.shape
    if "pascal" in name:
        return PascalVOC.shape
    return ModDataset.get_data_shape(name.replace("clevr6", "clevr_with_masks"), var)


def int_or_float_or_str(p):
    try:
        return int(p)
    except ValueError:
        pass
    try:
        return float(p)
    except ValueError:
        pass
    return str(p)


def maybe_parse_tuple(s):
    if not s:
        return s
    if isinstance(s, (tuple, list)):
        return tuple(s)
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")) or (
        s.startswith("[") and s.endswith("]")
    ):
        return s
    s = s.lstrip("(").lstrip("[").rstrip(")").rstrip("]").strip()
    return tuple(int_or_float_or_str(p.strip()) for p in s.split(","))


class Wrapper:
    @property
    def shape(self):
        return self.d.shape

    def __init__(self, dataset, transforms=None, add_rng_color=0):
        self.d = dataset
        self.transforms = transforms
        self.cmap = None
        if add_rng_color:
            rng = np.random.Generator(np.random.SFC64(seed=3730728833043713873))
            self.cmap = torch.from_numpy(
                rng.random((len(self.d), add_rng_color, 4, 1, 1))
            ).to(torch.float)

    def __len__(self):
        return len(self.d)

    def __getitem__(self, ind):
        itm = self.d[ind]
        if self.transforms is not None:
            itm = self.transforms(itm)
        if self.cmap is not None:
            itm = (*itm, self.cmap[ind, :, : itm[0].shape[0]])
        return itm


class DataSpec:
    SUBBG_TOKENS = ("sub_bg", "subbg", "bbg", "sbg")  # Subtract (make black) background
    RNGBG_TOKENS = ("rng_bg", "rngbg", "rbg")  # Make background RNG colour
    RNGDO_TOKENS = (
        "rbg_dob",
        "rngdob",
        "rdo",
    )  # Make Objects RNG colour (random colour per object)
    RNGSO_TOKENS = (
        "rng_sob",
        "rngsob",
        "rso",
    )  # Make objects same RNG colour (random colour per image)
    CCROP_TOKENS = ("crop_c", "crop", "c", "cc")  # Center-crop the dataset
    FLIP_TOKENS = ("flip",)  # Center-crop the dataset
    OBJ_TOKENS = ("obj", "onlyobject")
    SIZE_MAP = {
        "tiny": (48, 64),
        "small": (96, 128),
        "medium": (144, 192),
        "large": (192, 256),
        "huge": (240, 320),
        "custom": (120, 160)
    }

    def __init__(self, dataspec: str):
        self.dataset = []
        datavar, *rest = dataspec.split("-")
        if datavar.startswith("concat(") and datavar.endswith(")"):
            datavar = datavar[7:-1].strip()
            datavars = [dv.strip() for dv in datavar.split(",")]
        else:
            datavars = [datavar]
        # print(datavars)
        for datavar in datavars:
            dataset, *var = datavar.split(":")
            if dataset == "clevr":
                dataset = "clevr_with_masks"
            if not len(var):
                var = None
            else:
                var = ":".join(var)
            self.dataset.append((dataset, var))
        flags = set(r.strip().lower() for r in rest if r.strip())
        # print(flags)
        self.sub_bg = any(t in flags for t in self.SUBBG_TOKENS)
        for t in self.SUBBG_TOKENS:
            flags.discard(t)

        self.rng_bg = any(t in flags for t in self.RNGBG_TOKENS)
        for t in self.RNGBG_TOKENS:
            flags.discard(t)

        if self.sub_bg and self.rng_bg:
            raise ValueError(f"sub_bg and rng_bg are mutually exclusive")

        self.max = -1
        for f in flags:
            if f.startswith("max"):
                try:
                    n = int(f[3:])
                except ValueError:
                    continue
                self.max = n
                if not (self.sub_bg or self.rng_bg):
                    raise ValueError(
                        f"Max {n} object filter requires bg replacament options"
                    )
        flags.discard(f"max{self.max}")

        self.rng_sob = any(t in flags for t in self.RNGSO_TOKENS)
        for t in self.RNGSO_TOKENS:
            flags.discard(t)

        self.rng_dob = any(t in flags for t in self.RNGDO_TOKENS)
        for t in self.RNGDO_TOKENS:
            flags.discard(t)

        self.centre_crop = any(t in flags for t in self.CCROP_TOKENS)
        for t in self.CCROP_TOKENS:
            flags.discard(t)

        self.flip = any(t in flags for t in self.FLIP_TOKENS)
        for t in self.FLIP_TOKENS:
            flags.discard(t)

        self.obj = any(t in flags for t in self.OBJ_TOKENS)
        for t in self.OBJ_TOKENS:
            flags.discard(t)

        if self.rng_dob and self.rng_sob:
            raise ValueError(f"rng_dob and rng_sob are mutually exclusive")

        self.size = None
        if len(flags):
            self.size = maybe_parse_tuple(flags.pop())
            if not isinstance(self.size, (list, tuple)):
                self.size = self.SIZE_MAP[self.size]

        if self.rng_bg and self.rng_sob or self.rng_dob:
            warnings.warn(
                f"RNG colour is applied to objects and background, are you sure you don't want to just run multi_dsprites?"
            )

        if len(flags):
            raise ValueError(f"Unknown flags {flags}")

    @property
    def shape(self):
        s = shape_from_name(self.dataset[0][0], self.dataset[0][1])
        if self.size:
            s = s[0], *self.size
        elif self.centre_crop:
            # Only calculate for
            s = CentreCrop(
                ModDataset.get_crop_fraction(self.dataset[0][0])
            ).croping_bounds(s)
        return s

    def __str__(self):
        acc = []
        for dataset, var in self.dataset:
            r = dataset
            if dataset == "clevr_with_masks":
                r = "cle"
            elif dataset == "clevrtex":
                r = "clt"
            if var is not None and var not in {"cc", "new"}:
                r += f"_{var}"
            acc.append(r)
        r = "+".join(acc)
        if len(acc) > 1:
            r += "-"
        if self.centre_crop:
            r += "c"
        if self.flip:
            r += "flip"
        if self.size is not None:
            h = self.size[0]
            for size in self.SIZE_MAP:
                if h <= self.SIZE_MAP[size][0]:
                    r += size[0]
                    break
            else:
                r += "-cs"

        if self.sub_bg:
            r += "-bbg"
        if self.rng_bg:
            r += "-rbg"
        if self.rng_dob:
            r += "-rdo"
        if self.rng_sob:
            r += "-rso"
        if self.max >= 0:
            r += f"-m{self.max}"
        if self.obj:
            r += "-obj"
        return r

    def get_dataset(
        self,
        path: Path,
        subset=None,
        device=None,
        dataset=None,
        var=None,
        metapath=None,
    ):
        """metapath is the location where the dataset (potentially tar) was sourced... on SLURM it is different from path"""
        transforms = []
        if dataset is None:
            dataset = self.dataset[0][0]
            var = self.dataset[0][1]
        if self.centre_crop:
            c_fraction = ModDataset.get_crop_fraction(dataset)  # Default 1.0
            transforms.append(CentreCrop(crop_fraction=c_fraction))
        if self.flip:
            transforms.append(HFlip())
        if self.obj:
            transforms.append(ObjectOnly())
        if self.size is not None:
            transforms.append(Resize(self.size))
        if self.max >= 0:
            transforms.append(MaxObjectsFilter(self.max))
        if transforms:
            transforms = ChainTransforms(transforms)
        else:
            transforms = None
        assert subset in {None, "train", "val", "test"}
        if dataset == "mnist":
            d = FlatMnist(path, device=device, split=subset)
        elif dataset == "pascal":
            d = PascalVOC(path, split=subset)
        elif dataset == "clevr6":
            if metapath is None:
                metapath = path
            d = FilteredDataset(
                lambda img, masks, vis: vis.to(int).sum() <= 7,
                f"clevr6_{subset}",
                metapath,
                path,
                "clevr_with_masks",
                var,
                split=subset,
            )
        elif dataset == "clevrtex":
            d = WrappedCLEVRTEX(path, dataset_variant=var, split=subset)
            transforms = None
        else:
            d = ModDataset(path, dataset, var, split=subset)
        clr_count = int(self.rng_bg)
        if self.rng_dob:
            clr_count += MAX_OBGJECTS
        elif self.rng_sob:
            clr_count += 1
        wrapped = Wrapper(d, transforms, add_rng_color=clr_count)
        return wrapped

    @property
    def post_processing(self):
        pp_fn = lambda batch: batch
        fns = []
        clr_ind = 0
        if self.sub_bg:
            fns.append(lambda batch: (batch[0] * (1.0 - batch[1][:, 0]), *batch[1:]))
        elif self.rng_bg:
            clr_ind += 1
            fns.append(
                lambda batch: (
                    batch[0] * (1.0 - batch[1][:, 0])
                    + batch[-1][:, 0] * batch[1][:, 0],
                    *batch[1:],
                )
            )
        if self.rng_sob:
            fns.append(
                lambda batch: (
                    batch[0] * batch[1][:, 0]
                    + batch[-1][:, clr_ind] * (1.0 - batch[1][:, 0]),
                    *batch[1:],
                )
            )
        elif self.rng_dob:
            fns.append(
                lambda batch: (
                    batch[0] * batch[1][:, 0]
                    + (batch[-1][:, clr_ind : batch[1].shape[1]] * batch[1][:, 1:]).sum(
                        1
                    ),
                    *batch[1:],
                )
            )
        if len(fns) and (self.rng_bg or self.rng_sob or self.rng_dob):
            fns.append(lambda batch: batch[:-1])  # Remove clr
        if len(fns):

            def pp_fn(batch):
                for fn in fns:
                    batch = fn(batch)
                return batch

        return pp_fn

    def __load_dataset(self, subset, device, dataset, var):
        for path in ool.env.get_data_paths():
            try:
                dataset = self.get_dataset(
                    path,
                    subset=subset,
                    device=device,
                    dataset=dataset,
                    var=var,
                    metapath=ool.env.get_meta_path(),
                )
                break
            except DatasetReadError:
                continue
        else:
            output = ool.env.get_data_paths()[0]
            if dataset == "mnist":
                fname = FlatMnist.NAME
            elif dataset == "clevr6":
                fname = ModDataset.get_archive("clevr_with_masks", subset, var)
            else:
                fname = ModDataset.get_archive(dataset, subset, var)

            searchpaths = ool.env.get_archived_data_paths()
            for sp in searchpaths:
                target = sp / fname
                if target.exists():
                    break
            else:
                raise ValueError(
                    f"Cannot find {fname} for {dataset}:{var}; Looked: {':'.join(str(p) for p in searchpaths)}"
                )
            ool.data.utils.retrieve_data(target, output)
            dataset = self.get_dataset(
                output,
                subset=subset,
                device=device,
                dataset=dataset,
                var=var,
                metapath=ool.env.get_meta_path(),
            )
        return dataset

    def prepare(self):
        print(f"{ool.env.print_prefix()} - Preparing data")
        [self.__load_dataset("train", "cpu", dataset=d, var=v) for d, v in self.dataset]
        [self.__load_dataset("val", "cpu", dataset=d, var=v) for d, v in self.dataset]
        print(f"{ool.env.print_prefix()} - Data prepared")

    def get_dataloader(
        self,
        batch_size,
        workers=-1,
        shuffle=True,
        subset=None,
        device=None,
        drop_last=False,
    ):
        if len(self.dataset) == 1:
            dataset = self.__load_dataset(
                subset, device, dataset=self.dataset[0][0], var=self.dataset[0][1]
            )
        else:
            dataset = torch.utils.data.ConcatDataset(
                [
                    self.__load_dataset(subset, device, dataset=d, var=v)
                    for d, v in self.dataset
                ]
            )
        if isinstance(dataset.d, FlatMnist):
            workers = "no"
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=ool.data.utils.worker_spec_to_cpu(workers, batch_size),
        )

    def get_dataloaders(
        self, batch_size, workers=-1, shuffle=True, subset=None, device=None
    ):
        return [
            torch.utils.data.DataLoader(
                self.__load_dataset(subset, device, dataset=d, var=v),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=ool.data.utils.worker_spec_to_cpu(workers, batch_size),
            )
            for d, v in self.dataset
        ]
