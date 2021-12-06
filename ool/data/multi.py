from pathlib import Path

import torch
import numpy as np
import json
import warnings

from ool.data.utils import DatasetReadError, index_with_bias_and_limit
from ool.clevrtex_eval import CLEVRTEX

DATASETS = {
    'multi_dsprites': {
        'fname': "multi_dsprites_{variant}.tar.gz",
        # 'variants': ('colored_on_colored', 'binarized', 'colored_on_grayscale')
        'variants': {
            'colored_on_colored': {},
            'binarized': {},
            'colored_on_grayscale': {}
        }
    },
    'objects_room': {
        'fname': "objects_room_{variant}.tar.gz",
        # 'variants': ('train', 'six_objects', 'empty_room', 'identical_color')
        'variants': {
            'train': {},
            'six_objects': {},
            'empty_room': {},
            'identical_color': {},
        }
    },
    'clevr_with_masks': {
        'fname': "clevr_with_masks.tar.gz",
        'ccrop_frac': 0.8,
        'variants': {
            None: {
                'shape': (3, 240, 320),
                'flags': {'div_mask'},
            }
        },
        'splits': {
            'test': (0, 15000),
            'val': (15000, 30000),
            'train': (30000, 1.)
        }
    },
    'tetrominoes': {
        'fname': "tetrominoes.tar.gz",
        'variants': {
            None: {}
        }
    },
    'clevrtex': {
        'fname': "clevrtex_{variant}.tar.gz",
        'ccrop_frac': 0.8,
        'splits': {
            'test': (0., 0.1),
            'val': (0.1, 0.2),
            'train': (0.2, 1.)
        },
        'variants': {
            'full': {'shape': (3, 240, 320), 'flags': {'drop_last'},},
            'pbg': {'shape': (3, 240, 320), 'flags': {'drop_last'},},
            'vbg': {'shape': (3, 240, 320), 'flags': {'drop_last'},},
            'grassbg': {'shape': (3, 240, 320), 'flags': {'drop_last'},},
            'camo': {'shape': (3, 240, 320), 'flags': {'drop_last'},},
            'test': {'shape': (3, 240, 320), 'flags': {'drop_last'},},

            # Old iterations of the data
            'v0': {'shape': (3, 480, 640), 'flags': {'fix_mask'}},
            'v1': {'shape': (3, 240, 320), 'flags': {'expand'}},
            'v2': {'shape': (3, 240, 320), 'flags': {'expand'},},
            'pbgv2': {'shape': (3, 240, 320), 'flags': {'expand'},
                    # 'splits': {'train': 'train', 'val': 'val', 'test': 'test'}
                    },
            'vbgv2': {'shape': (3, 240, 320), 'flags': {'expand'},
                    # 'splits': {'train': 'train', 'val': 'val', 'test': 'test'}
                    },
            'grassbgv2': {'shape': (3, 240, 320), 'flags': {'expand'},
                        # 'splits': {'train': 'train', 'val': 'val', 'test': 'test'}
                        },
            'camov2': {'shape': (3, 240, 320), 'flags': {'expand'},
                     # 'splits': {'train': 'train', 'val': 'val', 'test': 'test'}
                     },
            'old': {'shape': (3, 240, 320), 'flags': {'expand'},
                    # 'splits': {'train': 'train', 'val': 'val', 'test': 'test'}
                    },
        },
    }
}


class ModDataset:
    @staticmethod
    def _resolve(dataset, split, dataset_variant=None):
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset}")
        meta = DATASETS[dataset]
        var = None
        bias = 0.
        limit = 1.
        if dataset_variant not in meta['variants']:
            raise ValueError(f"Unknown variant {var}")
        var = dataset_variant
        if 'splits' in meta['variants'][dataset_variant] and split in meta['variants'][dataset_variant]['splits']:
            split_meta = meta['variants']['splits'][split]
        else:
            split_meta = meta['splits'][split]
        if isinstance(split_meta, (list, tuple)):
            bias, limit = split_meta
        else:
            var = f'{var}_{split_meta}'
        fname = meta['fname'].format(variant=var)
        return var, fname, bias, limit

    @staticmethod
    def get_data_shape(dataset, dataset_variant=None):
        return DATASETS[dataset]['variants'][dataset_variant].get('shape', None)

    @staticmethod
    def get_crop_fraction(dataset):
        return DATASETS.get(dataset, {}).get("ccrop_frac", 1.0)

    @staticmethod
    def get_archive(dataset, split, dataset_variant=None):
        _, fname, *__ = ModDataset._resolve(dataset, split, dataset_variant=dataset_variant)
        return fname

    def reindex(self):
        print(f'Indexing {self.basepath}')
        new_index = {}
        for npz_path in self.basepath.glob('**/*.npz'):
            rel_path = npz_path.relative_to(self.basepath)
            try:
                indstr = str(rel_path.name).replace('.npz', '').split("_")[-1]
                indstr = indstr.lstrip('0')
                if indstr:
                    ind = int(indstr)
                else:
                    ind = 0
            except ValueError:
                print(f"Could not parse {rel_path}")
                continue
            new_index[str(ind)] = str(rel_path)
        if len(new_index) == 0:
            raise DatasetReadError()
        print(f"Found {len(new_index)} values")
        return new_index

    def __init__(self, path: Path, dataset, dataset_variant=None, split='train'):
        var, _, bias, limit = self._resolve(dataset, split, dataset_variant=dataset_variant)
        subfolder = dataset
        if var:
            subfolder += f"_{var}"
        self.basepath = path / subfolder
        if not self.basepath.exists():
            raise DatasetReadError()
        try:
            with (self.basepath / 'manifest_ind.json').open('r') as inf:
                self.index = json.load(inf)
        except (json.JSONDecodeError, IOError, FileNotFoundError):
            self.index = self.reindex()
        print(f"Sourced {dataset}{dataset_variant} ({split}) from {self.basepath}")
        self.dataset = dataset
        self.dataset_variant = dataset_variant
        self.split = split
        if isinstance(bias, float):
            bias = int(bias * len(self.index))
        if isinstance(limit, float):
            limit = int(limit * len(self.index))
        self.limit = limit
        self.bias = bias
        self.shape = ModDataset.get_data_shape(self.dataset, self.dataset_variant)
        self.flags = DATASETS[dataset]['variants'][dataset_variant].get('flags', set())

    def __len__(self):
        return self.limit - self.bias

    def __getitem__(self, ind):
        ind = index_with_bias_and_limit(ind, self.bias, self.limit)
        path = self.index[str(ind)]
        itm = np.load(str(self.basepath / path))
        img = torch.from_numpy(itm['image']).transpose(-1, -2).transpose(-2, -3).to(torch.float) / 255.
        masks = torch.from_numpy(itm['mask']).transpose(-1, -2).transpose(-2, -3).to(torch.float)

        if 'mult_mask' in self.flags:
            masks = masks * 255.
        if 'div_mask' in self.flags:
            masks = masks / 255.

        if len(masks.shape) == 3:
            masks = masks[:, None, :, :]

        if 'expand' in self.flags:
            masks = torch.cat(
                [masks, torch.zeros(11 - masks.shape[0], *masks.shape[1:], dtype=masks.dtype, device=masks.device)],
                dim=0)

        if 'drop_last' in self.flags:
            masks = masks[:-1]

        if 'visibility' in itm:
            vis = torch.from_numpy(itm['visibility']).to(bool)
            vis = vis[:masks.shape[0]]
        else:
            # Assume all are visible (in objects room)
            vis = torch.tensor([True] * masks.shape[0])

        return img, masks, vis


class FilteredDataset(ModDataset):
    def __init__(self, filter, filter_kw, metapath :Path, *args, **kwargs):
        super(FilteredDataset, self).__init__(*args, **kwargs)
        self.filtidx = []
        index_path = metapath / f'{filter_kw}_index.json'
        if index_path.exists() and index_path.is_file():
            with index_path.open('r') as inf:
                self.filtidx = json.load(inf)
            print(f'Found index {index_path} : {len(self.filtidx)}')
        else:
            print("Filtering")
            for i in range(super(FilteredDataset, self).__len__()):
                img, masks, vis = super(FilteredDataset, self).__getitem__(i)
                if filter(img, masks, vis):
                    self.filtidx.append(i)
            if index_path.parent.exists():
                try:
                    with index_path.open('w') as outf:
                        json.dump(self.filtidx, outf)
                    print(f'Cached results at {index_path}')
                except:
                    print(f'Failed to cache results at {index_path}')
                    pass
            else:
                print(f'Cache path {index_path.parent} not found')
    
    def __len__(self):
        return len(self.filtidx)
    
    def __getitem__(self, ind):
        return super(FilteredDataset, self).__getitem__(self.filtidx[ind])

class WrappedCLEVRTEX(CLEVRTEX):
    def __int__(self, *args, **kwargs):
        super(WrappedCLEVRTEX, self).__int__(*args, **kwargs, return_metadata=False)
    
    def __getitem__(self, item):
        _, img, mask, *__ = super(WrappedCLEVRTEX, self).__getitem__(item)
        mask = (torch.arange(11, device=mask.device).view(-1, 1, 1, 1) == mask).to(torch.float)
        vis = mask.flatten(1).to(bool).any(-1)
        return (img, mask, vis)
