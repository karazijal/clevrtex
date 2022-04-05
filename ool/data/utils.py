import os
from pathlib import Path
import subprocess
from functools import lru_cache
import datetime

import filelock

from ool.env import is_slurm, print_prefix


class DatasetReadError(ValueError):
    pass


TAR_SP = [Path("/usr/bin/tar"), Path("/bin/tar")]


@lru_cache(None)
def tarbin():
    for p in TAR_SP:
        if p.exists():
            return str(p)
    return "tar"


def worker_spec_to_cpu(spec: int, batchsize=None) -> int:
    if isinstance(spec, str):
        if spec == "no":
            print(f"No workers requisted; using 0")
            return 0
        else:
            spec = -1
    num_workers = os.cpu_count()
    if spec <= 0:
        num_workers = os.cpu_count()
        if batchsize is not None:
            if batchsize <= 32:
                num_workers = min(num_workers, 4)
            elif batchsize <= 64:
                num_workers = min(num_workers, 8)
            elif is_slurm():
                num_workers = 8
        print(f"Unspecified worker count {spec}; using ", num_workers)
        return num_workers
    return spec


def retrieve_data(target, output):
    print(f"{print_prefix()} - Retrieving {target} to {output}")
    if not output.exists():
        output.mkdir(exist_ok=True)
    fl_path = Path(str(output / target.name) + ".lock")
    with filelock.FileLock(fl_path):
        print(f"{print_prefix()} - Grabbed filelock {fl_path}")
        probably_the_output_folder = Path(
            str(output / target.name).replace(".tar", "").replace(".gz", "")
        )
        if not probably_the_output_folder.exists():
            subprocess.check_call(
                [tarbin(), "-C", str(output), "-xzf", str(target)], close_fds=True
            )
            print(f"{print_prefix()} - Done retrieving {target} to {output}")
        else:
            print(
                f"{print_prefix()} - Found {probably_the_output_folder}; not retrieving"
            )
    print(f"{print_prefix()} - Released filelock {fl_path}")


def index_with_bias_and_limit(idx, bias, limit):
    if idx >= 0:
        idx += bias
        if idx >= limit:
            raise IndexError()
    else:
        idx = limit + idx
        if idx < bias:
            raise IndexError()
    return idx
