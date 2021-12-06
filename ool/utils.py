from pathlib import Path
import tarfile
import os
from datetime import datetime

from ool.env import get_expe_path, is_slurm, is_external_user, unique_id, print_prefix

class RunningMean:
    def __init__(self):
        self.v = 0.
        self.n = 0

    def update(self, v, n=1):
        self.v += v * n
        self.n += n

    def value(self):
        return self.v / (self.n + 1e-12)

    def __str__(self):
        return str(self.value())

def set_owner(path):
    if is_external_user() and is_slurm():
        pass

def exp_log_dir(name, no_unique=False, dir=None, dist_safe=False):
    prefix = dir or get_expe_path()
    pb = prefix / name
    # create parents and set their access mode
    for s in reversed(pb.relative_to(prefix).parents):
        p = prefix / s
        if not p.exists():
            p.mkdir(parents=False, exist_ok=True, mode=0o777)
            set_owner(p)
        else:
            try:
                p.chmod(0o777)
            except PermissionError:
                print(f"{print_prefix()} - Permission error; can't chmod {str(p)}")
    p = pb
    i = 0
    if not no_unique:
        if dist_safe:
            if "OOL_EXPERIMENT_PATH" in os.environ:
                p = Path(os.environ["OOL_EXPERIMENT_PATH"])
            else:
                p = Path(str(pb) + f"__id{unique_id()}")  # generate new unique path
                p.mkdir(parents=False, exist_ok=True, mode=0o777)
        else:
            while p.exists():
                p = Path(str(pb) + f"_{i}")
                i += 1
            p.mkdir(parents=False, exist_ok=False, mode=0o777)
    return str(p)


def watermark_source(dst_base, source_dir=None):
    p = Path(source_dir or Path.cwd())
    d = f"code_{datetime.now().strftime('%y%m%d_%H%M%S')}.tar.gz"
    dst = Path(dst_base).expanduser() / d
    with tarfile.open(dst, 'w:gz') as tar:
        tar.add(p, arcname=p.name)
    set_owner(dst)
    return dst
