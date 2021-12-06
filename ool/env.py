"""Various utilities to multiplex misc env where this might be running"""
"""Edit this file to describe locations where things should go"""
import os
from pathlib import Path
import subprocess
import base64
import uuid
import datetime

from functools import lru_cache

import torch


EXTERNAL = 'OOL_EXTERNAL'

class SLURM:
    ARRAY_JOB_ID='SLURM_ARRAY_JOB_ID'
    JOB_ACCOUNT='SLURM_JOB_ACCOUNT'
    JOB_ID='SLURM_JOB_ID'
    JOB_NAME='SLURM_JOB_NAME'
    JOB_NODELIST='SLURM_JOB_NODELIST'
    JOB_NUM_NODES='SLURM_JOB_NUM_NODES'
    JOB_PARTITION='SLURM_JOB_PARTITION'
    JOB_UID='SLURM_JOB_UID'
    JOB_USER='SLURM_JOB_USER'
    RESTART_COUNT='SLURM_RESTART_COUNT'
    PROCID='SLURM_PROCID'
    NODEID='SLURM_NODEID'
    LOCALID='SLURM_LOCALID'
    STEP_ID='SLURM_STEP_ID'
    STEP_NUM_TASKS='SLURM_STEP_NUM_TASKS'

@lru_cache(None)
def __hostname():
    if 'HOST' in os.environ:
        return str(os.environ['HOST'])
    if 'HOSTNAME' in os.environ:
        return str(os.environ['HOSTNAME'])
    else:
        return str(subprocess.check_output('hostname', shell=True).decode().strip())

def timestamp():
    return datetime.datetime.now().strftime('%d_%H:%M:%S.%f')

def is_aims():
    "Is AIMS CDT's cluster env?"
    return False

def is_slurm():
    return SLURM.ARRAY_JOB_ID in os.environ or SLURM.JOB_ID in os.environ

def is_docker():
    return os.environ['USER'] == 'root'  #TODO: should probably use something more sophisticated

def env_report():
    print(f"SLURM {is_slurm()} DOCKER {is_docker()} EXTERNAL {is_external_user()}")
    if torch.cuda.is_available():
        print(f"{print_prefix()} CUDA is available: see {torch.cuda.device_count()} GPU(s) using {torch.get_default_dtype()} by default")
        if is_slurm():
            print(print_prefix(), subprocess.check_output('nvidia-smi', shell=True).decode())
    else:
        print(f"{print_prefix()} CUDA is unavailable: using {torch.get_default_dtype()} by default")
    if is_slurm():
        p = Path(os.environ['TMPDIR']).absolute()
        if p.exists():
            print(f"{print_prefix()} tmp: {', '.join(([str(c) for c in p.iterdir()]))}")
        p = p / 'ool_data'
        if p.exists():
            print(f"{print_prefix()} {', '.join(([str(c) for c in p.iterdir()]))}")


def is_external_user():
    """Is this being run on behalf of some1 else"""
    return EXTERNAL in os.environ and str(os.environ[EXTERNAL]) == '1'

def get_expe_path():
    p = Path.cwd() / 'output'
    p.mkdir(parents=False, mode=0o777, exist_ok=True)
    return p

def get_meta_path():
    return Path.cwd() / 'ool_data'




def get_data_paths():
    return [Path.cwd() / 'ool_data']




def get_archived_data_paths():
    return []


def print_prefix():
    if is_slurm():
        return f'{__hostname()}: ' \
               f'GR={os.environ.get(SLURM.PROCID,"procid")} ' \
               f'NR={os.environ.get(SLURM.NODEID,"nodeid")} ' \
               f'LR={os.environ.get(SLURM.LOCALID,"localid")} ' \
               f'- {timestamp()}'
    else:
        return f'{__hostname()}: - {timestamp()}'

def unique_id():
    if is_slurm():
        s = os.environ.get(SLURM.PROCID, f"[{timestamp()}]")
        return f'{os.environ[SLURM.JOB_ID]}-{s}'
    else:
        return base64.urlsafe_b64encode(uuid.uuid1().bytes).decode()
