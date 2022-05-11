from os import path
from glob import glob
from pathlib import Path


def search_for_run(run_path, mode="last"):
    if run_path is None: return None
    if ".ckpt" in run_path: return run_path
    ckpts = map(str, Path(run_path).rglob("*.ckpt"))
    ckpts = filter(lambda e: mode in e, ckpts)
    return sorted(ckpts)[-1]


print(search_for_run(None))