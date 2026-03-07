from pathlib import Path
import os
import re

import cached_conv as cc
import gin
import torch
import acids_dataset as ad


BASE_PATH: Path = Path(__file__).parent.resolve()
CUSTOM_PATH: Path = (Path(__file__).parent.parent / "configs").resolve()

gin.add_config_file_search_path(BASE_PATH)
gin.add_config_file_search_path(CUSTOM_PATH)
gin.add_config_file_search_path(BASE_PATH.joinpath('configs'))

# AUGMENTATION_CONFIG_PATH = BASE_PATH.joinpath('configs', 'augmentations')
# AUGMENTATION_CONFIG_PATH = Path(ad.__file__).parent.resolve() / "configs" / "transforms"
# if (AUGMENTATION_CONFIG_PATH.resolve() != target_augment_config_path) or (not AUGMENTATION_CONFIG_PATH.exists()): 
#     if (AUGMENTATION_CONFIG_PATH.exists()):
#         os.remove(AUGMENTATION_CONFIG_PATH)
#     AUGMENTATION_CONFIG_PATH.symlink_to(target_augment_config_path)


def __safe_configurable(name):
    try: 
        setattr(cc, name, gin.get_configurable(f"cc.{name}"))
    except ValueError:
        setattr(cc, name, gin.external_configurable(getattr(cc, name), module="cc"))

if ad.__file__:
    AUGMENTATION_CONFIG_PATH = Path(ad.__file__).parent.resolve() / "configs" / "transforms"
    gin.add_config_file_search_path(AUGMENTATION_CONFIG_PATH)
else:
    print('[Warning] Could not locate acids_dataset ; augmentations not available')


__safe_configurable("get_padding")
__safe_configurable("Conv1d")
__safe_configurable("ConvTranspose1d")

from .blocks import *
from .discriminator import *
from .model import RAVE, BetaWarmupCallback
from .pqmf import *
from .balancer import *
from . import core

# configs that can be parsed even if system is resumed.
RAVE_OVERRIDE_CONFIGS = [
    "no_encoder_freeze.gin", 
    "adv_at_start.gin", 
    "full_beta.gin"
]


def get_run_name(run_path): 
    checkpoint_path = (Path(run_path) / "..").resolve()
    if checkpoint_path.stem != "checkpoints": 
        return None
    version_path = (checkpoint_path / "..").resolve()
    if checkpoint_path.stem.startswith("version_"):
        return None
    return ((version_path / "..").resolve()).stem

def load_rave_checkpoint(model_path, n_channels=1, ema=False, name="last.ckpt", remove_keys=None, configs=[], overrides=[]):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    if model_path.suffix == ".ts":
        return torch.jit.load(model_path)
    with core.GinEnv():
        config_file = core.search_for_config(model_path) 
        if config_file is None:
            print('no configuration file found at address :'%model_path)
        gin.parse_config_files_and_bindings([config_file] + configs, overrides)
        run_path = core.search_for_run(model_path, name=name)
        if run_path is None: 
            raise FileNotFoundError("no model found with name: %s"%name)
        rave_model = RAVE()
        checkpoint = torch.load(run_path, map_location='cpu', weights_only=True)
        if remove_keys is not None: 
            if not isinstance(remove_keys, list): remove_keys = [remove_keys]
            weights_to_remove = []
            for k in checkpoint['state_dict']: 
                if set([re.match(f, k) for f in remove_keys]) != {None}: 
                    weights_to_remove.append(k)
            for w in weights_to_remove: 
                del checkpoint['state_dict'][w]
        if ema and "EMA" in checkpoint["callbacks"]:
            rave_model.load_state_dict(
                checkpoint["callbacks"]["EMA"],
                strict=False,
            )
        else:
            rave_model.load_state_dict(
                checkpoint["state_dict"],
                strict=False,
            )
    return rave_model, run_path

from rave.dataset import get_augmentations, parse_transform
def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name

def parse_augmentations(augmentations, sr):
    for a in augmentations:
        with core.GinEnv(paths=[Path(__file__).parent / "configs" / "augmentations"]):
            gin.parse_config_file(a)
            parse_transform()
    return get_augmentations()


