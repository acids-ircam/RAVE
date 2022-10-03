import cached_conv as cc
import gin
import torch
import os
import pathlib

rave_install_path = pathlib.Path(os.path.dirname(__file__)).parent

gin.add_config_file_search_path(str(rave_install_path))
gin.register(torch.nn.Conv1d, module="torch.nn")
gin.register(torch.nn.Conv2d, module="torch.nn")

cc.get_padding = gin.external_configurable(cc.get_padding, module="cc")
cc.Conv1d = gin.external_configurable(cc.Conv1d, module="cc")
cc.ConvTranspose1d = gin.external_configurable(cc.ConvTranspose1d, module="cc")

from .blocks import *
from .discriminator import *
from .model import RAVE
from .pqmf import *
