import gin
import cached_conv as cc
import torch

gin.register(torch.nn.Conv1d, module="torch.nn")
gin.register(torch.nn.Conv2d, module="torch.nn")

cc.get_padding = gin.external_configurable(cc.get_padding, module="cc")
cc.Conv1d = gin.external_configurable(cc.Conv1d, module="cc")
cc.ConvTranspose1d = gin.external_configurable(cc.ConvTranspose1d, module="cc")

from .model import RAVE
from .blocks import *
from .discriminator import *
from .pqmf import *
