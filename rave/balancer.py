import torch.nn as nn
import gin.torch


@gin.configurable
class Balancer(nn.Module):
    def __init__(self):
        super().__init__(self)

    def forward(self, *args, **kwargs):
        raise RuntimeError('Balancer has been disabled in newest RAVE version. \n' \
                           'If you try to import checkpoint trained with a previous version, remove it from configuration.')