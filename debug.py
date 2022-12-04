import torch
from rave.blocks import ResidualEncoder

re = ResidualEncoder(16, 32, [4, 4, 2, 2], 512, 1, 7, [1, 3, 9])
#
x = torch.randn(1, 16, 2048)

print(re(x).shape)
#
import cached_conv as cc

print(cc.get_padding(4,2))