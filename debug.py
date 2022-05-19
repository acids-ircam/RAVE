# %%
from rave.blocks import ResidualStack
import torch
import cached_conv as cc

rs = ResidualStack(16, [3, 5, 7], [[1, 1], [3, 1], [5, 1]], "centered")

x = torch.randn(1, 16, 128)
rs(x).shape

cc.test_equal(
    lambda: ResidualStack(16, [3, 5, 7], [[1, 1], [3, 1], [5, 1]], "centered"),
    x,
)

# %%
ResidualStack(16, [3, 5, 7], [[1, 1], [3, 1], [5, 1]], "centered")
# %%
