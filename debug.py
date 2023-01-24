# %%
%load_ext autoreload
%autoreload 2
import torch
from rave.quantization import ResidualVectorQuantization
# %%

rvq = ResidualVectdorQuantization(
    4,
    dim=128,
    codebook_size=256,
)
# %%
x = torch.randn(2,128,512)
q, d, c = rvq(x)
# %%
print(q.shape)
print(d)
print(c.shape)
# %%
c
# %%
