# %%
%load_ext autoreload
%autoreload 2
import torch
from rave.quantization import ResidualVectorQuantization
# %%

rvq = ResidualVectorQuantization(
    4,
    dim=2,
    codebook_size=32,
)
# %%
x = torch.randn(2,2,512)
q, d, c = rvq(x)
# %%
print(q.shape)
print(d)
print(c.shape)
# %%
c
# %%
codes = rvq.encode(x)

# %%
print(codes.shape)
# %%
y = rvq.decode(codes)
# %%
y.shape
# %%
rvq = torch.jit.script(rvq)
# %%
print(rvq)
# %%
