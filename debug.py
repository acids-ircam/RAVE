# %%
import torch
import torch.nn as nn

model = torch.jit.load("vae.ts")
x = torch.randn(1, 1, 4096)
# %%
model(x).shape
# %%
