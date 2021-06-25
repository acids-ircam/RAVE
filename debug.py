#%%
import pytest
import torch
torch.set_grad_enabled(False)
import torch.nn as nn
from fd.flows.ar_flow import StrictCausalConv, ARFlow, SequentialFlow, ActNorm1d
from fd.flows.ar_model import ARModel
from einops import rearrange
import matplotlib.pyplot as plt

arflow = ARFlow(StrictCausalConv(16, 32, 3)).eval()
actnorm = ActNorm1d(16).eval()

armodel = ARFlow(ARModel(1, 20, 24, 3, 4, 3)).eval()

x = torch.randn(1, 1, 128)
y = armodel(x)[0]
z = armodel.inverse(y)

# %%
plt.plot(x.reshape(-1))
plt.plot(z.reshape(-1))
# %%
plt.plot(x.reshape(-1)- z.reshape(-1))
# %%
