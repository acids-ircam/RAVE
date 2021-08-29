# %%
import torch
from fd.parallel_model.core import mod_sigmoid
import matplotlib.pyplot as plt

x = torch.linspace(-10, 10, 1000)

plt.plot(x, 20 * torch.log10(mod_sigmoid(x)))
# %%
20