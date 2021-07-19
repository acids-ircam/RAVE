# %%
import torch

torch.set_grad_enabled(False)
from fd.parallel_model.pqmf import CachedPQMF
import matplotlib.pyplot as plt

x = torch.randn(1, 1, 1024)
pqmf = CachedPQMF(100, 8)

y = pqmf(x)
z = pqmf.inverse(y)

print(pqmf.hk.shape)
plt.plot(x.reshape(-1))
plt.plot(z.reshape(-1))
# %%
