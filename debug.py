#%%
import librosa as li
import matplotlib.pyplot as plt
from fd.parallel_model.pqmf import PQMF
import torch

torch.set_grad_enabled(False)

x, sr = li.load(li.example("trumpet"), 24000)
x = x[:2**16]

x = torch.from_numpy(x).float().reshape(1, 1, -1)

pqmf = PQMF(100, 4)

y = pqmf(x)

z = pqmf.inverse(y)

# %%
plt.plot(x.reshape(-1)[:1000])
plt.plot(z.reshape(-1)[:1000])
# %%
x=  torch.randn(1,4,4)
print(x)
x[:,1::2,::2] *= -1
print(x)
# %%
