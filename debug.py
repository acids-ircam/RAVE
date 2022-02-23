# %%
import torch
torch.set_grad_enabled(False)
from rave.model import RAVE
import math
#%%

model = RAVE.load_from_checkpoint(
    "runs/engine/rave/version_1/checkpoints/last-v1.ckpt",
    strict=False,
).eval()
# %%
n_noise = 0
for n, p in model.named_parameters():
    if "noise_scale" in n:
        p  = torch.nn.functional.softplus(
                p) / math.sqrt(2)
        plt.plot(p.numpy())
        n_noise+=1

plt.legend(range(n_noise))
# %%

import matplotlib.pyplot as plt

t = torch.linspace(-5, 5, 100)
plt.plot(t, torch.nn.functional.softplus(t) / math.sqrt(2))

# %%
