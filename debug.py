# %%
import numpy as np
from rave.core import get_beta_kl_cyclic_annealed
import matplotlib.pyplot as plt

t = np.arange(1000000, step=100)

beta = list(
    map(
        lambda x: get_beta_kl_cyclic_annealed(
            step=x,
            cycle_size=50000,
            warmup=500000,
            min_beta=1e-4,
            max_beta=1e-1,
        ), t))

plt.plot(t, beta)
plt.yscale("log")
plt.grid()

# %%
