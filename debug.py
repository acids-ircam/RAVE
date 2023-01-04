# %%
import torch
from rave import blocks
import matplotlib.pyplot as plt
import numpy as np

x = torch.randn(1, 128, 256)
x = x / torch.norm(x, p=2, dim=1, keepdim=True)

angles = blocks.unit_norm_vector_to_angles(x)
noise = torch.randn_like(angles) / 10
noisy_angles = blocks.wrap_around_value(angles + noise)
y = blocks.angles_to_unit_norm_vector(angles)

# %%
print(torch.allclose(x, y, atol=1e-3, rtol=1e-3))
# %%
plt.plot(noisy_angles[0].T)
plt.show()
plt.plot(noise[0].T)
plt.show()
# %%
t = torch.linspace(-10, 10, 1000)
plt.plot(blocks.wrap_around_value(t, 1))
# %%
