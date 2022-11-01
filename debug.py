# %%
import matplotlib.pyplot as plt
import torch

torch.set_grad_enabled(False)

model = torch.jit.load(
    '../ppo/runs/ppv_spec_cd324d5008/ppv_spec_cd324d5008_streaming.ts').eval()
x = torch.zeros(1, 1, 2**17)
y = model(x).reshape(-1).numpy()

plt.plot(y)
# %%
