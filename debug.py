#%%
import torch
import prior
import rave.core
import gin
import matplotlib.pyplot as plt
import cached_conv as cc

torch.set_grad_enabled(False)
cc.use_cached_conv(True)

device = torch.device(f"cuda:2")

gin.parse_config_file("configs/prior.gin")
z = torch.zeros(1, 2**13)
model = prior.Prior.load_from_checkpoint(
    rave.core.search_for_run("runs/prior_piano/rave")).eval()
model(z)
model.to(device)

# %%
import librosa as li
import soundfile as sf
import math
import numpy as np

x, sr = li.load("nocturne.wav", sr=None)
x = x[:2**int(math.floor(math.log2(len(x))))]
x /= np.max(abs(x))

x = torch.from_numpy(x).reshape(1, 1, -1)

rave_pretrained = torch.jit.load("piano.ts")

# %%
z = rave_pretrained.encode(x).permute(0, 2, 1).reshape(1, -1).to(device)
model(z)  # CACHE PREVIOUS STEPS
z_continuation = model.generate(torch.zeros(1, 4096).to(device))
# %%
z_full = torch.cat([z, z_continuation], -1)
z_full = z_full.reshape(-1, 4).transpose(0, 1).reshape(1, 4, -1).cpu()
y_full = rave_pretrained.decode(z_full.contiguous()).reshape(-1).numpy()
sf.write("continuation.wav", y_full, sr)

# %%
