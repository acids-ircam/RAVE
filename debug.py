# %%
import torch
import torch.nn as nn
import librosa as li
from random import choice
from glob import glob
import matplotlib.pyplot as plt
import sounddevice as sd

from fd.parallel_model import use_buffer_conv

use_buffer_conv(True)

from fd.parallel_model.model import ParallelModel
import numpy as np
from scipy.signal import resample
from export_model import model

torch.set_grad_enabled(False)

audio = choice(glob("/Users/acaillon/Desktop/out_24k/*.wav"))
x, sr = li.load(audio, None)
assert sr == 24000

x = x[:2**16]

# model = ParallelModel.load_from_checkpoint(
#     glob("lightning_logs/version_0/checkpoints/*.ckpt")[0])
# model.eval()
# model = torch.jit.load("traced_model_24kHz_16z.torchscript").eval()

x_ = torch.from_numpy(x).float().reshape(1, 1, -1)
# mean, scale = model.encoder(x_)
# z, _ = model.reparametrize(mean, scale)
z = model.encode(x_)

# z = z - model.latent_mean.reshape(1, -1, 1)
# z = nn.functional.conv1d(z, model.latent_pca.unsqueeze(-1))

plt.plot(z[0].T + 3 * torch.arange(16), "k-")
plt.show()

# z = nn.functional.conv1d(z, model.latent_pca.T.unsqueeze(-1))
# z = z + model.latent_mean.reshape(1, -1, 1)

# y = model.decoder(z).numpy().reshape(-1)
y = model.decode(z).numpy().reshape(-1)

sd.play(x, sr)
sd.wait()
sd.play(y, sr)

print(x.reshape(-1).shape[0] // z.reshape(-1).shape[0])

# %%
N = 64
x = np.random.randn(1, 16, N)
x = resample(x, 128, axis=-1)
z = torch.from_numpy(x).float()

# z = nn.functional.conv1d(z, model.latent_pca.T.unsqueeze(-1))

# y = model.decoder(z).numpy().reshape(-1)
y = model.decode(z).numpy().reshape(-1)
sd.play(y, sr)

# %%
