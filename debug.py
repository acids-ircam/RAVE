# %%
import torch

torch.set_grad_enabled(False)
import torch.nn as nn

from fd.parallel_model.model import ParallelModel

model = ParallelModel.load_from_checkpoint(
    "lightning_logs/version_0/checkpoints/last.ckpt").eval()


def generate(z, generator):
    z = generator.pre_net(z)
    z = generator.net(z)
    z = generator.post_net(z)

    loudness = z[:, :1]
    waveform = z[:, 1:]

    waveform = torch.tanh(waveform) * loudness

    return waveform, loudness


# %%
import librosa as li
from random import choice
from glob import glob
import matplotlib.pyplot as plt
import torch

x, sr = li.load(
    choice(
        glob("/slow-2/antoine/dataset/ljspeech/LJSpeech-1.1/out_24k/*.wav")),
    None)

x = x[:2**16]
x = torch.from_numpy(x).float().reshape(1, 1, -1)

z = model.reparametrize(*model.encoder(model.pqmf(x)))[0]

l_true = model.loudness(x)

y, l_fake = generate(z, model.decoder)

# plt.plot(l_true.reshape(-1))
plt.plot(l_fake.reshape(-1))
