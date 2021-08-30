# %%
import torch
import torch.nn as nn
from os import system
import matplotlib.pyplot as plt

system(
    "python export_model.py --run last-v1.ckpt --sr 48000 --cached true --latent-size 128"
)

#%%
torch.set_grad_enabled(False)

import librosa as li
from random import choice
from glob import glob

import soundfile as sf
from tqdm import tqdm

from fd.parallel_model.model import ParallelModel

audio = choice(glob("/slow-2/antoine/dataset/wheel/out_48k/*.wav"))
x, sr = li.load(audio, 48000)
x = x[:2**18]

model_script = torch.jit.load("vae.ts").eval()
model_check = ParallelModel.load_from_checkpoint("last-v1.ckpt",
                                                 strict=False).eval()


def generate(x):
    if model_check.pqmf is not None:
        x = model_check.pqmf(x)

    mean, scale = model_check.encoder(x)
    z, _ = model_check.reparametrize(mean, scale)
    y = model_check.decoder(z)

    if model_check.pqmf is not None:
        y = model_check.pqmf.inverse(y)
    return y


x = torch.from_numpy(x).float().reshape(1, 1, -1)

y_c = generate(x)
y_s = model_script(x)

y = torch.cat([y_c, y_s], -1).reshape(-1).numpy()
sf.write("eval.wav", y, sr)
