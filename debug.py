import torch

torch.set_grad_enabled(False)

import librosa as li
from random import choice
from glob import glob

import sounddevice as sd
from tqdm import tqdm

from fd.parallel_model.model import ParallelModel


def play(x):
    sd.play(x.reshape(-1).numpy(), 48000)
    sd.wait()


audio = choice(
    glob("/Users/acaillon/Downloads/The Wheel of Time/00 - New Spring/*.wav"))
x, sr = li.load(audio, 48000)
x = x[:2**16]

model_script = torch.jit.load("vae.ts").eval()
model_check = ParallelModel.load_from_checkpoint("checkpoints/last-v1.ckpt", strict=False).eval()

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
y_script = model_script(x)
y_check = generate(x)

play(x)
play(y_check)
play(y_script)
