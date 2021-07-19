# %%
import torch
import librosa as li
import sounddevice as sd


def play(x):
    sd.play(x.reshape(-1).cpu().numpy(), 24000)
    sd.wait()


torch.set_grad_enabled(False)

x, sr = li.load("/Users/acaillon/Desktop/out_24k/LJ001-0001_0000.wav", None)
model = torch.jit.load("vae.ts").eval()

x = torch.from_numpy(x).reshape(1, 1, -1).float()

z = model.encode(x)
z[:,9:] = torch.randn_like(z[:,9:])

play(model.decode(z))
# %%
