# %%
import torch
import librosa as li
from random import choice
from glob import glob
import sounddevice as sd
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

audio = choice(glob("/Users/acaillon/Desktop/out_24k/*.wav"))
x, sr = li.load(audio, None)
assert sr == 24000

x = x[:2**16].reshape(1, 1, -1)
x = torch.from_numpy(x).float()

model = torch.jit.load("traced_model_24kHz_16z.torchscript").eval()

y = model(x)

xs = torch.split(x, 4096, -1)
ys = []
for _x in xs:
    ys.append(model(_x))
ys = torch.cat(ys, -1)


plt.plot(ys.reshape(-1))
sd.play(ys.reshape(-1).numpy(), sr)
sd.wait()
sd.play(y.reshape(-1).numpy(), sr)

# %%
