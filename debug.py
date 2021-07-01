# %%
import torch
import librosa as li
from random import choice
from glob import glob
import sounddevice as sd
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_grad_enabled(False)

audio = choice(glob("/Users/acaillon/Desktop/out_24k/*.wav"))
x, sr = li.load(audio, sr=48000)
# assert sr == 24000

x = x[:2**16].reshape(1, 1, -1)
x = torch.from_numpy(x).float()

model = torch.jit.load("vae.ts").eval()

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
from time import time

N = 2**14
sr = 24000
x = torch.randn(1, 1, N)
mean = 0
nel = 0
for i in tqdm(range(100)):
    st = time()
    model(x)
    st = time() - st
    nel += 1
    mean += (st - mean) / nel

print("\n")
print(f"RTF: {N / sr / mean:.2f}")
# %%
