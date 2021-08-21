# %%
from fd.parallel_model.core import Loudness
import torch
import matplotlib.pyplot as plt
import librosa as li
import sounddevice as sd

x, sr = li.load("/Users/acaillon/Desktop/out_24k/LJ001-0173_0000.wav", None)
sd.play(x, sr)
sd.wait()

x = torch.from_numpy(x).reshape(1, 1, -1)

l = Loudness(24000, 512)
y = l(x)
# %%
plt.plot(y.reshape(-1))
# %%
