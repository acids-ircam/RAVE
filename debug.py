# %%
import torch
import numpy as np
import matplotlib.pyplot as plt


def derivative(x: torch.Tensor) -> torch.Tensor:
    return x[..., 1:] - x[..., :-1]


def unwrap(x: torch.Tensor) -> torch.Tensor:
    x = derivative(x)
    x = (x + np.pi) % (2 * np.pi)
    return (x - np.pi).cumsum(-1)


# %%
import librosa as li
import numpy as np

x, sr = li.load(li.example("sweetwaltz"), sr=None)

f = (np.ones(10 * sr) * 440 + np.random.randn(10*sr)*10).cumsum()
x = np.cos(2 * np.pi * f / sr)

x = x / np.max(abs(x))
x = torch.from_numpy(x).float().reshape(-1)

x = torch.stft(x, 2048, return_complex=True, normalized=True)

ifreq = derivative(unwrap(x.angle()))
mask = torch.clip(torch.log1p(x.abs()[..., 2:]), 0, 1)

plt.imshow(ifreq * mask, aspect="auto", origin="lower")
# %%
plt.imshow(mask, aspect="auto", origin="lower")

# %%
