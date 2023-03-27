#%%
import librosa as li
import matplotlib.pyplot as plt
import torch
from torchaudio.transforms import Spectrogram, MelSpectrogram

x, sr = li.load(li.example("trumpet"), sr=44100)

x = torch.from_numpy(x).reshape(1, -1).float()
# %%

spec = Spectrogram(2048, 2048, 512, normalized=True)
spec = MelSpectrogram(44100, 2048, 2048, 512, normalized=True, n_mels=128)
s = torch.log1p(spec(x))
# %%
plt.matshow(s[0])
plt.colorbar()
# %%
# %%
