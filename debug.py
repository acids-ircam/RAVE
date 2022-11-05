import torchaudio
import torch
from rave.core import MultiScaleSTFT
import numpy as np

x = torch.randn(1, 1, 2**16)
stft = MultiScaleSTFT(
    2**np.arange(5, 12),
    44100,
    magnitude=False,
    num_mels=64,
)

for y in stft(x):
    print(y.dtype, y.shape)