from scipy.signal import kaiserord, firwin
import numpy as np

import torch
import torch.nn as nn

from .buffer_conv import CachedConv1d
from .core import get_padding


def kaiser_filter(wc, atten, N=None):
    """
    Computes a kaiser lowpass filter
    Parameters
    ----------
    wc: float
        Angular frequency
    
    atten: float
        Attenuation (dB, positive)
    """
    N_, beta = kaiserord(atten, wc / np.pi)
    N_ = 2 * (N_ // 2) + 1
    N = N if N is not None else N_
    h = firwin(N, wc, window=('kaiser', beta), scale=False, nyq=np.pi)
    return h


class Resampling(nn.Module):
    def __init__(self, target_sr, source_sr):
        super().__init__()
        ratio = target_sr // source_sr
        assert int(ratio) == ratio

        if target_sr == source_sr:
            self.identity = True
            return
        else:
            self.identity = False

        wc = np.pi / ratio
        filt = kaiser_filter(wc, 140)
        filt = torch.from_numpy(filt).float()

        self.downsample = CachedConv1d(
            1,
            1,
            len(filt),
            stride=ratio,
            padding=get_padding(len(filt), ratio),
        )

        self.downsample.weight.data.copy_(filt.reshape(1, 1, -1))
        self.downsample.bias.data.zero_()

        pad = len(filt) % ratio

        filt = nn.functional.pad(filt, (pad, 0))
        filt = filt.reshape(-1, ratio).permute(1, 0)  # ratio  x T

        pad = (filt.shape[-1] + 1) % 2
        filt = nn.functional.pad(filt, (pad, 0)).unsqueeze(1)

        self.upsample = CachedConv1d(
            1,
            2,
            filt.shape[-1],
            stride=1,
            padding=get_padding(filt.shape[-1]),
        )

        self.upsample.weight.data.copy_(filt)
        self.upsample.bias.data.zero_()

        self.ratio = ratio

    def from_target_sampling_rate(self, x):
        if self.identity:
            return x
        return self.downsample(x)

    def to_target_sampling_rate(self, x):
        if self.identity:
            return x
        x = self.upsample(x)  # B x 2 x T
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1).unsqueeze(1)
        return x