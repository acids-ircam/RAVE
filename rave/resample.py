from scipy.signal import kaiserord, firwin
import numpy as np

import torch
import torch.nn as nn

from .pqmf import kaiser_filter

from cached_conv import CachedConv1d, get_padding


class Resampling(nn.Module):
    def __init__(self, target_sr, source_sr):
        super().__init__()
        self.source_sr = source_sr
        self.taget_sr = target_sr

        ratio = target_sr // source_sr
        assert int(ratio) == ratio
        self.identity = target_sr == source_sr

        if self.identity:
            self.upsample = nn.Identity()
            self.downsample = nn.Identity()
            return

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
        return self.downsample(x)

    def to_target_sampling_rate(self, x):
        x = self.upsample(x)  # B x 2 x T
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1).unsqueeze(1)
        return x