import cached_conv as cc
import numpy as np
import torch
import torch.nn as nn

from .pqmf import kaiser_filter


class Resampler(nn.Module):

    def __init__(self, target_sr, model_sr):
        super().__init__()
        assert target_sr != model_sr, "identical source and target rates"

        self.model_sr = model_sr
        self.taget_sr = target_sr

        ratio = target_sr // model_sr
        assert int(ratio) == ratio

        if ratio % 2 and cc.USE_BUFFER_CONV:
            raise ValueError(
                f"When using streaming mode, resampling ratio must be a power of 2, got {ratio}"
            )

        wc = np.pi / ratio
        filt = kaiser_filter(wc, 140)
        filt = torch.from_numpy(filt).float()

        self.downsample = cc.Conv1d(
            1,
            1,
            len(filt),
            stride=ratio,
            padding=cc.get_padding(len(filt), ratio),
            bias=False,
        )

        self.downsample.weight.data.copy_(filt.reshape(1, 1, -1))

        pad = len(filt) % ratio

        filt = nn.functional.pad(filt, (pad, 0))
        filt = filt.reshape(-1, ratio).permute(1, 0)

        pad = (filt.shape[-1] + 1) % 2
        filt = nn.functional.pad(filt, (pad, 0)).unsqueeze(1)

        self.upsample = cc.Conv1d(1,
                                  ratio,
                                  filt.shape[-1],
                                  stride=1,
                                  padding=cc.get_padding(filt.shape[-1]),
                                  bias=False)

        self.upsample.weight.data.copy_(filt)

        self.ratio = ratio

    def to_model_sampling_rate(self, x):
        return self.downsample(x)

    def from_model_sampling_rate(self, x):
        x = self.upsample(x)  # B x 2 x T
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1).unsqueeze(1)
        return x
