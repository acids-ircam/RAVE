"""
Sampling rate conversion between a RAVE model and its host (``export --sr``).

The resampling filters live in acids-cc (``cc.Resample1d``, eager or cached
depending on ``cc.use_cached_conv``, acids-cc >= 3.2). This module only keeps
the RAVE interface, for retro-compatibility:

- ``Resampler(target_sr, model_sr)`` with ``to_model_sampling_rate`` /
  ``from_model_sampling_rate``, as used by ``rave.scripting`` ;
- ``model_sr`` / ``target_sr`` attributes (and the legacy ``taget_sr``
  spelling), read back from scripted models by ``rave.uncompiling`` ;
- a gin configurable ``rave.Resampler``, so that the filter can be tuned from
  a gin file (``rave.Resampler.lowpass_filter_width = 64``).

Models exported before keep the resampler compiled in their TorchScript file.
"""
import cached_conv as cc
import gin
import torch.nn as nn


def _is_power_of_2(n: int) -> bool:
    return n > 0 and not (n & (n - 1))


@gin.configurable(module="rave")
class Resampler(nn.Module):
    """
    Converts audio between the host rate ``target_sr`` and the model rate
    ``model_sr``. Extra keyword arguments (``lowpass_filter_width``,
    ``rolloff``, ``attenuation``, ``beta``) are given to ``cc.Resample1d``.

    In streaming mode (``cc.use_cached_conv(True)``), each audio buffer must
    be a multiple of the ``down`` factor of each direction; since nn~ / host
    buffers are powers of 2, these factors must be powers of 2 too.
    """

    def __init__(self, target_sr: int, model_sr: int, **kwargs):
        super().__init__()
        assert target_sr != model_sr, "identical source and target rates"

        self.model_sr = int(model_sr)
        self.target_sr = int(target_sr)
        self.taget_sr = self.target_sr  # legacy attribute name

        self.downsample = cc.Resample1d(target_sr, model_sr, **kwargs)
        self.upsample = cc.Resample1d(model_sr, target_sr, **kwargs)

        if cc.is_cached():
            for name, m in [("encode", self.downsample),
                            ("decode", self.upsample)]:
                if not _is_power_of_2(m.down):
                    raise ValueError(
                        f"When using streaming mode, resampling from "
                        f"{target_sr} to {model_sr} needs {name} buffers "
                        f"multiple of {m.down}: the resampling ratio must "
                        f"be a power of 2")

        # integer ratio (former attribute), 0 for rational ratios
        self.ratio = self.target_sr // self.model_sr \
            if not self.target_sr % self.model_sr else 0

    def to_model_sampling_rate(self, x):
        return self.downsample(x)

    def from_model_sampling_rate(self, x):
        return self.upsample(x)
