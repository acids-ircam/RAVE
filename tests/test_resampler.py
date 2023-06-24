import cached_conv as cc
import gin
import pytest
import torch

from rave.resampler import Resampler

configs = [(44100, 22050), (48000, 16000)]


@pytest.mark.parametrize("target_sr,model_sr", configs)
def test_resampler(target_sr, model_sr):
    gin.clear_config()
    cc.use_cached_conv(False)

    resampler = Resampler(target_sr, model_sr)

    x = torch.randn(1, 1, 2**12 * 3)

    y = resampler.to_model_sampling_rate(x)
    z = resampler.from_model_sampling_rate(y)

    assert x.shape == z.shape

    cc.use_cached_conv(True)

    try:
        resampler = Resampler(target_sr, model_sr)

        x = torch.randn(1, 1, 2**12 * 3)

        y = resampler.to_model_sampling_rate(x)
        z = resampler.from_model_sampling_rate(y)

        assert x.shape == z.shape

    except ValueError:
        pass
