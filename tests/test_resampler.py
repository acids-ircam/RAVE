"""rave.resampler.Resampler: RAVE interface over cc.Resample1d."""
import math

import cached_conv as cc
import gin
import pytest
import torch

from rave.resampler import Resampler

# (target_sr, model_sr): ratios 2, 3, 4, 8 and a non power of 2 even one
INTEGER = [(44100, 22050), (66150, 22050), (88200, 22050), (176400, 22050),
           (48000, 16000), (96000, 16000)]
RATIONAL = [(48000, 44100), (44100, 48000)]
POW2 = [(44100, 22050), (88200, 22050), (176400, 22050)]


def ids(configs):
    return [f"{t}<->{m}" for t, m in configs]


@pytest.fixture(autouse=True)
def clean_state():
    gin.clear_config()
    initial = cc.is_cached()
    with torch.no_grad():
        yield
    cc.use_cached_conv(initial)
    gin.clear_config()


def tone(freq, sr, length):
    t = torch.arange(length, dtype=torch.float64) / sr
    return torch.sin(2 * math.pi * freq * t).float()[None, None]


def rms_db(x):
    return 20 * math.log10(x.pow(2).mean().sqrt().item() + 1e-12)


@pytest.mark.parametrize("target_sr,model_sr", INTEGER + RATIONAL,
                         ids=ids(INTEGER + RATIONAL))
def test_offline(target_sr, model_sr):
    cc.use_cached_conv(False)
    r = Resampler(target_sr, model_sr)
    assert (r.model_sr, r.target_sr, r.taget_sr) == \
        (model_sr, target_sr, target_sr)
    length = 3 * 2**12 * r.downsample.down
    x = torch.randn(2, 1, length)
    y = r.to_model_sampling_rate(x)
    assert y.shape[-1] == length * model_sr // target_sr
    z = r.from_model_sampling_rate(y)
    assert z.shape == x.shape


@pytest.mark.parametrize("target_sr,model_sr", POW2, ids=ids(POW2))
def test_streaming(target_sr, model_sr):
    """cached resampler, fed by buffers, matches the offline one"""
    cc.use_cached_conv(False)
    eager = Resampler(target_sr, model_sr)
    cc.use_cached_conv(True)
    cached = Resampler(target_sr, model_sr)

    buffer = 2048
    x = torch.randn(1, 1, 32 * buffer)
    for direction, rate in [("to", target_sr), ("from", model_sr)]:
        f_eager = getattr(eager, f"{direction}_model_sampling_rate")
        f_cached = getattr(cached, f"{direction}_model_sampling_rate")
        y = f_eager(x)
        cy = torch.cat([f_cached(b) for b in torch.split(x, buffer, -1)], -1)
        assert cy.shape == y.shape
        m = getattr(cached, "downsample" if direction == "to" else
                    "upsample")
        d = m.cumulative_delay
        torch.testing.assert_close(cy[..., d:], y[..., :y.shape[-1] - d],
                                   rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("target_sr,model_sr",
                         [(66150, 22050), (48000, 16000), (48000, 44100)],
                         ids=["ratio3", "ratio3b", "rational"])
def test_streaming_rejects_non_pow2(target_sr, model_sr):
    cc.use_cached_conv(True)
    with pytest.raises(ValueError):
        Resampler(target_sr, model_sr)


@pytest.mark.parametrize("target_sr,model_sr", INTEGER + RATIONAL,
                         ids=ids(INTEGER + RATIONAL))
def test_antialiasing(target_sr, model_sr):
    """acids-ircam/RAVE#359: tones above the model Nyquist are removed"""
    cc.use_cached_conv(False)
    r = Resampler(target_sr, model_sr)
    length = r.downsample.down * 2**13
    margin = 1024
    nyq_model, nyq_target = model_sr / 2, target_sr / 2
    if nyq_target <= nyq_model * 1.05:
        pytest.skip("no stopband to test")
    freq = nyq_model + .5 * (nyq_target - nyq_model)
    y = r.to_model_sampling_rate(tone(freq, target_sr, length))
    assert rms_db(y[..., margin:-margin]) < -40
    # passband tone keeps its level through a round trip
    x = tone(.3 * nyq_model, target_sr, length)
    z = r.from_model_sampling_rate(r.to_model_sampling_rate(x))
    assert abs(rms_db(z[..., margin:-margin]) -
               rms_db(x[..., margin:-margin])) < .1


def test_gin_options():
    gin.parse_config("rave.Resampler.lowpass_filter_width = 8")
    short = Resampler(44100, 22050)
    gin.clear_config()
    default = Resampler(44100, 22050)
    assert short.downsample.kernel.shape[-1] < \
        default.downsample.kernel.shape[-1]


@pytest.mark.parametrize("target_sr,model_sr", [(44100, 22050)])
def test_script(target_sr, model_sr):
    for cached in (False, True):
        cc.use_cached_conv(cached)
        r = Resampler(target_sr, model_sr)
        x = torch.zeros(1, 1, 2048)
        r.from_model_sampling_rate(r.to_model_sampling_rate(x))
        torch.jit.script(r)


@pytest.mark.parametrize("scale", [2, 4])
def test_msd_discriminator(scale):
    """DescriptDiscriminator(rates=...) builds MSDs, with cc.Resample1d"""
    from rave.descript_discriminator import DescriptDiscriminator
    d = DescriptDiscriminator(rates=[scale], periods=[], fft_sizes=[],
                              n_channels=2)
    msd = d.discriminators[0]
    assert msd.downsampler.down == scale
    x = torch.randn(1, 2, 2**14)
    assert msd.downsampler(x).shape[-1] == 2**14 // scale
    d(x)
