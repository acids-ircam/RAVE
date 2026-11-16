"""V4 output-head defaults and compatibility with saved legacy configurations."""

import cached_conv as cc
import gin
import pytest
import torch
from torch import nn

import rave
from rave import blocks


@pytest.fixture(autouse=True)
def clean_config():
    gin.clear_config()
    cc.use_cached_conv(False)
    yield
    gin.clear_config()
    cc.use_cached_conv(False)


def small_decoder(**overrides):
    kwargs = dict(data_size=2, n_freqs=64, capacity=2, n_channels=1,
                  ratios_time=[2, 2], ratios_freq=[2, 2],
                  ratios_channels=[1, 2, 2], latent_size=8, kernel_size=3)
    kwargs.update(overrides)
    return blocks.Generator2D(**kwargs)


def test_generator_constructor_keeps_legacy_head_and_state_keys():
    model = small_decoder()
    assert tuple(model.outconv.weight.shape) == (2, 4, 7, 7)
    assert isinstance(model.output_smoothing, nn.Identity)
    assert not any(name.startswith('output_smoothing.') for name in model.state_dict())
    restored = small_decoder()
    restored.load_state_dict(model.state_dict(), strict=True)
    z = torch.randn(2, 8, 4)
    torch.testing.assert_close(model(z), restored(z), rtol=0, atol=0)


def test_new_v4_config_defaults():
    gin.parse_config_files_and_bindings(['v4.gin'], [])
    assert gin.query_parameter('%CAPACITY') == 32
    decoder = gin.get_configurable('blocks.Generator2D')(n_channels=1)
    assert tuple(decoder.outconv.weight.shape) == (2, 64, 3, 3)
    assert tuple(decoder.output_smoothing.weight.shape) == (2, 2, 7, 7)


@pytest.mark.parametrize('capacity', [32, 64])
def test_saved_legacy_config_and_checkpoint_roundtrip(tmp_path, capacity):
    # Simulate the fully resolved config saved before the new head options existed.
    gin.parse_config_files_and_bindings(['v4.gin'], [
        f'CAPACITY={capacity}',
        'blocks.Generator2D.output_kernel_size=7',
        'blocks.Generator2D.output_smoothing_kernel_size=0',
    ])
    original = rave.RAVE().eval()
    saved_config = '\n'.join(line for line in gin.config_str().splitlines()
                             if 'Generator2D.output_kernel_size' not in line
                             and 'Generator2D.output_smoothing_kernel_size' not in line)
    assert 'output_smoothing' not in saved_config
    (tmp_path / 'config.gin').write_text(saved_config)
    torch.save({'state_dict': original.state_dict()}, tmp_path / 'last.ckpt')
    gin.clear_config()
    restored, _ = rave.load_rave_checkpoint(tmp_path, keep_config=True)
    restored.eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    assert restored.decoder.outconv.weight.shape == original.decoder.outconv.weight.shape
    assert tuple(restored.decoder.outconv.weight.shape) == (2, 2*capacity, 7, 7)
    assert isinstance(restored.decoder.output_smoothing, nn.Identity)
    with torch.inference_mode():
        z = torch.randn(1, 128, 1)
        torch.testing.assert_close(original.decoder(z), restored.decoder(z), rtol=0, atol=0)


def test_new_checkpoint_config_preserves_new_head(tmp_path):
    gin.parse_config_files_and_bindings(['v4.gin'], ['CAPACITY=2', 'LATENT_SIZE=8'])
    original = rave.RAVE().eval()
    (tmp_path / 'config.gin').write_text(gin.config_str())
    torch.save({'state_dict': original.state_dict()}, tmp_path / 'last.ckpt')
    gin.clear_config()
    restored, _ = rave.load_rave_checkpoint(tmp_path)
    restored.load_state_dict(original.state_dict(), strict=True)
    assert tuple(restored.decoder.outconv.weight.shape[-2:]) == (3, 3)
    assert tuple(restored.decoder.output_smoothing.weight.shape) == (2, 2, 7, 7)


@pytest.mark.parametrize('channels', [1, 2])
def test_two_convolution_head_shape_and_gradients(channels):
    model = small_decoder(n_channels=channels, output_kernel_size=3,
                          output_smoothing_kernel_size=7)
    z = torch.randn(2, 8, 4, requires_grad=True)
    y = model(z)
    assert y.shape == (2, channels, 2, 32, 16)
    y.square().mean().backward()
    assert torch.isfinite(z.grad).all()
    for prefix in ['outconv.', 'output_smoothing.']:
        parameters = [p for name, p in model.named_parameters() if name.startswith(prefix)]
        assert parameters and all(p.grad is not None and torch.isfinite(p.grad).all()
                                  for p in parameters)


@pytest.mark.parametrize('smoothing', [0, 7])
def test_streaming_head_is_scriptable_and_matches_eager(smoothing):
    cc.use_cached_conv(True)
    eager = small_decoder(output_kernel_size=3 if smoothing else 7,
                          output_smoothing_kernel_size=smoothing).eval()
    target = small_decoder(output_kernel_size=3 if smoothing else 7,
                           output_smoothing_kernel_size=smoothing).eval()
    target.load_state_dict(eager.state_dict(), strict=True)
    # Cached-conv allocates its streaming buffers on the first forward call.
    with torch.inference_mode():
        warmup = torch.zeros(1, 8, 1)
        eager(warmup)
        target(warmup)
    scripted = torch.jit.script(target)
    with torch.inference_mode():
        for _ in range(3):
            z = torch.randn(1, 8, 1)
            expected = eager(z)
            actual = scripted(z)
            assert actual.shape == (1, 1, 2, 32, 4)
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('kwargs', [dict(output_kernel_size=0), dict(output_kernel_size=4),
                                   dict(output_smoothing_kernel_size=-1),
                                   dict(output_smoothing_kernel_size=4)])
def test_invalid_output_kernel_sizes(kwargs):
    with pytest.raises(ValueError, match='odd integer'):
        small_decoder(**kwargs)


def test_new_head_full_streaming_export(tmp_path):
    from rave.scripting import export

    gin.parse_config_files_and_bindings(['v4.gin'], ['CAPACITY=2', 'LATENT_SIZE=8'])
    model = rave.RAVE().eval()
    (tmp_path / 'config.gin').write_text(gin.config_str())
    torch.save({'state_dict': model.state_dict()}, tmp_path / 'last.ckpt')
    gin.clear_config()
    target = export(tmp_path, out_path=tmp_path / 'new_head.ts',
                    streaming=True, latent_size=8)
    scripted = torch.jit.load(str(target)).eval()
    with torch.inference_mode():
        for _ in range(3):
            y = scripted(torch.randn(1, 1, 2048) * 0.1)
            assert y.shape == (1, 1, 2048)
            assert torch.isfinite(y).all()
