import torch
import pytest

from rave.stft import CausalSTFT, StreamableSTFT


@pytest.mark.parametrize("skip_features", [None, -1, 1])
def test_streamable_stft_multichannel_roundtrip(skip_features):
    torch.manual_seed(0)
    stft = StreamableSTFT(nfft=256,
                          hop_size=64,
                          stream=False,
                          skip_features=skip_features)
    x = torch.randn(2, 3, 2048)

    spec = stft(x)
    y = stft.inverse(spec)

    expected_freq = 129
    if skip_features is not None:
        expected_freq -= abs(skip_features)

    assert spec.shape[:3] == (2, 3, 2)
    assert spec.shape[-2:] == (expected_freq, 32)
    assert y.shape == x.shape
    if skip_features is None:
        assert torch.allclose(y[..., :-stft.nfft], x[..., :-stft.nfft],
                              atol=1e-4, rtol=1e-4)
    else:
        assert torch.isfinite(y).all()


def test_streamable_stft_streaming_matches_non_streaming():
    torch.manual_seed(0)
    nfft = 1024
    hop_size = 512
    n_fade = 4
    x = torch.randn(2, 3, hop_size * 40)

    full = StreamableSTFT(nfft=nfft,
                          hop_size=hop_size,
                          stream=False,
                          skip_features=-1)
    stream = StreamableSTFT(nfft=nfft,
                            hop_size=hop_size,
                            stream=True,
                            skip_features=-1)

    full_y = full.inverse(full(x))

    chunks = x.split(hop_size, dim=-1)
    stream_y = torch.cat(
        [stream.inverse(stream(chunk)) for chunk in chunks],
        dim=-1,
    )

    delay = n_fade * hop_size + (nfft - hop_size) // 2
    trim = delay + nfft
    assert stream_y.shape == x.shape

    assert torch.allclose(
        stream_y[..., delay + trim:],
        full_y[..., trim:-delay],
        atol=1e-3,
        rtol=1e-3,
    )


def test_causal_stft_exact_streaming_overlap_add():
    torch.manual_seed(0)
    nfft = 256
    hop_size = 64
    x = torch.randn(2, 3, hop_size * 48)

    full = CausalSTFT(nfft=nfft,
                      hop_size=hop_size,
                      skip_features=None,
                      stream=False)
    stream = CausalSTFT(nfft=nfft,
                        hop_size=hop_size,
                        skip_features=None,
                        stream=True)

    full_y = full.inverse(full(x))
    stream_y = torch.cat(
        [stream.inverse(stream(chunk)) for chunk in x.split(hop_size, -1)],
        -1,
    )

    assert stream_y.shape == x.shape
    assert torch.allclose(stream_y[..., nfft:], full_y[..., nfft:],
                          atol=1e-5, rtol=1e-5)

    delay = full.delay
    assert torch.allclose(
        full_y[..., delay:-nfft],
        x[..., :-delay - nfft],
        atol=1e-5,
        rtol=1e-5,
    )


def test_causal_stft_inverse_exact_from_stream_start():
    # no amplification of the window edges on the first frames (silent frames before the start)
    torch.manual_seed(0)
    nfft, hop_size = 1024, 512
    x = torch.randn(1, 1, hop_size * 16)
    for stream in (False, True):
        stft = CausalSTFT(nfft=nfft, hop_size=hop_size, skip_features=None, stream=stream)
        if stream:
            y = torch.cat([stft.inverse(stft(c)) for c in x.split(hop_size * 4, -1)], -1)
        else:
            y = stft.inverse(stft(x))
        assert torch.allclose(y[..., stft.delay:], x[..., :-stft.delay], atol=1e-4, rtol=1e-4)
        # a spectrum that is not the STFT of a signal (e.g. a decoder output): the first frames
        # must not be divided by the (near-zero) edges of a single squared window
        spec = torch.randn(1, 1, 2, nfft // 2 + 1, 16) * 0.1
        stft = CausalSTFT(nfft=nfft, hop_size=hop_size, skip_features=None, stream=stream)
        y = stft.inverse(spec) if not stream else torch.cat(
            [stft.inverse(s) for s in spec.split(4, -1)], -1)
        steady = y[..., nfft:].abs().max()
        assert y[..., :nfft].abs().max() < 2 * steady


def test_causal_stft_buffers_do_not_keep_previous_graph():
    torch.manual_seed(0)
    stft = CausalSTFT(nfft=256, hop_size=64, skip_features=-1, stream=False)

    for _ in range(2):
        x = torch.randn(2, 1, 64 * 8, requires_grad=True)
        spec = stft(x)
        loss = spec.square().mean()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_causal_stft_inverse_buffers_do_not_keep_previous_graph():
    torch.manual_seed(0)
    stft = CausalSTFT(nfft=256, hop_size=64, skip_features=-1, stream=False)

    for _ in range(2):
        spec = torch.randn(2, 1, 2, 128, 8, requires_grad=True)
        y = stft.inverse(spec)
        loss = y.square().mean()
        loss.backward()
        assert spec.grad is not None
        assert torch.isfinite(spec.grad).all()


def test_causal_stft_non_streaming_does_not_reuse_state():
    torch.manual_seed(0)
    stft = CausalSTFT(nfft=256,
                      hop_size=64,
                      skip_features=None,
                      stream=False)
    x = torch.randn(2, 1, 64 * 8)

    y1 = stft.inverse(stft(x))
    y2 = stft.inverse(stft(x))

    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    test_streamable_stft_streaming_matches_non_streaming()
    test_streamable_stft_multichannel_roundtrip()
