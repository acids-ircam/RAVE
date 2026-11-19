"""CAM prior: transformer streaming, flow, scripting and export in a RAVE model.

    pytest tests/test_cam_prior.py
"""
import math
from pathlib import Path

import gin
import pytest
import torch
import cached_conv as cc

import rave
from rave.prior.cam import CAMCore, CAMStreamer, LatentCodec, add_noise
from rave.prior.cam_networks import CAMTransformer, window_mask

LATENT = 8


@pytest.fixture(autouse=True)
def seed():
    # other tests of the suite may leave autograd disabled
    with torch.enable_grad():
        torch.manual_seed(0)
        yield


@pytest.fixture(scope="module")
def parsed_config():
    gin.clear_config()
    gin.parse_config_files_and_bindings(
        [str(rave.BASE_PATH / "configs" / "prior" / "cam.gin")],
        ["CAMTransformer.embed_dim = 32", "CAMTransformer.n_heads = 2", "CAMTransformer.n_layers = 2",
         "CAMTransformer.local_attention_size = 6", "FlowHead.model_channels = 32", "FlowHead.n_blocks = 2"],
        finalize_config=False)
    yield
    gin.clear_config()


def make_core(parsed_config) -> CAMCore:
    core = CAMCore(LATENT, 16)
    with torch.no_grad():
        # the head is initialised with a zero output: perturb it, as after some training
        for p in core.head.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return core.eval()


def test_window_mask():
    mask = window_mask(5, 5, 3, torch.device("cpu"))
    assert mask[4].tolist() == [False, False, True, True, True]
    assert mask[0].tolist() == [True, False, False, False, False]
    # queries are the last frames of the keys
    assert torch.equal(window_mask(2, 5, 3, torch.device("cpu")), window_mask(5, 5, 3, torch.device("cpu"))[-2:])
    assert window_mask(4, 4, 0, torch.device("cpu")).sum() == 10  # unlimited: plain causal


def test_causal_and_local(parsed_config):
    net = CAMTransformer(LATENT, 16).eval()  # window: 6
    x = torch.randn(2, LATENT, 20)
    y = net(x)
    # future frames do not change the past
    x2 = x.clone()
    x2[..., 12:] = torch.randn_like(x2[..., 12:])
    assert torch.allclose(net(x2)[..., :12], y[..., :12], atol=1e-5)
    # frames out of the window of a context do not change it (frame 0 is 6 steps before frame 6...). The window
    # applies at each layer, so the receptive field is n_layers * (window - 1) + 1 = 11 frames
    x3 = x.clone()
    x3[..., 0] = torch.randn_like(x3[..., 0])
    assert torch.allclose(net(x3)[..., 11:], y[..., 11:], atol=1e-5)
    assert not torch.allclose(net(x3)[..., :6], y[..., :6], atol=1e-5)


@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_streaming_matches_full(parsed_config, chunk):
    net = CAMTransformer(LATENT, 16).eval()
    x = torch.randn(2, LATENT, 21)
    full = net(x)
    cache = net.init_cache(2, x.device)
    outs = []
    for i in range(0, x.shape[-1], chunk):
        out, cache = net.step(x[..., i:i + chunk], cache)
        outs.append(out)
        assert cache.shape[4] <= net.window - 1  # the cache is the attention window
    assert torch.allclose(torch.cat(outs, -1), full, atol=1e-5)


def test_noise_modes():
    x = torch.randn(4, LATENT, 10)
    assert torch.equal(add_noise(x, "none"), x)
    assert add_noise(x, "convex").shape == x.shape
    assert add_noise(x, "linear").shape == x.shape
    with pytest.raises(ValueError):
        add_noise(x, "other")


def test_flow_loss_and_sampling(parsed_config):
    core = make_core(parsed_config).train()
    z = torch.randn(3, LATENT, 12)
    loss = core.flow_loss(z)
    loss.backward()
    assert loss.ndim == 0 and math.isfinite(loss.item())
    assert all(p.grad is not None for p in core.net.parameters())
    assert all(p.grad is not None for p in core.head.parameters())
    out = core.sample(10, batch_size=2, steps=3)
    assert out.shape == (2, LATENT, 10)
    cont = core.sample(10, batch_size=2, steps=3, initial=z[:2, :, :4])
    assert cont.shape == (2, LATENT, 10) and torch.equal(cont[..., :4], z[:2, :, :4])


def test_streamer_script(parsed_config):
    streamer = CAMStreamer(make_core(parsed_config), ratio=2048)
    scripted = torch.jit.script(streamer)
    temp = torch.zeros(1, 1, 5)
    z = scripted.generate(temp, 3)
    assert z.shape == (1, LATENT, 5)
    # the state persists between calls (cache = window), and resets with the batch size
    assert streamer.core.net.window - 1 >= scripted.cache.shape[4]
    scripted.generate(torch.zeros(2, 1, 4), 3).shape == (2, LATENT, 4)
    scripted.reset()
    assert scripted.generate(temp, 3).shape == (1, LATENT, 5)


def test_streamer_matches_core(parsed_config):
    """streamed generation == core.sample (same seed)."""
    core = make_core(parsed_config)
    streamer = CAMStreamer(core, ratio=2048)
    torch.manual_seed(1)
    a = streamer.generate(torch.zeros(1, 1, 9), 4)
    torch.manual_seed(1)
    b = core.sample(9, batch_size=1, steps=4)
    assert torch.allclose(a, b, atol=1e-5)


def test_listen_weight_zero_is_prior(parsed_config):
    core = make_core(parsed_config)
    streamer = CAMStreamer(core, ratio=2048)
    guide = torch.randn(1, LATENT, 6)
    torch.manual_seed(2)
    free = streamer.generate(torch.zeros(1, 1, 6), 4)
    streamer.reset()
    torch.manual_seed(2)
    listened = streamer.generate_listen(guide, 4, 1., 0.)
    assert torch.allclose(free, listened, atol=1e-5)
    streamer.reset()
    torch.manual_seed(2)
    driven = streamer.generate_listen(guide, 4, 1., 1.5)
    assert not torch.allclose(free, driven, atol=1e-3)


def test_listen_is_causal(parsed_config):
    """output frame t is guided by the listened frames < t only."""
    streamer = CAMStreamer(make_core(parsed_config), ratio=2048)
    guide = torch.randn(1, LATENT, 6)
    other = guide.clone()
    other[..., 3:] = torch.randn_like(other[..., 3:])
    torch.manual_seed(3)
    a = streamer.generate_listen(guide, 4, 1., 1.)
    streamer.reset()
    torch.manual_seed(3)
    b = streamer.generate_listen(other, 4, 1., 1.)
    assert torch.allclose(a[..., :4], b[..., :4], atol=1e-5)
    assert not torch.allclose(a[..., 4:], b[..., 4:], atol=1e-5)


def _untrained_run(path: Path) -> Path:
    """small untrained v2 run (config.gin + last.ckpt)."""
    path.mkdir(parents=True, exist_ok=True)
    cc.use_cached_conv(False)
    with rave.core.GinEnv():
        gin.parse_config_files_and_bindings([str(rave.BASE_PATH / "configs" / "v2.gin")], ["CAPACITY = 16"])
        model = rave.RAVE()
        (path / "config.gin").write_text(gin.config_str())
    torch.save({"state_dict": model.state_dict(), "callbacks": {}}, path / "last.ckpt")
    return path


def test_export_with_cam_prior(tmp_path):
    """rave export --prior <CAM run>: prior / prior_full / prior_listen methods, with the same latent
    space as encode_full / decode_full."""
    from rave.scripting import export
    gin.clear_config()  # (the config of the other tests is locked)
    run = _untrained_run(tmp_path / "run")

    # a CAM prior "run": checkpoint + config (as written by `rave train_prior`)
    probe = torch.jit.load(export(run, out_path=tmp_path / "probe", name="probe"))
    full_size = probe.full_latent_size
    ratio = 2**14 // probe.encode_full(torch.zeros(1, 1, 2**14)).shape[-1]
    with rave.core.GinEnv():
        gin.parse_config_files_and_bindings(
            [str(rave.BASE_PATH / "configs" / "prior" / "cam.gin")],
            ["CAMTransformer.embed_dim = 32", "CAMTransformer.n_heads = 2", "CAMTransformer.n_layers = 2",
             "CAMTransformer.local_attention_size = 6", "FlowHead.model_channels = 32", "FlowHead.n_blocks = 2"])
        prior_dir = tmp_path / "prior"
        prior_dir.mkdir()
        (prior_dir / "config.gin").write_text(gin.config_str())
        core = CAMCore(full_size, 16)
    state = {f"core.{k}": v for k, v in core.state_dict().items()}
    state["ratio"] = torch.tensor(ratio)
    torch.save({"state_dict": state}, prior_dir / "last.ckpt")

    ts_path = export(run, out_path=tmp_path / "out", name="m", streaming=True, prior=str(prior_dir), latent_size=8)
    model = torch.jit.load(ts_path).eval()
    with torch.no_grad():
        control = torch.zeros(1, 1, 5)
        assert model.prior(control).shape == (1, 8, 5)  # projected, like encode
        assert model.prior_full(control).shape == (1, full_size, 5)
        assert model.decode_full(model.prior_full(control)).shape[-1] == 5 * ratio
        z = model.encode_full(torch.randn(1, 1, 5 * ratio) * .1)
        model.set_prior_listen_weight(1.)
        assert model.prior_listen(z).shape == (1, full_size, 5)
        assert model.set_prior_steps(3) == 0 and model.get_prior_steps() == 3
