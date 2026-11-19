"""Compare regular training updates with the original joint-discriminator objective."""
import ast
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchaudio.transforms import Spectrogram
from einops import rearrange
ROOT = Path(__file__).resolve().parents[1]

# Extract production code without importing optional dataset/training dependencies.
def load_nodes(path, scope, names=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and (names is None or n.name in names)]
    for n in ast.walk(ast.Module(body=nodes, type_ignores=[])):
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)):
            n.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), scope)
scope = dict(torch=torch, nn=nn, F=F, np=np, Spectrogram=Spectrogram, rearrange=rearrange)
load_nodes(Path('rave/stft.py'), scope)
load_nodes(Path('rave/spectro_discriminator.py'), scope)
load_nodes(Path('rave/core.py'), scope, {'mean_difference', 'hinge_gan'})
module = ast.parse((ROOT / 'rave/model.py').read_text())
cls = next((n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'RAVE'))
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {'training_step', 'split_features'}]
for n in methods:
    n.decorator_list = []
scope.update(nullcontext=nullcontext, Profiler=lambda: SimpleNamespace(tick=lambda *a: None), _pqmf_encode=lambda p, x: x)
exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), 'original_training_step', 'exec'), scope)

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 2, 3, padding=1)

    def forward(self, x):
        return self.conv(x)

    def set_warmed_up(self, x):
        pass

    def reparametrize(self, z):
        mean, scale = z.chunk(2, 1)
        return (mean + 0.01 * torch.randn_like(mean) * scale.sigmoid(), 0.01 * (mean.square() + scale.square()).mean())

class Decoder(nn.Sequential):

    def set_warmed_up(self, x):
        pass

class Harness(nn.Module):
    split_features = scope['split_features']

    def __init__(self, dtype):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(nn.Conv1d(1, 4, 3, padding=1), nn.Tanh(), nn.Conv1d(4, 1, 3, padding=1))
        self.discriminator = scope['SpectroDiscriminator'](capacity=2, channels=[1, 2, 4, 4, 4, 8], n_ffts=[128, 256, 512, 1024, 2048, 4096], hop_lengths=[64, 128, 256, 512, 1024, 2048])
        self.double() if dtype == torch.float64 else self.float()
        self.opts = [torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.0001, betas=(0.9, 0.999)), torch.optim.AdamW(self.discriminator.parameters(), lr=0.0001, betas=(0.8, 0.9))]
        self.warmed_up = True
        self.update_discriminator_every = 4
        self.output_mode = 'raw'
        self.pqmf = None
        self.n_channels = 1
        self.valid_signal_crop = False
        self.multiband_audio_distance = None
        self.num_skipped_features = 0
        self.beta_factor = 1.0
        self.weights = {'feature_matching': 100.0}
        self.gradient_clip_gen = self.gradient_clip_dis = None
        self.metrics = {}

    def encode(self, x, return_mb=False):
        z = self.encoder(x)
        return (z, x) if return_mb else z

    def audio_distance(self, x, y):
        return {'mse': (x - y).square().mean()}

    def feature_matching_fun(self, x, y):
        return scope['mean_difference'](x, y, relative=True)

    def gan_loss(self, x, y):
        return scope['hinge_gan'](x, y)

    def optimizers(self):
        return self.opts

    def loss_weight(self, k, step):
        return self.weights.get(k, 1.0)

    def log(self, k, v):
        self.metrics[k] = v.detach().clone() if torch.is_tensor(v) else v

    def log_dict(self, d):
        for k, v in d.items():
            self.log(k, v)

# Reference objective from before the memory optimization.
def original(self, batch, batch_idx):
    x = batch.requires_grad_(True)
    self.encoder.set_warmed_up(self.warmed_up)
    self.decoder.set_warmed_up(self.warmed_up)
    z, reg = self.encoder.reparametrize(self.encoder(x))
    y = self.decoder(z)
    losses = {'fullband_mse': (x - y).square().mean(), 'regularization': reg}
    if self.warmed_up:
        real, fake = self.split_features(self.discriminator(torch.cat([x, y])))
        fm = sum((sum((self.feature_matching_fun(a, b) for a, b in zip(r[:-1], f[:-1]))) / len(r) for r, f in zip(real, fake))) / len(real)
        pairs = [self.gan_loss(r[-1], f[-1]) for r, f in zip(real, fake)]
        dis_loss = 2 * sum((p[0] for p in pairs)) / len(pairs)
        losses.update(feature_matching=fm, adversarial=sum((p[1] for p in pairs)) / len(pairs))
        self.log('loss_dis', dis_loss)
    if self.warmed_up and batch_idx % self.update_discriminator_every == 0:
        self.opts[1].zero_grad()
        dis_loss.backward()
        self.opts[1].step()
    else:
        self.opts[0].zero_grad()
        sum((v * self.loss_weight(k, batch_idx) for k, v in losses.items())).backward()
        self.opts[0].step()
    self.log_dict(losses)

def compare(a, b, name, rtol, atol):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=lambda m: name + ': ' + m)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), name
        for k in a:
            compare(a[k], b[k], name + '/' + str(k), rtol, atol)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, name + '/' + str(i), rtol, atol)
    else:
        assert a == b, (name, a, b)

@pytest.mark.parametrize('dtype,rtol,atol', [(torch.float64, 1e-07, 1e-09), (torch.float32, 0.003, 3e-05)])
@pytest.mark.parametrize('seed', [17, 43])
def test_regular_adversarial_updates_match_original(dtype, rtol, atol, seed):
    torch.manual_seed(seed)
    a = Harness(dtype)
    b = copy.deepcopy(a)
    for step in range(9):
        torch.manual_seed(1000 + seed + step)
        x = torch.randn(3, 1, 16384, dtype=dtype)
        rng = torch.random.get_rng_state()
        a.metrics = {}
        b.metrics = {}
        before = {n: p.detach().clone() for n, p in a.named_parameters()}
        original(a, x.clone(), step)
        torch.random.set_rng_state(rng)
        scope['training_step'](b, x.clone(), step)
        dis = step % 4 == 0
        assert any((not torch.equal(before[n], p) for n, p in a.named_parameters() if n.startswith('discriminator.') == dis)), 'Active optimizer must update weights'
        assert all((torch.equal(before[n], p) for n, p in a.named_parameters() if n.startswith('discriminator.') != dis)), 'Inactive weights must stay unchanged'
        for k in ['loss_dis'] if dis else ['fullband_mse', 'feature_matching', 'adversarial', 'regularization']:
            compare(a.metrics[k], b.metrics[k], 'loss/' + k, rtol, atol)
        for (name, p), (name2, q) in zip(a.named_parameters(), b.named_parameters()):
            if name.startswith('discriminator.') == dis:
                compare(p.grad, q.grad, 'gradient/' + name, rtol, atol)
            compare(p, q, 'parameter/' + name, rtol, atol)
        compare(a.state_dict(), b.state_dict(), 'state', rtol, atol)
        for i in range(2):
            compare(a.opts[i].state_dict(), b.opts[i].state_dict(), 'optimizer', rtol, atol)
        assert all((p.requires_grad for p in b.discriminator.parameters()))


def test_discriminator_step_skips_generator_graph_and_reconstruction():
    model = Harness(torch.float64)
    events = []
    model.encoder.register_forward_hook(
        lambda *args: events.append(torch.is_grad_enabled()))

    def unused_reconstruction(*args):
        raise AssertionError('Reconstruction is unused on discriminator updates')

    model.audio_distance = unused_reconstruction
    scope['training_step'](model, torch.randn(3, 1, 16384, dtype=torch.float64), 0)
    assert events == [False]
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())


def test_generator_step_uses_fixed_real_features_and_restores_parameter_flags():
    model = Harness(torch.float64)
    first_parameter = next(model.discriminator.parameters())
    first_parameter.requires_grad_(False)
    flags = [p.requires_grad for p in model.discriminator.parameters()]
    events = []
    model.discriminator.register_forward_pre_hook(
        lambda module, inputs: events.append((torch.is_grad_enabled(),
                                              inputs[0].requires_grad,
                                              any(p.requires_grad for p in module.parameters()))))
    scope['training_step'](model, torch.randn(3, 1, 16384, dtype=torch.float64), 1)
    assert events == [(False, False, False), (True, True, False)]
    assert [p.requires_grad for p in model.discriminator.parameters()] == flags
    assert all(p.grad is None for p in model.discriminator.parameters())
    assert any(p.grad is not None for p in model.decoder.parameters())


def test_reconstruction_phase_update_matches_original():
    torch.manual_seed(17)
    baseline = Harness(torch.float64)
    baseline.warmed_up = False
    optimized = copy.deepcopy(baseline)
    batch = torch.randn(3, 1, 16384, dtype=torch.float64)
    rng = torch.random.get_rng_state()
    original(baseline, batch.clone(), 0)
    torch.random.set_rng_state(rng)
    scope['training_step'](optimized, batch.clone(), 0)
    compare(baseline.state_dict(), optimized.state_dict(), 'state', 1e-7, 1e-9)
    for left, right in zip(baseline.opts, optimized.opts):
        compare(left.state_dict(), right.state_dict(), 'optimizer', 1e-7, 1e-9)
