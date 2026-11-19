"""Exercise the fine-tuning path with small networks, without audio dependencies."""
import ast
import copy
import re
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


# Load the actual methods without importing optional audio packages.
source = ast.parse((Path(__file__).parents[1] / 'rave/model.py').read_text())
model = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'RAVE')
names = {'configure_optimizers', '_prepare_finetune_step', '_finetune_update',
         'training_step', 'on_save_checkpoint', 'on_load_checkpoint',
         '_finetune_discriminator_backward'}
methods = [node for node in model.body if isinstance(node, ast.FunctionDef) and node.name in names]
for method in methods:
    method.decorator_list = []
namespace = {'torch': torch, 'nullcontext': nullcontext, 'Profiler': lambda: SimpleNamespace(tick=lambda *args: None),
             '_pqmf_encode': lambda pqmf, x: x}
exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
             'rave/model.py', 'exec'), namespace)


class Encoder(nn.Linear):
    beta = 1.

    def set_warmed_up(self, state):
        pass

    def reparametrize(self, z):
        return z, self.beta * z.square().mean()


class Decoder(Encoder):
    pass


class Discriminator(nn.Linear):
    calls = 0

    def forward(self, x):
        self.calls += 1
        output = super().forward(x)
        return [[output.tanh(), output]]


core_source = ast.parse((Path(__file__).parents[1] / 'rave/core.py').read_text())
key_functions = [node for node in core_source.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'get_finetune_keys', 'parse_weight_shortcut',
                                   'parse_dict_with_regexps'}]
key_namespace = {'re': re}
exec(compile(ast.fix_missing_locations(ast.Module(body=key_functions, type_ignores=[])),
             'rave/core.py', 'exec'), key_namespace)


class Toy(nn.Module):
    def __init__(self, mode, delay=0, freeze=(), total=8):
        super().__init__()
        self.encoder = Encoder(3, 3)
        self.decoder = Decoder(3, 3)
        self.discriminator = Discriminator(3, 1)
        self.finetune_mode = mode
        self.finetune_steps = 0
        self.finetune_total_steps = total
        self.encoder_start_step = delay
        self.finetune_lrs = (1e-3, 1e-2, 1e-2)
        self.update_discriminator_every = 4
        self.warmed_up = True
        self.output_mode = 'raw'
        self.pqmf = None
        self.multiband_audio_distance = None
        self.valid_signal_crop = False
        self.feature_matching_fun = lambda x, y: (x - y).abs().mean()
        self.num_skipped_features = 0
        self.beta_factor = 1.
        self.weights = {}
        self.gradient_clip_gen = self.gradient_clip_dis = None
        self.losses = {}
        selected = key_namespace['get_finetune_keys'](self, mode, ['all'], freeze)
        self.opts = self.configure_optimizers(selected)

    def encode(self, x, return_mb=False):
        z = self.encoder(x)
        return (z, x) if return_mb else z

    def split_features(self, features):
        return ([[f[:2] for f in scale] for scale in features],
                [[f[2:] for f in scale] for scale in features])

    def gan_loss(self, real, fake):
        return (real - 1).square().mean() + fake.square().mean(), (fake - 1).square().mean()

    def audio_distance(self, x, y):
        return {'mse': (x - y).square().mean()}

    def optimizers(self):
        return self.opts

    def manual_backward(self, loss):
        loss.backward()

    def loss_weight(self, key, step):
        return 1.

    def log(self, *args):
        pass

    def log_dict(self, losses):
        self.losses = losses


for name in names:
    setattr(Toy, name, namespace[name])


def changed(model, before):
    return {name.split('.')[0] for name, p in model.named_parameters()
            if not torch.equal(p, before[name])}


def step(model):
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.training_step(torch.randn(2, 3), 0)
    return changed(model, before)


def test_encoder_reconstruction_and_regularization():
    model = Toy('encoder')
    assert step(model) == {'encoder'}
    assert 'adversarial' not in model.losses
    assert model.losses['regularization'].item() > 0
    model.encoder.beta = 0.
    assert step(model) == {'encoder'}
    assert 'regularization' not in model.losses


def test_decoder_trains_discriminator_on_configured_cadence():
    model = Toy('decoder')
    assert len(model.opts) == 2
    assert step(model) == {'discriminator'}
    for _ in range(3):
        assert step(model) == {'decoder'}
        assert 'adversarial' in model.losses
        assert 'feature_matching' in model.losses
    assert step(model) == {'discriminator'}


def test_decoder_can_explicitly_freeze_discriminator():
    model = Toy('decoder', freeze=['discriminator'])
    assert len(model.opts) == 1
    for _ in range(5):
        assert step(model) == {'decoder'}
    assert all(not p.requires_grad for p in model.discriminator.parameters())


def test_beta_regularization_and_resume():
    model = Toy('encoder')
    model.encoder.beta = 0.25
    z = torch.ones(2, 3)
    assert model.encoder.reparametrize(z)[1].item() == 0.25
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    assert checkpoint['finetune']['beta'] == 0.25
    resumed = Toy('encoder')
    with pytest.raises(ValueError):
        resumed.on_load_checkpoint(checkpoint)
    resumed.encoder.beta = 0.25
    resumed.on_load_checkpoint(checkpoint)


def test_batch_end_without_scheduler():
    class Base:
        def on_train_batch_end(self, outputs, batch, batch_idx):
            self.batch_ended = True

    hook = next(node for node in model.body
                if isinstance(node, ast.FunctionDef) and node.name == 'on_train_batch_end')
    cls = ast.ClassDef(name='HookModel', bases=[ast.Name(id='Base', ctx=ast.Load())],
                       keywords=[], body=[hook], decorator_list=[])
    scope = {'Base': Base}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 'rave/model.py', 'exec'), scope)
    instance = scope['HookModel']()
    instance.lr_schedulers = lambda: None
    instance.on_train_batch_end(None, None, 0)
    assert instance.batch_ended
    steps = []
    instance.lr_schedulers = lambda: SimpleNamespace(step=lambda: steps.append(1))
    instance.on_train_batch_end(None, None, 1)
    assert steps == [1]


def test_full_resume_preserves_discriminator_cadence():
    model = Toy('full')
    assert step(model) == {'discriminator'}
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    resumed = Toy('full')
    resumed.on_load_checkpoint(checkpoint)
    assert step(resumed) == {'decoder'}


def test_validation_updates_pca_from_small_dataset():
    import numpy as np
    from einops import rearrange
    from sklearn.decomposition import PCA

    hooks = [node for node in model.body if isinstance(node, ast.FunctionDef)
             and node.name in {'on_validation_batch_end', 'on_validation_epoch_end', '_compute_pca', '_record_vintage_pca', '_record_pca'}]
    scope = {'torch': torch, 'np': np, 'rearrange': rearrange, 'PCA': PCA,
             'MAX_LATENT_FOR_VALIDATION': 2000, 'MAX_AUDIO_FOR_LOGGING': 32,
             'blocks': SimpleNamespace(VariationalEncoder=Encoder),
             'RunningStage': SimpleNamespace(SANITY_CHECKING='sanity')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=hooks, type_ignores=[])),
                 'rave/model.py', 'exec'), scope)
    instance = SimpleNamespace(
        receptive_field=torch.ones(2),
        trainer=SimpleNamespace(state=SimpleNamespace(stage='validation'), val_dataloaders=SimpleNamespace(batch_size=1)),
        encoder=Encoder(3, 3), warmed_up=True, no_freeze_when_warmed_up=True,
        update_pca=True, latent_val_buffer=[], audio_val_buffer=[],
        latent_mean=torch.zeros(3), latent_pca=torch.eye(3), fidelity=torch.zeros(3),
        integrator=None, eval_number=0, sr=44100, _invert=lambda audio: audio,
        logger=SimpleNamespace(experiment=SimpleNamespace(add_audio=lambda *args: None, add_histogram=lambda *args: None)),
        projections={}, _log_latent_statistics=lambda *args: None, _log_fidelity=lambda *args: None,
        log=lambda *args: None)
    z = torch.tensor([[[1., 2., 3., 4.], [2., 1., 4., 2.], [0., 3., 2., 1.]]])
    from types import MethodType
    for name in ('_compute_pca', '_record_vintage_pca', '_record_pca'):
        setattr(instance, name, MethodType(scope[name], instance))
    instance.encoder.std_from_scale = torch.nn.functional.softplus
    posterior = torch.cat([z, torch.zeros_like(z)], dim=1)
    scope['on_validation_batch_end'](instance, (torch.zeros(1, 8), posterior))
    scope['on_validation_epoch_end'](instance)
    assert torch.allclose(instance.latent_mean, z.mean(dim=(0, 2)))
    assert instance.fidelity[-1] == pytest.approx(1.)
    assert not torch.equal(instance.latent_pca, torch.eye(3))
    assert not instance.latent_val_buffer


def test_full_decoder_then_encoder_and_learning_rates():
    model = Toy('full', total=8)
    assert [g['lr'] for g in model.opts[0].param_groups] == [1e-3, 1e-2]
    assert step(model) == {'discriminator'}
    for _ in range(3):
        assert step(model) == {'decoder'}
        assert 'feature_matching' in model.losses
    for _ in range(4):
        assert step(model) == {'encoder'}
        assert 'adversarial' not in model.losses
        assert 'feature_matching' not in model.losses
    assert model.finetune_steps == 8


@pytest.mark.parametrize('completed', [1, 3, 4, 5])
def test_full_resume_preserves_phase_and_optimizer_state(completed):
    model = Toy('full', total=8)
    for _ in range(completed):
        step(model)
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    resumed = Toy('full', total=8)
    resumed.load_state_dict(model.state_dict())
    for original, restored in zip(model.opts, resumed.opts):
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
    resumed.on_load_checkpoint(checkpoint)
    torch.manual_seed(42)
    expected = step(model)
    torch.manual_seed(42)
    assert step(resumed) == expected
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key])


def test_full_resume_checks_saved_duration_and_accepts_older_checkpoints():
    model = Toy('full', total=8)
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    with pytest.raises(ValueError, match='same max_steps'):
        Toy('full', total=10).on_load_checkpoint(checkpoint)
    del checkpoint['finetune']['total_steps']
    resumed = Toy('full', total=8)
    resumed.on_load_checkpoint(checkpoint)
    assert step(resumed) == {'discriminator'}


@pytest.mark.parametrize('mode', ['alternating', 'full_asymetric'])
def test_removed_modes_rejected(mode):
    with pytest.raises(ValueError, match='encoder, decoder or full'):
        step(Toy(mode))
