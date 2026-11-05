"""Exercise the fine-tuning path with small networks, without audio dependencies."""
import ast
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
namespace = {'torch': torch, 'Profiler': lambda: SimpleNamespace(tick=lambda *args: None),
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


class Toy(nn.Module):
    def __init__(self, mode, delay=0):
        super().__init__()
        self.encoder = Encoder(3, 3)
        self.decoder = Decoder(3, 3)
        self.discriminator = Discriminator(3, 1)
        self.finetune_mode = mode
        self.finetune_steps = 0
        self.encoder_start_step = delay
        self.finetune_lrs = (1e-3, 1e-2, 1e-2)
        self.finetune_regularization = None
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
        selected = [name for name, _ in self.named_parameters()
                    if mode != 'encoder' or name.startswith('encoder.')]
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
    model.finetune_regularization = 0.
    assert step(model) == {'encoder'}
    assert model.losses['regularization'].item() == 0


def test_full_delay_and_learning_rates():
    model = Toy('full', delay=1)
    assert [group['lr'] for group in model.opts[0].param_groups] == [1e-3, 1e-2]
    assert step(model) == {'discriminator'}
    assert model.discriminator.calls == 1
    assert step(model) == {'encoder', 'decoder'}
    assert model.discriminator.calls == 2
    assert step(model) == {'encoder', 'decoder'}
    assert step(model) == {'encoder', 'decoder'}
    assert step(model) == {'discriminator'}
    assert model.discriminator.calls == 5


def test_full_encoder_stays_frozen_during_delay():
    model = Toy('full', delay=3)
    assert step(model) == {'discriminator'}
    assert step(model) == {'decoder'}
    assert step(model) == {'decoder'}
    assert step(model) == {'encoder', 'decoder'}


def test_alternating_and_resume():
    model = Toy('alternating')
    assert step(model) == {'decoder', 'discriminator'}
    assert 'feature_matching' not in model.losses
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    resumed = Toy('alternating')
    resumed.on_load_checkpoint(checkpoint)
    assert resumed.finetune_steps == 1
    assert step(resumed) == {'encoder'}
    assert 'adversarial' not in resumed.losses
    assert resumed.losses['regularization'].item() > 0
    assert step(resumed) == {'decoder', 'discriminator'}
    with pytest.raises(ValueError):
        Toy('full').on_load_checkpoint(checkpoint)


def test_beta_regularization_and_resume():
    model = Toy('alternating')
    model.encoder.beta = 0.25
    z = torch.ones(2, 3)
    assert model.encoder.reparametrize(z)[1].item() == 0.25
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    assert checkpoint['finetune']['beta'] == 0.25
    resumed = Toy('alternating')
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


def test_alternating_discriminator_backward_precedes_generator_forward():
    model = Toy('alternating')
    events = []
    model.discriminator.register_forward_hook(lambda *args: events.append('forward'))
    model.discriminator.weight.register_hook(lambda grad: events.append('backward'))
    assert step(model) == {'decoder', 'discriminator'}
    assert events == ['forward', 'backward', 'forward']


def test_full_resume_preserves_discriminator_cadence():
    model = Toy('full')
    assert step(model) == {'discriminator'}
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    resumed = Toy('full')
    resumed.on_load_checkpoint(checkpoint)
    assert step(resumed) == {'encoder', 'decoder'}


def test_validation_updates_pca_from_small_dataset():
    import numpy as np
    from einops import rearrange
    from sklearn.decomposition import PCA

    hooks = [node for node in model.body if isinstance(node, ast.FunctionDef)
             and node.name in {'on_validation_batch_end', 'on_validation_epoch_end'}]
    scope = {'torch': torch, 'np': np, 'rearrange': rearrange, 'PCA': PCA,
             'MAX_LATENT_FOR_VALIDATION': 2000, 'MAX_AUDIO_FOR_LOGGING': 32,
             'blocks': SimpleNamespace(VariationalEncoder=Encoder),
             'RunningStage': SimpleNamespace(SANITY_CHECKING='sanity')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=hooks, type_ignores=[])),
                 'rave/model.py', 'exec'), scope)
    instance = SimpleNamespace(
        receptive_field=torch.ones(2),
        trainer=SimpleNamespace(state=SimpleNamespace(stage='validation')),
        encoder=Encoder(3, 3), warmed_up=True, no_freeze_when_warmed_up=True,
        update_pca=True, latent_val_buffer=[], audio_val_buffer=[],
        latent_mean=torch.zeros(3), latent_pca=torch.eye(3), fidelity=torch.zeros(3),
        integrator=None, eval_number=0, sr=44100,
        logger=SimpleNamespace(experiment=SimpleNamespace(add_audio=lambda *args: None)),
        log=lambda *args: None)
    z = torch.tensor([[[1., 2., 3., 4.], [2., 1., 4., 2.], [0., 3., 2., 1.]]])
    scope['on_validation_batch_end'](instance, (torch.zeros(1, 8), z))
    scope['on_validation_epoch_end'](instance)
    assert torch.allclose(instance.latent_mean, z.mean(dim=(0, 2)))
    assert instance.fidelity[-1] == pytest.approx(1.)
    assert not torch.equal(instance.latent_pca, torch.eye(3))
    assert not instance.latent_val_buffer
