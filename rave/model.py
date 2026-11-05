import math
from absl import logging
import os
from time import time
from typing import Callable, Optional, Iterable, Dict


import gin, pdb
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from sklearn.decomposition import PCA
from pytorch_lightning.trainer.states import RunningStage


import rave.core

from . import blocks

MAX_LATENT_FOR_VALIDATION = 2000
MAX_AUDIO_FOR_LOGGING = 32


_default_loss_weights = {
    'audio_distance': 1.,
    'multiband_audio_distance': 1.,
    'adversarial': 1.,
    'feature_matching' : 20,
}

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep

@gin.configurable
class WarmupCallback(pl.Callback):

    def __init__(self, adversarial_at_start: bool = False) -> None:
        super().__init__()
        self.adversarial_at_start = adversarial_at_start
        self.state = {'training_steps': 0}

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:
        if self.adversarial_at_start and not pl_module.warmed_up:
            pl_module.warmed_up = True
        if not pl_module.warmed_up and self.state['training_steps'] >= pl_module.warmup:
            pl_module.warmed_up = True
            checkpoint_path = os.path.join(trainer.log_dir, "checkpoints")
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_path, "pre_warmup.ckpt")
            trainer.save_checkpoint(checkpoint_path)
        self.state['training_steps'] += 1

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)


class FirstCheckpointCallback(pl.Callback):

    def __init__(self, filename: str = "last.ckpt") -> None:
        super().__init__()
        self.filename = filename
        self.state = {"saved": False}

    def on_train_start(self, trainer, pl_module) -> None:
        if self.state["saved"]:
            return

        checkpoint_path = os.path.join(trainer.log_dir, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        trainer.save_checkpoint(os.path.join(checkpoint_path, self.filename))
        self.state["saved"] = True

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)


class QuantizeCallback(WarmupCallback):

    def on_train_batch_(self, trainer, pl_module, batch,
                             batch_idx) -> None:

        if pl_module.warmup_quantize is None: return

        if self.state['training_steps'] >= pl_module.warmup_quantize:
            if isinstance(pl_module.encoder, blocks.DiscreteEncoder):
                pl_module.encoder.enabled = torch.tensor(1).type_as(
                    pl_module.encoder.enabled)
        self.state['training_steps'] += 1


@gin.configurable
class BetaWarmupCallback(pl.Callback):

    def __init__(self, initial_value: float = .2,
                       target_value: float = .2,
                       warmup_len: int = 1,
                       multiplier: float = 1.,
                       log: bool = True) -> None:
        super().__init__()
        self.state = {'training_steps': 0}
        self.warmup_len = warmup_len
        self.initial_value = initial_value
        self.target_value = target_value
        self.multiplier = multiplier
        self.log_warmup = log

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:
        self.state['training_steps'] += 1

        warmup_ratio = self.state["training_steps"] / self.warmup_len
        target_value = self.multiplier * self.target_value

        if (self.state["training_steps"] >= self.warmup_len):
            pl_module.beta_factor = target_value
            return

        if self.log_warmup: 
            beta = math.log(self.initial_value) * (1 - warmup_ratio) + math.log(
               target_value) * warmup_ratio
            pl_module.beta_factor = math.exp(beta)
        else:
            beta = warmup_ratio * (target_value - self.initial_value) + self.initial_value
            pl_module.beta_factor = min(beta, target_value)

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)


@torch.fx.wrap
def _pqmf_encode(pqmf, x: torch.Tensor):
    batch_size = x.shape[:-2]
    x_multiband = x.reshape(-1, 1, x.shape[-1])
    x_multiband = pqmf(x_multiband)
    x_multiband = x_multiband.reshape(*batch_size, -1, x_multiband.shape[-1])
    return x_multiband


@torch.fx.wrap
def _pqmf_decode(pqmf, x: torch.Tensor, batch_size: Iterable[int], n_channels: int):
    x = x.reshape(x.shape[0] * n_channels, -1, x.shape[-1])
    x = pqmf.inverse(x)
    x = x.reshape(*batch_size, n_channels, -1)
    return x


@gin.configurable
class RAVE(pl.LightningModule):

    def __init__(
        self,
        latent_size,
        sampling_rate,
        encoder,
        decoder,
        discriminator,
        phase_1_duration,
        gan_loss,
        valid_signal_crop,
        feature_matching_fun,
        num_skipped_features,
        audio_distance: Callable[[], nn.Module],
        multiband_audio_distance: Optional[Callable[[], nn.Module]],
        n_bands: int = 16,
        balancer = None,
        weights: Optional[Dict[str, float]] = None,
        warmup_quantize: Optional[int] = None,
        pqmf: Optional[Callable[[], nn.Module]] = None,
        spectrogram: Optional[Callable] = None,
        update_discriminator_every: int = 2,
        n_channels: int = 1,
        input_mode: str = "pqmf",
        output_mode: str = "pqmf",
        audio_monitor_epochs: int = 1,
        # for retro-compatibility
        enable_pqmf_encode: Optional[bool] = None,
        enable_pqmf_decode: Optional[bool] = None,
        is_mel_input: Optional[bool] = None,
        no_freeze_when_warmed_up: bool = False, 
        loss_weights = None,
        update_pca=True,
        gradient_clip_gen: Optional[float] = None,
        gradient_clip_dis: Optional[float] = None,
    ):
        super().__init__()
        self.pqmf = pqmf(n_channels=n_channels) if pqmf is not None else None
        self.spectrogram = None
        if spectrogram is not None:
            self.spectrogram = spectrogram
        assert input_mode in ['pqmf', 'mel', 'raw', 'complex_stft']
        assert output_mode in ['raw', 'pqmf', 'complex_stft']
        self.input_mode = input_mode
        self.output_mode = output_mode
        # retro-compatibility
        if (enable_pqmf_encode is not None) or (enable_pqmf_decode is not None):
            self.input_mode = "pqmf" if enable_pqmf_encode else "raw"
            self.output_mode = "pqmf" if enable_pqmf_decode else "raw"
        if (is_mel_input) is not None:
            self.input_mode = "mel"
        if loss_weights is not None:
            weights = loss_weights
        assert weights is not None, "RAVE model requires either weights or loss_weights (depreciated) keyword"

        # setup model
        self.encoder = encoder(n_channels=n_channels, no_freeze_when_warmed_up=no_freeze_when_warmed_up)
        self.no_freeze_when_warmed_up = no_freeze_when_warmed_up
        self.decoder = decoder(n_channels=n_channels)
        self.discriminator = discriminator(n_channels=n_channels)
        self.gradient_clip_gen = gradient_clip_gen
        self.gradient_clip_dis = gradient_clip_dis

        self.audio_distance = audio_distance()
        self.multiband_audio_distance = multiband_audio_distance() if multiband_audio_distance is not None else None

        self.gan_loss = gan_loss

        self.latent_val_buffer = []
        self.audio_val_buffer = []
        self.register_buffer("latent_pca", torch.eye(latent_size))
        self.register_buffer("latent_mean", torch.zeros(latent_size))
        self.register_buffer("fidelity", torch.zeros(latent_size))

        self.latent_size = latent_size

        self.automatic_optimization = False
        self.update_pca = update_pca

        # SCHEDULE
        self.warmup = phase_1_duration
        self.warmup_quantize = warmup_quantize
        self.weights = _default_loss_weights
        self.weights.update(weights)
        self.warmed_up = False

        # CONSTANTS
        self.sr = sampling_rate
        self.valid_signal_crop = valid_signal_crop
        self.n_channels = n_channels
        self.feature_matching_fun = feature_matching_fun
        self.num_skipped_features = num_skipped_features
        self.update_discriminator_every = update_discriminator_every

        self.eval_number = 0
        self.beta_factor = 1.
        self.integrator = None

        self.register_buffer("receptive_field", torch.tensor([0, 0]).long())
        self.audio_monitor_epochs = audio_monitor_epochs

    @gin.configurable(module="rave")
    def configure_optimizers(self, weight_list = None, lr_gen=1.e-4, lr_dis = 1.e-4):
        def log_optimizer_params(prefix, gen_named_p, dis_named_p):
            gen_count = sum(p.numel() for _, p in gen_named_p)
            dis_count = sum(p.numel() for _, p in dis_named_p)
            logging.info(
                "%s optimizer parameters: generator=%d tensors / %d scalars, "
                "discriminator=%d tensors / %d scalars",
                prefix,
                len(gen_named_p),
                gen_count,
                len(dis_named_p),
                dis_count,
            )
            logging.info(
                "%s generator parameter names:\n%s",
                prefix,
                "\n".join(
                    f"  {name}: shape={tuple(param.shape)}, numel={param.numel()}"
                    for name, param in gen_named_p),
            )
            logging.info(
                "%s discriminator parameter names:\n%s",
                prefix,
                "\n".join(
                    f"  {name}: shape={tuple(param.shape)}, numel={param.numel()}"
                    for name, param in dis_named_p),
            )

        if getattr(self, 'finetune_mode', None) is not None:
            selected = set(weight_list if weight_list is not None else dict(self.named_parameters()))
            groups = []
            dis_p = []
            for prefix, module, lr in zip(
                    ['encoder', 'decoder', 'discriminator'],
                    [self.encoder, self.decoder, self.discriminator], self.finetune_lrs):
                parameters = []
                for name, parameter in module.named_parameters():
                    parameter.requires_grad_(f'{prefix}.{name}' in selected)
                    if parameter.requires_grad:
                        parameters.append(parameter)
                if parameters:
                    if prefix == 'discriminator':
                        dis_p = parameters
                    else:
                        groups.append({'params': parameters, 'lr': lr})
            self.finetune_parameters = [p for p in self.parameters() if p.requires_grad]
            if not groups:
                raise ValueError('Fine-tuning requires encoder or decoder parameters')
            optimizers = [torch.optim.Adam(groups, betas=(.5, .9))]
            if dis_p:
                optimizers.append(torch.optim.Adam(dis_p, self.finetune_lrs[2], (.5, .9)))
            return optimizers

        if weight_list is None:
            gen_named_p = [(f"encoder.{name}", param)
                           for name, param in self.encoder.named_parameters()]
            gen_named_p += [(f"decoder.{name}", param)
                            for name, param in self.decoder.named_parameters()]
            dis_named_p = [(f"discriminator.{name}", param)
                           for name, param in self.discriminator.named_parameters()]
            log_optimizer_params("full", gen_named_p, dis_named_p)
            gen_p = [param for _, param in gen_named_p]
            dis_p = [param for _, param in dis_named_p]
            gen_opt = torch.optim.AdamW(gen_p, lr_gen, (.9, .999))
            dis_opt = torch.optim.AdamW(dis_p, lr_dis, (.8, .9))
            return ({'optimizer': gen_opt,
                    'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.LinearLR(gen_opt, start_factor=1.0, end_factor=1., total_iters=self.warmup)}},
                    {'optimizer':dis_opt})

        else: 
            params = dict(self.named_parameters())
            weight_names = list(filter(lambda x: x in params, weight_list))
            gen_names = list(filter(lambda x: not x.startswith('discriminator'), weight_names))
            dis_names = list(filter(lambda x: x.startswith('discriminator'), weight_names))
            gen_named_p = [(name, self.get_parameter(name)) for name in gen_names]
            dis_named_p = [(name, self.get_parameter(name)) for name in dis_names]
            gen_p = [param for _, param in gen_named_p]
            dis_p = [param for _, param in dis_named_p]
            logging.info(
                "partial optimizer parameter names: requested=%d, matched=%d",
                len(weight_list),
                len(weight_names),
            )
            log_optimizer_params("partial", gen_named_p, dis_named_p)
            optimizers = []
            #TODO not efficient but would require too much re-factor
            if len(gen_p) > 0: 
                gen_opt = torch.optim.Adam(gen_p, lr_gen, (.5, .9))
            else:
                gen_opt = torch.optim.Adam(gen_p, 0., (.5, .9))
            optimizers.append({'optimizer': gen_opt, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.LinearLR(gen_opt, start_factor=1.0, end_factor=0.1, total_iters=self.warmup)}})
            if len(dis_p) > 0:
                dis_opt = torch.optim.Adam(dis_p, lr_dis, (.5, .9))
            else:
                dis_opt = torch.optim.Adam(gen_p, 0, (.5, .9))
                self.update_discriminator_every = 1
            optimizers.append({'optimizer': dis_opt})
            return tuple(optimizers)

    def import_weights(self, weight_dict, strict=False):
        state_dict = self.state_dict()
        invalid_keys = []
        valid_keys = []
        for k, v in weight_dict.items(): 
            assert k in state_dict, "cannot update weight %s k: does not exist in model"%k
            if strict:
                if v.shape != state_dict[k].shape:
                    invalid_keys.append(k)
                    continue
            valid_keys.append(k)

        if len(invalid_keys) > 0:
            error_description = "\n\t".join(["imported shape: %s, model shape: %s"%(weight_dict[i].shape, state_dict[i].shape) for i in invalid_keys])
            error_description = 'Cannot import keys, size mismatch : \n%s'%error_description
            if strict:
                raise RuntimeError(error_description)
            else:
                logging.warning(error_description)

        for k in valid_keys:
            try: 
                self.get_parameter(k).data = v
            except AttributeError: 
                try:
                    self.get_buffer(k).data = v
                except Exception as e:
                    raise e
                

    def _mel_encode(self, x: torch.Tensor):
        batch_size = x.shape[:-2]
        x = self.spectrogram(x)[..., :-1]
        x = torch.log1p(x).reshape(*batch_size, -1, x.shape[-1])
        return x
        
    def encode(self, x, return_mb: bool = False):
        x_enc = x
        if self.input_mode == "pqmf":
            x_enc = _pqmf_encode(self.pqmf, x_enc)
        elif self.input_mode == "mel":
            x_enc = self._mel_encode(x)
        elif self.input_mode == "complex_stft":
            x_enc = self.spectrogram(x)
            
        z = self.encoder(x_enc)
        if return_mb:
            if self.input_mode == "pqmf":
                return z, x_enc
            else:
                if self.multiband_audio_distance is not None:
                    x_multiband = _pqmf_encode(self.pqmf, x)
                else:
                    x_multiband = None
                return z, x_multiband
        return z

    def latent_size_from_fidelity(self, fidelity):
        if isinstance(self.encoder, rave.blocks.VariationalEncoder):
            if fidelity is None:
                latent_size = self.full_latent_size
            else:
                latent_size = max(
                    np.argmax(self.fidelity.cpu().numpy() > fidelity), 1)
                latent_size = 2**math.ceil(math.log2(latent_size))

        elif isinstance(self.encoder, rave.blocks.DiscreteEncoder):
            latent_size = self.encoder.num_quantizers

        elif isinstance(self.encoder, rave.blocks.WasserteinEncoder):
            latent_size = self.latent_size

        elif isinstance(self.encoder, rave.blocks.SphericalEncoder):
            latent_size = self.latent_size - 1
        
        else:
            raise TypeError('Cannot get latent size from encoder : %s'%(self.latent_size))

        return latent_size

    def loss_weight(self, loss_name, batch_idx = None, clamp = True):
        loss_weight = self.weights[loss_name]

        if isinstance(loss_weight, list):
            assert len(loss_weight) == 3, "if weight is list then it must be [init, end, n_steps]. Got : %s"%loss_weight
            loss_init, loss_target, n_steps = loss_weight
            batch_idx = self.trainer.global_step
            loss_weight = (loss_target - loss_init) / n_steps * batch_idx + loss_init
            if clamp: loss_weight = np.clip(loss_weight, min(loss_init, loss_target), max(loss_init, loss_target))
        return loss_weight

    def encode_compressed(self, x, fidelity = 0.99):
        z = self.encode(x, return_mb=False)
        z = self.encoder.reparametrize(z)[0] 
        z = z - self.latent_mean.unsqueeze(-1)
        z = torch.nn.functional.conv1d(z, self.latent_pca.unsqueeze(-1))
        z = z[:, :self.latent_size_from_fidelity(fidelity)]
        return z

    def decode(self, z):
        batch_size = z.shape[:-2]
        y = self.decoder(z)
        if self.output_mode == "pqmf":
            y = _pqmf_decode(self.pqmf, y, batch_size=batch_size, n_channels=self.n_channels)
        elif self.output_mode == "complex_stft":
            y = self.spectrogram.inverse(y)
        return y

    def forward(self, x):
        z = self.encode(x, return_mb=False)
        z = self.encoder.reparametrize(z)[0]
        return self.decode(z)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def split_features(self, features):
        feature_real = []
        feature_fake = []
        for scale in features:
            true, fake = zip(*map(
                lambda x: torch.split(x, x.shape[0] // 2, 0),
                scale,
            ))
            feature_real.append(true)
            feature_fake.append(fake)
        return feature_real, feature_fake

    def on_save_checkpoint(self, checkpoint):
        if getattr(self, 'finetune_mode', None) is not None:
            checkpoint['finetune'] = {
                'steps': self.finetune_steps,
                'mode': self.finetune_mode,
                'encoder_start_step': self.encoder_start_step,
                'lrs': self.finetune_lrs,
                'regularization': self.finetune_regularization,
                'beta': getattr(self.encoder, 'beta', None),
            }

    def on_load_checkpoint(self, checkpoint):
        if getattr(self, 'finetune_mode', None) is not None:
            state = checkpoint.get('finetune')
            if state is None:
                raise ValueError('Resume requires a fine-tuning checkpoint; use --run for source weights')
            expected = (self.finetune_mode, self.encoder_start_step,
                        tuple(self.finetune_lrs), self.finetune_regularization)
            restored = (state['mode'], state['encoder_start_step'],
                        tuple(state['lrs']), state['regularization'])
            if expected != restored:
                raise ValueError('Resume with the same fine-tuning mode, delay, learning rates and regularization')
            if 'beta' in state and state['beta'] != getattr(self.encoder, 'beta', None):
                raise ValueError('Resume with the same encoder beta')
            self.finetune_steps = state['steps']

    def _prepare_finetune_step(self):
        encoder_turn = self.finetune_mode == 'encoder' or (
            self.finetune_mode == 'alternating' and self.finetune_steps % 2 == 1)
        encoder_active = self.finetune_steps >= self.encoder_start_step
        adversarial = self.finetune_mode != 'encoder' and not encoder_turn
        self.finetune_discriminator_turn = (
            self.finetune_mode == 'full' and
            self.finetune_steps % self.update_discriminator_every == 0 and
            any(id(p) in self.finetune_parameter_ids for p in self.discriminator.parameters()))
        for prefix, module in [('encoder', self.encoder), ('decoder', self.decoder),
                               ('discriminator', self.discriminator)]:
            active = ((prefix == 'encoder' and encoder_active and (
                encoder_turn or self.finetune_mode == 'full')) or
                (prefix != 'encoder' and adversarial))
            if self.finetune_mode == 'full':
                active = active and (self.finetune_discriminator_turn if prefix == 'discriminator'
                                     else not self.finetune_discriminator_turn)
            module.train(active)
            for parameter in module.parameters():
                parameter.requires_grad_(active and id(parameter) in self.finetune_parameter_ids)
        return encoder_turn, adversarial

    def _finetune_discriminator_backward(self, loss_dis):
        dis_opt = self.optimizers()[1]
        dis_opt.zero_grad()
        self.manual_backward(loss_dis)
        if self.gradient_clip_dis is not None:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip_dis)

    def _finetune_update(self, loss_gen, loss_dis, adversarial):
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        gen_opt = optimizers[0]
        if self.finetune_discriminator_turn:
            self._finetune_discriminator_backward(loss_dis)
            optimizers[1].step()
        else:
            gen_opt.zero_grad()
            loss = sum(value * (self.loss_weight(key, self.finetune_steps)
                               if key in self.weights else 1.) for key, value in loss_gen.items())
            if loss.requires_grad:
                self.manual_backward(loss)
                if self.gradient_clip_gen is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()), self.gradient_clip_gen)
                gen_opt.step()
            if adversarial and self.finetune_mode == 'alternating' and len(optimizers) > 1:
                optimizers[1].step()
        self.log('finetune_steps', float(self.finetune_steps + 1))
        self.finetune_steps += 1

    def training_step(self, batch, batch_idx):
        p = Profiler()
        finetune = getattr(self, 'finetune_mode', None) is not None
        if finetune:
            self.finetune_parameter_ids = {id(p) for p in self.finetune_parameters}
            encoder_turn, adversarial = self._prepare_finetune_step()
        else:
            gen_opt, dis_opt = self.optimizers()
            adversarial = self.warmed_up
        x_raw = batch
        x_raw.requires_grad = not (finetune and self.finetune_discriminator_turn)

        batch_size = x_raw.shape[:-2]
        self.encoder.set_warmed_up(self.warmed_up)
        self.decoder.set_warmed_up(self.warmed_up)

        # ENCODE INPUT
        # get multiband in case
        z, x_multiband = self.encode(x_raw, return_mb=True)

        z, reg = self.encoder.reparametrize(z)[:2]
        p.tick('encode')

        # DECODE LATENT
        y = self.decoder(z)
        if self.output_mode == "pqmf":
            y_multiband = y
            y_raw = _pqmf_decode(self.pqmf, y, batch_size=batch_size, n_channels=self.n_channels)
        elif self.output_mode== "complex_stft":
            y_multiband = None
            y_raw = self.spectrogram.inverse(y)
        else:
            y_raw = y 
            y_multiband = _pqmf_encode(self.pqmf, y)

        # TODO this has been added for training with num_samples = 65536 samples, output padding seems to mess with output dimensions. 
        # this may probably conflict with cached_conv
        
        y_raw = y_raw[..., :x_raw.shape[-1]]
        if y_multiband is not None:
            y_multiband = y_multiband[..., :x_multiband.shape[-1]]

        p.tick('decode')

        if self.valid_signal_crop and self.receptive_field.sum():
            x_multiband = rave.core.valid_signal_crop(
                x_multiband,
                *self.receptive_field,
            )
            y_multiband = rave.core.valid_signal_crop(
                y_multiband,
                *self.receptive_field,
            )
        p.tick('crop')

        # DISTANCE BETWEEN INPUT AND OUTPUT
        distances = {}
        
        if self.multiband_audio_distance is not None:
            multiband_distance =  self.multiband_audio_distance(
                x_multiband, y_multiband)
            p.tick('mb distance')
            for k, v in multiband_distance.items():
                distances[f'multiband_{k}'] = self.loss_weight('multiband_audio_distance', batch_idx) * v

        fullband_distance = self.audio_distance(x_raw, y_raw)
        p.tick('fb distance')

        for k, v in fullband_distance.items():
            distances[f'fullband_{k}'] = self.loss_weight('audio_distance', batch_idx) *  v

        feature_matching_distance = 0.

        if adversarial:  # DISCRIMINATION
            xy = torch.cat([x_raw, y_raw], 0)
            if finetune and self.finetune_mode == 'alternating':
                dis_optimizers = self.optimizers()
                if isinstance(dis_optimizers, (list, tuple)) and len(dis_optimizers) > 1:
                    dis_features = self.discriminator(xy.detach())
                    dis_real, dis_fake = self.split_features(dis_features)
                    dis_loss = 2. * sum(self.gan_loss(real[-1], fake[-1])[0]
                                       for real, fake in zip(dis_real, dis_fake)) / len(dis_real)
                    self._finetune_discriminator_backward(dis_loss)
                    dis_loss = dis_loss.detach()
                    del dis_features, dis_real, dis_fake
                else:
                    dis_loss = xy.new_zeros(())
                for parameter in self.discriminator.parameters():
                    parameter.requires_grad_(False)
            features = self.discriminator(
                xy.detach() if finetune and self.finetune_discriminator_turn else xy)

            feature_real, feature_fake = self.split_features(features)

            loss_dis = 0
            loss_adv = 0

            pred_real = 0
            pred_fake = 0

            for scale_real, scale_fake in zip(feature_real, feature_fake):
                current_feature_distance = sum(
                    map(
                        self.feature_matching_fun,
                        scale_real[self.num_skipped_features:-1],
                        scale_fake[self.num_skipped_features:-1],
                    )) / len(scale_real[self.num_skipped_features:])

                feature_matching_distance = feature_matching_distance + current_feature_distance

                _dis, _adv = self.gan_loss(scale_real[-1], scale_fake[-1])

                pred_real = pred_real + scale_real[-1].mean()
                pred_fake = pred_fake + scale_fake[-1].mean()

                loss_dis = loss_dis + _dis
                loss_adv = loss_adv + _adv

            feature_matching_distance = feature_matching_distance / len(
                feature_real)
            loss_adv = loss_adv / len(feature_real)
            loss_dis = 2. * loss_dis / len(feature_real)
            if finetune and self.finetune_mode == 'alternating':
                loss_dis = dis_loss

        else:
            pred_real = torch.tensor(0.).to(x_raw)
            pred_fake = torch.tensor(0.).to(x_raw)
            loss_dis = torch.tensor(0.).to(x_raw)
            loss_adv = torch.tensor(0.).to(x_raw)
        p.tick('discrimination')

        # COMPOSE GEN LOSS
        loss_gen = {}
        loss_gen.update(distances)
        p.tick('update loss gen dict')

        if reg.item():
            regularization = self.beta_factor
            if finetune:
                regularization = self.finetune_regularization
                if regularization is None:
                    regularization = 1.
                if self.finetune_mode == 'alternating' and not encoder_turn:
                    regularization = 0.
            loss_gen['regularization'] = reg * regularization

        if adversarial:
            loss_gen['feature_matching'] = feature_matching_distance #REMOVE DOUBLE WEIGHTING * self.loss_weight('feature_matching', batch_idx) * 
            loss_gen['adversarial'] = loss_adv  #REMOVE DOUBLE WEIGHTING self.loss_weight('adversarial', batch_idx) * 

        # OPTIMIZATION
        if finetune:
            # Alternating decoder steps use reconstruction + adversarial only.
            if self.finetune_mode == 'alternating':
                loss_gen.pop('feature_matching', None)
            self._finetune_update(loss_gen, loss_dis, adversarial)
        elif not (batch_idx %
                self.update_discriminator_every) and self.warmed_up:
            dis_opt.zero_grad()
            loss_dis.backward()
            if self.gradient_clip_dis is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    self.gradient_clip_dis,
                )
            dis_opt.step()
            p.tick('dis opt')
        else:
            gen_opt.zero_grad()
            loss_gen_value = 0.
            for k, v in loss_gen.items():
                # print("loss", k, "weight", self.loss_weight(k, batch_idx) if k in self.weights else "not defined")
                weight = 1. if not k in self.weights else self.loss_weight(k, batch_idx)
                loss_gen_value += v * weight
            loss_gen_value.backward()
            if self.gradient_clip_gen is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.decoder.parameters()),
                    self.gradient_clip_gen,
                )
            gen_opt.step()

        # LOGGING
        self.log("beta_factor", self.beta_factor)

        if self.warmed_up:
            self.log("loss_dis", loss_dis)
            self.log("pred_real", pred_real.mean())
            self.log("pred_fake", pred_fake.mean())

        self.log_dict(loss_gen)
        p.tick('logging')

    def validation_step(self, batch, batch_idx):
        """pytorch lightning version compatibility -_-'"""
        return self.on_validation_step(batch, batch_idx)

    def on_validation_step(self, x, batch_idx):
        z = self.encode(x)
        if isinstance(self.encoder, blocks.VariationalEncoder):
            mean = torch.split(z, z.shape[1] // 2, 1)[0]
        else:
            mean = None

        z = self.encoder.reparametrize(z)[0]
        y = self.decode(z)

        distance = self.audio_distance(x, y)
        full_distance = sum(distance.values())

        if self.trainer is not None:
            self.log('validation', full_distance)

        return torch.cat([x, y], -1), mean

    def on_validation_batch_end(self, out, *args, **kwargs):
        if not self.receptive_field.sum():
            print("Computing receptive field for this configuration...")
            lrf, rrf = rave.core.get_rave_receptive_field(self, n_channels=self.n_channels)
            self.receptive_field[0] = lrf
            self.receptive_field[1] = rrf
            print(
                f"Receptive field: {1000*lrf/self.sr:.2f}ms <-- x --> {1000*rrf/self.sr:.2f}ms"
            )

        if not len(out): return

        audio, z = out
        audio = list(map(lambda x: x.detach().cpu(), audio))

        if self.trainer.state.stage == RunningStage.SANITY_CHECKING:
            return

        remaining = MAX_LATENT_FOR_VALIDATION - len(self.latent_val_buffer)
        if z is not None and remaining > 0:
            self.latent_val_buffer.extend(z[:remaining].detach().cpu())
        
        remaining_audio = MAX_AUDIO_FOR_LOGGING - len(self.audio_val_buffer)
        if remaining_audio > 0:
            self.audio_val_buffer.extend(audio[:remaining_audio])

    def on_validation_epoch_end(self):

        if self.trainer.state.stage == RunningStage.SANITY_CHECKING:
            return

        audio = torch.cat(self.audio_val_buffer, 0)
        if isinstance(self.encoder, blocks.VariationalEncoder):
            if not (self.warmed_up and not self.no_freeze_when_warmed_up) and (self.update_pca):
                z = torch.stack(self.latent_val_buffer, 0)
                z = rearrange(z, "b c t -> (b t) c")

                self.latent_mean.copy_(z.mean(0))
                z = z - self.latent_mean.cpu()

                pca = PCA(z.shape[-1]).fit(z)

                components = pca.components_
                components = torch.from_numpy(components).to(self.latent_mean)
                self.latent_pca.copy_(components)

                var = pca.explained_variance_ / np.sum(pca.explained_variance_)
                var = np.cumsum(var)

                self.fidelity.copy_(torch.from_numpy(var).to(self.fidelity))

                var_percent = [.8, .9, .95, .99]
                for p in var_percent:
                    self.log(
                        f"fidelity_{p}",
                        np.argmax(var > p).astype(np.float32),
                    )

        y = audio.reshape(-1).numpy()
        if self.integrator is not None:
            y = self.integrator(y)
        self.logger.experiment.add_audio("audio_val", y, self.eval_number,
                                        self.sr)
        self.eval_number += 1
        self.audio_val_buffer = []
        self.latent_val_buffer = []

    def on_fit_start(self):
        tb = self.logger.experiment

        config = gin.operative_config_str()
        config = config.split('\n')
        config = ['```'] + config + ['```']
        config = '\n'.join(config)
        tb.add_text("config", config)

        model = str(self)
        model = model.split('\n')
        model = ['```'] + model + ['```']
        model = '\n'.join(model)
        tb.add_text("model", model)
