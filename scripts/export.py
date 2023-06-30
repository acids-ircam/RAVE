import logging
import math
import os

logging.basicConfig(level=logging.INFO)
logging.info("library loading")
logging.info("DEBUG")
import torch

torch.set_grad_enabled(False)

import cached_conv as cc
import gin
import nn_tilde
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from absl import flags

import rave
import rave.blocks
import rave.core
import rave.resampler

FLAGS = flags.FLAGS

flags.DEFINE_string('run',
                    default=None,
                    help='Path to the run to export',
                    required=True)
flags.DEFINE_bool('streaming',
                  default=False,
                  help='Enable the model streaming mode')
flags.DEFINE_float(
    'fidelity',
    default=.95,
    lower_bound=.1,
    upper_bound=.999,
    help='Fidelity to use during inference (Variational mode only)')
flags.DEFINE_bool(
    'stereo',
    default=False,
    help='Enable fake stereo mode (one encoding, double decoding')
flags.DEFINE_bool('ema_weights',
                  default=False,
                  help='Use ema weights if available')
flags.DEFINE_integer('sr',
                     default=None,
                     help='Optional resampling sample rate')


class ScriptedRAVE(nn_tilde.Module):

    def __init__(self,
                 pretrained: rave.RAVE,
                 stereo: bool,
                 fidelity: float = .95,
                 target_sr: bool = None) -> None:
        super().__init__()
        self.stereo = stereo

        self.pqmf = pretrained.pqmf
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder

        self.sr = pretrained.sr

        self.resampler = None

        if target_sr is not None:
            if target_sr != self.sr:
                assert not target_sr % self.sr, "Incompatible target sampling rate"
                self.resampler = rave.resampler.Resampler(target_sr, self.sr)
                self.sr = target_sr

        self.full_latent_size = pretrained.latent_size

        self.is_using_adain = False
        for m in self.modules():
            if isinstance(m, rave.blocks.AdaptiveInstanceNormalization):
                self.is_using_adain = True
                break

        if self.is_using_adain and stereo:
            raise ValueError("Stereo mode not yet supported with AdaIN")

        self.register_attribute("learn_target", False)
        self.register_attribute("reset_target", False)
        self.register_attribute("learn_source", False)
        self.register_attribute("reset_source", False)

        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)
        self.register_buffer("fidelity", pretrained.fidelity)

        if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
            latent_size = max(
                np.argmax(pretrained.fidelity.numpy() > fidelity), 1)
            latent_size = 2**math.ceil(math.log2(latent_size))
            self.latent_size = latent_size

        elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
            self.latent_size = pretrained.encoder.num_quantizers

        elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
            self.latent_size = pretrained.latent_size

        elif isinstance(pretrained.encoder, rave.blocks.SphericalEncoder):
            self.latent_size = pretrained.latent_size - 1

        else:
            raise ValueError(
                f'Encoder type {pretrained.encoder.__class__.__name__} not supported'
            )

        x_len = 2**14
        x = torch.zeros(1, 1, x_len)

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        x_m = x.clone() if self.pqmf is None else self.pqmf(x)

        z = self.encoder(x_m)

        ratio_encode = x_len // z.shape[-1]

        channels = ["(L)", "(R)"] if stereo else ["(mono)"]

        self.fake_adain = rave.blocks.AdaptiveInstanceNormalization(0)

        self.register_method(
            "encode",
            in_channels=1,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=ratio_encode,
            input_labels=['(signal) Input audio signal'],
            output_labels=[
                f'(signal) Latent dimension {i}'
                for i in range(self.latent_size)
            ],
        )
        self.register_method(
            "decode",
            in_channels=self.latent_size,
            in_ratio=ratio_encode,
            out_channels=2 if stereo else 1,
            out_ratio=1,
            input_labels=[
                f'(signal) Latent dimension {i}'
                for i in range(self.latent_size)
            ],
            output_labels=[
                f'(signal) Reconstructed audio signal {channel}'
                for channel in channels
            ],
        )

        self.register_method(
            "forward",
            in_channels=1,
            in_ratio=1,
            out_channels=2 if stereo else 1,
            out_ratio=1,
            input_labels=['(signal) Input audio signal'],
            output_labels=[
                f'(signal) Reconstructed audio signal {channel}'
                for channel in channels
            ],
        )

    def post_process_latent(self, z):
        raise NotImplementedError

    def pre_process_latent(self, z):
        raise NotImplementedError

    def update_adain(self):
        for m in self.modules():
            if isinstance(m, rave.blocks.AdaptiveInstanceNormalization):
                m.learn_x.zero_()
                m.learn_y.zero_()

                if self.learn_target[0]:
                    m.learn_y.add_(1)
                if self.learn_source[0]:
                    m.learn_x.add_(1)

                if self.reset_target[0]:
                    m.reset_y()
                if self.reset_source[0]:
                    m.reset_x()

        self.reset_source = False,
        self.reset_target = False,

    @torch.jit.export
    def encode(self, x):
        if self.is_using_adain:
            self.update_adain()

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        if self.pqmf is not None:
            x = self.pqmf(x)

        z = self.encoder(x)
        z = self.post_process_latent(z)
        return z

    @torch.jit.export
    def decode(self, z, from_forward: bool = False):
        if self.is_using_adain and not from_forward:
            self.update_adain()

        if self.stereo:
            z = torch.cat([z, z], 0)

        z = self.pre_process_latent(z)
        y = self.decoder(z)

        if self.pqmf is not None:
            y = self.pqmf.inverse(y)

        if self.resampler is not None:
            y = self.resampler.from_model_sampling_rate(y)

        if self.stereo:
            y = torch.cat(y.chunk(2, 0), 1)

        return y

    def forward(self, x):
        return self.decode(self.encode(x), from_forward=True)

    @torch.jit.export
    def get_learn_target(self) -> bool:
        return self.learn_target[0]

    @torch.jit.export
    def set_learn_target(self, learn_target: bool) -> int:
        self.learn_target = (learn_target, )
        return 0

    @torch.jit.export
    def get_learn_source(self) -> bool:
        return self.learn_source[0]

    @torch.jit.export
    def set_learn_source(self, learn_source: bool) -> int:
        self.learn_source = (learn_source, )
        return 0

    @torch.jit.export
    def get_reset_target(self) -> bool:
        return self.reset_target[0]

    @torch.jit.export
    def set_reset_target(self, reset_target: bool) -> int:
        self.reset_target = (reset_target, )
        return 0

    @torch.jit.export
    def get_reset_source(self) -> bool:
        return self.reset_source[0]

    @torch.jit.export
    def set_reset_source(self, reset_source: bool) -> int:
        self.reset_source = (reset_source, )
        return 0


class VariationalScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        z = self.encoder.reparametrize(z)[0]
        z = z - self.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.latent_pca.unsqueeze(-1))
        z = z[:, :self.latent_size]
        return z

    def pre_process_latent(self, z):
        noise = torch.randn(
            z.shape[0],
            self.full_latent_size - self.latent_size,
            z.shape[-1],
        ).type_as(z)
        z = torch.cat([z, noise], 1)
        z = F.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        z = z + self.latent_mean.unsqueeze(-1)
        return z


class DiscreteScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        z = self.encoder.rvq.encode(z)
        return z.float()

    def pre_process_latent(self, z):
        z = torch.clamp(z, 0,
                        self.encoder.rvq.layers[0].codebook_size - 1).long()
        z = self.encoder.rvq.decode(z)
        if self.encoder.noise_augmentation:
            noise = torch.randn(z.shape[0], self.encoder.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)
        return z


class WasserteinScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        return z

    def pre_process_latent(self, z):
        if self.encoder.noise_augmentation:
            noise = torch.randn(z.shape[0], self.encoder.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)
        return z


class SphericalScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        return rave.blocks.unit_norm_vector_to_angles(z)

    def pre_process_latent(self, z):
        return rave.blocks.angles_to_unit_norm_vector(z)


def main(argv):
    cc.use_cached_conv(FLAGS.streaming)

    logging.info("building rave")

    gin.parse_config_file(os.path.join(FLAGS.run, "config.gin"))
    checkpoint = rave.core.search_for_run(FLAGS.run)

    pretrained = rave.RAVE()
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        if FLAGS.ema_weights and "EMA" in checkpoint["callbacks"]:
            pretrained.load_state_dict(
                checkpoint["callbacks"]["EMA"],
                strict=False,
            )
        else:
            pretrained.load_state_dict(
                checkpoint["state_dict"],
                strict=False,
            )
    else:
        print("No checkpoint found, RAVE will remain randomly initialized")
    pretrained.eval()

    if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
        script_class = VariationalScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
        script_class = DiscreteScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
        script_class = WasserteinScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.SphericalEncoder):
        script_class = SphericalScriptedRAVE
    else:
        raise ValueError(f"Encoder type {type(pretrained.encoder)} "
                         "not supported for export.")

    logging.info("warmup pass")

    x = torch.zeros(1, 1, 2**14)
    pretrained(x)

    logging.info("optimize model")

    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)
    logging.info("script model")

    scripted_rave = script_class(
        pretrained=pretrained,
        stereo=FLAGS.stereo,
        fidelity=FLAGS.fidelity,
        target_sr=FLAGS.sr,
    )

    logging.info("save model")
    model_name = os.path.basename(os.path.normpath(FLAGS.run))
    if FLAGS.streaming:
        model_name += "_streaming"
    if FLAGS.stereo:
        model_name += "_stereo"
    model_name += ".ts"

    scripted_rave.export_to_ts(os.path.join(FLAGS.run, model_name))

    logging.info(
        f"all good ! model exported to {os.path.join(FLAGS.run, model_name)}")
