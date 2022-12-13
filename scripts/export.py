import logging
import pdb
import math
import os
import sys

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
from absl import flags, app
from typing import Union

import rave
import rave.blocks
import rave.core
import rave.scripted_vq

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
flags.DEFINE_integer(
    'channels',
    default=None,
    help='overrides model channels')


class ScriptedRAVE(nn_tilde.Module):

    def __init__(self, pretrained: rave.RAVE, channels: Union[int, None] = None) -> None:
        super().__init__()
        self.pqmf = pretrained.pqmf
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        self.channels = channels or pretrained.n_channels

        self.sr = pretrained.sr

        self.full_latent_size = pretrained.latent_size

        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)
        self.register_buffer("fidelity", pretrained.fidelity)

        if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
            latent_size = max(
                np.argmax(pretrained.fidelity.numpy() > FLAGS.fidelity), 1)
            latent_size = 2**math.ceil(math.log2(latent_size))
            self.latent_size = latent_size

        elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
            self.latent_size = pretrained.encoder.num_quantizers
            self.quantizer = rave.scripted_vq.SimpleQuantizer([
                *map(
                    lambda l: l._codebook.embed.squeeze(0),
                    pretrained.encoder.rvq.layers,
                )
            ])
            del self.encoder.rvq

        elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
            self.latent_size = pretrained.latent_size

        else:
            raise ValueError(
                f'Encoder type {pretrained.encoder.__class__.__name__} not supported'
            )

        x = torch.zeros(1, self.channels, 2**14)
        z = self.encode(x)
        ratio_encode = x.shape[-1] // z.shape[-1]

        self.register_method(
            "encode",
            in_channels=self.channels,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=ratio_encode,
            input_labels=['(signal) Channel %d'%d for d in range(self.channels)],
            output_labels=[
                f'(signal) Latent dimension {i}'
                for i in range(self.latent_size)
            ],
        )
        self.register_method(
            "decode",
            in_channels=self.latent_size,
            in_ratio=ratio_encode,
            out_channels=self.channels,
            out_ratio=1,
            input_labels=[
                f'(signal) Latent dimension {i}'
                for i in range(self.latent_size)
            ],
            output_labels=[
                f'(signal) Reconstructed audio signal {channel}'
                for channel in range(self.channels)
            ],
        )

        self.register_method(
            "forward",
            in_channels=self.channels,
            in_ratio=1,
            out_channels=self.channels,
            out_ratio=1,
            input_labels=['(signal) Channel %d'%d for d in range(self.channels)],
            output_labels=[
                f'(signal) Reconstructed audio signal {channel}'
                for channel in range(channels)
            ],
        )

        self.register_attribute('dumb', 0)

    def get_dumb(self) -> int:
        return 0
    def set_dumb(self, value: int) -> None:
        return

    def post_process_latent(self, z):
        raise NotImplementedError

    def pre_process_latent(self, z):
        raise NotImplementedError

    @torch.jit.export
    def encode(self, x):
        batch_size = x.shape[0]
        x = x.reshape(-1, 1, x.shape[-1])
        x = self.pqmf(x)
        x = x.reshape(batch_size, -1, x.shape[-1])
        z = self.encoder(x)
        z = self.post_process_latent(z)
        return z

    @torch.jit.export
    def decode(self, z):
        batch_size = z.shape[0]
        z = self.pre_process_latent(z)
        y = self.decoder(z)
        y = y.reshape(y.shape[0] * self.channels, -1, y.shape[-1])
        y = self.pqmf.inverse(y)
        y = y.reshape(batch_size, self.channels, -1)
        return y

    def forward(self, x):
        return self.decode(self.encode(x))


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
        )
        z = torch.cat([z, noise], 1)
        z = F.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        z = z + self.latent_mean.unsqueeze(-1)
        return z


class DiscreteScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        z = self.quantizer.residual_quantize(z)
        return z.float()

    def pre_process_latent(self, z):
        z = torch.clamp(z, 0, self.quantizer.n_codes - 1).long()
        z = self.quantizer.residual_dequantize(z)
        z = self.encoder.add_noise_to_vector(z)
        return z


class WasserteinScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        return z

    def pre_process_latent(self, z):
        return z


def main(argv):
    cc.use_cached_conv(FLAGS.streaming)

    logging.info("building rave")

    gin.parse_config_file(os.path.join(FLAGS.run, "config.gin"),
                          #      skip_unknown=True,
                          )
    if FLAGS.channels:
        gin.bind_parameter('RAVE.n_channels', FLAGS.channels)
    checkpoint = rave.core.search_for_run(FLAGS.run)

    pretrained = rave.RAVE()
    if checkpoint is not None:
        pretrained.load_state_dict(
            torch.load(checkpoint, map_location='cpu')["state_dict"])
    else:
        print("No checkpoint found, RAVE will remain randomly initialized")
    pretrained.eval()

    if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
        script_class = VariationalScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
        script_class = DiscreteScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
        script_class = WasserteinScriptedRAVE
    else:
        raise ValueError(f"Encoder type {type(pretrained.encoder)} "
                         "not supported for export.")

    logging.info("warmup pass")

    x = torch.zeros(1, pretrained.n_channels, 2**14)
    pretrained(x)

    logging.info("remove weightnorm")

    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)

    logging.info("script model")

    scripted_rave = script_class(pretrained=pretrained, channels=FLAGS.channels)

    logging.info("save model")
    model_name = os.path.basename(os.path.normpath(FLAGS.run))
    if FLAGS.streaming:
        model_name += "_streaming"
    model_name += ".ts"

    scripted_rave.export_to_ts(os.path.join(FLAGS.run, model_name))

    logging.info(
        f"all good ! model exported to {os.path.join(FLAGS.run, model_name)}")

if __name__ == '__main__':
    app.run(main)
