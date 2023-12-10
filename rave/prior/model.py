import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import gin
from tqdm import tqdm
import math
import numpy as np

from .residual_block import ResidualBlock
from .core import DiagonalShift, QuantizedNormal


import cached_conv as cc

class Prior(pl.LightningModule):

    def __init__(self, resolution, res_size, skp_size, kernel_size, cycle_size,
                 n_layers, pretrained_vae=None, fidelity=None, n_channels=1, latent_size=None, sr=44100):
        super().__init__()

        self.diagonal_shift = DiagonalShift()
        self.quantized_normal = QuantizedNormal(resolution)

        self.synth = pretrained_vae
        self.sr = sr

        if latent_size is not None:
            self.latent_size = 2**math.ceil(math.log2(latent_size))
        elif fidelity is not None:
            assert pretrained_vae, "giving fidelity keyword needs the pretrained_vae keyword to be given"
            latent_size = torch.where(pretrained_vae.fidelity > fidelity)[0][0]
            self.latent_size = 2**math.ceil(math.log2(latent_size))
        else:
            raise RuntimeError('please init Prior with either fidelity or latent_size keywords')

        self.pre_net = nn.Sequential(
            cc.Conv1d(
                resolution * self.latent_size,
                res_size,
                kernel_size,
                padding=cc.get_padding(kernel_size, mode="causal"),
                groups=self.latent_size,
            ),
            nn.LeakyReLU(.2),
        )

        self.residuals = nn.ModuleList([
            ResidualBlock(
                res_size,
                skp_size,
                kernel_size,
                2**(i % cycle_size),
            ) for i in range(n_layers)
        ])

        self.post_net = nn.Sequential(
            cc.Conv1d(skp_size, skp_size, 1),
            nn.LeakyReLU(.2),
            cc.Conv1d(
                skp_size,
                resolution * self.latent_size,
                1,
                groups=self.latent_size,
            ),
        )

        self.n_channels = n_channels
        self.val_idx = 0
        rf = (kernel_size - 1) * sum(2**(np.arange(n_layers) % cycle_size)) + 1
        if pretrained_vae is not None:
            ratio = self.get_model_ratio()
            self.min_receptive_field = 2**math.ceil(math.log2(rf * ratio))

    def get_model_ratio(self):
        x_len = 2**14
        x = torch.zeros(1, self.n_channels, x_len)
        z = self.encode(x)
        ratio_encode = x_len // z.shape[-1]
        return ratio_encode

    def configure_optimizers(self):
        p = []
        p.extend(list(self.pre_net.parameters()))
        p.extend(list(self.residuals.parameters()))
        p.extend(list(self.post_net.parameters()))
        return torch.optim.Adam(p, lr=1e-4)

    @torch.no_grad()
    def encode(self, x):
        self.synth.eval()
        z = self.synth.encode(x)
        z = self.post_process_latent(z)
        return z

    @torch.no_grad()
    def decode(self, z):
        self.synth.eval()
        z = self.pre_process_latent(z)
        return self.synth.decode(z)

    def forward(self, x):
        res = self.pre_net(x)
        skp = torch.tensor(0.).to(x)
        for layer in self.residuals:
            res, skp = layer(res, skp)
        x = self.post_net(skp)
        return x

    @torch.no_grad()
    def generate(self, x, argmax: bool = False):
        for i in tqdm(range(x.shape[-1] - 1)):
            if cc.USE_BUFFER_CONV:
                start = i
            else:
                start = None

            pred = self.forward(x[..., start:i + 1])

            if not cc.USE_BUFFER_CONV:
                pred = pred[..., -1:]

            pred = self.post_process_prediction(pred, argmax=argmax)

            x[..., i + 1:i + 2] = pred
        return x

    def split_classes(self, x):
        # B x D*C x T
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], self.latent_size, -1)
        x = x.permute(0, 2, 1, 3)  # B x D x T x C
        return x

    def post_process_prediction(self, x, argmax: bool = False):
        x = self.split_classes(x)
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])

        if argmax:
            x = torch.argmax(x, -1)
        else:
            x = torch.softmax(x - torch.logsumexp(x, -1, keepdim=True), -1)
            x = torch.multinomial(x, 1, True).squeeze(-1)

        x = x.reshape(shape[0], shape[1], shape[2])
        x = self.quantized_normal.to_stack_one_hot(x)
        return x

    def training_step(self, batch, batch_idx):
        x = self.encode(batch)
        x = self.quantized_normal.encode(self.diagonal_shift(x))
        pred = self.forward(x)

        x = torch.argmax(self.split_classes(x[..., 1:]), -1)
        pred = self.split_classes(pred[..., :-1])

        loss = nn.functional.cross_entropy(
            pred.reshape(-1, self.quantized_normal.resolution),
            x.reshape(-1),
        )

        self.log("latent_prediction", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.encode(batch)
        x = self.quantized_normal.encode(self.diagonal_shift(x))
        pred = self.forward(x)

        x = torch.argmax(self.split_classes(x[..., 1:]), -1)
        pred = self.split_classes(pred[..., :-1])

        loss = nn.functional.cross_entropy(
            pred.reshape(-1, self.quantized_normal.resolution),
            x.reshape(-1),
        )

        self.log("validation", loss)
        return batch

    def validation_epoch_end(self, out):
        x = torch.randn_like(self.encode(out[0]))
        x = self.quantized_normal.encode(self.diagonal_shift(x))
        z = self.generate(x)
        z = self.diagonal_shift.inverse(self.quantized_normal.decode(z))

        y = self.decode(z)
        self.logger.experiment.add_audio(
            "generation",
            y.reshape(-1),
            self.val_idx,
            self.synth.sr,
        )
        self.val_idx += 1

    @abc.abstractmethod
    def post_process_latent(self, z):
        raise NotImplementedError()

    @abc.abstractmethod
    def pre_process_latent(self, z):
        raise NotImplementedError()



@gin.configurable
class VariationalPrior(Prior):

    def post_process_latent(self, z):
        z = self.synth.encoder.reparametrize(z)[0]
        z = z - self.synth.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.synth.latent_pca.unsqueeze(-1))
        z = z[:, :self.latent_size]
        return z

    def pre_process_latent(self, z):
        noise = torch.randn(
            z.shape[0],
            self.synth.latent_size - z.shape[1],
            z.shape[-1],
        ).type_as(z)
        z = torch.cat([z, noise], 1)
        z = F.conv1d(z, self.synth.latent_pca.T.unsqueeze(-1))
        z = z + self.synth.latent_mean.unsqueeze(-1)
        return z