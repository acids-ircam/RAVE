import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm

from .residual_block import ResidualBlock
from .core import DiagonalShift, QuantizedNormal

import cached_conv as cc


class Model(pl.LightningModule):

    def __init__(self, resolution, res_size, skp_size, kernel_size, cycle_size,
                 n_layers, pretrained_vae):
        super().__init__()
        self.save_hyperparameters()

        self.diagonal_shift = DiagonalShift()
        self.quantized_normal = QuantizedNormal(resolution)

        self.synth = torch.jit.load(pretrained_vae)
        self.sr = self.synth.sampling_rate.item()
        data_size = self.synth.cropped_latent_size

        self.pre_net = nn.Sequential(
            cc.Conv1d(
                resolution * data_size,
                res_size,
                kernel_size,
                padding=cc.get_padding(kernel_size, mode="causal"),
                groups=data_size,
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
                resolution * data_size,
                1,
                groups=data_size,
            ),
        )

        self.data_size = data_size

        self.val_idx = 0

    def configure_optimizers(self):
        p = []
        p.extend(list(self.pre_net.parameters()))
        p.extend(list(self.residuals.parameters()))
        p.extend(list(self.post_net.parameters()))
        return torch.optim.Adam(p, lr=1e-4)

    @torch.no_grad()
    def encode(self, x):
        self.synth.eval()
        return self.synth.encode(x)

    @torch.no_grad()
    def decode(self, z):
        self.synth.eval()
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
        x = x.reshape(x.shape[0], x.shape[1], self.data_size, -1)
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
            self.synth.sampling_rate.item(),
        )
        self.val_idx += 1
