import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import gin

import core

import cached_conv as cc


@gin.configurable
class Prior(pl.LightningModule):
    def __init__(self,
                 pre_net,
                 post_net,
                 residual_block,
                 n_layers,
                 n_quantizer,
                 codebook_dim,
                 sampling_rate,
                 cycle_size,
                 decode_fun=None):
        super().__init__()
        self.pre_net = pre_net()
        self.post_net = post_net()
        self.residuals = nn.ModuleList([
            residual_block(dilation=2**(i % cycle_size))
            for i in range(n_layers)
        ])
        self.n_quantizer = n_quantizer
        self.codebook_dim = codebook_dim
        self.decode_fun = decode_fun
        self.sampling_rate = sampling_rate

        self.register_buffer("receptive_field", torch.tensor(0).long())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x, offset=0, one_hot_encoding=True):
        if one_hot_encoding:
            x = F.one_hot(x, self.codebook_dim).permute(0, 2, 1).float()
        res = self.pre_net(x)
        skp = torch.tensor(0.).to(x)
        for layer in self.residuals:
            res, skp = layer(res, skp, offset)
        x = self.post_net(skp)
        return x

    @torch.no_grad()
    def generate(self, x, sample: bool = True):
        for i in tqdm(range(x.shape[-1] - 1)):

            start = i if cc.USE_BUFFER_CONV else None
            offset = i if cc.USE_BUFFER_CONV else 0

            pred = self.forward(x[..., start:i + 1], offset=offset)

            if not cc.USE_BUFFER_CONV:
                pred = pred[..., -1:]

            pred = self.post_process_prediction(pred, sample=sample)

            x[..., i + 1:i + 2] = pred
        return x

    def post_process_prediction(self, x, sample: bool = True):
        if sample: x = F.gumbel_softmax(x, hard=True, dim=1)
        x = torch.argmax(x, dim=1, keepdim=True)
        return x

    def training_step(self, batch, batch_idx):
        batch = batch.permute(0, 2, 1).reshape(batch.shape[0], -1)
        pred = self.forward(batch).permute(0, 2, 1)

        batch = batch[:, self.receptive_field:-1]
        pred = pred[:, self.receptive_field + 1:]

        loss = nn.functional.cross_entropy(
            pred.reshape(-1, self.codebook_dim),
            batch.reshape(-1),
        )

        self.log("prior_prediction", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.permute(0, 2, 1).reshape(batch.shape[0], -1)
        pred = self.forward(batch).permute(0, 2, 1)

        loss = nn.functional.cross_entropy(
            pred.reshape(-1, self.codebook_dim),
            batch.reshape(-1),
        )

        self.log("validation", loss)
        return batch

    def validation_epoch_end(self, out):
        if not self.receptive_field.sum():
            print("Computing receptive field for this configuration...")
            lrf = core.get_prior_receptive_field(self)[0]
            self.receptive_field = lrf
            print(f"Receptive field: {1000*lrf/self.sr:.2f}ms <-- z")

        if self.decode_fun is None: return

        batch = out[0]

        z = self.generate(batch)
        z = z.reshape(z.shape[0], -1, self.n_quantizer).permute(0, 2, 1)
        y = self.decode_fun(z)

        self.logger.experiment.add_audio(
            "generation",
            y.reshape(-1),
            self.global_step,
            self.sampling_rate.item(),
        )
