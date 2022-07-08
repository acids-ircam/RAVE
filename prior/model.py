import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import trange
import gin

from .core import get_prior_receptive_field

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
                 dilation_factor,
                 decode_fun=None):
        super().__init__()
        self.pre_net = pre_net()
        self.post_net = post_net()
        self.residuals = nn.ModuleList([
            residual_block(dilation=dilation_factor**(i % cycle_size))
            for i in range(n_layers)
        ])
        self.n_quantizer = n_quantizer
        self.codebook_dim = codebook_dim
        self.decode_fun = decode_fun
        self.sampling_rate = sampling_rate

        self.use_cached_conv = cc.USE_BUFFER_CONV

        self.register_buffer("receptive_field", torch.tensor(0).long())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x, offset: int = 0, one_hot_encoding: bool = True):
        if one_hot_encoding:
            x = F.one_hot(x.long(), num_classes=self.codebook_dim)
            x = x.permute(0, 2, 1).float()
        res = self.pre_net(x)
        skp = torch.tensor(0.).to(x)
        for layer in self.residuals:
            res, skp = layer(res, skp, offset)
        x = self.post_net(skp)
        return x

    @torch.jit.export
    @torch.no_grad()
    def generate(self, x, sample: bool = True):
        for i in trange(x.shape[-1] - 1):

            start = i if self.use_cached_conv else None
            offset = i if self.use_cached_conv else 0

            pred = self.forward(x[..., start:i + 1],
                                offset=offset,
                                one_hot_encoding=True)
            pred = pred[..., -1:]
            pred = self.post_process_prediction(pred, sample=sample)

            x[:, i + 1:i + 2] = pred
        return x

    def post_process_prediction(self, x, sample: bool = True):
        if sample: x = F.gumbel_softmax(x, hard=True, dim=1)
        x = torch.argmax(x, dim=1, keepdim=False)
        return x

    def training_step(self, batch, batch_idx):
        batch = batch.permute(0, 2, 1).reshape(batch.shape[0], -1)  # B x (T D)
        pred = self.forward(batch).permute(0, 2, 1)  # B x (T D) x C

        batch = batch[:, self.receptive_field + 1:]
        pred = pred[:, self.receptive_field:-1]

        loss = nn.functional.cross_entropy(
            pred.reshape(-1, self.codebook_dim),
            batch.reshape(-1).long(),
        )

        self.log("prior_prediction", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.permute(0, 2, 1).reshape(batch.shape[0], -1)
        pred = self.forward(batch).permute(0, 2, 1)

        batch = batch[:, self.receptive_field + 1:]
        pred = pred[:, self.receptive_field:-1]

        loss = nn.functional.cross_entropy(
            pred.reshape(-1, self.codebook_dim),
            batch.reshape(-1).long(),
        )

        self.log("validation", loss)
        return batch

    def validation_epoch_end(self, out):
        if not self.receptive_field.sum():
            print("Computing receptive field for this configuration...")
            lrf = get_prior_receptive_field(self)[0]
            self.receptive_field += lrf
            print(f"Receptive field: {lrf} steps <-- z")

        if self.decode_fun is None: return

        batch = out[0][..., :512]

        z = self.generate(batch.reshape(batch.shape[0], -1), sample=True)
        z = z.reshape(z.shape[0], -1, self.n_quantizer).permute(0, 2, 1)
        y = self.decode_fun(z)

        self.logger.experiment.add_audio(
            "generation",
            y.reshape(-1),
            self.global_step,
            self.sampling_rate,
        )
