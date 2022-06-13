import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
import gin

import cached_conv as cc


@gin.configurable
class Model(pl.LightningModule):

    def __init__(self, pre_net, post_net, residual_block, n_layers):
        super().__init__()
        self.pre_net = pre_net()
        self.post_net = post_net()
        self.residuals = nn.ModuleList(
            [residual_block() for _ in range(n_layers)])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

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
