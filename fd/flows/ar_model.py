import torch
import torch.nn as nn
from .ar_flow import StrictCausalConv, SequentialFlow, ARFlow, ActNorm1d
import pytorch_lightning as pl
import math


class Block(nn.Module):
    def __init__(self, in_size, res_size, skp_size, kernel_size, dilation):
        super().__init__()
        self.causal_conv = nn.Conv1d(
            in_size,
            2 * res_size,
            kernel_size,
            padding=((kernel_size - 1) * dilation) // 2,
            dilation=dilation,
        )
        self.res_conv = nn.Conv1d(res_size, res_size, 1)
        self.skp_conv = nn.Conv1d(res_size, skp_size, 1)

    def forward(self, x):
        res = x.clone()
        x = self.causal_conv(x)
        xa, xb = torch.split(x, x.shape[1] // 2, 1)
        x = torch.sigmoid(xa) * torch.tanh(xb)

        res = res + self.res_conv(x)
        skp = self.skp_conv(x)

        return res, skp


class ARModel(nn.Module):
    def __init__(self, in_size, res_size, skp_size, kernel_size, n_block,
                 dilation_cycle):
        super().__init__()

        self.prenet = nn.Sequential(StrictCausalConv(in_size, res_size, 3),
                                    nn.LeakyReLU(.2))

        blocks = []
        for i in range(n_block):
            blocks.append(
                Block(
                    res_size,
                    res_size,
                    skp_size,
                    kernel_size,
                    2**(i % dilation_cycle),
                ))

        self.blocks = nn.ModuleList(blocks)

        self.postnet = nn.Sequential(
            nn.LeakyReLU(.2),
            nn.Conv1d(skp_size, skp_size, 1),
            nn.LeakyReLU(.2),
            nn.Conv1d(skp_size, 2 * in_size, 1),
        )

        self.postnet[-1].weight.data.zero_()
        self.postnet[-1].bias.data.zero_()

    def forward(self, x):
        x = self.prenet(x)

        skp = 0
        for block in self.blocks:
            x, _skp = block(x)
            skp = skp + _skp

        return self.postnet(skp)


class TeacherFlow(pl.LightningModule):
    def __init__(self, in_size, res_size, skp_size, kernel_size, n_block,
                 dilation_cycle, n_flow):
        super().__init__()
        self.save_hyperparameters()
        flows = []
        for _ in range(n_flow):
            flows.append(
                ARFlow(
                    ARModel(
                        in_size,
                        res_size,
                        skp_size,
                        kernel_size,
                        n_block,
                        dilation_cycle,
                    )))
            flows.append(ActNorm1d(in_size))

        self.flows = SequentialFlow(nn.ModuleList(flows))

    def normal_ll(self, x):
        return -.5 * (math.log(2 * math.pi) + x * x)

    def logpx(self, x):
        T = x.shape[-1]

        y, logdet = self.flows(x)
        logpy = self.normal_ll(y).reshape(y.shape[0], -1).sum(-1) / T
        logdet = logdet / T

        logpy = logpy.mean()
        logdet = logdet.mean()

        logpx = logpy + logdet

        return logpx, logpy, logdet

    def training_step(self, batch, batch_idx):
        x = batch.unsqueeze(1)

        logpx, logpy, logdet = self.logpx(x)

        self.log("logpy", logpy)
        self.log("logpx", logpx)
        self.log("logdet", logdet)

        return -logpx

    def validation_step(self, batch, batch_idx):
        x = batch.unsqueeze(1)

        logpx, logpy, logdet = self.logpx(x)

        self.log("validation", -logpx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)