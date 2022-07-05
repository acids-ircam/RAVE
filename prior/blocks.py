import torch
import torch.nn as nn
import gin
import cached_conv as cc


@gin.register
class ResidualBlock(nn.Module):
    def __init__(self, res_size, skp_size, kernel_size, dilation, n_dim):
        super().__init__()
        fks = (kernel_size - 1) * dilation + 1

        self.dconv = cc.Conv1d(
            res_size,
            2 * res_size,
            kernel_size,
            padding=cc.get_padding(kernel_size, 1, dilation, "causal"),
            dilation=dilation,
        )
        self.rconv = nn.Conv1d(res_size, res_size, 1)
        self.sconv = nn.Conv1d(res_size, skp_size, 1)

        self.dim_embedding = nn.Embedding(n_dim, res_size)

        self.n_dim = n_dim

    def forward(self, x, skp, offset=0):
        res = x.clone()

        idx = (torch.arange(
            x.shape[-1],
            device=x.device,
        ) + offset) % self.n_dim
        bias = self.dim_embedding(idx).transpose(0, 1)

        x = x + bias

        x = self.dconv(x)
        xa, xb = torch.split(x, x.shape[1] // 2, 1)

        x = torch.sigmoid(xa) * torch.tanh(xb)
        res = res + self.rconv(x)
        skp = skp + self.sconv(x)
        return res, skp


@gin.register
def pre_net(in_dim, res_size, kernel_size):
    return nn.Sequential(
        cc.Conv1d(in_dim,
                  res_size,
                  kernel_size,
                  padding=cc.get_padding(kernel_size, mode="causal")),
        nn.LeakyReLU(.2),
    )


@gin.register
def post_net(skp_size, out_dim):
    return nn.Sequential(
        cc.Conv1d(skp_size, skp_size, 1),
        nn.LeakyReLU(.2),
        cc.Conv1d(
            skp_size,
            out_dim,
            1,
        ),
    )