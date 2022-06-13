import torch
import torch.nn as nn
import gin
import cached_conv as cc


class ResidualBlock(nn.Module):

    def __init__(self, res_size, skp_size, kernel_size, dilation):
        super().__init__()
        fks = (kernel_size - 1) * dilation + 1

        self.dconv = cc.Conv1d(
            res_size,
            2 * res_size,
            kernel_size,
            padding=(fks - 1, 0),
            dilation=dilation,
        )

        self.rconv = nn.Conv1d(res_size, res_size, 1)
        self.sconv = nn.Conv1d(res_size, skp_size, 1)

    def forward(self, x, skp):
        res = x.clone()

        x = self.dconv(x)
        xa, xb = torch.split(x, x.shape[1] // 2, 1)

        x = torch.sigmoid(xa) * torch.tanh(xb)
        res = res + self.rconv(x)
        skp = skp + self.sconv(x)
        return res, skp


@gin.register
def pre_net(dim, res_size, kernel_size):
    return nn.Sequential(
        cc.Conv1d(dim,
                  res_size,
                  kernel_size,
                  padding=cc.get_padding(kernel_size, mode="causal")),
        nn.LeakyReLU(.2),
    )


@gin.register
def post_net(skp_size, dim):
    return nn.Sequential(
        cc.Conv1d(skp_size, skp_size, 1),
        nn.LeakyReLU(.2),
        cc.Conv1d(
            skp_size,
            dim,
            1,
        ),
    )
