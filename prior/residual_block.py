import torch
import torch.nn as nn
from cached_conv import Conv1d, CachedConv1d, USE_BUFFER_CONV

Conv1d = CachedConv1d if USE_BUFFER_CONV else Conv1d


class ResidualBlock(nn.Module):
    def __init__(self, res_size, skp_size, kernel_size, dilation):
        super().__init__()
        fks = (kernel_size - 1) * dilation + 1

        self.dconv = Conv1d(
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
