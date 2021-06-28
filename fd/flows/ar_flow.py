import torch
import torch.nn as nn
from tqdm import tqdm


class StrictCausalConv(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        self.causal_padding = self.kernel_size[0]

    def forward(self, x):
        x = nn.functional.pad(x, (self.causal_padding, 0))
        x = super().forward(x)[..., :-1]
        return x


class ARFlow(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def extract_parameters(self, x):
        out = self.net(x)
        mean, var = torch.split(out, out.shape[1] // 2, 1)
        var = torch.sigmoid(var)
        return mean, var

    def apply_transform(self, x):
        mean, var = self.extract_parameters(x)
        x = (x - mean) * var
        _logdet = torch.log(var).reshape(var.shape[0], -1).sum(-1)
        return x, _logdet

    def forward(self, x, logdet=0):
        x, _logdet = self.apply_transform(x)
        return x, logdet + _logdet

    @torch.no_grad()
    def inverse(self, x):
        for t in tqdm(range(x.shape[-1])):
            mean, var = self.extract_parameters(x[..., :t + 1])
            x[..., t:t + 1] = x[..., t:t + 1] / var[..., -1:] + mean[..., -1:]
        return x


class SequentialFlow(nn.Module):
    def __init__(self, flows: nn.ModuleList):
        super().__init__()
        self.flows = flows

    def forward(self, x, logdet=0):
        for f in self.flows:
            x, logdet = f(x, logdet)
        return x, logdet

    @torch.no_grad()
    def inverse(self, x):
        for f in self.flows[::-1]:
            x = f.inverse(x)
        return x


class ActNormNd(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(0))
        self.dim = dim

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dim})"

    @property
    def shape(self):
        raise NotImplementedError

    def forward(self, x, logdet=0):
        if not self.initialized:
            with torch.no_grad():
                scale = x.transpose(0, 1).reshape(x.shape[0], -1).std()
                scale = -torch.log(torch.clamp(scale, 1e-2))
                self.scale.copy_(scale.detach())

                bias = x.transpose(0, 1).reshape(x.shape[0], -1).mean()
                self.bias.copy_(-bias.detach())

                self.initialized.fill_(1)

        scale = torch.exp(self.scale.reshape(self.shape))

        bias = self.bias.reshape(self.shape)
        x = scale * (x + bias)

        _logdet = torch.log(scale).sum() * torch.prod(torch.tensor(
            x.shape[2:]))

        return x, logdet + _logdet.reshape(-1)

    @torch.no_grad()
    def inverse(self, x):
        assert self.initialized, "actnorm is not initialized"
        scale = torch.exp(-self.scale.reshape(self.shape))
        bias = self.bias.reshape(self.shape)

        x = x * scale - bias
        return x


class ActNorm1d(ActNormNd):
    @property
    def shape(self):
        return [1, -1, 1]