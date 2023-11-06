import math
import torch
import torch.nn as nn


class QuantizedNormal(nn.Module):
    def __init__(self, resolution, dither=True):
        super().__init__()
        self.resolution = resolution
        self.dither = dither
        self.clamp = 4

    def from_normal(self, x):
        return .5 * (1 + torch.erf(x / math.sqrt(2)))

    def to_normal(self, x):
        x = torch.erfinv(2 * x - 1) * math.sqrt(2)
        return torch.clamp(x, -self.clamp, self.clamp)

    def encode(self, x):
        x = self.from_normal(x)
        x = torch.floor(x * self.resolution)
        x = torch.clamp(x, 0, self.resolution - 1)
        return self.to_stack_one_hot(x.long())

    def to_stack_one_hot(self, x):
        x = nn.functional.one_hot(x, self.resolution)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1).float()
        return x

    def decode(self, x):
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1, self.resolution)
        x = torch.argmax(x, -1) / self.resolution
        if self.dither:
            x = x + torch.rand_like(x) / self.resolution
        x = self.to_normal(x)
        x = x.permute(0, 2, 1)
        return x


class DiagonalShift(nn.Module):
    def __init__(self, groups=1):
        super().__init__()
        assert isinstance(groups, int)
        assert groups > 0
        self.groups = groups

    def shift(self, x: torch.Tensor, i: int, n_dim: int):
        i = i // self.groups
        n_dim = n_dim // self.groups
        start = i
        end = -n_dim + i + 1
        end = end if end else None
        return x[..., start:end]

    def forward(self, x):
        n_dim = x.shape[1]
        x = torch.split(x, 1, 1)
        x = [
            self.shift(_x, i, n_dim) for _x, i in zip(
                x,
                torch.arange(n_dim).flip(0),
            )
        ]
        x = torch.cat(list(x), 1)
        return x

    def inverse(self, x):
        x = x.flip(1)
        x = self.forward(x)
        x = x.flip(1)
        return x