import torch
import torch.nn as nn
from fd.parallel_model.buffer_conv import CachedConv1d, CachedConvTranspose1d
import pytest


def test_conv():
    x = torch.randn(1, 16, 256)
    conv = nn.Conv1d(16, 16, 3, padding=2)

    cconv = CachedConv1d(16, 16, 3, padding=1)
    cconv.weight.data.copy_(conv.weight.data)
    cconv.bias.data.copy_(conv.bias.data)

    xs = torch.split(x, 64, -1)
    ys = []

    for _x in xs:
        ys.append(cconv(_x))

    ys = torch.cat(ys, -1)

    truth = conv(x)[..., :256]

    assert torch.allclose(ys, truth, 1e-4, 1e-4)


def test_conv_t():
    x = torch.randn(1, 16, 256)
    conv = nn.ConvTranspose1d(16, 16, 4, 2, 1)
    cconv = CachedConvTranspose1d(16, 16, 4, stride=2)

    cconv.weight.data.copy_(conv.weight.data)
    cconv.bias.data.copy_(conv.bias.data)

    xs = torch.split(x, 64, -1)
    ys = []

    for _x in xs:
        _x = cconv(_x)
        ys.append(_x)

    ys = torch.cat(ys, -1)[..., 1:]
    truth = conv(x)[..., :-1]

    assert torch.allclose(ys, truth, 1e-4, 1e-4)
