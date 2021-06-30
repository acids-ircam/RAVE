import torch

torch.set_grad_enabled(False)

import torch.nn as nn
from fd.parallel_model.buffer_conv import CachedConv1d, CachedConvTranspose1d, Conv1d
from fd.parallel_model.core import get_padding
import pytest
import matplotlib.pyplot as plt

convs = [
    Conv1d(16, 16, 3, stride=1, padding=get_padding(3, 1, 1)),
    Conv1d(16, 16, 1, stride=2, padding=get_padding(1, 2, 1)),
    Conv1d(16, 16, 3, stride=2, padding=get_padding(3, 2, 1)),
    Conv1d(16, 16, 3, stride=1, padding=get_padding(3, 1, 2), dilation=2),
    Conv1d(16, 16, 5, stride=4, padding=get_padding(5, 4, 1), dilation=1),
    Conv1d(16, 16, 5, stride=2, padding=get_padding(5, 2, 1), dilation=1),
    Conv1d(16, 16, 3, stride=2, padding=get_padding(3, 2, 2), dilation=2),
]

id_extraction = lambda conv: f"k:{conv.kernel_size[0]}, s:{conv.stride[0]}, p:{conv._pad}, d:{conv.dilation[0]}"
ids = list(map(id_extraction, convs))


@pytest.mark.parametrize("conv", convs, ids=ids)
def test_conv(conv: nn.Conv1d):
    x = torch.randn(1, conv.in_channels, 256)
    cconv = CachedConv1d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size[0],
        stride=conv.stride[0],
        padding=conv._pad,
        dilation=conv.dilation[0],
    )
    cconv.weight.data.copy_(conv.weight.data)
    cconv.bias.data.copy_(conv.bias.data)

    y = conv(x)

    xs = torch.split(x, 64, -1)
    ys = []

    for _x in xs:
        ys.append(cconv(_x))

    ys = torch.cat(ys, -1)

    assert y.shape == torch.Size(
        [1, conv.out_channels, x.shape[-1] // conv.stride[0]])
    assert ys.shape == torch.Size(
        [1, conv.out_channels, x.shape[-1] // conv.stride[0]])

    if (p := conv._pad[-1]) != 0:
        ys = ys[..., p:]
        y = y[..., :-p]

    # plt.plot(y.reshape(-1))
    # plt.plot(ys.reshape(-1))
    # plt.show()

    assert torch.allclose(y, ys, 1e-4, 1e-4)


convts = [
    nn.ConvTranspose1d(16, 16, 4, 2, 1),
    nn.ConvTranspose1d(16, 16, 8, 4, 2),
    nn.ConvTranspose1d(16, 16, 16, 8, 4),
]

id_extraction = lambda conv: f"k:{conv.kernel_size[0]}, s:{conv.stride[0]}"


@pytest.mark.parametrize("conv", convts, ids=list(map(id_extraction, convts)))
def test_conv_t(conv: nn.ConvTranspose1d):
    x = torch.randn(1, conv.in_channels, 128)
    cconv = CachedConvTranspose1d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
    )

    cconv.weight.data.copy_(conv.weight.data)
    cconv.bias.data.copy_(conv.bias.data)

    xs = torch.split(x, 64, -1)
    ys = []

    for _x in xs:
        _x = cconv(_x)
        ys.append(_x)

    ys = torch.cat(ys, -1)[..., conv.padding[0]:]
    y = conv(x)[..., :-conv.padding[0]]

    # plt.plot(y.reshape(-1))
    # plt.plot(ys.reshape(-1))
    # plt.show()

    assert torch.allclose(ys, y, 1e-4, 1e-4)
