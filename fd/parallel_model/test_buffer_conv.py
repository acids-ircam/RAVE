import torch

torch.set_grad_enabled(False)

import torch.nn as nn
from fd.parallel_model.buffer_conv import CachedConv1d, CachedConvTranspose1d, Conv1d
from fd.parallel_model.core import get_padding
import pytest
import matplotlib.pyplot as plt

convs = [
    Conv1d(1, 1, 3, stride=1, padding=get_padding(3, 1, 1)),
    Conv1d(1, 1, 1, stride=2, padding=get_padding(1, 2, 1)),
    Conv1d(1, 1, 3, stride=2, padding=get_padding(3, 2, 1)),
    Conv1d(1, 1, 3, stride=1, padding=get_padding(3, 1, 2), dilation=2),
    Conv1d(1, 1, 5, stride=4, padding=get_padding(5, 4, 1), dilation=1),
    Conv1d(1, 1, 5, stride=2, padding=get_padding(5, 2, 1), dilation=1),
    Conv1d(1, 1, 3, stride=2, padding=get_padding(3, 2, 2), dilation=2),
]

xs = len(convs) * [torch.randn(1, 1, 128)]
id_extraction = lambda conv: f"k:{conv.kernel_size[0]}, s:{conv.stride[0]}, p:{conv.padding[0]}, d:{conv.dilation[0]}"
ids = list(map(id_extraction, convs))


@pytest.mark.parametrize("x,conv", zip(xs, convs), ids=ids)
def test_conv(x, conv: nn.Conv1d):

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

    assert y.shape == torch.Size([1, 1, x.shape[-1] // conv.stride[0]])
    assert ys.shape == torch.Size([1, 1, x.shape[-1] // conv.stride[0]])

    if (p := conv._pad[-1]) != 0:
        ys = ys[..., p:]
        y = y[..., :-p]

    plt.plot(y.reshape(-1))
    plt.plot(ys.reshape(-1))
    plt.show()

    assert torch.allclose(y, ys, 1e-4, 1e-4)


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
