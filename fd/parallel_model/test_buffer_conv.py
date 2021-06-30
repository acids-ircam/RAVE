import torch
import torch.nn as nn
from fd.parallel_model.buffer_conv import CachedConv1d, CachedConvTranspose1d
from fd.parallel_model.core import get_padding
import pytest
import matplotlib.pyplot as plt

convs = [
    nn.Conv1d(1, 1, 3, stride=1, padding=1),
    nn.Conv1d(1, 1, 1, stride=2, padding=0),
    nn.Conv1d(1, 1, 3, stride=2, padding=1),
    nn.Conv1d(1, 1, 3, stride=1, padding=2, dilation=2),
    nn.Conv1d(1, 1, 3, stride=2, padding=2, dilation=2),
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
        padding=conv.padding[0],
        dilation=conv.dilation[0],
    )
    cconv.weight.data.copy_(conv.weight.data)
    cconv.bias.data.copy_(conv.bias.data)

    xs = torch.split(x, 16, -1)
    ys = []

    for _x in xs:
        ys.append(cconv(_x))

    ys = torch.cat(ys, -1)
    truth = conv(x)

    if p := conv.padding[0] != 0:
        ys = ys[..., p:]
        truth = truth[..., :-p]

    plt.plot(ys.reshape(-1).detach())
    plt.plot(truth.reshape(-1).detach())
    plt.legend(["ys", "truth"])
    plt.show()

    assert ys.shape == truth.shape
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
