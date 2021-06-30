import torch
import torch.nn as nn
import pytest
from fd.parallel_model.core import get_padding
from fd.parallel_model.buffer_conv import Conv1d


def test_padding():
    assert get_padding(kernel_size=3, stride=1, dilation=1) == (1, 1)
    assert get_padding(kernel_size=3, stride=1, dilation=2) == (2, 2)
    assert get_padding(kernel_size=3, stride=2, dilation=1) == (1, 0)
    assert get_padding(kernel_size=3, stride=2, dilation=1) == (1, 0)
    assert get_padding(kernel_size=1, stride=2, dilation=1) == (0, 0)
    assert get_padding(kernel_size=3, stride=2, dilation=2) == (2, 1)


convs = [
    Conv1d(1, 1, 3, stride=1, dilation=1, padding=get_padding(3, 1, 1)),
    Conv1d(1, 1, 1, stride=1, dilation=1, padding=get_padding(1, 1, 1)),
    Conv1d(1, 1, 3, stride=2, dilation=1, padding=get_padding(3, 2, 1)),
    Conv1d(1, 1, 3, stride=2, dilation=2, padding=get_padding(3, 2, 2)),
    Conv1d(1, 1, 5, stride=4, dilation=1, padding=get_padding(5, 4, 1)),
    Conv1d(1, 1, 3, stride=4, dilation=2, padding=get_padding(3, 4, 2)),
]


@pytest.mark.parametrize(
    "conv",
    convs,
    ids=list(
        map(
            lambda c: f"k:{c.kernel_size}, s:{c.stride}, p:{c.padding}",
            convs,
        )))
def test_conv(conv):
    x = torch.randn(1, 1, 256)
    y = conv(x)
    assert y.shape == torch.Size([1, 1, 256 // conv.stride[0]])
