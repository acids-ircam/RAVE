import itertools

import cached_conv as cc
import pytest
import torch

from rave.blocks import *

kernel_size = [
    1,
    3,
]

dilations = [[1, 1], [3, 1]]

kernel_sizes = [
    [3],
    [3, 5],
]

dilations_list = [
    [[1, 1]],
    [[1, 1], [3, 1], [5, 1]],
]


@pytest.mark.parametrize('kernel_sizes,dilations_list',
                         itertools.product(kernel_sizes, dilations_list))
def test_residual_stack(kernel_sizes, dilations_list):
    x = torch.randn(1, 1, 32)
    cc.use_cached_conv(False)
    stack_regular = ResidualStack(
        dim=1,
        kernel_sizes=[3],
        dilations_list=[[1, 1], [3, 1], [5, 1]],
    )

    cc.use_cached_conv(True)
    stack_stream = ResidualStack(
        dim=1,
        kernel_sizes=[3],
        dilations_list=[[1, 1], [3, 1], [5, 1]],
    )

    for p1, p2 in zip(stack_regular.parameters(), stack_stream.parameters()):
        p2.data.copy_(p1.data)

    delay = stack_stream.cumulative_delay

    y_regular = stack_regular(x)
    y_stream = stack_stream(x)

    if delay:
        y_regular = y_regular[..., delay:-delay]
        y_stream = y_stream[..., delay + delay:]

    assert torch.allclose(y_regular, y_stream, 1e-4, 1e-4)


@pytest.mark.parametrize('kernel_size,dilations_list',
                         itertools.product(kernel_size, dilations))
def test_residual_layer(kernel_size, dilations_list):
    x = torch.randn(1, 1, 32)

    cc.use_cached_conv(False)
    layer_regular = ResidualLayer(1, kernel_size, dilations_list)

    cc.use_cached_conv(True)
    layer_stream = ResidualLayer(1, kernel_size, dilations_list)

    for p1, p2 in zip(layer_regular.parameters(), layer_stream.parameters()):
        p2.data.copy_(p1.data)

    delay = layer_stream.cumulative_delay

    y_regular = layer_regular(x)
    y_stream = layer_stream(x)

    if delay:
        y_regular = y_regular[..., delay:-delay]
        y_stream = y_stream[..., delay + delay:]

    assert torch.allclose(y_regular, y_stream, 1e-3, 1e-4)
