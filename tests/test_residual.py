import itertools

import cached_conv as cc
import gin
import pytest
import torch

from rave.blocks import *

gin.enter_interactive_mode()

kernel_size = [
    1,
    3,
]

dilations = [[1, 1], [3, 1]]

kernel_sizes = [
    [3],
    [3, 5],
    [3, 5, 7],
]

dilations_list = [
    [[1, 1]],
    [[1, 1], [3, 1], [5, 1]],
]

ratios = [
    2,
    4,
    8,
]


@pytest.mark.parametrize('kernel_sizes,dilations_list',
                         itertools.product(kernel_sizes, dilations_list))
def test_residual_stack(kernel_sizes, dilations_list):
    dim = 16
    x = torch.randn(1, dim, 32)
    cc.use_cached_conv(False)
    stack_regular = ResidualStack(
        dim=dim,
        kernel_sizes=[3],
        dilations_list=[[1, 1], [3, 1], [5, 1]],
    )

    cc.use_cached_conv(True)
    stack_stream = ResidualStack(
        dim=dim,
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
    dim = 16
    x = torch.randn(1, dim, 32)

    cc.use_cached_conv(False)
    layer_regular = ResidualLayer(dim, kernel_size, dilations_list)

    cc.use_cached_conv(True)
    layer_stream = ResidualLayer(dim, kernel_size, dilations_list)

    for p1, p2 in zip(layer_regular.parameters(), layer_stream.parameters()):
        p2.data.copy_(p1.data)

    delay = layer_stream.cumulative_delay

    y_regular = layer_regular(x)
    y_stream = layer_stream(x)

    if delay:
        y_regular = y_regular[..., delay:-delay]
        y_stream = y_stream[..., delay + delay:]

    assert torch.allclose(y_regular, y_stream, 1e-3, 1e-4)


@pytest.mark.parametrize('ratio,', ratios)
def test_upsample_layer(ratio):
    dim = 16
    x = torch.randn(1, dim, 32)

    cc.use_cached_conv(False)
    upsample_regular = UpsampleLayer(dim, dim, ratio)

    cc.use_cached_conv(True)
    upsample_stream = UpsampleLayer(dim, dim, ratio)

    for p1, p2 in zip(upsample_regular.parameters(),
                      upsample_stream.parameters()):
        p2.data.copy_(p1.data)

    delay = upsample_stream.cumulative_delay

    y_regular = upsample_regular(x)
    y_stream = upsample_stream(x)

    if delay:
        y_regular = y_regular[..., delay:-delay]
        y_stream = y_stream[..., delay + delay:]

    assert torch.allclose(y_regular, y_stream, 1e-3, 1e-4)
