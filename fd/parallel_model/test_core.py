import pytest
from fd.parallel_model.core import get_padding


def test_padding():
    assert get_padding(kernel_size=3, stride=1, dilation=1) == (1, 1)
    assert get_padding(kernel_size=3, stride=1, dilation=2) == (2, 2)
    assert get_padding(kernel_size=3, stride=2, dilation=1) == (1, 0)
    assert get_padding(kernel_size=3, stride=2, dilation=1) == (1, 0)
    assert get_padding(kernel_size=1, stride=2, dilation=1) == (0, 0)
    assert get_padding(kernel_size=3, stride=2, dilation=2) == (2, 1)

