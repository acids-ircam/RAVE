import torch
from fd.flows.ar_flow import StrictCausalConv
from fd.flows.naf import NAF1d
import pytest
from einops import rearrange


def test_jacobian():
    model = StrictCausalConv(1, 3, 3)
    model = NAF1d(model).double()

    x = torch.randn(1, 1, 64).double()

    y, logdet = model(x, 0)
    true_logdet = torch.autograd.functional.jacobian(
        lambda x: model(x, 0)[0],
        x,
    )
    true_logdet = rearrange(
        true_logdet,
        "b1 c1 t1 b2 c2 t2 -> b1 b2 (c1 t1) (c2 t2)",
    ).sum(1)

    true_logdet = torch.logdet(true_logdet)
    assert torch.allclose(logdet, true_logdet, atol=1e-4, rtol=1e-4)