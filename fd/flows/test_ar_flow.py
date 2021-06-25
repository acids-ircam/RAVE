import pytest
import torch
import torch.nn as nn
from fd.flows.ar_flow import StrictCausalConv, ARFlow, SequentialFlow, ActNorm1d
from fd.flows.ar_model import ARModel
from einops import rearrange

arflow = ARFlow(StrictCausalConv(16, 32, 3)).eval().double()
actnorm = ActNorm1d(16).eval().double()

armodel = ARFlow(ARModel(16, 20, 24, 3, 4, 3)).eval().double()

multiflow = SequentialFlow(
    nn.ModuleList([
        ARFlow(StrictCausalConv(16, 32, 3)),
        ActNorm1d(16),
        ARFlow(StrictCausalConv(16, 32, 3)),
        ActNorm1d(16),
    ])).eval().double()

flows = [arflow, multiflow, actnorm, armodel]


@pytest.mark.parametrize(
    "model",
    flows,
    ids=[f.__class__.__name__ for f in flows],
)
def test_inversible(model):
    x = torch.randn(1, 16, 32).double()
    y, logdet = model(x)
    z = model.inverse(y)
    assert torch.allclose(x, z, 1e-4, 1e-4)


@pytest.mark.parametrize(
    "model",
    flows,
    ids=[f.__class__.__name__ for f in flows],
)
def test_jacobian(model):
    x = torch.randn(1, 16, 128).double()

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