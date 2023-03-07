import gin
import pytest
import torch

import rave

gin.enter_interactive_mode()

configs = [
    ["v1.gin"],
    ["v2.gin"],
    ["v2.gin", "wasserstein.gin"],
    ["v2.gin", "spherical.gin"],
    ["discrete.gin"],
    ["discrete.gin", "spectral_discriminator.gin"],
]

configs += [c + ["causal.gin"] for c in configs]


@pytest.mark.parametrize("config", configs, ids=lambda elm: " ".join(elm))
def test_config(config):
    gin.clear_config()
    gin.parse_config_files_and_bindings(config, [])

    model = rave.RAVE()

    x = torch.randn(1, 1, 2**15)
    z = model.encode(x)
    y = model.decode(z)
    score = model.discriminator(y)

    assert x.shape == y.shape
