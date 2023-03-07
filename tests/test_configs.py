import gin
import pytest
import torch
import torch.nn as nn

import rave
from scripts import export

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

    if isinstance(model.encoder, rave.blocks.VariationalEncoder):
        script_class = export.VariationalScriptedRAVE
    elif isinstance(model.encoder, rave.blocks.DiscreteEncoder):
        script_class = export.DiscreteScriptedRAVE
    elif isinstance(model.encoder, rave.blocks.WasserteinEncoder):
        script_class = export.WasserteinScriptedRAVE
    elif isinstance(model.encoder, rave.blocks.SphericalEncoder):
        script_class = export.SphericalScriptedRAVE
    else:
        raise ValueError(f"Encoder type {type(model.encoder)} "
                         "not supported for export.")

    x = torch.zeros(1, 1, 2**14)
    model(x)

    for m in model.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)

    scripted_rave = script_class(pretrained=model, stereo=False)
