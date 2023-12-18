import itertools
import os
import tempfile

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
    ["v2.gin", "adain.gin"],
    ["v2.gin", "wasserstein.gin"],
    ["v2.gin", "spherical.gin"],
    ["v2.gin", "hybrid.gin"],
    ["v2_small.gin", "adain.gin"],
    ["v2_small.gin", "wasserstein.gin"],
    ["v2_small.gin", "spherical.gin"],
    ["v2_small.gin", "hybrid.gin"],
    ["discrete.gin"],
    ["discrete.gin", "snake.gin"],
    ["discrete.gin", "snake.gin", "adain.gin"],
    ["discrete.gin", "snake.gin", "descript_discriminator.gin"],
    ["discrete.gin", "spectral_discriminator.gin"],
    ["discrete.gin", "noise.gin"],
    ["discrete.gin", "hybrid.gin"],
    ["v3.gin"],
    ["v3.gin", "hybrid.gin"]
]

configs += [c + ["causal.gin"] for c in configs]

model_sampling_rate = [44100, 22050]
stereo = [True, False]

configs = list(itertools.product(configs, model_sampling_rate, stereo))


@pytest.mark.parametrize(
    "config,sr,stereo",
    configs,
    ids=map(
        lambda e: " ".join(e[0]) + f" [{e[1]}] " +
        ("stereo" if e[2] else "mono"), configs),
)
def test_config(config, sr, stereo):

    gin.clear_config()
    gin.parse_config_files_and_bindings(config, [
        f"SAMPLING_RATE={sr}",
        "CAPACITY=2",
    ])

    n_channels = 2 if stereo else 1
    model = rave.RAVE(n_channels=n_channels)

    x = torch.randn(1, n_channels, 2**15)
    z, _ = model.encode(x, return_mb=True)
    z, _ = model.encoder.reparametrize(z)[:2]
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

    x = torch.zeros(1, n_channels, 2**14)

    model(x)

    for m in model.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)

    scripted_rave = script_class(
        pretrained=model,
        channels=n_channels,
    )

    scripted_rave_resampled = script_class(
        pretrained=model,
        channels=n_channels,
        target_sr=44100,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        scripted_rave.export_to_ts(os.path.join(tmpdir, "ori.ts"))
        scripted_rave_resampled.export_to_ts(
            os.path.join(tmpdir, "resampled.ts"))
