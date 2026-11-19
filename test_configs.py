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
    # ["v2.gin", "hybrid.gin"], NOT READY YET
    ["discrete.gin"],
    ["discrete.gin", "snake.gin"],
    ["discrete.gin", "snake.gin", "adain.gin"],
    ["discrete.gin", "snake.gin", "descript_discriminator.gin"],
    ["discrete.gin", "spectral_discriminator.gin"],
    ["discrete.gin", "noise.gin"],
    ["v3.gin"],
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
    if any(map(lambda x: "adain" in x, config)) and stereo:
        pytest.skip()

    gin.clear_config()
    gin.parse_config_files_and_bindings(config, [
        f"SAMPLING_RATE={sr}",
        "CAPACITY=2",
    ])

    model = rave.RAVE()

    if stereo:
        for m in model.modules():
            if isinstance(m, rave.blocks.AdaptiveInstanceNormalization):
                pytest.skip()

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

    scripted_rave = script_class(
        pretrained=model,
        stereo=stereo,
    )

    scripted_rave_resampled = script_class(
        pretrained=model,
        stereo=stereo,
        target_sr=44100,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        scripted_rave.export_to_ts(os.path.join(tmpdir, "ori.ts"))
        scripted_rave_resampled.export_to_ts(
            os.path.join(tmpdir, "resampled.ts"))
