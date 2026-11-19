import gin
import torch
import torch.nn as nn

from rave import spectro_discriminator

gin.enter_interactive_mode()

def _parse_v4_config(bindings=None):
    gin.clear_config()
    gin.parse_config_files_and_bindings(["v4.gin"], bindings or [])


def test_v4_spectro_discriminator_forward():
    _parse_v4_config([
        "spectro_discriminator.SpectroDiscriminator.capacity = 2",
    ])

    discriminator = gin.get_configurable(
        "spectro_discriminator.SpectroDiscriminator")()
    x = torch.randn(2, 1, 4096*16)

    with torch.no_grad():
        features = discriminator(x)

    assert len(features) == 6
    for scale_features in features:
        assert len(scale_features) == 7
        for feature_map in scale_features:
            assert feature_map.shape[0] == x.shape[0]
            assert feature_map.ndim == 4
            assert torch.isfinite(feature_map).all()
