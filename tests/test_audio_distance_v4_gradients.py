import gin
import torch

import rave

gin.enter_interactive_mode()


def _grad_norm(parameters):
    grads = [
        p.grad.detach().norm()
        for p in parameters
        if p.grad is not None
    ]
    if not grads:
        return torch.tensor(0.0)
    return torch.stack(grads).sum()


def test_audio_distance_v4_backpropagates_to_v4_model_weights():
    torch.manual_seed(0)
    gin.clear_config()
    gin.parse_config_files_and_bindings(["v4.gin"], [
        "CAPACITY=2",
        "LATENT_SIZE=4",
        "spectro_discriminator.SpectroDiscriminator.capacity=2",
    ])

    model = rave.RAVE(n_channels=1)
    model.train()

    x = torch.randn(1, 1, 8192)
    y = model(x)
    distances = model.audio_distance(x, y)
    loss = sum(distances.values())

    print("AudioDistanceV4 loss:", float(loss.detach()))
    loss.backward()

    encoder_grad_norm = _grad_norm(model.encoder.parameters())
    decoder_grad_norm = _grad_norm(model.decoder.parameters())
    print("encoder grad norm:", float(encoder_grad_norm))
    print("decoder grad norm:", float(decoder_grad_norm))

    assert torch.isfinite(loss)
    assert torch.isfinite(encoder_grad_norm)
    assert torch.isfinite(decoder_grad_norm)
    assert encoder_grad_norm > 0
    assert decoder_grad_norm > 0
