import torch

torch.set_grad_enabled(False)
import torch.nn.functional as F
import cached_conv as cc
import rave
import rave.blocks
import rave.core
import gin
from effortless_config import Config
import os
import math
import numpy as np


class ScriptedRAVE(torch.nn.Module):

    def __init__(self, pretrained: rave.RAVE) -> None:
        super().__init__()
        self.pqmf = pretrained.pqmf
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder

        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)
        self.register_buffer("fidelity", pretrained.fidelity)

        if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
            self.mode = "variational"
            latent_size = np.argmax(
                pretrained.fidelity.numpy() > args.FIDELITY)
            latent_size = 2**math.ceil(math.log2(latent_size))
            self.latent_size = latent_size
        elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
            self.mode = "discrete"
            self.latent_size = pretrained.encoder.num_quantizers

        x = torch.randn(1, 1, 2**14)
        x = self.pqmf(x)
        z = self.encoder(x)
        ratio_encode = x.shape[-1] // z.shape[-1]

        self.register_buffer(
            "encode_params",
            torch.tensor([1, 1, self.latent_size, ratio_encode]))

    @torch.jit.export
    def encode(self, x):
        x = self.pqmf(x)
        z = self.encoder(x)
        if self.mode == "variational":
            z, _ = self.encoder.reparametrize(z)
            z = z - self.latent_mean.unsqueeze(-1)
            z = F.conv1d(z, self.latent_pca.unsqueeze(-1))
            z = z[:, :self.latent_size]
        elif self.mode == "discrete":
            _, _, z = self.encoder.reparametrize(z)
        return z


class args(Config):
    NAME = None
    STREAMING = False
    FIDELITY = .95
    STEREO = False


args.parse_args()
cc.use_cached_conv(args.STREAMING)

assert args.NAME is not None, "YOU HAD ONE JOB: GIVE A NAME !!"
root = os.path.join("runs", args.NAME, "rave")
gin.parse_config_file(os.path.join(root, "config.gin"))
checkpoint = rave.core.search_for_run(root)

pretrained = rave.RAVE()
pretrained.load_state_dict(torch.load(checkpoint)["state_dict"])
pretrained.eval()

scripted_rave = ScriptedRAVE(pretrained)
scripted_rave = torch.jit.script(scripted_rave)

torch.jit.save(scripted_rave, "debug.ts")