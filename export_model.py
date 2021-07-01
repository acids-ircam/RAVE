import torch
import torch.nn as nn

from fd import parallel_model

parallel_model.use_buffer_conv(True)

from fd.parallel_model.model import ParallelModel, Residual
from fd.parallel_model.buffer_conv import CachedConv1d, CachedConvTranspose1d
from effortless_config import Config


class args(Config):
    RUN = None


class TraceModel(ParallelModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = None
        x = torch.randn(1, 1, 2**11)
        self(x)

        n_cache = 0

        cached_modules = [
            CachedConv1d,
            CachedConvTranspose1d,
            Residual,
        ]

        for m in self.modules():
            if any(list(map(lambda c: isinstance(m, c), cached_modules))):
                m.script_cache()
                n_cache += 1

        print(f"{n_cache} cached modules found and scripted !")
        self.encoder = torch.jit.trace(
            self.encoder,
            torch.zeros(1, 1, 2**14),
            check_trace=False,
        )
        self.decoder = torch.jit.trace(
            self.decoder,
            torch.zeros(1, self.latent_size, 128),
            check_trace=False,
        )

    @torch.jit.export
    def encode(self, x):
        mean, scale = self.encoder(x)
        z = self.reparametrize(mean, scale)[0]
        z = nn.functional.conv1d(z, self.latent_pca.unsqueeze(-1))
        return z

    @torch.jit.export
    def decode(self, z):
        z = nn.functional.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        x = self.decoder(z)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


if __name__ == "__main__":
    args.parse_args()

    model = TraceModel.load_from_checkpoint(args.RUN, strict=True).eval()
    model = torch.jit.script(model)

    torch.jit.save(
        model,
        f"traced_model_{model.sr//1000}kHz_{model.latent_size}z.torchscript",
    )
