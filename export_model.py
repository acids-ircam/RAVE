import torch
import torch.nn as nn

from fd import parallel_model

parallel_model.use_buffer_conv(False)

from fd.parallel_model.model import ParallelModel
from fd.parallel_model.buffer_conv import CachedConv1d, CachedConvTranspose1d
from effortless_config import Config


class args(Config):
    RUN = None


class TraceModel(nn.Module):
    def __init__(self, trained: ParallelModel):
        super().__init__()
        self.encoder = trained.encoder
        self.decoder = trained.decoder
        self.sr = trained.sr

    def reparametrize(self, mean, scale):
        std = nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return z, kl

    @torch.jit.export
    def encode(self, x):
        mean, scale = self.encoder(x)
        z = self.reparametrize(mean, scale)[0]
        return z

    @torch.jit.export
    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


if __name__ == "__main__":
    args.parse_args()

    model = ParallelModel.load_from_checkpoint(args.RUN, strict=False)
    model = TraceModel(model)

    x = torch.randn(1, 1, 1024)
    z = model.encode(x)
    y = model.decode(z)

    model(x)

    for m in model.modules():
        if isinstance(m, CachedConv1d) or isinstance(m, CachedConvTranspose1d):
            m.script_cache()

    latent_size = int(z.shape[1])
    sr = model.sr

    model = torch.jit.script(model)

    torch.jit.save(model, f"traced_model_{sr}Hz_{latent_size}z.torchscript")