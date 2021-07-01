import torch
import torch.nn as nn

from fd import parallel_model

parallel_model.use_buffer_conv(True)

from fd.parallel_model.model import ParallelModel, Residual
from fd.parallel_model.buffer_conv import CachedConv1d, CachedConvTranspose1d
from fd.parallel_model.resample import Resampling
from effortless_config import Config


class args(Config):
    RUN = None
    SR = None


class TraceModel(nn.Module):
    def __init__(self, pretrained: ParallelModel, resample: Resampling):
        super().__init__()
        latent_size = pretrained.latent_size
        self.resample = resample
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder

        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)
        self.register_buffer("latent_size", torch.tensor(latent_size))
        self.register_buffer("sampling_rate", torch.tensor(pretrained.sr))

    def reparametrize(self, mean, scale):
        std = nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return z, kl

    @torch.jit.export
    def encode(self, x):
        x = self.resample.from_target_sampling_rate(x)
        mean, scale = self.encoder(x)
        z = self.reparametrize(mean, scale)[0]
        z = z - self.latent_mean.unsqueeze(-1)
        z = nn.functional.conv1d(z, self.latent_pca.unsqueeze(-1))
        return z

    @torch.jit.export
    def decode(self, z):
        z = nn.functional.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        z = z + self.latent_mean.unsqueeze(-1)
        x = self.decoder(z)
        x = self.resample.to_target_sampling_rate(x)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


if __name__ == "__main__":
    args.parse_args()

    print("loading model from checkpoint")
    model = ParallelModel.load_from_checkpoint(args.RUN, strict=True).eval()

    print("warmup forward pass")
    x = torch.zeros(1, 1, 1024)
    mean, scale = model.encoder(x)
    y = model.decoder(mean)

    print("scripting cached modules")
    n_cache = 0

    cached_modules = [
        CachedConv1d,
        CachedConvTranspose1d,
        Residual,
    ]

    model.discriminator = None

    for n, m in model.named_modules():
        if any(list(map(lambda c: isinstance(m, c), cached_modules))):
            if not m.cache.initialized:
                print(f"warmup failed for module {n}")
            m.script_cache()
            n_cache += 1

    print(f"{n_cache} cached modules found and scripted !")

    sr = model.sr

    if args.SR is not None:
        target_sr = int(args.SR)
    else:
        target_sr = sr

    print("build resampling model")
    resample = Resampling(target_sr, sr)
    x = torch.randn(1, 1, 2**14)
    resample.to_target_sampling_rate(resample.from_target_sampling_rate(x))
    resample.upsample.script_cache()
    resample.downsample.script_cache()

    print("script model")
    model = TraceModel(model, resample)
    model = torch.jit.script(model)

    print("save model")
    model.save("vae.ts")