import torch
import torch.nn as nn
from effortless_config import Config
from glob import glob
from os import path
from fd import parallel_model


class args(Config):
    RUN = None
    SR = None
    CACHED = False
    LATENT_SIZE = 8


args.parse_args()

parallel_model.use_buffer_conv(args.CACHED)

from fd.parallel_model.model import ParallelModel, Residual
from fd.parallel_model.buffer_conv import CachedConv1d, CachedConvTranspose1d, AlignBranches
from fd.parallel_model.resample import Resampling
from fd.parallel_model.pqmf import CachedPQMF


class TraceModel(nn.Module):
    def __init__(self, pretrained: ParallelModel, resample: Resampling):
        super().__init__()
        latent_size = pretrained.latent_size
        self.resample = resample

        self.pqmf = pretrained.pqmf
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder

        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)
        self.register_buffer("latent_size", torch.tensor(latent_size))
        self.register_buffer(
            "sampling_rate",
            torch.tensor(self.resample.taget_sr),
        )

        self.cropped_latent_size = args.LATENT_SIZE

        x = torch.zeros(1, 1, 2**14)
        z = self.encode(x)
        ratio = x.shape[-1] // z.shape[-1]

        self.register_buffer(
            "encode_params",
            torch.tensor([1, 1, self.cropped_latent_size, ratio]))

        self.register_buffer(
            "decode_params",
            torch.tensor([self.cropped_latent_size, ratio, 1, 1]))

        self.register_buffer("forward_params", torch.tensor([1, 1, 1, 1]))

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

        if self.pqmf is not None:
            x = self.pqmf(x)

        mean, scale = self.encoder(x)
        z = self.reparametrize(mean, scale)[0]
        z = z - self.latent_mean.unsqueeze(-1)
        z = nn.functional.conv1d(z, self.latent_pca.unsqueeze(-1))

        z = z[:, :self.cropped_latent_size]
        return z

    @torch.jit.export
    def decode(self, z):
        pad_size = self.latent_size.item() - self.cropped_latent_size
        z = torch.cat([z, torch.randn(z.shape[0], pad_size, z.shape[-1])], 1)

        z = nn.functional.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        z = z + self.latent_mean.unsqueeze(-1)
        x = self.decoder(z)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x)

        x = self.resample.to_target_sampling_rate(x)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


print("loading model from checkpoint")

if ".ckpt" in (RUN := str(args.RUN)):
    pass
elif "checkpoints" in RUN:
    RUN = path.join(RUN, "*.ckpt")
    RUN = glob(RUN)[-1]
else:
    RUN = path.join(RUN, "checkpoints", "*.ckpt")
    RUN = glob(RUN)[-1]

print(f"found {RUN}")
model = ParallelModel.load_from_checkpoint(RUN, strict=False).eval()

print("flattening weights")
for m in model.modules():
    if hasattr(m, "weight_g"):
        nn.utils.remove_weight_norm(m)

print("warmup forward pass")
x = torch.zeros(1, 1, 2**14)
if model.pqmf is not None:
    x = model.pqmf(x)
mean, scale = model.encoder(x)
y = model.decoder(mean)
if model.pqmf is not None:
    y = model.pqmf.inverse(y)

print("scripting cached modules")
n_cache = 0

cached_modules = [
    CachedConv1d,
    CachedConvTranspose1d,
    Residual,
    CachedPQMF,
    AlignBranches,
]

model.discriminator = None

for n, m in model.named_modules():
    if any(list(map(lambda c: isinstance(m, c),
                    cached_modules))) and args.CACHED:
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
x = torch.zeros(1, 1, 2**14)
resample.to_target_sampling_rate(resample.from_target_sampling_rate(x))

if not resample.identity and args.CACHED:
    resample.upsample.script_cache()
    resample.downsample.script_cache()

print("script model")
model = TraceModel(model, resample)
model = torch.jit.script(model)

print("save model")
model.save("vae.ts")