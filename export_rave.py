import torch
import torch.nn as nn
from effortless_config import Config
from glob import glob
from os import path
import logging
from termcolor import colored
import cached_conv

logging.basicConfig(level=logging.INFO,
                    format=colored("[%(relativeCreated).2f] ", "green") +
                    "%(message)s")

logging.info("exporting model")


class args(Config):
    RUN = None
    SR = None
    CACHED = False
    FIDELITY = .95
    NAME = "vae"
    STEREO = False


args.parse_args()
cached_conv.use_buffer_conv(args.CACHED)

from rave.model import RAVE
from cached_conv import CachedConv1d, CachedConvTranspose1d, AlignBranches
from rave.resample import Resampling
from rave.pqmf import CachedPQMF

from rave.core import search_for_run

import numpy as np
import math


class TraceModel(nn.Module):
    def __init__(self, pretrained: RAVE, resample: Resampling,
                 fidelity: float):
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

        latent_size = np.argmax(pretrained.fidelity.numpy() > fidelity)
        latent_size = 2**math.ceil(math.log2(latent_size))
        self.cropped_latent_size = latent_size

        x = torch.zeros(1, 1, 2**14)
        z = self.encode(x)
        ratio = x.shape[-1] // z.shape[-1]

        self.register_buffer(
            "encode_params",
            torch.tensor([
                1,
                1,
                self.cropped_latent_size,
                ratio,
            ]))

        self.register_buffer(
            "decode_params",
            torch.tensor([
                self.cropped_latent_size,
                ratio,
                2 if args.STEREO else 1,
                1,
            ]))

        self.register_buffer("forward_params", torch.tensor([1, 1, 1, 1]))

        self.stereo = args.STEREO

    def post_process_distribution(self, mean, scale):
        std = nn.functional.softplus(scale) + 1e-4
        return mean, std

    def reparametrize(self, mean, std):
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
        mean, std = self.post_process_distribution(mean, scale)

        z = self.reparametrize(mean, std)[0]

        z = z - self.latent_mean.unsqueeze(-1)
        z = nn.functional.conv1d(z, self.latent_pca.unsqueeze(-1))

        z = z[:, :self.cropped_latent_size]
        return z

    @torch.jit.export
    def encode_amortized(self, x):
        x = self.resample.from_target_sampling_rate(x)

        if self.pqmf is not None:
            x = self.pqmf(x)

        mean, scale = self.encoder(x)
        mean, std = self.post_process_distribution(mean, scale)
        var = std * std

        mean = mean - self.latent_mean.unsqueeze(-1)

        mean = nn.functional.conv1d(mean, self.latent_pca.unsqueeze(-1))
        var = nn.functional.conv1d(var, self.latent_pca.unsqueeze(-1).pow(2))

        mean = mean[:, :self.cropped_latent_size]
        var = var[:, :self.cropped_latent_size]
        std = var.sqrt()

        return mean, std

    @torch.jit.export
    def decode(self, z: torch.Tensor):
        if self.stereo and z.shape[0] == 1:
            z = z.expand(2, z.shape[1], z.shape[2])

        pad_size = self.latent_size.item() - z.shape[1]
        z = torch.cat([
            z,
            torch.randn(
                z.shape[0],
                pad_size,
                z.shape[-1],
                device=z.device,
            )
        ], 1)

        z = nn.functional.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        z = z + self.latent_mean.unsqueeze(-1)
        x = self.decoder(z)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x)

        x = self.resample.to_target_sampling_rate(x)

        if self.stereo:
            x = x.permute(1, 0, 2)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


logging.info("loading model from checkpoint")

RUN = search_for_run(args.RUN)
logging.info(f"using {RUN}")
model = RAVE.load_from_checkpoint(RUN, strict=False).eval()

logging.info("flattening weights")
for m in model.modules():
    if hasattr(m, "weight_g"):
        nn.utils.remove_weight_norm(m)

logging.info("warmup forward pass")
x = torch.zeros(1, 1, 2**14)
if model.pqmf is not None:
    x = model.pqmf(x)
mean, scale = model.encoder(x)

if args.STEREO:
    mean = mean.expand(2, *mean.shape[1:])

y = model.decoder(mean)
if model.pqmf is not None:
    y = model.pqmf.inverse(y)

logging.info("scripting cached modules")
n_cache = 0

cached_modules = [
    CachedConv1d,
    CachedConvTranspose1d,
    CachedPQMF,
    AlignBranches,
]

model.discriminator = None

for n, m in model.named_modules():
    if any(list(map(lambda c: isinstance(m, c),
                    cached_modules))) and args.CACHED:
        m.script_cache()
        n_cache += 1

logging.info(f"{n_cache} cached modules found and scripted")

sr = model.sr

if args.SR is not None:
    target_sr = int(args.SR)
else:
    target_sr = sr

logging.info("build resampling model")
resample = Resampling(target_sr, sr)
x = torch.zeros(1, 1, 2**14)
resample.to_target_sampling_rate(resample.from_target_sampling_rate(x))

if not resample.identity and args.CACHED:
    resample.upsample.script_cache()
    resample.downsample.script_cache()

logging.info("script model")
model = TraceModel(model, resample, args.FIDELITY)
model(x)

model = torch.jit.script(model)
logging.info(f"save rave_{args.NAME}.ts")
model.save(f"rave_{args.NAME}.ts")