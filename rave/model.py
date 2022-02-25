import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from .core import multiscale_stft, Loudness
from .core import get_beta_kl_cyclic_annealed
from .pqmf import CachedPQMF as PQMF
from sklearn.decomposition import PCA
from einops import rearrange

from time import time

from cached_conv import USE_BUFFER_CONV
from cached_conv import CachedConv1d, CachedConvTranspose1d, Conv1d

from .blocks import Encoder, Generator, StackDiscriminators

Conv1d = CachedConv1d if USE_BUFFER_CONV else Conv1d
ConvTranspose1d = CachedConvTranspose1d if USE_BUFFER_CONV else nn.ConvTranspose1d


class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep


class RAVE(pl.LightningModule):

    def __init__(self,
                 data_size,
                 capacity,
                 latent_size,
                 ratios,
                 bias,
                 loud_stride,
                 use_noise,
                 noise_ratios,
                 noise_bands,
                 d_capacity,
                 d_multiplier,
                 d_n_layers,
                 warmup,
                 mode,
                 no_latency=False,
                 min_kl=1e-4,
                 max_kl=5e-1,
                 cropped_latent_size=0,
                 sr=24000):
        super().__init__()
        self.save_hyperparameters()

        if data_size == 1:
            self.pqmf = None
        else:
            self.pqmf = PQMF(70 if no_latency else 100, data_size)

        self.loudness = Loudness(sr, 512)

        encoder_out_size = cropped_latent_size if cropped_latent_size else latent_size

        self.encoder = Encoder(
            data_size,
            capacity,
            encoder_out_size,
            ratios,
            "causal" if no_latency else "centered",
            bias,
        )
        self.decoder = Generator(
            latent_size,
            capacity,
            data_size,
            ratios,
            loud_stride,
            use_noise,
            noise_ratios,
            noise_bands,
            "causal" if no_latency else "centered",
            bias,
        )

        self.discriminator = StackDiscriminators(
            3,
            in_size=1,
            capacity=d_capacity,
            multiplier=d_multiplier,
            n_layers=d_n_layers,
        )

        self.idx = 0

        self.register_buffer("latent_pca", torch.eye(encoder_out_size))
        self.register_buffer("latent_mean", torch.zeros(encoder_out_size))
        self.register_buffer("fidelity", torch.zeros(encoder_out_size))

        self.latent_size = latent_size

        self.automatic_optimization = False

        self.warmup = warmup
        self.warmed_up = False
        self.sr = sr
        self.mode = mode
        self.step = 0

        self.min_kl = min_kl
        self.max_kl = max_kl
        self.cropped_latent_size = cropped_latent_size

    def configure_optimizers(self):
        gen_p = list(self.encoder.parameters())
        gen_p += list(self.decoder.parameters())
        dis_p = list(self.discriminator.parameters())

        gen_opt = torch.optim.Adam(gen_p, 1e-4, (.5, .9))
        dis_opt = torch.optim.Adam(dis_p, 1e-4, (.5, .9))

        return gen_opt, dis_opt

    def lin_distance(self, x, y):
        return torch.norm(x - y) / torch.norm(x)

    def log_distance(self, x, y):
        return abs(torch.log(x + 1e-7) - torch.log(y + 1e-7)).mean()

    def distance(self, x, y):
        scales = [2048, 1024, 512, 256, 128]
        x = multiscale_stft(x, scales, .75)
        y = multiscale_stft(y, scales, .75)

        lin = sum(list(map(self.lin_distance, x, y)))
        log = sum(list(map(self.log_distance, x, y)))

        return lin + log

    def reparametrize(self, mean, scale):
        std = nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        if self.cropped_latent_size:
            noise = torch.randn(
                z.shape[0],
                self.latent_size - self.cropped_latent_size,
                z.shape[-1],
            ).to(z.device)
            z = torch.cat([z, noise], 1)
        return z, kl

    def adversarial_combine(self, score_real, score_fake, mode="hinge"):
        if mode == "hinge":
            loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
            loss_dis = loss_dis.mean()
            loss_gen = -score_fake.mean()
        elif mode == "square":
            loss_dis = (score_real - 1).pow(2) + score_fake.pow(2)
            loss_dis = loss_dis.mean()
            loss_gen = (score_fake - 1).pow(2).mean()
        else:
            raise NotImplementedError
        return loss_dis, loss_gen

    def training_step(self, batch, batch_idx):
        p = Profiler()
        step = len(self.train_dataloader()) * self.current_epoch + batch_idx
        self.step = step

        gen_opt, dis_opt = self.optimizers()
        x = batch.unsqueeze(1)

        if self.pqmf is not None:  # MULTIBAND DECOMPOSITION
            x = self.pqmf(x)
            p.tick("pqmf")

        if self.warmed_up:  # EVAL ENCODER
            self.encoder.eval()

        # ENCODE INPUT
        z, kl = self.reparametrize(*self.encoder(x))
        p.tick("encode")

        if self.warmed_up:  # FREEZE ENCODER
            z = z.detach()
            kl = kl.detach()

        # DECODE LATENT
        y = self.decoder(z, add_noise=self.warmed_up)
        p.tick("decode")

        # DISTANCE BETWEEN INPUT AND OUTPUT
        distance = self.distance(x, y)
        p.tick("mb distance")

        if self.pqmf is not None:  # FULL BAND RECOMPOSITION
            x = self.pqmf.inverse(x)
            y = self.pqmf.inverse(y)
            distance = distance + self.distance(x, y)
            p.tick("fb distance")

        loud_x = self.loudness(x)
        loud_y = self.loudness(y)
        loud_dist = (loud_x - loud_y).pow(2).mean()
        distance = distance + loud_dist
        p.tick("loudness distance")

        feature_matching_distance = 0
        if self.warmed_up:  # DISCRIMINATION
            feature_true = self.discriminator(x)
            feature_fake = self.discriminator(y)

            loss_dis = 0
            loss_adv = 0

            pred_true = 0
            pred_fake = 0

            for scale_true, scale_fake in zip(feature_true, feature_fake):
                feature_matching_distance = feature_matching_distance + 10 * sum(
                    map(
                        lambda x, y: abs(x - y).mean(),
                        scale_true,
                        scale_fake,
                    )) / len(scale_true)

                _dis, _adv = self.adversarial_combine(
                    scale_true[-1],
                    scale_fake[-1],
                    mode=self.mode,
                )

                pred_true = pred_true + scale_true[-1].mean()
                pred_fake = pred_fake + scale_fake[-1].mean()

                loss_dis = loss_dis + _dis
                loss_adv = loss_adv + _adv

        else:
            pred_true = torch.tensor(0.).to(x)
            pred_fake = torch.tensor(0.).to(x)
            loss_dis = torch.tensor(0.).to(x)
            loss_adv = torch.tensor(0.).to(x)

        # COMPOSE GEN LOSS
        beta = get_beta_kl_cyclic_annealed(
            step=step,
            cycle_size=5e4,
            warmup=self.warmup // 2,
            min_beta=self.min_kl,
            max_beta=self.max_kl,
        )
        loss_gen = distance + feature_matching_distance + loss_adv + beta * kl
        p.tick("gen loss compose")

        # OPTIMIZATION
        if step % 2 and self.warmed_up:
            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()
        else:
            gen_opt.zero_grad()
            loss_gen.backward()
            gen_opt.step()
        p.tick("optimization")

        # LOGGING
        self.log("loss_dis", loss_dis)
        self.log("loss_gen", loss_gen)
        self.log("loud_dist", loud_dist)
        self.log("regularization", kl)
        self.log("pred_true", pred_true.mean())
        self.log("pred_fake", pred_fake.mean())
        self.log("distance", distance)
        self.log("beta", beta)
        self.log("feature_matching", feature_matching_distance)
        p.tick("log")

        # print(p)

    def encode(self, x):
        if self.pqmf is not None:
            x = self.pqmf(x)

        mean, scale = self.encoder(x)
        z, _ = self.reparametrize(mean, scale)
        return z

    def decode(self, z):
        y = self.decoder(z, add_noise=True)
        if self.pqmf is not None:
            y = self.pqmf.inverse(y)
        return y

    def validation_step(self, batch, batch_idx):
        x = batch.unsqueeze(1)

        if self.pqmf is not None:
            x = self.pqmf(x)

        mean, scale = self.encoder(x)
        z, _ = self.reparametrize(mean, scale)
        y = self.decoder(z, add_noise=self.warmed_up)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x)
            y = self.pqmf.inverse(y)

        distance = self.distance(x, y)

        if self.trainer is not None:
            self.log("validation", distance)

        return torch.cat([x, y], -1), mean

    def validation_epoch_end(self, out):
        step = len(self.train_dataloader()) * self.current_epoch
        audio, z = list(zip(*out))

        # LATENT SPACE ANALYSIS
        if not self.warmed_up:
            z = torch.cat(z, 0)
            z = rearrange(z, "b c t -> (b t) c")

            self.latent_mean.copy_(z.mean(0))
            z = z - self.latent_mean

            pca = PCA(z.shape[-1]).fit(z.cpu().numpy())

            components = pca.components_
            components = torch.from_numpy(components).to(z)
            self.latent_pca.copy_(components)

            var = pca.explained_variance_ / np.sum(pca.explained_variance_)
            var = np.cumsum(var)

            self.fidelity.copy_(torch.from_numpy(var).to(self.fidelity))

            var_percent = [.8, .9, .95, .99]
            for p in var_percent:
                self.log(f"{p}%_manifold", np.argmax(var > p))

        if step > self.warmup:
            self.warmed_up = True

        y = torch.cat(audio, 0)[:64].reshape(-1)
        self.logger.experiment.add_audio("audio_val", y, self.idx, self.sr)

        for n, p in self.named_parameters():
            if "scale" in n or "constant" in n:
                self.logger.experiment.add_histogram(
                    n,
                    p.reshape(-1).data.detach(),
                    self.idx,
                )

        self.idx += 1
