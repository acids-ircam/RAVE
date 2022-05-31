import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from .core import multiscale_stft, Loudness
from .core import get_beta_kl_cyclic_annealed
from .pqmf import CachedPQMF as PQMF
from sklearn.decomposition import PCA
from einops import rearrange

from .blocks import Generator, Encoder
from .discriminator import FullDiscriminator

import gin


@gin.configurable
class RAVE(pl.LightningModule):

    def __init__(self, latent_size, pqmf, sampling_rate, loudness, encoder,
                 decoder, discriminator, phase_1_duration):
        super().__init__()

        self.pqmf = pqmf()
        self.loudness = loudness()
        self.encoder = encoder()
        self.decoder = decoder()
        self.discriminator = discriminator()

        self.idx = 0

        self.register_buffer("latent_pca", torch.eye(latent_size))
        self.register_buffer("latent_mean", torch.zeros(latent_size))
        self.register_buffer("fidelity", torch.zeros(latent_size))

        self.latent_size = latent_size

        self.automatic_optimization = False

        self.warmup = phase_1_duration
        self.warmed_up = False
        self.sr = sampling_rate

        self.register_buffer("saved_step", torch.tensor(0))

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
        return abs(torch.log(x + 1) - torch.log(y + 1)).mean()

    def distance(self, x, y):
        scales = [2048, 1024, 512, 256, 128]
        x = multiscale_stft(x, scales, .75)
        y = multiscale_stft(y, scales, .75)

        lin = sum(list(map(self.lin_distance, x, y)))
        log = sum(list(map(self.log_distance, x, y)))

        return lin + log

    def reparametrize(self, z):
        if self.regularization == "kl":
            return self.reparametrize_kl(z)
        else:
            return self.reparametrize_vq(z)

    def adversarial_combine(self, score_real, score_fake, mode="hinge"):
        if mode == "hinge":
            loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
            loss_dis = loss_dis.mean()
            loss_gen = -score_fake.mean()
        elif mode == "square":
            loss_dis = (score_real - 1).pow(2) + score_fake.pow(2)
            loss_dis = loss_dis.mean()
            loss_gen = (score_fake - 1).pow(2).mean()
        elif mode == "nonsaturating":
            score_real = torch.clamp(torch.sigmoid(score_real), 1e-7, 1 - 1e-7)
            score_fake = torch.clamp(torch.sigmoid(score_fake), 1e-7, 1 - 1e-7)
            loss_dis = -(torch.log(score_real) +
                         torch.log(1 - score_fake)).mean()
            loss_gen = -torch.log(score_fake).mean()
        else:
            raise NotImplementedError
        return loss_dis, loss_gen

    def split_features(self, features):
        feature_true = []
        feature_fake = []
        for scale in features:
            true, fake = zip(*map(
                lambda x: torch.split(x, x.shape[0] // 2, 0),
                scale,
            ))
            feature_true.append(true)
            feature_fake.append(fake)
        return feature_true, feature_fake

    def training_step(self, batch, batch_idx):
        self.saved_step += 1
        self.warmed_up = True

        gen_opt, dis_opt = self.optimizers()
        x = batch.unsqueeze(1)

        if self.pqmf is not None:  # MULTIBAND DECOMPOSITION
            x = self.pqmf(x)

        if self.warmed_up:  # EVAL ENCODER
            self.encoder.eval()
            if self.rvq is not None:
                self.rvq.eval()

        # ENCODE INPUT
        z, reg = self.reparametrize(self.encoder(x))

        # if self.warmed_up:  # FREEZE ENCODER
        #     z = z.detach()
        #     reg = reg.detach()

        # DECODE LATENT
        y = self.decoder(z, add_noise=self.warmed_up)

        # DISTANCE BETWEEN INPUT AND OUTPUT
        distance = self.distance(x, y)

        if self.pqmf is not None:  # FULL BAND RECOMPOSITION
            x = self.pqmf.inverse(x)
            y = self.pqmf.inverse(y)
            distance = distance + self.distance(x, y)

        loud_x = self.loudness(x)
        loud_y = self.loudness(y)
        loud_dist = (loud_x - loud_y).pow(2).mean()
        distance = distance + loud_dist

        feature_matching_distance = 0.
        if self.warmed_up:  # DISCRIMINATION
            xy = torch.cat([x, y], 0)
            features = self.discriminator(xy)
            feature_true, feature_fake = self.split_features(features)

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
            step=self.global_step,
            cycle_size=5e4,
            warmup=self.warmup // 2,
            min_beta=self.min_kl,
            max_beta=self.max_kl,
        )
        loss_gen = distance + loss_adv + beta * reg
        if self.feature_match:
            loss_gen = loss_gen + feature_matching_distance

        # OPTIMIZATION
        if self.saved_step % 2 and self.warmed_up:
            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()
        else:
            gen_opt.zero_grad()
            loss_gen.backward()
            gen_opt.step()

        # LOGGING
        self.log("loss_dis", loss_dis)
        self.log("loss_gen", loss_gen)
        self.log("loud_dist", loud_dist)
        self.log("regularization", reg)
        self.log("pred_true", pred_true.mean())
        self.log("pred_fake", pred_fake.mean())
        self.log("distance", distance)
        self.log("beta", beta)
        self.log("feature_matching", feature_matching_distance)

    def encode(self, x):
        if self.pqmf is not None:
            x = self.pqmf(x)

        z, _ = self.reparametrize(self.encoder(x))
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

        z = self.encoder(x)
        if self.regularization == "kl":
            mean = torch.split(z, z.shape[1] // 2, 1)[0]
        else:
            mean = None

        z, _ = self.reparametrize(z)
        y = self.decoder(z, add_noise=self.warmed_up)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x)
            y = self.pqmf.inverse(y)

        distance = self.distance(x, y)

        if self.trainer is not None:
            self.log("validation", distance)
        return torch.cat([x, y], -1), mean

    def validation_epoch_end(self, out):
        if not len(out): return
        audio, z = list(zip(*out))
        if self.saved_step > self.warmup:
            self.warmed_up = True

        # LATENT SPACE ANALYSIS
        if not self.warmed_up and self.regularization == "kl":
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
                self.log(
                    f"fidelity_{p}",
                    np.argmax(var > p).astype(np.float32),
                )

        y = torch.cat(audio, 0)[:64].reshape(-1)
        self.logger.experiment.add_audio("audio_val", y,
                                         self.saved_step.item(), self.sr)
        self.idx += 1
