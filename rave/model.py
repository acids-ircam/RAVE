import torch
import numpy as np
import pytorch_lightning as pl
from .core import multiscale_stft
from sklearn.decomposition import PCA
from einops import rearrange

from .blocks import VariationalEncoder

import gin
from time import time


class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        total = self.ticks[-1][0] - self.ticks[0][0]
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += f"TOTAL: {total*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep


@gin.configurable
class RAVE(pl.LightningModule):

    def __init__(self, latent_size, pqmf, sampling_rate, loudness, encoder,
                 decoder, discriminator, phase_1_duration, gan_loss,
                 feature_match):
        super().__init__()

        self.pqmf = pqmf()
        self.loudness = loudness()
        self.encoder = encoder()
        self.decoder = decoder()
        self.discriminator = discriminator()

        self.gan_loss = gan_loss

        self.idx = 0

        self.register_buffer("latent_pca", torch.eye(latent_size))
        self.register_buffer("latent_mean", torch.zeros(latent_size))
        self.register_buffer("fidelity", torch.zeros(latent_size))

        self.latent_size = latent_size

        self.automatic_optimization = False

        self.warmup = phase_1_duration
        self.warmed_up = False
        self.sr = sampling_rate
        self.feature_match = feature_match

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
        x = multiscale_stft(x)
        y = multiscale_stft(y)

        lin = sum(list(map(self.lin_distance, x, y)))
        log = sum(list(map(self.log_distance, x, y)))

        return lin + log

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
        p = Profiler()

        self.saved_step += 1

        gen_opt, dis_opt = self.optimizers()
        p.tick("get optimizers")
        x = batch.unsqueeze(1)

        x = self.pqmf(x)
        p.tick("pqmf")

        self.encoder.set_warmed_up(self.warmed_up)

        # ENCODE INPUT
        z, reg = self.encoder.reparametrize(self.encoder(x))[:2]
        p.tick("encode")

        # DECODE LATENT
        y = self.decoder(z, add_noise=self.warmed_up)
        p.tick("decode")

        # DISTANCE BETWEEN INPUT AND OUTPUT
        distance = self.distance(x, y)
        p.tick("mb distance")

        x = self.pqmf.inverse(x)
        y = self.pqmf.inverse(y)
        p.tick("recomposition")

        distance = distance + self.distance(x, y)
        p.tick("fb distance")

        loud_x = self.loudness(x)
        loud_y = self.loudness(y)
        loud_dist = (loud_x - loud_y).pow(2).mean()
        distance = distance + loud_dist
        p.tick("loudness")

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

                _dis, _adv = self.gan_loss(scale_true[-1], scale_fake[-1])

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
        loss_gen = distance + loss_adv + reg
        p.tick("loss")

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
        p.tick("optim")

        # LOGGING
        self.log("loss_dis", loss_dis)
        self.log("loss_gen", loss_gen)
        self.log("loud_dist", loud_dist)
        self.log("regularization", reg)
        self.log("pred_true", pred_true.mean())
        self.log("pred_fake", pred_fake.mean())
        self.log("distance", distance)
        self.log("feature_matching", feature_matching_distance)
        p.tick("logging")

        # print(p)

    def encode(self, x):
        x = self.pqmf(x)
        z, _ = self.encoder.reparametrize(self.encoder(x))[:2]
        return z

    def decode(self, z):
        y = self.decoder(z, add_noise=True)
        y = self.pqmf.inverse(y)
        return y

    def validation_step(self, batch, batch_idx):
        x = batch.unsqueeze(1)
        x = self.pqmf(x)
        z = self.encoder(x)

        if isinstance(self.encoder, VariationalEncoder):
            mean = torch.split(z, z.shape[1] // 2, 1)[0]
        else:
            mean = None

        z = self.encoder.reparametrize(z)[0]
        y = self.decoder(z, add_noise=self.warmed_up)

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
        if not self.warmed_up and isinstance(self.encoder, VariationalEncoder):
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
