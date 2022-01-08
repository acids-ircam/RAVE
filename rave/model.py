import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as wn
import numpy as np
import pytorch_lightning as pl
from .core import multiscale_stft, Loudness, mod_sigmoid
from .core import amp_to_impulse_response, fft_convolve
from .pqmf import CachedPQMF as PQMF
from sklearn.decomposition import PCA
from einops import rearrange

import math

from time import time

from cached_conv import USE_BUFFER_CONV, get_padding
from cached_conv import CachedConv1d, CachedConvTranspose1d, Conv1d, CachedPadding1d, AlignBranches, CachedSequential

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


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        future = module.future_compensation
        self.aligned = AlignBranches(
            module,
            nn.Identity(),
            futures=[future, 0],
        )
        self.future_compensation = future

    def forward(self, x):
        x_net, x_res = self.aligned(x)
        return x_net + x_res


class ResidualStack(nn.Module):
    def __init__(self, dim, kernel_size, padding_mode, bias=False):
        super().__init__()
        net = []
        for i in range(3):
            net.append(
                Residual(
                    CachedSequential(
                        nn.LeakyReLU(.2),
                        wn(
                            Conv1d(
                                dim,
                                dim,
                                kernel_size,
                                padding=get_padding(
                                    kernel_size,
                                    dilation=3**i,
                                    mode=padding_mode,
                                ),
                                dilation=3**i,
                                bias=bias,
                            )),
                        nn.LeakyReLU(.2),
                        wn(
                            Conv1d(
                                dim,
                                dim,
                                kernel_size,
                                padding=get_padding(kernel_size,
                                                    mode=padding_mode),
                                bias=bias,
                            )),
                    )))

        self.net = CachedSequential(*net)
        self.future_compensation = self.net.future_compensation

    def forward(self, x):
        return self.net(x)


class UpsampleLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ratio, padding_mode, bias=False):
        super().__init__()
        net = [nn.LeakyReLU(.2)]
        if ratio > 1:
            net.append(
                wn(
                    ConvTranspose1d(
                        in_dim,
                        out_dim,
                        2 * ratio,
                        stride=ratio,
                        padding=ratio // 2,
                        bias=bias,
                    )))
        else:
            net.append(
                wn(
                    Conv1d(
                        in_dim,
                        out_dim,
                        3,
                        padding=get_padding(3, mode=padding_mode),
                        bias=bias,
                    )))

        self.net = CachedSequential(*net)
        self.future_compensation = self.net.future_compensation

    def forward(self, x):
        return self.net(x)


class NoiseGenerator(nn.Module):
    def __init__(self, in_size, data_size, ratios, noise_bands, padding_mode):
        super().__init__()
        net = []
        channels = [in_size] * len(ratios) + [data_size * noise_bands]
        for i, r in enumerate(ratios):
            net.append(
                Conv1d(
                    channels[i],
                    channels[i + 1],
                    3,
                    padding=get_padding(3, r, mode=padding_mode),
                    stride=r,
                ))
            if i != len(ratios) - 1:
                net.append(nn.LeakyReLU(.2))

        self.net = CachedSequential(*net)
        self.data_size = data_size
        self.future_compensation = self.net.future_compensation

        self.register_buffer(
            "target_size",
            torch.tensor(np.prod(ratios)).long(),
        )

    def forward(self, x):
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(amp.shape[0], amp.shape[1], self.data_size, -1)

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        return noise


class Generator(nn.Module):
    def __init__(self,
                 latent_size,
                 capacity,
                 data_size,
                 ratios,
                 loud_stride,
                 use_noise,
                 noise_ratios,
                 noise_bands,
                 padding_mode,
                 bias=False):
        super().__init__()
        net = [
            wn(
                Conv1d(
                    latent_size,
                    2**len(ratios) * capacity,
                    7,
                    padding=get_padding(7, mode=padding_mode),
                    bias=bias,
                ))
        ]
        for i, r in enumerate(ratios):
            in_dim = 2**(len(ratios) - i) * capacity
            out_dim = 2**(len(ratios) - i - 1) * capacity

            net.append(UpsampleLayer(in_dim, out_dim, r, padding_mode))
            net.append(ResidualStack(out_dim, 3, padding_mode))

        self.net = CachedSequential(*net)

        wave_gen = wn(
            Conv1d(out_dim,
                   data_size,
                   7,
                   padding=get_padding(7, mode=padding_mode),
                   bias=bias))

        loud_gen = wn(
            Conv1d(out_dim,
                   1,
                   2 * loud_stride + 1,
                   stride=loud_stride,
                   padding=get_padding(2 * loud_stride + 1,
                                       loud_stride,
                                       mode=padding_mode),
                   bias=bias))

        branches = [wave_gen, loud_gen]

        if use_noise:
            noise_gen = NoiseGenerator(
                out_dim,
                data_size,
                noise_ratios,
                noise_bands,
                padding_mode=padding_mode,
            )
            branches.append(noise_gen)

        self.synth = AlignBranches(*branches)
        self.use_noise = use_noise
        self.loud_stride = loud_stride

    def forward(self, x, add_noise: bool = True):
        x = self.net(x)

        if self.use_noise:
            waveform, loudness, noise = self.synth(x)
        else:
            waveform, loudness = self.synth(x)
            noise = torch.zeros_like(waveform)

        loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform) * mod_sigmoid(loudness)

        if add_noise:
            waveform = waveform + noise

        return waveform


class Encoder(nn.Module):
    def __init__(self,
                 data_size,
                 capacity,
                 latent_size,
                 ratios,
                 padding_mode,
                 bias=False):
        super().__init__()
        net = [
            Conv1d(data_size,
                   capacity,
                   7,
                   padding=get_padding(7, mode=padding_mode),
                   bias=bias)
        ]

        for i, r in enumerate(ratios):
            in_dim = 2**i * capacity
            out_dim = 2**(i + 1) * capacity

            net.append(nn.BatchNorm1d(in_dim))
            net.append(nn.LeakyReLU(.2))
            net.append(
                Conv1d(
                    in_dim,
                    out_dim,
                    2 * r + 1,
                    padding=get_padding(2 * r + 1, r, mode=padding_mode),
                    stride=r,
                    bias=bias,
                ))

        net.append(nn.LeakyReLU(.2))
        net.append(
            Conv1d(
                out_dim,
                2 * latent_size,
                5,
                padding=get_padding(5, mode=padding_mode),
                groups=2,
                bias=bias,
            ))

        self.net = CachedSequential(*net)
        self.future_compensation = self.net.future_compensation

    def forward(self, x):
        z = self.net(x)
        return torch.split(z, z.shape[1] // 2, 1)


class Discriminator(nn.Module):
    def __init__(self, in_size, capacity, multiplier, n_layers):
        super().__init__()

        net = [nn.Conv1d(in_size, capacity, 15, padding=7)]
        net.append(nn.LeakyReLU(.2))

        for i in range(n_layers):
            net.append(
                Conv1d(
                    capacity * multiplier**i,
                    min(1024, capacity * multiplier**(i + 1)),
                    41,
                    stride=multiplier,
                    padding=get_padding(41, multiplier),
                    groups=multiplier**(i + 1),
                ))
            net.append(nn.LeakyReLU(.2))

        net.append(
            Conv1d(
                min(1024, capacity * multiplier**(i + 1)),
                min(1024, capacity * multiplier**(i + 1)),
                5,
                padding=get_padding(5),
            ))
        net.append(nn.LeakyReLU(.2))
        net.append(Conv1d(min(1024, capacity * multiplier**(i + 1)), 1, 1))
        self.net = nn.ModuleList(net)

    def forward(self, x):
        feature = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, Conv1d):
                feature.append(x)
        return feature


class StackDiscriminators(nn.Module):
    def __init__(self, n_dis, *args, **kwargs):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [Discriminator(*args, **kwargs) for i in range(n_dis)], )

    def forward(self, x):
        features = []
        for layer in self.discriminators:
            features.append(layer(x))
            x = nn.functional.avg_pool1d(x, 2)
        return features


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
                 sr=24000):
        super().__init__()
        self.save_hyperparameters()

        if data_size == 1:
            self.pqmf = None
        else:
            self.pqmf = PQMF(40 if no_latency else 100, data_size)

        self.loudness = Loudness(sr, 512)

        self.encoder = Encoder(data_size, capacity, latent_size, ratios,
                               "causal" if no_latency else "centered", bias)
        self.decoder = Generator(latent_size, capacity, data_size, ratios,
                                 loud_stride, use_noise, noise_ratios,
                                 noise_bands,
                                 "causal" if no_latency else "centered", bias)

        self.discriminator = StackDiscriminators(
            3,
            in_size=1,
            capacity=d_capacity,
            multiplier=d_multiplier,
            n_layers=d_n_layers,
        )

        self.idx = 0

        self.register_buffer("latent_pca", torch.eye(latent_size))
        self.register_buffer("latent_mean", torch.zeros(latent_size))
        self.register_buffer("fidelity", torch.zeros(latent_size))

        self.latent_size = latent_size

        self.automatic_optimization = False

        self.warmup = warmup
        self.warmed_up = False
        self.sr = sr
        self.mode = mode
        self.step = 0

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
        loss_gen = distance + feature_matching_distance + loss_adv + 1e-1 * kl
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
        audio, z = list(zip(*out))

        # LATENT SPACE ANALYSIS
        if not self.warmed_up:
            z = torch.cat(z, 0)
            z = rearrange(z, "b c t -> (b t) c")

            self.latent_mean.copy_(z.mean(0))
            z = z - self.latent_mean

            pca = PCA(self.latent_size).fit(z.cpu().numpy())

            components = pca.components_
            components = torch.from_numpy(components).to(z)
            self.latent_pca.copy_(components)

            var = pca.explained_variance_ / np.sum(pca.explained_variance_)
            var = np.cumsum(var)

            self.fidelity.copy_(torch.from_numpy(var).to(self.fidelity))

            var_percent = [.8, .9, .95, .99]
            for p in var_percent:
                self.log(f"{p}%_manifold", np.argmax(var > p))

        if self.step > self.warmup:
            self.warmed_up = True

        y = torch.cat(audio, 0)[:64].reshape(-1)
        self.logger.experiment.add_audio("audio_val", y, self.idx, self.sr)
        self.idx += 1
