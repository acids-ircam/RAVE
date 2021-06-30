import torch
import torch.nn as nn
import pytorch_lightning as pl
from .core import multiscale_stft, get_padding
from sklearn.decomposition import PCA
from einops import rearrange

from . import USE_BUFFER_CONV
from .buffer_conv import CachedConv1d, CachedConvTranspose1d, Conv1d, CachedPadding1d

Conv1d = CachedConv1d if USE_BUFFER_CONV else Conv1d
ConvTranspose1d = CachedConvTranspose1d if USE_BUFFER_CONV else nn.ConvTranspose1d


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        if USE_BUFFER_CONV:
            pad = 0
            for m in self.module.modules():
                if isinstance(m, CachedConv1d):
                    pad = pad + m.future_compensation
            self.cache = CachedPadding1d(pad)
            self.pad = pad
        else:
            self.cache = None

    def forward(self, x):
        x_net = self.module(x)
        if self.cache is not None:
            x = self.cache(x)[..., :-self.pad]
        return x_net + x


class ResidualStack(nn.Module):
    def __init__(self, dim, kernel_size, bias=False):
        super().__init__()
        net = []
        for i in range(3):
            net.append(
                Residual(
                    nn.Sequential(
                        nn.LeakyReLU(.2),
                        Conv1d(
                            dim,
                            dim,
                            kernel_size,
                            padding=get_padding(
                                kernel_size,
                                dilation=3**i,
                            ),
                            dilation=3**i,
                            bias=bias,
                        ),
                        nn.LeakyReLU(.2),
                        Conv1d(
                            dim,
                            dim,
                            kernel_size,
                            padding=get_padding(kernel_size),
                            bias=bias,
                        ),
                    )))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class UpsampleLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ratio, bias=False):
        super().__init__()
        net = [nn.LeakyReLU(.2)]
        if ratio > 1:
            net.append(
                ConvTranspose1d(
                    in_dim,
                    out_dim,
                    2 * ratio,
                    stride=ratio,
                    padding=ratio // 2,
                    bias=bias,
                ))
        else:
            net.append(
                Conv1d(
                    in_dim,
                    out_dim,
                    3,
                    padding=get_padding(3),
                    bias=bias,
                ))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, latent_size, capacity, data_size, ratios, bias=False):
        super().__init__()
        self.pre_net = Conv1d(
            latent_size,
            2**len(ratios) * capacity,
            7,
            padding=get_padding(7),
            bias=bias,
        )

        net = []
        for i, r in enumerate(ratios):
            in_dim = 2**(len(ratios) - i) * capacity
            out_dim = 2**(len(ratios) - i - 1) * capacity

            net.append(UpsampleLayer(in_dim, out_dim, r))
            net.append(ResidualStack(out_dim, 3))

        self.net = nn.Sequential(*net)

        self.post_net = nn.Sequential(
            Conv1d(out_dim, data_size, 7, padding=get_padding(7), bias=bias),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.pre_net(x)
        x = self.net(x)
        x = self.post_net(x)
        return x


class Encoder(nn.Module):
    def __init__(self, data_size, capacity, latent_size, ratios, bias=False):
        super().__init__()
        net = [
            Conv1d(data_size, capacity, 7, padding=get_padding(7), bias=bias)
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
                    padding=get_padding(2 * r + 1, r),
                    stride=r,
                    bias=bias,
                ))

        net.append(nn.LeakyReLU(.2))
        net.append(
            Conv1d(
                out_dim,
                2 * latent_size,
                5,
                padding=get_padding(5),
                groups=2,
                bias=bias,
            ))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        z = self.net(x)
        return torch.split(z, z.shape[1] // 2, 1)


class Discriminator(nn.Module):
    def __init__(self, capacity, multiplier, n_layers):
        super().__init__()

        net = [nn.Conv1d(1, capacity, 15, padding=7)]
        net.append(nn.LeakyReLU(.2))

        for i in range(n_layers):
            net.append(
                nn.Conv1d(
                    capacity * multiplier**i,
                    min(1024, capacity * multiplier**(i + 1)),
                    41,
                    stride=multiplier,
                    padding=get_padding(41, multiplier),
                    groups=multiplier**(i + 1),
                ))
            net.append(nn.LeakyReLU(.2))

        net.append(
            nn.Conv1d(
                min(1024, capacity * multiplier**(i + 1)),
                min(1024, capacity * multiplier**(i + 1)),
                5,
                padding=get_padding(5),
            ))
        net.append(nn.LeakyReLU(.2))
        net.append(nn.Conv1d(min(1024, capacity * multiplier**(i + 1)), 1, 1))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ParallelModel(pl.LightningModule):
    def __init__(self,
                 data_size,
                 capacity,
                 latent_size,
                 ratios,
                 bias,
                 d_capacity,
                 d_multiplier,
                 d_n_layers,
                 warmup,
                 sr=24000):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(data_size, capacity, latent_size, ratios, bias)
        self.decoder = Generator(latent_size, capacity, data_size, ratios,
                                 bias)

        self.discriminator = Discriminator(d_capacity, d_multiplier,
                                           d_n_layers)

        self.idx = 0

        self.register_buffer("latent_pca", torch.eye(latent_size))
        self.register_buffer("latent_mean", torch.zeros(latent_size))

        self.latent_size = latent_size

        self.automatic_optimization = False

        self.warmup = warmup
        self.sr = sr

    def configure_optimizers(self):
        gen_p = list(self.encoder.parameters())
        gen_p += list(self.decoder.parameters())
        dis_p = self.discriminator.parameters()

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

    def training_step(self, batch, batch_idx):
        step = len(self.train_dataloader()) * self.current_epoch + batch_idx

        gen_opt, dis_opt = self.optimizers()

        x = batch.unsqueeze(1)
        z, kl = self.reparametrize(*self.encoder(x))

        y = self.decoder(z)

        distance = self.distance(x, y)

        if step > self.warmup:
            pred_true = self.discriminator(x).mean()
            pred_fake = self.discriminator(y).mean()
        else:
            pred_true = torch.tensor(0.).to(x)
            pred_fake = torch.tensor(0.).to(x)

        if step % 2 and step > self.warmup:
            loss_dis = torch.relu(1 - pred_true) + torch.relu(1 + pred_fake)

            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()
            self.log("loss_dis", loss_dis)
        else:
            loss_gen = distance - pred_fake + 1e-2 * kl

            gen_opt.zero_grad()
            loss_gen.backward()
            gen_opt.step()
            self.log("loss_gen", loss_gen)

        self.log("distance", distance)
        self.log("regularization", kl)
        self.log("pred_true", pred_true)
        self.log("pred_fake", pred_fake)

    def validation_step(self, batch, batch_idx):
        x = batch.unsqueeze(1)
        mean, scale = self.encoder(x)
        z, _ = self.reparametrize(mean, scale)
        y = self.decoder(z)

        distance = self.distance(x, y)

        self.log("validation", distance)

        return torch.cat([x, y], -1), mean

    def validation_epoch_end(self, out):
        audio, z = list(zip(*out))

        z = torch.cat(z, 0)
        z = rearrange(z, "b c t -> (b t) c")

        self.latent_mean.copy_(z.mean(0))
        z = z - self.latent_mean

        pca = PCA(self.latent_size).fit(z.cpu().numpy()).components_
        pca = torch.from_numpy(pca).to(z)

        self.latent_pca.copy_(pca)

        y = torch.cat(audio, 0)[:64].reshape(-1)
        self.logger.experiment.add_audio("audio_val", y, self.idx, 24000)
        self.idx += 1
