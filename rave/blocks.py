import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as wn
import numpy as np
from .core import amp_to_impulse_response, fft_convolve, mod_sigmoid

import math

from cached_conv import USE_BUFFER_CONV, get_padding
from cached_conv import CachedConv1d, CachedConvTranspose1d, Conv1d, AlignBranches, CachedSequential

Conv1d = CachedConv1d if USE_BUFFER_CONV else Conv1d
ConvTranspose1d = CachedConvTranspose1d if USE_BUFFER_CONV else nn.ConvTranspose1d


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


class AlignModulation(AlignBranches):

    def __init__(self, *branches, futures=None):
        super().__init__(*branches, futures=futures)

    def forward(self, x, z):
        x = self.branches[0](x)
        x = self.paddings[0](x)

        z = self.branches[1](z)
        z = self.paddings[1](z)

        z, mean, scale = torch.split(z, z.shape[1] // 3, 1)

        return x, z, mean, scale


class ModulationLayer(nn.Module):

    def __init__(self, in_size, out_size, stride, padding_mode) -> None:
        super().__init__()
        self.net = CachedSequential(
            Conv1d(in_size, in_size, 3, padding=get_padding(3)),
            nn.BatchNorm1d(in_size),
            UpsampleLayer(
                in_size,
                out_size,
                stride,
                padding_mode=padding_mode,
                bias=True,
            ),
        )

        self.proj = nn.Sequential(
            nn.LeakyReLU(.2),
            nn.Conv1d(out_size, 2 * out_size, 1),
        )

        self.future_compensation = stride if USE_BUFFER_CONV else 0

    def forward(self, x):
        x = self.net(x)
        mod = self.proj(x)
        mean, scale = torch.split(mod, mod.shape[1] // 2, 1)
        scale = torch.nn.functional.softplus(scale) / math.log(2)

        return torch.cat([x, mean, scale], 1)


class NoiseLayer(nn.Module):

    def __init__(self, dim) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        scale = torch.nn.functional.softplus(self.scale) / math.log(2)
        return x + scale.unsqueeze(-1) * torch.randn_like(x)


class ModulatedGenerator(nn.Module):

    def __init__(self, main, modulation, latent_size,
                 noise_dimensions) -> None:
        super().__init__()
        assert len(main) == len(modulation)

        self.blocks = nn.ModuleList(
            [AlignModulation(m1, m2) for m1, m2 in zip(main, modulation)])

        self.noise = nn.ModuleList([NoiseLayer(d) for d in noise_dimensions])

        self.constant = nn.Parameter(torch.zeros(latent_size))

    def forward(self, z):
        x = self.constant.reshape(1, -1, 1).expand_as(z)

        for block, noise in zip(self.blocks, self.noise):
            x = noise(x)
            x, z, mean, scale = block(x, z)
            x = x * scale + mean

        return x


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
        main_net = [
            wn(
                Conv1d(
                    latent_size,
                    2**len(ratios) * capacity,
                    7,
                    padding=get_padding(7, mode=padding_mode),
                    bias=bias,
                ))
        ]

        noise_dimensions = [latent_size]
        modulation_net = [
            ModulationLayer(latent_size, 2**len(ratios) * capacity, 1,
                            padding_mode)
        ]

        for i, r in enumerate(ratios):
            in_dim = 2**(len(ratios) - i) * capacity
            out_dim = 2**(len(ratios) - i - 1) * capacity

            main_net.append(
                CachedSequential(
                    UpsampleLayer(in_dim, out_dim, r, padding_mode),
                    ResidualStack(out_dim, 3, padding_mode),
                ))
            noise_dimensions.append(in_dim)
            modulation_net.append(
                ModulationLayer(in_dim, out_dim, r, padding_mode))

        self.net = ModulatedGenerator(main_net, modulation_net, latent_size,
                                      noise_dimensions)

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

            net.append(
                CachedSequential(
                    nn.BatchNorm1d(in_dim),
                    nn.LeakyReLU(.2),
                    Conv1d(
                        in_dim,
                        out_dim,
                        2 * r + 1,
                        padding=get_padding(2 * r + 1, r, mode=padding_mode),
                        stride=r,
                        bias=bias,
                    ),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(.2),
                    Conv1d(
                        out_dim,
                        out_dim,
                        3,
                        padding=get_padding(3, 1, mode=padding_mode),
                        bias=bias,
                    ),
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

        net = [wn(Conv1d(in_size, capacity, 15, padding=get_padding(15)))]
        net.append(nn.LeakyReLU(.2))

        for i in range(n_layers):
            net.append(
                wn(
                    Conv1d(
                        capacity * multiplier**i,
                        min(1024, capacity * multiplier**(i + 1)),
                        41,
                        stride=multiplier,
                        padding=get_padding(41, multiplier),
                        groups=multiplier**(i + 1),
                    )))
            net.append(nn.LeakyReLU(.2))

        net.append(
            wn(
                Conv1d(
                    min(1024, capacity * multiplier**(i + 1)),
                    min(1024, capacity * multiplier**(i + 1)),
                    5,
                    padding=get_padding(5),
                )))
        net.append(nn.LeakyReLU(.2))
        net.append(wn(Conv1d(min(1024, capacity * multiplier**(i + 1)), 1, 1)))
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
