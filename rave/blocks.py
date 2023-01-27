from functools import partial
from typing import Callable, Optional, Sequence

import cached_conv as cc
import gin
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .core import amp_to_impulse_response, fft_convolve, mod_sigmoid


@gin.configurable
def normalization(module: nn.Module, mode: str = 'identity'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        return weight_norm(module)
    else:
        raise Exception(f'Normalization mode {mode} not supported')


class SampleNorm(nn.Module):

    def forward(self, x):
        return x / torch.norm(x, 2, 1, keepdim=True)


class Residual(nn.Module):

    def __init__(self, module, cumulative_delay=0):
        super().__init__()
        additional_delay = module.cumulative_delay
        self.aligned = cc.AlignBranches(
            module,
            nn.Identity(),
            delays=[additional_delay, 0],
        )
        self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x):
        x_net, x_res = self.aligned(x)
        return x_net + x_res


class ResidualLayer(nn.Module):

    def __init__(self, dim, kernel_size, dilations, cumulative_delay=0):
        super().__init__()
        net = []
        cd = 0
        for d in dilations:
            net.append(nn.LeakyReLU(.2))
            net.append(
                normalization(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        dilation=d,
                        padding=cc.get_padding(kernel_size, dilation=d),
                        cumulative_delay=cd,
                    )))
            cd = net[-1].cumulative_delay
        self.net = Residual(
            cc.CachedSequential(*net),
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)


class DilatedUnit(nn.Module):

    def __init__(self, dim: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        net = [
            nn.LeakyReLU(.2),
            normalization(
                cc.Conv1d(dim,
                          dim,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          padding=cc.get_padding(
                              kernel_size,
                              dilation=dilation,
                          ))),
            nn.LeakyReLU(.2),
            normalization(cc.Conv1d(dim, dim, kernel_size=1)),
        ]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = net[1].cumulative_delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 dilations_list,
                 cumulative_delay=0) -> None:
        super().__init__()
        layers = []
        cd = 0

        for dilations in dilations_list:
            layers.append(
                ResidualLayer(
                    dim,
                    kernel_size,
                    dilations,
                    cumulative_delay=cd,
                ))
            cd = layers[-1].cumulative_delay

        self.net = cc.CachedSequential(
            *layers,
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)


@gin.configurable
class ResidualStack(nn.Module):

    def __init__(self,
                 dim,
                 kernel_sizes,
                 dilations_list,
                 cumulative_delay=0) -> None:
        super().__init__()
        blocks = []
        for k in kernel_sizes:
            blocks.append(ResidualBlock(dim, k, dilations_list))
        self.net = cc.AlignBranches(*blocks, cumulative_delay=cumulative_delay)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        x = self.net(x)
        x = torch.stack(x, 0).sum(0)
        return x


class UpsampleLayer(nn.Module):

    def __init__(self, in_dim, out_dim, ratio, cumulative_delay=0):
        super().__init__()
        net = [nn.LeakyReLU(.2)]
        if ratio > 1:
            net.append(
                normalization(
                    cc.ConvTranspose1d(in_dim,
                                       out_dim,
                                       2 * ratio,
                                       stride=ratio,
                                       padding=ratio // 2)))
        else:
            net.append(
                normalization(
                    cc.Conv1d(in_dim, out_dim, 3, padding=cc.get_padding(3))))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

    def forward(self, x):
        return self.net(x)


@gin.configurable
class NoiseGenerator(nn.Module):

    def __init__(self, in_size, data_size, ratios, noise_bands):
        super().__init__()
        net = []
        channels = [in_size] * len(ratios) + [data_size * noise_bands]
        cum_delay = 0
        for i, r in enumerate(ratios):
            net.append(
                cc.Conv1d(
                    channels[i],
                    channels[i + 1],
                    3,
                    padding=cc.get_padding(3, r),
                    stride=r,
                    cumulative_delay=cum_delay,
                ))
            cum_delay = net[-1].cumulative_delay
            if i != len(ratios) - 1:
                net.append(nn.LeakyReLU(.2))

        self.net = cc.CachedSequential(*net)
        self.data_size = data_size
        self.cumulative_delay = self.net.cumulative_delay * int(
            np.prod(ratios))

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

    def __init__(
        self,
        latent_size,
        capacity,
        data_size,
        ratios,
        loud_stride,
        use_noise,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()
        net = [
            normalization(
                cc.Conv1d(
                    latent_size,
                    2**len(ratios) * capacity,
                    7,
                    padding=cc.get_padding(7),
                ))
        ]

        if recurrent_layer is not None:
            net.append(
                recurrent_layer(
                    dim=2**len(ratios) * capacity,
                    cumulative_delay=net[0].cumulative_delay,
                ))

        for i, r in enumerate(ratios):
            in_dim = 2**(len(ratios) - i) * capacity
            out_dim = 2**(len(ratios) - i - 1) * capacity

            net.append(
                UpsampleLayer(
                    in_dim,
                    out_dim,
                    r,
                    cumulative_delay=net[-1].cumulative_delay,
                ))
            net.append(
                ResidualStack(out_dim,
                              cumulative_delay=net[-1].cumulative_delay))

        self.net = cc.CachedSequential(*net)

        wave_gen = normalization(
            cc.Conv1d(out_dim, data_size, 7, padding=cc.get_padding(7)))

        loud_gen = normalization(
            cc.Conv1d(
                out_dim,
                1,
                2 * loud_stride + 1,
                stride=loud_stride,
                padding=cc.get_padding(2 * loud_stride + 1, loud_stride),
            ))

        branches = [wave_gen, loud_gen]

        if use_noise:
            noise_gen = NoiseGenerator(out_dim, data_size)
            branches.append(noise_gen)

        self.synth = cc.AlignBranches(
            *branches,
            cumulative_delay=self.net.cumulative_delay,
        )

        self.use_noise = use_noise
        self.loud_stride = loud_stride
        self.cumulative_delay = self.synth.cumulative_delay

        self.register_buffer("warmed_up", torch.tensor(0))

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x):
        x = self.net(x)

        if self.use_noise:
            waveform, loudness, noise = self.synth(x)
        else:
            waveform, loudness = self.synth(x)
            noise = torch.zeros_like(waveform)

        if self.loud_stride != 1:
            loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform) * mod_sigmoid(loudness)

        if self.warmed_up and self.use_noise:
            waveform = waveform + noise

        return waveform


class Encoder(nn.Module):

    def __init__(
        self,
        data_size,
        capacity,
        latent_size,
        ratios,
        n_out,
        sample_norm,
        repeat_layers,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()
        net = [cc.Conv1d(data_size, capacity, 7, padding=cc.get_padding(7))]

        for i, r in enumerate(ratios):
            in_dim = 2**i * capacity
            out_dim = 2**(i + 1) * capacity

            if sample_norm:
                net.append(SampleNorm())
            else:
                net.append(nn.BatchNorm1d(in_dim))
            net.append(nn.LeakyReLU(.2))
            net.append(
                cc.Conv1d(
                    in_dim,
                    out_dim,
                    2 * r + 1,
                    padding=cc.get_padding(2 * r + 1, r),
                    stride=r,
                    cumulative_delay=net[-3].cumulative_delay,
                ))

            for i in range(repeat_layers - 1):
                if sample_norm:
                    net.append(SampleNorm())
                else:
                    net.append(nn.BatchNorm1d(out_dim))
                net.append(nn.LeakyReLU(.2))
                net.append(
                    cc.Conv1d(
                        out_dim,
                        out_dim,
                        3,
                        padding=cc.get_padding(3),
                        cumulative_delay=net[-3].cumulative_delay,
                    ))

        net.append(nn.LeakyReLU(.2))

        if recurrent_layer is not None:
            net.append(
                recurrent_layer(
                    dim=out_dim,
                    cumulative_delay=net[-2].cumulative_delay,
                ))
            net.append(nn.LeakyReLU(.2))

        net.append(
            cc.Conv1d(
                out_dim,
                latent_size * n_out,
                5,
                padding=cc.get_padding(5),
                groups=n_out,
                cumulative_delay=net[-2].cumulative_delay,
            ))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        z = self.net(x)
        return z


class EncoderV2(nn.Module):

    def __init__(
        self,
        data_size: int,
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        n_out: int,
        kernel_size: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
    ) -> None:
        super().__init__()
        net = [
            normalization(
                cc.Conv1d(
                    data_size,
                    capacity,
                    kernel_size=kernel_size * 2 + 1,
                    padding=cc.get_padding(kernel_size * 2 + 1),
                )),
        ]

        num_channels = capacity
        for r in ratios:
            # ADD RESIDUAL DILATED UNITS
            for d in dilations:
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=num_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                        )))

            # ADD DOWNSAMPLING UNIT
            net.append(nn.LeakyReLU(.2))

            if keep_dim:
                out_channels = num_channels * r
            else:
                out_channels = num_channels * 2
            net.append(
                normalization(
                    cc.Conv1d(
                        num_channels,
                        out_channels,
                        kernel_size=2 * r,
                        stride=r,
                        padding=cc.get_padding(2 * r, r),
                    )))

            num_channels = out_channels

        net.append(nn.LeakyReLU(.2))
        net.append(
            normalization(
                cc.Conv1d(
                    num_channels,
                    latent_size * n_out,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size),
                )))

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size * n_out))

        self.net = cc.CachedSequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeneratorV2(nn.Module):

    def __init__(
        self,
        data_size: int,
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        kernel_size: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
    ) -> None:
        super().__init__()
        ratios = ratios[::-1]

        if keep_dim:
            num_channels = np.prod(ratios) * capacity
        else:
            num_channels = 2**len(ratios) * capacity

        net = []

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size))

        net.append(
            normalization(
                cc.Conv1d(
                    latent_size,
                    num_channels,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size),
                )), )

        for r in ratios:
            # ADD UPSAMPLING UNIT
            if keep_dim:
                out_channels = num_channels // r
            else:
                out_channels = num_channels // 2
            net.append(nn.LeakyReLU(.2))
            net.append(
                normalization(
                    cc.ConvTranspose1d(num_channels,
                                       out_channels,
                                       2 * r,
                                       stride=r,
                                       padding=r // 2)))

            num_channels = out_channels

            # ADD RESIDUAL DILATED UNITS
            for d in dilations:
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=num_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                        )))

        net.append(nn.LeakyReLU(.2))
        net.append(
            normalization(
                cc.Conv1d(
                    num_channels,
                    data_size,
                    kernel_size=kernel_size * 2 + 1,
                    padding=cc.get_padding(kernel_size * 2 + 1),
                )))

        self.net = cc.CachedSequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))

    def set_warmed_up(self, state: bool):
        pass


class VariationalEncoder(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder()
        self.register_buffer("warmed_up", torch.tensor(0))

    def reparametrize(self, z):
        mean, scale = z.chunk(2, 1)
        std = nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return z, kl

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        if self.warmed_up:
            z = z.detach()
        return z


class WasserteinEncoder(nn.Module):

    def __init__(self, encoder_cls):
        super().__init__()
        self.encoder = encoder_cls()
        self.register_buffer("warmed_up", torch.tensor(0))

    def compute_mean_kernel(self, x, y):
        kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
        return torch.exp(-kernel_input).mean()

    def compute_mmd(self, x, y):
        x_kernel = self.compute_mean_kernel(x, x)
        y_kernel = self.compute_mean_kernel(y, y)
        xy_kernel = self.compute_mean_kernel(x, y)
        mmd = x_kernel + y_kernel - 2 * xy_kernel
        return mmd

    def reparametrize(self, z):
        z_reshaped = z.permute(0, 2, 1).reshape(-1, z.shape[1])
        reg = self.compute_mmd(z_reshaped, torch.randn_like(z_reshaped))
        return z, reg.mean()

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        if self.warmed_up:
            z = z.detach()
        return z


class DiscreteEncoder(nn.Module):

    def __init__(self, encoder_cls, vq_cls, num_quantizers):
        super().__init__()
        self.encoder = encoder_cls()
        self.rvq = vq_cls()
        self.num_quantizers = num_quantizers
        self.register_buffer("warmed_up", torch.tensor(0))
        self.register_buffer("enabled", torch.tensor(0))

    @torch.jit.ignore
    def reparametrize(self, z):
        if self.enabled:
            q, diff, _ = self.rvq(z)
            return q, diff.mean()
        else:
            return z, torch.zeros_like(z).mean()

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x):
        z = self.encoder(x)
        return z


class SphericalEncoder(nn.Module):

    def __init__(self, encoder_cls: Callable[[], nn.Module]) -> None:
        super().__init__()
        self.encoder = encoder_cls()

    def reparametrize(self, z):
        norm_z = z / torch.norm(z, p=2, dim=1, keepdim=True)
        reg = torch.zeros_like(z).mean()
        return norm_z, reg

    def set_warmed_up(self, state: bool):
        pass

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return z


def unit_norm_vector_to_angles(x: torch.Tensor) -> torch.Tensor:
    norms = x.flip(1).pow(2)
    norms[:, 1] += norms[:, 0]
    norms = norms[:, 1:]
    norms = norms.cumsum(1).flip(1).sqrt()
    angles = torch.arccos(x[:, :-1] / norms)
    angles[:, -1] = torch.where(
        x[:, -1] >= 0,
        angles[:, -1],
        2 * np.pi - angles[:, -1],
    )
    angles[:, :-1] = angles[:, :-1] / np.pi
    angles[:, -1] = angles[:, -1] / (2 * np.pi)
    return 2 * (angles - .5)


def angles_to_unit_norm_vector(angles: torch.Tensor) -> torch.Tensor:
    angles = (angles / 2 + .5) % 1
    angles[:, :-1] = angles[:, :-1] * np.pi
    angles[:, -1] = angles[:, -1] * (2 * np.pi)
    cos = angles.cos()
    sin = angles.sin().cumprod(dim=1)
    cos = torch.cat([
        cos,
        torch.ones(cos.shape[0], 1, cos.shape[-1]).type_as(cos),
    ], 1)
    sin = torch.cat([
        torch.ones(sin.shape[0], 1, sin.shape[-1]).type_as(sin),
        sin,
    ], 1)
    return cos * sin


def wrap_around_value(x: torch.Tensor, value: float = 1) -> torch.Tensor:
    return (x + value) % (2 * value) - value
