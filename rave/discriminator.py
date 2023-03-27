from typing import Callable, Optional, Sequence, Tuple, Type

import cached_conv as cc
import numpy as np
import torch
import torch.nn as nn
import torchaudio

from .blocks import normalization


def spectrogram(n_fft: int):
    return torchaudio.transforms.Spectrogram(
        n_fft,
        hop_length=n_fft // 4,
        power=None,
        normalized=True,
        center=False,
        pad_mode=None,
    )


def rectified_2d_conv_block(
    capacity,
    kernel_sizes,
    strides: Optional[Tuple[int, int]] = None,
    dilations: Optional[Tuple[int, int]] = None,
    in_size: Optional[int] = None,
    out_size: Optional[int] = None,
    activation: bool = True,
):
    if dilations is None:
        paddings = kernel_sizes[0] // 2, kernel_sizes[1] // 2
    else:
        fks = (kernel_sizes[0] - 1) * dilations[0], (kernel_sizes[1] -
                                                     1) * dilations[1]
        paddings = fks[0] // 2, fks[1] // 2

    conv = normalization(
        nn.Conv2d(
            in_size or capacity,
            out_size or capacity,
            kernel_size=kernel_sizes,
            stride=strides or (1, 1),
            dilation=dilations or (1, 1),
            padding=paddings,
        ))

    if not activation: return conv

    return nn.Sequential(conv, nn.LeakyReLU(.2))


class EncodecConvNet(nn.Module):

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            rectified_2d_conv_block(capacity, (9, 3), in_size=2),
            rectified_2d_conv_block(capacity, (9, 3), (2, 1), (1, 1)),
            rectified_2d_conv_block(capacity, (9, 3), (2, 1), (1, 2)),
            rectified_2d_conv_block(capacity, (9, 3), (2, 1), (1, 4)),
            rectified_2d_conv_block(capacity, (3, 3)),
            rectified_2d_conv_block(capacity, (3, 3),
                                    out_size=1,
                                    activation=False),
        )

    def forward(self, x):
        features = []
        for layer in self.net:
            x = layer(x)
            features.append(x)
        return features


class ConvNet(nn.Module):

    def __init__(self, in_size, out_size, capacity, n_layers, kernel_size,
                 stride, conv) -> None:
        super().__init__()
        channels = [in_size]
        channels += list(capacity * 2**np.arange(n_layers))

        if isinstance(stride, int):
            stride = n_layers * [stride]

        net = []
        for i in range(n_layers):
            if not isinstance(kernel_size, int):
                pad = (cc.get_padding(kernel_size[0],
                                      stride[i],
                                      mode="centered")[0], 0)
                s = (stride[i], 1)
            else:
                pad = cc.get_padding(kernel_size, stride[i],
                                     mode="centered")[0]
                s = stride[i]
            net.append(
                normalization(
                    conv(
                        channels[i],
                        channels[i + 1],
                        kernel_size,
                        stride=s,
                        padding=pad,
                    )))
            net.append(nn.LeakyReLU(.2))
        net.append(conv(channels[-1], out_size, 1))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        features = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.modules.conv._ConvNd):
                features.append(x)
        return features


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, n_discriminators, convnet, n_channels=1) -> None:
        super().__init__()
        layers = []
        for i in range(n_discriminators):
            layers.append(convnet(in_size=n_channels))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = []
        for layer in self.layers:
            features.append(layer(x))
            x = nn.functional.avg_pool1d(x, 2)
        return features


class MultiScaleSpectralDiscriminator(nn.Module):

    def __init__(self, scales: Sequence[int],
                 convnet: Callable[[], nn.Module]) -> None:
        super().__init__()
        self.specs = nn.ModuleList([spectrogram(n) for n in scales])
        self.nets = nn.ModuleList([convnet() for _ in scales])

    def forward(self, x):
        features = []
        for spec, net in zip(self.specs, self.nets):
            spec_x = spec(x)
            spec_x = torch.cat([spec_x.real, spec_x.imag], 1)
            features.append(net(spec_x))
        return features


class MultiScaleSpectralDiscriminator1d(nn.Module):

    def __init__(self, scales: Sequence[int],
                 convnet: Callable[[int], nn.Module]) -> None:
        super().__init__()
        self.specs = nn.ModuleList([spectrogram(n) for n in scales])
        self.nets = nn.ModuleList([convnet(n + 2) for n in scales])

    def forward(self, x):
        features = []
        for spec, net in zip(self.specs, self.nets):
            spec_x = spec(x).squeeze(1)
            spec_x = torch.cat([spec_x.real, spec_x.imag], 1)
            features.append(net(spec_x))
        return features


class MultiPeriodDiscriminator(nn.Module):

    def __init__(self, periods, convnet, n_channels=1) -> None:
        super().__init__()
        layers = []
        self.periods = periods

        for _ in periods:
            layers.append(convnet(in_size=n_channels))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = []
        for layer, n in zip(self.layers, self.periods):
            features.append(layer(self.fold(x, n)))
        return features

    def fold(self, x, n):
        pad = (n - (x.shape[-1] % n)) % n
        x = nn.functional.pad(x, (0, pad))
        return x.reshape(*x.shape[:2], -1, n)


class CombineDiscriminators(nn.Module):

    def __init__(self, discriminators: Sequence[Type[nn.Module]], n_channels=1) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(disc_cls(n_channels=n_channels)
                                            for disc_cls in discriminators)

    def forward(self, x):
        features = []
        for disc in self.discriminators:
            features.extend(disc(x))
        return features
