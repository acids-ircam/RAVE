from turtle import forward
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as wn
import numpy as np
import cached_conv as cc


class ConvNetNd(nn.Module):

    @property
    def ConvNd(self) -> nn.modules.conv._ConvNd.__class__:
        raise NotImplementedError

    def __init__(self, in_size, out_size, capacity, n_layers, kernel_size,
                 stride) -> None:
        super().__init__()
        channels = [in_size]
        channels += list(capacity * 2**np.arange(n_layers))

        if isinstance(stride, int):
            stride = n_layers * [stride]

        net = []
        for i in range(n_layers):
            if not isinstance(kernel_size, int):
                pad = (cc.get_padding(kernel_size[0], stride[i])[0], 0)
                s = (stride[i], 1)
            else:
                pad = cc.get_padding(kernel_size, stride[i])[0]
                s = stride[i]
            net.append(
                self.ConvNd(
                    channels[i],
                    channels[i + 1],
                    kernel_size,
                    stride=s,
                    padding=pad,
                ))
            net.append(nn.LeakyReLU(.2))
        net.append(self.ConvNd(channels[-1], out_size, 1))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        features = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.modules.conv._ConvNd):
                features.append(x)
        return features


class ConvNet1d(ConvNetNd):

    @property
    def ConvNd(self) -> nn.modules.conv._ConvNd.__class__:
        return nn.Conv1d


class ConvNet2d(ConvNetNd):

    @property
    def ConvNd(self) -> nn.modules.conv._ConvNd.__class__:
        return nn.Conv2d


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, capacity, n_discriminators, n_layers, kernel_size,
                 stride) -> None:
        super().__init__()
        layers = []
        for i in range(n_discriminators):
            layers.append(
                ConvNet1d(1, 1, capacity, n_layers, kernel_size, stride))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = []
        for layer in self.layers:
            features.append(layer(x))
            x = nn.functional.avg_pool1d(x, 2)
        return features


class MultiPeriodDiscriminator(nn.Module):

    def __init__(self, capacity, periods, n_layers, kernel_size,
                 stride) -> None:
        super().__init__()
        layers = []
        self.periods = periods

        for _ in periods:
            layers.append(
                ConvNet2d(1, 1, capacity, n_layers, kernel_size, stride))

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


class FullDiscriminator(nn.Module):

    def __init__(self, capacity, periods, n_scale, n_layers, scale_kernel_size,
                 period_kernel_size, stride) -> None:
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(
            capacity,
            periods,
            n_layers,
            period_kernel_size,
            stride,
        )
        self.msd = MultiScaleDiscriminator(
            capacity,
            n_scale,
            n_layers,
            scale_kernel_size,
            stride,
        )

    def forward(self, x):
        features = []
        features.extend(self.mpd(x))
        features.extend(self.msd(x))
        return features
