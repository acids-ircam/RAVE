# Many ideas taken from https://github.com/acids-ircam/cached_conv/blob/master/cached_conv/convs.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import cached_conv as cc

class CachedPadding2d(nn.Module):

    def __init__(self, padding, crop=False):
        super().__init__()

        self.padding = padding
        self.crop = crop
        self.register_buffer('pad', torch.empty(0, 0, 0, padding))

    def forward(self, x):
        if self.padding:
            if self.pad.shape[0] == 0:
                self.pad = x.new_zeros((x.shape[0], x.shape[1], x.shape[2], self.padding))
            elif x.shape[0] > self.pad.shape[0]:
                extra = x.new_zeros((x.shape[0] - self.pad.shape[0], x.shape[1], x.shape[2], self.padding))
                self.pad = torch.cat((self.pad, extra), dim=0)
            x = torch.cat([self.pad[:x.shape[0]], x], -1)

            self.pad[:x.shape[0]].copy_(x[..., -self.padding:])

            if self.crop:
                x = x[..., :-self.padding]
        return x

class CausalConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=True,
                 padding_time=(0, 0),
                 padding_vert=(0, 0),
                 **kwargs):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        if (isinstance(padding_time, list) or isinstance(padding_time, tuple)):
            padding_left, padding_right = padding_time
        else:
            padding_left = padding_time

            padding_right = padding_time

        if (isinstance(padding_vert, list) or isinstance(padding_vert, tuple)):
            padding_vert = padding_vert
        elif isinstance(padding_vert, str) and padding_vert == "same":
            padding_vert = (kernel_size[0] - 1) // 2
            padding_vert = (padding_vert, padding_vert)
        elif isinstance(padding_vert, str) and padding_vert == "none":
            padding_vert = (0, 0)
        else:
            padding_vert = (padding_vert, padding_vert)

        self.padding_left = padding_left
        self.padding_right = padding_right
        self.padding_vert = padding_vert  

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         padding=0,
                         dilation=dilation,
                         bias=bias)


    def forward(self, x):
        # Pad time dimension (W) on the right and left
        pad = (self.padding_left, self.padding_right, self.padding_vert[0],
               self.padding_vert[1])  # (left, right, top, bottom)
        x = F.pad(x, pad)
        x = F.conv2d(x,
                     self.weight,
                     self.bias,
                     stride=self.stride,
                     padding=0,
                     dilation=self.dilation,
                     groups=self.groups)
        return x


def __prepare_scriptable__(self):
    for hook in self._forward_pre_hooks.values():
        if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
            torch.nn.utils.remove_weight_norm(self)
    return self


def normalization(module: nn.Module, mode: str = 'weight_norm'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        layer = torch.nn.utils.weight_norm(module)
        layer.__prepare_scriptable__ = __prepare_scriptable__.__get__(layer)
        return layer
    else:
        raise Exception(f'Normalization mode {mode} not supported')


class CachedConv2d(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if cc.USE_BUFFER_CONV:

            if "padding" in list(kwargs.keys()):
                padding = kwargs.pop("padding")
                padding_time = padding
                padding_vert = padding
            else:
                padding_time = kwargs.get("padding_time", 0)
                padding_vert = kwargs.get("padding_vert", 0)

            cumulative_delay = kwargs.pop("cumulative_delay", 0)

            if isinstance(padding_time, int):
                r_pad = padding_time
                padding = 2 * r_pad
            elif (isinstance(padding_time, list)
                  or isinstance(padding_time, tuple)):
                r_pad = padding_time[1]
                padding = padding_time[1] + padding_time[0]
            else:
                print(padding)
                raise NotImplementedError

            kwargs["padding_vert"] = padding_vert
            kwargs["padding_time"] = 0

            # super().__init__(*args, **kwargs)
            if kwargs["normalization"]:
                causal_conv = CausalConv2d(*args, **kwargs)
                self.causal_conv = normalization(causal_conv)
            else:
                self.causal_conv = CausalConv2d(*args, **kwargs)

            s = self.causal_conv.stride[1]
            cd = cumulative_delay

            stride_delay = (s - ((r_pad + cd) % s)) % s

            self.cumulative_delay = (r_pad + stride_delay + cd) // s

            self.cache = CachedPadding2d(padding)
            self.downsampling_delay = CachedPadding2d(stride_delay, crop=True)
        else:
            kwargs.pop("cumulative_delay", 0)
            if "padding" in list(kwargs.keys()):
                padding = kwargs.pop("padding")
                padding_time = padding
                padding_vert = padding
            else:
                padding_time = kwargs.get("padding_time", 0)
                padding_vert = kwargs.get("padding_vert", 0)

            kwargs["padding_vert"] = padding_vert
            kwargs["padding_time"] = padding_time

            # super().__init__(*args, **kwargs)
            if kwargs["normalization"]:
                self.causal_conv = normalization(CausalConv2d(*args, **kwargs))
            else:
                self.causal_conv = CausalConv2d(*args, **kwargs)
            self.cumulative_delay = 0
            self.cache = nn.Identity()
            self.downsampling_delay = nn.Identity()

    # def __prepare_scriptable__(self):
    #     print("preparing zzzz")
    #     torch.nn.utils.remove_weight_norm(self.causal_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsampling_delay(x)
        x = self.cache(x)
        x = self.causal_conv(x)
        return x

class CachedConvTranspose2d(nn.ConvTranspose2d):
    """
    Implementation of a ConvTranspose1d operation with cached padding
    """

    def __init__(self, *args, **kwargs):
        cd = kwargs.pop("cumulative_delay", 0)
        super().__init__(*args, **kwargs)

        if cc.USE_BUFFER_CONV:
            stride = self.stride[1]
            self.cumulative_delay = self.padding[1] + cd * stride
            self._padding = [self.padding[0], 0]
            self.time_pad = self.padding[1]
            self.use_cache = True

        else:
            self.cumulative_delay = 0
            self._padding = [self.padding[0], 0]
            self.time_pad = self.padding[1]
            self.use_cache = False
        self.register_buffer("cache", torch.empty(0, 0, 0, 2 * self.time_pad), persistent=False)

    def forward(self, x):
        x = F.conv_transpose2d(
            x,
            self.weight,
            None,
            self.stride,
            self._padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

        padding = 2 * self.time_pad

        if self.use_cache and padding > 0:
            if self.cache.shape[0] == 0:
                self.cache = x.new_zeros((x.shape[0], x.shape[1], x.shape[2], padding))
            elif x.shape[0] > self.cache.shape[0]:
                extra = x.new_zeros((x.shape[0] - self.cache.shape[0], x.shape[1], x.shape[2], padding))
                self.cache = torch.cat((self.cache, extra), dim=0)
            x[..., :padding] += self.cache[:x.shape[0]]
            self.cache[:x.shape[0]].copy_(x[..., -padding:])

        if self.time_pad > 0:
            x = x[..., :-padding]

        bias = self.bias
        if bias is not None:
            x = x + bias.unsqueeze(-1).unsqueeze(-1)

        return x


class AlignBranches2d(nn.Module):

    def __init__(self, *branches, delays=None, cumulative_delay=0, stride=1):
        super().__init__()
        self.branches = nn.ModuleList(branches)

        if cc.USE_BUFFER_CONV:
            if delays is None:
                delays = list(map(lambda x: x.cumulative_delay, self.branches))

            max_delay = max(delays)
            self.addition_delay = max_delay // 2

            self.paddings = nn.ModuleList([
                CachedPadding2d((p), crop=True)
                for p in map(lambda f: max_delay - f, delays)
            ])

            self.cumulative_delay = int(cumulative_delay * stride) + max_delay
        else:
            self.paddings = nn.ModuleList([nn.Identity() for _ in branches])
            self.cumulative_delay = 0

    def forward(self, x):
        outs = []
        for branch, pad in zip(self.branches, self.paddings):
            delayed_x = pad(x)
            outs.append(branch(delayed_x))
        return outs
