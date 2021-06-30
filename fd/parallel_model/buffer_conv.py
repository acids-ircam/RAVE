import torch
import torch.nn as nn


class CachedPadding1d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.initialized = 0
        self.padding = padding

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, t = x.shape
        self.register_buffer("pad", torch.zeros(b, c, self.padding))
        self.initialized += 1

    def forward(self, x):
        if not self.initialized:
            self.init_cache(x)

        if self.padding == 0:
            return x

        padded_x = torch.cat([self.pad, x], -1)
        self.pad = padded_x[..., -self.padding:]

        return padded_x


class CachedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        padding = kwargs.get("padding", 0)

        kwargs["padding"] = 0

        super().__init__(*args, **kwargs)
        self.cache = CachedPadding1d(padding * 2)

    def script_cache(self):
        self.cache = torch.jit.script(self.cache)

    def forward(self, x):
        x = self.cache(x)
        return nn.functional.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CachedConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        self.cache = CachedPadding1d(1)

    def script_cache(self):
        self.cache = torch.jit.script(self.cache)

    def forward(self, x):
        stride = self.stride[0]
        x = self.cache(x)
        x = nn.functional.conv_transpose1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )
        x = x[..., stride:-stride]
        return x
