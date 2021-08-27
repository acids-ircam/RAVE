import torch
import torch.nn as nn


class CachedPadding1d(nn.Module):
    def __init__(self, padding, crop=False):
        super().__init__()
        self.initialized = 0
        self.padding = padding
        self.crop = crop

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, _ = x.shape
        self.register_buffer("pad", torch.zeros(b, c, self.padding))
        self.initialized += 1

    def forward(self, x):
        if not self.initialized:
            self.init_cache(x)

        if self.padding == 0:
            return x

        padded_x = torch.cat([self.pad, x], -1)
        self.pad = padded_x[..., -self.padding:]

        if self.crop:
            padded_x = padded_x[..., :-self.padding]

        return padded_x


class CachedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        padding = kwargs.get("padding", 0)

        kwargs["padding"] = 0

        super().__init__(*args, **kwargs)

        if isinstance(padding, int):
            self.future_compensation = padding
            padding = padding * 2
        elif isinstance(padding, list) or isinstance(padding, tuple):
            self.future_compensation = padding[1]
            padding = padding[0] + padding[1]

        self.cache = CachedPadding1d(padding)

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
    @property
    def future_compensation(self):
        raise Exception(
            f"Future compensation not implemented for {self.__class__.__name__} yet."
        )

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


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        self._pad = kwargs.get("padding", (0, 0))
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        self.future_compensation = 0

    def forward(self, x):
        x = nn.functional.pad(x, self._pad)
        return nn.functional.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class AlignBranches(nn.Module):
    def __init__(self, *branches, futures=None):
        super().__init__()
        self.branches = branches

        if futures is None:
            futures = list(map(lambda x: x.future_compensation, self.branches))

        max_future = max(futures)

        self.paddings = [
            CachedPadding1d(p, crop=True)
            for p in map(lambda f: max_future - f, futures)
        ]

    def forward(self, x):
        outs = []
        for b, p in zip(self.branches, self.paddings):
            outs.append(p(b(x)))
        return outs

    def script_cache(self):
        for p in self.paddings:
            p.script_cache()
