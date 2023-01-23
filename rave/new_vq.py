import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def ema_inplace(source, target, decay):
    source.data.mul_(decay).add_(target, alpha=(1 - decay))


class CachedExamples(nn.Module):

    def __init__(self, size: int, dim: int):
        super().__init__()
        self.register_buffer("cache", torch.zeros(size, dim))
        self.num_examples = 0
        self.is_init = False
        self.size = size

    def __call__(self, x: torch.Tensor):
        perm = torch.randperm(x.shape[0])[:self.size]
        x = x[perm]
        self.cache = self.cache.roll(-x.shape[0], 0)
        self.cache[-x.shape[0]:] = x

        if self.is_init:
            return self.cache
        else:
            self.num_examples += x.shape[0]
            self.is_init = self.num_examples >= self.size
            return None


class VQ(nn.Module):

    def __init__(self, dim, size, ema, usage_threshold=.2):
        super().__init__()
        self.register_buffer("embedding", torch.randn(size, dim))
        self.register_buffer("usage", torch.zeros(size))
        self.ema = ema
        self.size = size
        self.usage_threshold = usage_threshold
        self.buffer = CachedExamples(size, dim)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.embedding.transpose(0, 1)
        distance = (x.pow(2).sum(1, keepdim=True) - 2 * x @ embed +
                    embed.pow(2).sum(0, keepdim=True))
        return torch.argmin(distance, -1)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = F.embedding(codes, self.embedding)
        return quantized

    @torch.no_grad()
    def replace_dead_codes(self, x: torch.Tensor) -> torch.Tensor:
        buffer = self.buffer(x.clone())
        if buffer is None: return

        dead = self.usage < self.usage_threshold
        if not dead.any(): return

        self.embedding = torch.where(dead[:, None], buffer, self.embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1).reshape(-1, x.shape[1])

        codes = self.encode(x)
        quantized = self.decode(codes)
        
        diff = (x - quantized).pow(2).mean()
        quantized = quantized + x - x.detach()

        if self.training:
            self.replace_dead_codes(x)

            onehot = F.one_hot(codes, self.size).float()
            usage = onehot.sum(0)
            targets = (x.T @ onehot / usage).T

            targets = torch.where(torch.isnan(targets), self.embedding,
                                  targets)

            ema_inplace(self.embedding, targets, self.ema)
            ema_inplace(self.usage, usage, self.ema)

        quantized = quantized.reshape(
            batch_size,
            -1,
            quantized.shape[-1],
        ).permute(0, 2, 1)
        return quantized, diff.mean()