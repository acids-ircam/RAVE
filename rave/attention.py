import math
from typing import Tuple, Type, Callable

import cached_conv as cc
import einops
import gin
import torch
import torch.nn as nn


class MultiHeadAlibiAttention(nn.Module):

    def __init__(self,
                 n_head: int,
                 causal: bool,
                 cache_seq_len: int = 256) -> None:
        super().__init__()
        assert n_head >= 8, f'Alibi needs n_head > 8, got {n_head}'

        self._n_head = n_head
        self._causal = causal

        self._cached = cc.USE_BUFFER_CONV
        self._cache_seq_len = cache_seq_len

        self.register_cached_attention_bias()
        self.register_buffer("_keys_cache", torch.tensor([]))
        self.register_buffer("_values_cache", torch.tensor([]))
        self.register_buffer("_cache_length", torch.tensor(0))

    def bias_attention(self, attention: torch.Tensor):
        q_len, k_len = attention.shape[-2:]
        bias = torch.arange(k_len)[None] - torch.arange(q_len)[:, None]
        bias = torch.tril(bias).type_as(attention)
        bias = bias[None]

        m = torch.ones(self._n_head) * 2**(-8 / self._n_head)
        m = torch.cumprod(m, -1).to(attention)[:, None, None]

        bias = bias * m
        return attention + bias

    def register_cached_attention_bias(self):
        bias = -torch.arange(self._cache_seq_len).flip(0)
        slopes = torch.ones(self._n_head) * 2**(-8 / self._n_head)
        slopes = torch.cumprod(slopes, -1)[:, None, None]
        bias = bias * slopes
        self.register_buffer('_cached_attention_bias', bias)

    @torch.jit.unused
    def init_cache(self, k: torch.Tensor, v: torch.Tensor):
        self._keys_cache = torch.zeros(
            k.shape[0],
            k.shape[1],
            self._cache_seq_len,
            k.shape[-1],
        ).type_as(k)
        self._values_cache = torch.zeros(
            v.shape[0],
            v.shape[1],
            self._cache_seq_len,
            v.shape[-1],
        ).type_as(v)

    def update_cache(self, k: torch.Tensor, v: torch.Tensor):
        if not len(self._keys_cache) or not len(self._values_cache):
            self.init_cache(k, v)

        input_length = k.shape[2]

        if input_length > self._cache_seq_len:
            k = k[:, :, -self._cache_seq_len:]
            v = v[:, :, -self._cache_seq_len:]
            input_length = self._cache_seq_len

        self._keys_cache = self._keys_cache.roll(-input_length, dims=2)
        self._keys_cache[:, :, -input_length:] = k

        self._values_cache = self._values_cache.roll(-input_length, dims=2)
        self._values_cache[:, :, -input_length:] = v

        if self._cache_length < self._cache_seq_len:
            self._cache_length += input_length

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], self._n_head, -1)
        x = x.permute(0, 2, 1, 3)
        return x

    def gather_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x

    @torch.jit.unused
    def long_to_bool(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.bool()

    def reset(self):
        self._cache_length.zero_()

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        if self._cached:
            self.update_cache(k, v)
            k = self._keys_cache
            v = self._values_cache

            if q.shape[2] > 1:
                k = k[..., -q.shape[2]:, :]
                v = v[..., -q.shape[2]:, :]

        content_score = torch.einsum('bhtd,bhsd->bhts', q, k)
        attention = content_score / math.sqrt(q.shape[-1])

        # relative positional embedding
        if self._cached and q.shape[2] == 1:
            attention = attention + self._cached_attention_bias
        else:
            attention = self.bias_attention(attention)

        # causal masking
        if (self._causal and not self._cached) or (self._causal
                                                   and q.shape[2] > 1):
            mask = torch.triu(torch.ones_like(attention), 1).long()
            if not self._cached:
                mask = self.long_to_bool(mask)
            attention.masked_fill_(
                mask,
                -float('inf'),
            )
        elif self._cached:
            attention[..., :-self._cache_length] = -float('inf')

        attention = torch.softmax(attention, -1)

        out = torch.einsum('bhts,bhsd->bhtd', attention, v)
        out = self.gather_heads(out)

        return out


class TransformerLayer(nn.Module):

    def __init__(self, dim: int, attention: Callable[[], nn.Module],
                 dropout: float, n_head: int, head_dim: int) -> None:
        super().__init__()

        self._norm = nn.LayerNorm(dim)

        multihead_dim = head_dim * n_head

        self._q_proj = nn.Linear(dim, multihead_dim)
        self._k_proj = nn.Linear(dim, multihead_dim)
        self._v_proj = nn.Linear(dim, multihead_dim)

        self.attention = attention()

        self._dropout = nn.Dropout(p=dropout)
        self._predictor = nn.Linear(multihead_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._norm(x)

        q = self._q_proj(x)
        k = self._k_proj(x)
        v = self._v_proj(x)

        out = self.attention(q, k, v)

        out = self._dropout(out)
        out = self._predictor(out)

        return out


class OriginalFeedForwardModule(nn.Module):

    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                model_dim,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Residual(nn.Module):

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class Transformer(nn.Module):

    def __init__(self,
                 dim: int,
                 transformer_layer: Callable[[], nn.Module],
                 feed_forward_layer: Callable[[], nn.Module],
                 num_layers: int,
                 cumulative_delay: int = 0) -> None:
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(transformer_layer(dim))
            layers.append(feed_forward_layer(dim))

        self.net = nn.Sequential(*layers)
        self.cumulative_delay = cumulative_delay
        self.enabled = True
        self.temporal = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled: return x

        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x
