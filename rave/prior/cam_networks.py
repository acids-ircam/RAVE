"""Networks of the CAM prior.

Continuous Autoregressive Models (CAM): a causal transformer reads the previous latent frames and outputs a
context vector; a small MLP head turns that context into the next (continuous) latent frame by rectified flow.

    M. Pasini, J. Nistal, S. Lattner, G. Fazekas.
    "Continuous Autoregressive Models with Noise Augmentation Avoid Error Accumulation".
    arXiv:2411.18447 (2024), NeurIPS 2024 Audio Imagination Workshop.

Attention is causal and limited to the `local_attention_size` last frames. This is the only knob: when
streaming, the key / value cache is exactly the attention window.
"""
import math
from typing import List, Optional, Tuple

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Rotary position embedding (relative: windows can slide without re-indexing)
# --------------------------------------------------------------------------- #

class RotaryEmbedding(nn.Module):
    """RoPE on the first `dim` channels of each head. Keys are cached *unrotated* and positions are
    assigned inside the current window (keys: 0..Lk-1, queries: the last Lq positions); since RoPE only
    depends on relative positions, this is equivalent to absolute positions."""

    def __init__(self, dim: int, theta: float = 10000.):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        # x: (B, H, L, Dh)
        length = x.shape[-2]
        pos = torch.arange(offset, offset + length, device=x.device, dtype=x.dtype)
        angles = pos[:, None] * self.inv_freq.to(x.dtype)[None]  # L, dim/2
        cos, sin = angles.cos(), angles.sin()
        rot, keep = x[..., :self.dim], x[..., self.dim:]
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
        rot = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
        return torch.cat((rot, keep), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # the queries are the last frames of the keys
        return self._rotate(q, k.shape[-2] - q.shape[-2]), self._rotate(k, 0)


def window_mask(n_query: int, n_key: int, window: int, device: torch.device) -> torch.Tensor:
    """boolean (n_query, n_key) mask, True where attention is allowed: causal, and at most `window` frames
    back (the frame itself included). window <= 0 means unlimited. Queries are the last frames of the keys."""
    offset = n_key - n_query
    i = torch.arange(n_query, device=device)[:, None] + offset
    j = torch.arange(n_key, device=device)[None]
    mask = j <= i
    if window > 0:
        mask = mask & (i - j < window)
    return mask


# --------------------------------------------------------------------------- #
#  Transformer
# --------------------------------------------------------------------------- #

class SelfAttention(nn.Module):

    def __init__(self, embed_dim: int, n_heads: int, window: int, dropout: float):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.window = window
        self.dropout = dropout
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rotary = RotaryEmbedding(min(32, self.head_dim))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, E). cache: previous keys / values (2, B, H, L, Dh) or None.
        Returns the output and the new cache (the last window - 1 keys / values)."""
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._heads(q), self._heads(k), self._heads(v)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        new_cache = torch.stack((k, v))
        if self.window > 0:
            new_cache = new_cache[:, :, :, max(k.shape[2] - (self.window - 1), 0):]
        q, k_rot = self.rotary(q, k)
        mask = window_mask(q.shape[2], k.shape[2], self.window, x.device)
        out = F.scaled_dot_product_attention(
            q, k_rot, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.)
        out = out.transpose(1, 2).reshape(x.shape)
        return self.out(out), new_cache


class Block(nn.Module):

    def __init__(self, embed_dim: int, n_heads: int, window: int, mlp_multiplier: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, n_heads, window, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, cache = self.attention(self.norm1(x), cache)
        x = x + h
        return x + self.mlp(self.norm2(x)), cache


@gin.configurable
class CAMTransformer(nn.Module):
    """Causal transformer: latent frames (B, C, T) -> context vectors (B, D, T), D = out_dim.
    Context t only depends on frames <= t (and on the last `local_attention_size` ones)."""

    def __init__(self,
                 in_channels: int,
                 out_dim: int = 16,
                 embed_dim: int = 384,
                 n_layers: int = 6,
                 n_heads: int = 6,
                 mlp_multiplier: int = 3,
                 dropout: float = 0.1,
                 local_attention_size: Optional[int] = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.window = local_attention_size or 0
        self.embed = nn.Sequential(nn.Linear(in_channels, embed_dim), nn.LayerNorm(embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads, self.window, mlp_multiplier, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """full sequence, no cache (training)."""
        h = self.embed(x.transpose(1, 2))
        for block in self.blocks:
            h, _ = block(h, None)
        return self.proj(self.norm(h)).transpose(1, 2)

    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
                   ) -> torch.Tensor:
        return torch.zeros(self.n_layers, 2, batch_size, self.n_heads, 0, self.head_dim,
                           device=device, dtype=dtype)

    def step(self, x: torch.Tensor, cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """streaming: x (B, C, T) are the new frames, cache the state returned by the previous call (or
        init_cache). Returns the contexts of the new frames and the new cache. The result is the same as
        `forward` on the whole sequence."""
        h = self.embed(x.transpose(1, 2))
        caches: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            h, c = block(h, cache[i])
            caches.append(c)
        return self.proj(self.norm(h)).transpose(1, 2), torch.stack(caches)


class SOSReplacer(nn.Module):
    """learned "start of sequence" context, randomly substituted to the contexts during training, so that
    the head also learns to sample a frame without any history."""

    def __init__(self, dim: int):
        super().__init__()
        self.sos = nn.Parameter(torch.zeros(1, dim))
        nn.init.xavier_normal_(self.sos)

    def forward(self, z: torch.Tensor, replace_ratio: float) -> torch.Tensor:
        if replace_ratio <= 0.:
            return z
        mask = torch.rand(z.shape[0], 1, device=z.device) < replace_ratio
        return torch.where(mask, self.sos.to(z), z)


# --------------------------------------------------------------------------- #
#  Flow head
# --------------------------------------------------------------------------- #

def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimeEmbedding(nn.Module):

    def __init__(self, num_channels: int, max_positions: int = 10000, factor: float = 1000.):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.factor = factor

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(self.num_channels // 2, device=t.device).float()
        freqs = (1. / self.max_positions) ** (freqs / (self.num_channels // 2 - 1))
        angles = (t * self.factor)[:, None] * freqs.to(t.dtype)[None]
        return torch.cat((angles.cos(), angles.sin()), dim=1)


class ResBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(y).chunk(3, dim=-1)
        return x + gate * self.mlp(_modulate(self.norm(x), shift, scale))


@gin.configurable
class FlowHead(nn.Module):
    """velocity field v(x_t, t | context) of the next latent frame: MLP with adaptive layer norm."""

    def __init__(self, in_channels: int, context_dim: int = 16, model_channels: int = 256,
                 n_blocks: int = 6, time_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.time_embed = nn.Sequential(TimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(),
                                        nn.Linear(time_dim, model_channels))
        self.context_embed = nn.Linear(context_dim, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(model_channels)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels))
        self.out = nn.Linear(model_channels, in_channels)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # start as the identity of the residual stream, with a zero velocity
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        for m in (self.modulation[-1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (N, C), context: (N, D), t: (N,) in [0, 1] -> velocity (N, C)."""
        y = self.time_embed(t) + self.context_embed(context)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, y)
        shift, scale = self.modulation(y).chunk(2, dim=-1)
        return self.out(_modulate(self.norm(h), shift, scale))
