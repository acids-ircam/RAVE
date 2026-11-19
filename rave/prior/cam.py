"""CAM prior: autoregressive prior on the continuous latent space of a RAVE model.

Derived from Continuous Autoregressive Models:

    M. Pasini, J. Nistal, S. Lattner, G. Fazekas.
    "Continuous Autoregressive Models with Noise Augmentation Avoid Error Accumulation".
    arXiv:2411.18447 (2024), NeurIPS 2024 Audio Imagination Workshop.

The prior is trained on the full latent space of the model (`encode_full`: sampled latents, no projection).
Latents are encoded on the fly by the frozen RAVE (non-differentiable). For streaming, `CAMStreamer` is
embedded in the exported RAVE (see scripting.py): it generates full latents, which are then optionally
projected on the exported latent space (PCA...).
"""
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import gin
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cam_networks import CAMTransformer, FlowHead, SOSReplacer

import logging
_log = logging.getLogger(__name__)

NOISE_MODES = ("none", "convex", "linear")


def add_noise(x: torch.Tensor, mode: str) -> torch.Tensor:
    """noise augmentation of the autoregressive inputs: per frame, a random amount of noise is added,
    so that the model learns to correct its own errors."""
    if mode == "none":
        return x
    amount = torch.rand(x.shape[0], 1, x.shape[-1], device=x.device)
    if mode == "convex":
        return amount * torch.randn_like(x) + (1 - amount) * x
    if mode == "linear":
        return x + amount * torch.randn_like(x)
    raise ValueError(f"unknown noise mode {mode} (valid: {NOISE_MODES})")


class CAMCore(nn.Module):
    """transformer + flow head + learned start token. Contains everything needed for training and
    sampling; scriptable (see CAMStreamer)."""

    def __init__(self, latent_size: int, context_dim: int = 16):
        super().__init__()
        self.latent_size = latent_size
        self.context_dim = context_dim
        self.net = CAMTransformer(latent_size, out_dim=context_dim)
        self.head = FlowHead(latent_size, context_dim=context_dim)
        self.sos = SOSReplacer(context_dim)

    def flow_loss(self, z: torch.Tensor, p_drop_sos: float = 0.1, rf_batch_mul: int = 2,
                  noise_mode: str = "convex") -> torch.Tensor:
        """z: (B, C, T) latents. Teacher forcing: the context of frame t, computed on the (noised) frames
        <= t, conditions the flow to frame t + 1."""
        inputs = add_noise(z[..., :-1], noise_mode)
        context = self.net(inputs)
        target = z[..., 1:]
        b, c, t = target.shape
        target = target.permute(0, 2, 1).reshape(b * t, c)
        context = context.permute(0, 2, 1).reshape(b * t, -1)
        context = self.sos(context, p_drop_sos)
        if rf_batch_mul > 1:
            target = target.repeat(rf_batch_mul, 1)
            context = context.repeat(rf_batch_mul, 1)
        time = torch.rand(target.shape[0], device=z.device)
        base = torch.randn_like(target)
        x_t = (1 - time[:, None]) * base + time[:, None] * target
        velocity = self.head(x_t, context, time)
        return (velocity - (target - base)).square().mean()

    def sample_frame(self, context: torch.Tensor, steps: int, temperature: float,
                     guide: Optional[torch.Tensor] = None, weight: float = 0.) -> torch.Tensor:
        """next frame (B, C) from the context (B, D), by Euler integration of the flow.
        guide / weight: guidance by another context (see CAMStreamer.generate_listen), in the velocity space:
        v = v_context + weight * (v_guide - v_context)."""
        x = temperature * torch.randn(context.shape[0], self.latent_size, device=context.device,
                                      dtype=context.dtype)
        dt = 1. / steps
        for i in range(steps):
            time = torch.full((x.shape[0], ), i / steps, device=x.device, dtype=x.dtype)
            velocity = self.head(x, context, time)
            if guide is not None:
                velocity = velocity + weight * (self.head(x, guide, time) - velocity)
            x = x + dt * velocity
        return x

    @torch.no_grad()
    def sample(self, n_frames: int, batch_size: int = 1, steps: int = 20, temperature: float = 1.,
               initial: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, C, n_frames) latents, optionally continuing `initial` (B, C, T0), through the streaming path."""
        device = self.sos.sos.device
        cache = self.net.init_cache(batch_size, device)
        context = self.sos.sos.repeat(batch_size, 1)
        frames = torch.zeros(batch_size, self.latent_size, 0, device=device)
        if initial is not None:
            frames = initial.to(device)
            contexts, cache = self.net.step(frames, cache)
            context = contexts[..., -1]
        while frames.shape[-1] < n_frames:
            frame = self.sample_frame(context, steps, temperature)
            contexts, cache = self.net.step(frame.unsqueeze(-1), cache)
            context = contexts[..., -1]
            frames = torch.cat((frames, frame.unsqueeze(-1)), dim=-1)
        return frames[..., :n_frames]


# --------------------------------------------------------------------------- #
#  Frozen RAVE used as codec
# --------------------------------------------------------------------------- #

class LatentCodec:
    """Frozen, exported RAVE (non-streaming .ts) giving the full latents of the prior training. Not an
    nn.Module on purpose: it must stay out of the checkpoints of the prior."""

    def __init__(self, model: torch.jit.ScriptModule):
        from ..uncompiling import is_scripted_rave, unwrap_scripted, is_streaming
        if not is_scripted_rave(model):
            raise ValueError("the codec of the CAM prior must be a model exported by `rave export`")
        model = unwrap_scripted(model)
        if not hasattr(model, "encode_full"):
            raise ValueError("the CAM prior needs a continuous latent space (variational or wasserstein "
                             "model): this model has no encode_full method")
        if is_streaming(model):
            _log.warning("the codec is a streaming export: its cached convolutions keep a state between "
                         "training batches, which can degrade the latents. A non-streaming export (or a RAVE "
                         "run / checkpoint, exported for you) is recommended.")
        self.model = model.eval()
        self.sr = int(model.sr)
        self.n_channels = int(model.n_channels)
        self._device = torch.device("cpu")
        with torch.no_grad():
            z = self.model.encode_full(torch.zeros(1, self.n_channels, 2**14))
        self.latent_size = int(z.shape[1])
        self.ratio = 2**14 // int(z.shape[-1])

    def to(self, device) -> "LatentCodec":
        device = torch.device(device)
        if device != self._device:
            self.model = self.model.to(device)
            self._device = device
        return self

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        self.to(audio.device)
        return self.model.encode_full(audio).detach().float()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.to(z.device)
        return self.model.decode_full(z)


def load_codec(model_path: str, out_dir: Optional[str] = None, ckpt_name: str = "last.ckpt",
               ema_weights: bool = False) -> LatentCodec:
    """codec from a RAVE run / checkpoint (exported to out_dir/rave, non-streaming, reused if already
    there) or from a .ts."""
    model_path = str(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if os.path.splitext(model_path)[1] == ".ts":
        return LatentCodec(torch.jit.load(model_path, map_location="cpu"))
    from .. import core, get_run_name
    from ..scripting import export
    name = get_run_name(core.search_for_run(model_path, name=ckpt_name) or model_path) or Path(model_path).stem
    rave_dir = Path(out_dir or ".") / "rave"
    existing = sorted(rave_dir.glob(f"{name}*.ts")) if rave_dir.is_dir() else []
    existing = [p for p in existing if not p.stem.endswith("_streaming")]
    if existing:
        _log.info(f"reusing exported RAVE model {existing[0]}")
        return LatentCodec(torch.jit.load(str(existing[0]), map_location="cpu"))
    _log.info(f"exporting RAVE model {model_path} to {rave_dir}...")
    ts_path = export(model_path, out_path=str(rave_dir), name=name, streaming=False, ckpt_name=ckpt_name,
                     ema_weights=ema_weights)
    return LatentCodec(torch.jit.load(ts_path, map_location="cpu"))


# --------------------------------------------------------------------------- #
#  Training module
# --------------------------------------------------------------------------- #

@gin.configurable
class CAMPrior(pl.LightningModule):
    """Lightning module of the CAM prior.

    Args:
        codec: LatentCodec (frozen RAVE).
        n_frames: length of the training sequences, in latent frames (audio chunk: n_frames * ratio).
        context_dim: size of the context vector given by the transformer to the flow head.
        rf_batch_mul: the flow loss is computed on each frame this number of times (different times / noise).
        p_drop_sos: ratio of contexts replaced by the learned start token.
        noise_mode: noise augmentation of the inputs ("none", "convex", "linear").
        lr, lr_warmup, ema_decay, grad_clip: optimisation.
        sampling_steps: flow steps of the validation samples.
    """

    def __init__(self, codec: LatentCodec, n_frames: int = 64, context_dim: int = 16, rf_batch_mul: int = 2,
                 p_drop_sos: float = 0.1, noise_mode: str = "convex", lr: float = 1e-4, lr_warmup: int = 20000,
                 ema_decay: float = 0.999, grad_clip: float = 3., sampling_steps: int = 10):
        super().__init__()
        if noise_mode not in NOISE_MODES:
            raise ValueError(f"unknown noise mode {noise_mode} (valid: {NOISE_MODES})")
        self._codec = codec  # plain attribute: not part of state_dict / device moves
        self.core = CAMCore(codec.latent_size, context_dim)
        self.register_buffer("ratio", torch.tensor(codec.ratio))
        self.sr = codec.sr
        self.n_frames = n_frames
        self.n_signal = n_frames * codec.ratio
        self.rf_batch_mul = rf_batch_mul
        self.p_drop_sos = p_drop_sos
        self.noise_mode = noise_mode
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.ema_decay = ema_decay
        self.grad_clip = grad_clip
        self.sampling_steps = sampling_steps
        self.ema = None
        self._ema_state = None
        self.val_idx = 0

    # latents ---------------------------------------------------------------
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        z = self._codec.encode(audio)
        if z.shape[-1] < self.n_frames:
            raise RuntimeError(f"audio gives {z.shape[-1]} latent frames, less than n_frames={self.n_frames}")
        return z[..., :self.n_frames]

    # optimisation ----------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.core.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min((step + 1) / max(self.lr_warmup, 1), 1.))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _init_ema(self):
        from torch_ema import ExponentialMovingAverage
        self.ema = ExponentialMovingAverage(self.core.parameters(), decay=self.ema_decay)
        if self._ema_state is not None:
            self.ema.load_state_dict(self._ema_state)
            self._ema_state = None

    def on_fit_start(self):
        # the model is on its device: the shadow parameters of the EMA are created there
        self._init_ema()
        self._codec.to(self.device)

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.update()

    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["cam_ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self._ema_state = checkpoint.get("cam_ema")

    def on_validation_start(self):
        if self.ema is not None:
            self.ema.store(self.core.parameters())
            self.ema.copy_to(self.core.parameters())

    def on_validation_end(self):
        if self.ema is not None:
            self.ema.restore(self.core.parameters())

    # steps -----------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        z = self.encode(batch)
        loss = self.core.flow_loss(z, self.p_drop_sos, self.rf_batch_mul, self.noise_mode)
        self.log("latent_prediction", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z = self.encode(batch)
        loss = self.core.flow_loss(z, self.p_drop_sos, self.rf_batch_mul, "none")
        self.log("validation", loss)
        return z

    def on_validation_batch_end(self, out, batch, batch_idx, *args, **kwargs):
        if batch_idx != 0 or self.logger is None:
            return
        z = out[:min(4, out.shape[0])]
        generated = self.core.sample(self.n_frames, z.shape[0], self.sampling_steps)
        continued = self.core.sample(self.n_frames, z.shape[0], self.sampling_steps,
                                     initial=z[..., :self.n_frames // 2])
        for tag, latent in (("generation", generated), ("continuation", continued)):
            audio = self._codec.decode(latent.to(self.device))
            self.logger.experiment.add_audio(tag, audio[0].reshape(-1).cpu(), self.val_idx, self.sr)
        self.val_idx += 1


# --------------------------------------------------------------------------- #
#  Streaming (embedded in the exported RAVE)
# --------------------------------------------------------------------------- #

class CAMStreamer(nn.Module):
    """TorchScript-able streaming generator of full latents, embedded in ScriptedRAVE.

    The state (key / value caches of the transformer, last contexts) lives in buffers; it is reset when the
    batch size changes. The attention cache is exactly the attention window of the model."""

    def __init__(self, core: CAMCore, ratio: int):
        super().__init__()
        if core.net.window <= 0:
            raise ValueError("streaming requires a finite local_attention_size")
        self.core = core.eval()
        self.latent_size = core.latent_size
        self.ratio = int(ratio)
        n_layers, n_heads, head_dim = core.net.n_layers, core.net.n_heads, core.net.head_dim
        self.register_buffer("cache", torch.zeros(n_layers, 2, 1, n_heads, 0, head_dim))
        self.register_buffer("cache_listen", torch.zeros(n_layers, 2, 1, n_heads, 0, head_dim))
        self.register_buffer("context", core.sos.sos.detach().clone())
        self.register_buffer("context_listen", core.sos.sos.detach().clone())
        self.batch_size = 0

    @torch.jit.export
    def reset(self):
        self.batch_size = 0
        self.cache = self.cache[:, :, :, :, :0]
        self.cache_listen = self.cache_listen[:, :, :, :, :0]

    def _prepare(self, x: torch.Tensor):
        b = x.shape[0]
        if self.batch_size != b:
            n_layers, _, _, n_heads, _, head_dim = self.cache.shape
            self.cache = torch.zeros(n_layers, 2, b, n_heads, 0, head_dim, device=x.device, dtype=x.dtype)
            self.cache_listen = torch.zeros(n_layers, 2, b, n_heads, 0, head_dim, device=x.device,
                                            dtype=x.dtype)
            self.context = self.core.sos.sos.to(x).repeat(b, 1)
            self.context_listen = self.core.sos.sos.to(x).repeat(b, 1)
            self.batch_size = b

    @torch.jit.export
    def generate(self, temp: torch.Tensor, steps: int) -> torch.Tensor:
        """temp: (B, 1, T) control signal of the temperature (mapped as in nn~: softplus / log 2, so that 0 is 1).
        returns (B, latent_size, T)."""
        self._prepare(temp)
        temperature = float(nn.functional.softplus(temp.mean()) / math.log(2))
        frames = torch.jit.annotate(List[torch.Tensor], [])
        for _ in range(temp.shape[-1]):
            frame = self.core.sample_frame(self.context, steps, temperature)
            contexts, self.cache = self.core.net.step(frame.unsqueeze(-1), self.cache)
            self.context = contexts[..., 0]
            frames.append(frame.unsqueeze(-1))
        return torch.cat(frames, dim=-1)

    @torch.jit.export
    def generate_listen(self, z: torch.Tensor, steps: int, temperature: float, weight: float) -> torch.Tensor:
        """the generation follows a stream of full latents z (B, latent_size, T): the context that the stream
        gives to the next frame guides the flow, in the velocity space (weight 0: free prior; 1: the guide;
        > 1: extrapolation, like classifier-free guidance). The output frame t is guided by z[..., :t]."""
        self._prepare(z)
        listen_contexts, self.cache_listen = self.core.net.step(z, self.cache_listen)
        frames = torch.jit.annotate(List[torch.Tensor], [])
        for t in range(z.shape[-1]):
            guide = self.context_listen if t == 0 else listen_contexts[..., t - 1]
            frame = self.core.sample_frame(self.context, steps, temperature, guide, weight)
            contexts, self.cache = self.core.net.step(frame.unsqueeze(-1), self.cache)
            self.context = contexts[..., 0]
            frames.append(frame.unsqueeze(-1))
        self.context_listen = listen_contexts[..., -1]
        return torch.cat(frames, dim=-1)


def streamer_from_checkpoint(checkpoint_path: str, use_ema: bool = True) -> Tuple[CAMStreamer, int]:
    """CAMStreamer from the checkpoint of a CAMPrior run (the config.gin of the run must have been parsed).
    Returns (streamer, latent_size)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k[len("core."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("core.")}
    latent_size = state["net.embed.0.weight"].shape[1]
    context_dim = state["net.proj.weight"].shape[0]
    core = CAMCore(latent_size, context_dim)
    core.load_state_dict(state)
    if use_ema and "cam_ema" in checkpoint:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(core.parameters(), decay=0.999)
        ema.load_state_dict(checkpoint["cam_ema"])
        ema.copy_to(core.parameters())
    ratio = int(checkpoint["state_dict"]["ratio"])
    return CAMStreamer(core, ratio), latent_size
