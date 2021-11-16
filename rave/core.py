import torch
import torch.nn as nn
import torch.fft as fft
from einops import rearrange
import numpy as np
from random import random
from scipy.signal import lfilter
from pytorch_lightning.callbacks import ModelCheckpoint
import librosa as li
import math
from os import path
from glob import glob


def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7


def multiscale_stft(signal, scales, overlap):
    """
    Compute a stft on several scales, with a constant overlap value.
    Parameters
    ----------
    signal: torch.Tensor
        input signal to process ( B X C X T )
    
    scales: list
        scales to use
    overlap: float
        overlap between windows ( 0 - 1 )
    """
    signal = rearrange(signal, "b c t -> (b c) t")
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def random_angle(min_f=20, max_f=8000, sr=24000):
    min_f = np.log(min_f)
    max_f = np.log(max_f)
    rand = np.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * np.pi * rand / sr
    return rand


def pole_to_z_filter(omega, amplitude=.9):
    z0 = amplitude * np.exp(1j * omega)
    a = [1, -2 * np.real(z0), abs(z0)**2]
    b = [abs(z0)**2, -2 * np.real(z0), 1]
    return b, a


def random_phase_mangle(x, min_f, max_f, amp, sr):
    angle = random_angle(min_f, max_f, sr)
    b, a = pole_to_z_filter(angle, amp)
    return lfilter(b, a, x)


class EMAModelCheckPoint(ModelCheckpoint):
    def __init__(self, model: torch.nn.Module, alpha=.999, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()
        self.model = model
        self.alpha = alpha

    def on_train_batch_end(self, *args, **kwargs):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.shadow:
                    self.shadow[n] *= self.alpha
                    self.shadow[n] += (1 - self.alpha) * p.data

    def on_validation_epoch_start(self, *args, **kwargs):
        self.swap()

    def on_validation_epoch_end(self, *args, **kwargs):
        self.swap()

    def swap(self):
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                tmp = p.data.clone()
                p.data.copy_(self.shadow[n])
                self.shadow[n] = tmp

    def save_checkpoint(self, *args, **kwargs):
        self.swap()
        super().save_checkpoint(*args, **kwargs)
        self.swap()


class Loudness(nn.Module):
    def __init__(self, sr, block_size, n_fft=2048):
        super().__init__()
        self.sr = sr
        self.block_size = block_size
        self.n_fft = n_fft

        f = li.fft_frequencies(sr, n_fft) + 1e-7
        a_weight = li.A_weighting(f).reshape(-1, 1)

        self.register_buffer("a_weight", torch.from_numpy(a_weight).float())
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, x):
        x = torch.stft(
            x.squeeze(1),
            self.n_fft,
            self.block_size,
            self.n_fft,
            center=True,
            window=self.window,
            return_complex=True,
        ).abs()
        x = torch.log(x + 1e-7) + self.a_weight
        return torch.mean(x, 1, keepdim=True)


def amp_to_impulse_response(amp, target_size):
    """
    transforms frequecny amps to ir on the last dimension
    """
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(
        amp,
        (0, int(target_size) - int(filter_size)),
    )
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    """
    convolves signal by kernel on the last dimension
    """
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


def search_for_run(run_path):
    if ".ckpt" in run_path:
        pass
    elif "checkpoints" in run_path:
        run_path = path.join(run_path, "*.ckpt")
        run_path = glob(run_path)
        run_path = list(filter(lambda e: "last" in e, run_path))[-1]
    elif "version" in run_path:
        run_path = path.join(run_path, "checkpoints", "*.ckpt")
        run_path = glob(run_path)
        run_path = list(filter(lambda e: "last" in e, run_path))[-1]
    else:
        run_path = glob(path.join(run_path, "*"))
        run_path.sort()
        run_path = run_path[-1]
        run_path = path.join(run_path, "checkpoints", "*.ckpt")
        run_path = glob(run_path)
        run_path = list(filter(lambda e: "last" in e, run_path))[-1]
    return run_path
