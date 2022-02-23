import torch
import torch.nn as nn
import torch.fft as fft
from einops import rearrange
import numpy as np
from random import random, randint
from scipy.signal import lfilter
from pytorch_lightning.callbacks import ModelCheckpoint
from udls.transforms import Transform
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


def random_phase_mangle(x, min_f, max_f, amp, sr, axis=-1):
    angle = random_angle(min_f, max_f, sr)
    b, a = pole_to_z_filter(angle, amp)
    return lfilter(b, a, x, axis)

def simple_audio_preprocess(sampling_rate, N, mono=True):
    def preprocess(name):
        try:
            x, sr = li.load(name, sr=sampling_rate, mono=mono)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            return None

        if mono:
            pad = (N - (len(x) % N)) % N
            x = np.pad(x, (0, pad))

            x = x.reshape(-1, N)

            x = np.expand_dims(x, axis=1)
        else:
            pad = (N - (x.shape[1] % N)) % N
            x = np.pad(x, ((0, 0), (0, pad)))
            x = np.expand_dims(x, axis=0)
            x = np.split(x, x.shape[2] / N, axis=2)
            x = np.concatenate(x, axis=0)

        return x.astype(np.float32)

    return preprocess

class RandomCrop(Transform):
    """
    Randomly crops signal to fit n_signal samples
    """

    def __init__(self, n_signal):
        self.n_signal = n_signal

    def __call__(self, x: np.ndarray):
        in_point = randint(0, x.shape[-1] - self.n_signal)
        x = x[:, in_point:in_point + self.n_signal]
        return x


class Dequantize(Transform):
    def __init__(self, bit_depth):
        self.bit_depth = bit_depth

    def __call__(self, x: np.ndarray):
        x += np.random.rand(x.size).reshape(x.shape) / 2**self.bit_depth
        return x


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
    def __init__(self, sr, block_size, n_fft=2048, a_n_channels=1):
        super().__init__()
        self.sr = sr
        self.block_size = block_size
        self.n_fft = n_fft
        self.a_n_channels = a_n_channels

        f = np.linspace(0, sr / 2, n_fft // 2 + 1) + 1e-7
        a_weight = li.A_weighting(f).reshape(-1, 1)

        self.register_buffer("a_weight", torch.from_numpy(a_weight).float())
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, x):
        batch_size = x.size(0)

        if self.a_n_channels > 1:
            x = torch.cat(torch.split(x, 1, dim=1), dim=0)

        x = torch.stft(
            x.squeeze(1),
            self.n_fft,
            self.block_size,
            self.n_fft,
            center=True,
            window=self.window,
            return_complex=True,
        ).abs()

        x = x.unsqueeze(1)

        if self.a_n_channels > 1:
            x = torch.cat(torch.split(x, batch_size, dim=0), dim=1)

        x = torch.log(x + 1e-7) + self.a_weight
        return torch.mean(x, 2, keepdim=True)


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
    if run_path is None:
        return None

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
        if len(run_path):
            run_path = run_path[-1]
            run_path = path.join(run_path, "checkpoints", "*.ckpt")
            run_path = glob(run_path)
            run_path = list(filter(lambda e: "last" in e, run_path))[-1]
        else:
            run_path = None
    return run_path


def get_beta_kl(step, warmup, min_beta, max_beta):
    if step > warmup: return max_beta
    t = step / warmup
    min_beta_log = np.log(min_beta)
    max_beta_log = np.log(max_beta)
    beta_log = t * (max_beta_log - min_beta_log) + min_beta_log
    return np.exp(beta_log)


def get_beta_kl_cyclic(step, cycle_size, min_beta, max_beta):
    return get_beta_kl(step % cycle_size, cycle_size // 2, min_beta, max_beta)


def get_beta_kl_cyclic_annealed(step, cycle_size, warmup, min_beta, max_beta):
    min_beta = get_beta_kl(step, warmup, min_beta, max_beta)
    return get_beta_kl_cyclic(step, cycle_size, min_beta, max_beta)
