import torch
import torch.nn as nn
import torch.fft as fft
from einops import rearrange
import numpy as np
from random import random
from scipy.signal import lfilter
import librosa as li
from pathlib import Path
import gin
import udls
import udls.transforms as transforms
from torch.utils.data import random_split
import GPUtil as gpu
import os


def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7


def copy_config(source, destination):
    with open(source, "r") as source:
        with open(destination, "w") as destination:
            for l in source.read():
                destination.write(l)


@gin.configurable
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


@gin.configurable
class Loudness(nn.Module):

    def __init__(self, sr, block_size, n_fft=2048):
        super().__init__()
        self.sr = sr
        self.block_size = block_size
        self.n_fft = n_fft

        f = np.linspace(0, sr / 2, n_fft // 2 + 1) + 1e-7
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


def search_for_run(run_path, mode="last"):
    if run_path is None: return None
    if ".ckpt" in run_path: return run_path
    ckpts = map(str, Path(run_path).rglob("*.ckpt"))
    ckpts = filter(lambda e: mode in e, ckpts)
    ckpts = sorted(ckpts)
    if len(ckpts): return ckpts[-1]
    else: return None


def get_dataset(data_dir, preprocess_dir, sr, n_signal):
    preprocess = lambda name: udls.simple_audio_preprocess(
        sr,
        2 * n_signal,
    )(name).astype(np.float16)

    dataset = udls.SimpleDataset(
        preprocess_dir,
        data_dir,
        preprocess_function=preprocess,
        split_set="full",
        transforms=transforms.Compose([
            lambda x: x.astype(np.float32),
            transforms.RandomCrop(n_signal),
            transforms.RandomApply(
                lambda x: random_phase_mangle(x, 20, 2000, .99, sr),
                p=.8,
            ),
            transforms.Dequantize(16),
            lambda x: x.astype(np.float32),
        ]),
    )

    return dataset


def split_dataset(dataset, percent):
    split1 = max((percent * len(dataset)) // 100, 1)
    split2 = len(dataset) - split1
    split1, split2 = random_split(
        dataset,
        [split1, split2],
        generator=torch.Generator().manual_seed(42),
    )
    return split1, split2


def setup_gpu():
    CUDA = gpu.getAvailable(maxMemory=.05)
    VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if VISIBLE_DEVICES:
        use_gpu = int(int(VISIBLE_DEVICES) >= 0)
    elif len(CUDA):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])
        use_gpu = 1
    elif torch.cuda.is_available():
        print("Cuda is available but no fully free GPU found.")
        print("Training may be slower due to concurrent processes.")
        use_gpu = 1
    else:
        print("No GPU found.")
        use_gpu = 0

    return use_gpu


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


@gin.register
def hinge_gan(score_real, score_fake):
    loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
    loss_dis = loss_dis.mean()
    loss_gen = -score_fake.mean()
    return loss_dis, loss_gen


@gin.register
def ls_gan(score_real, score_fake):
    loss_dis = (score_real - 1).pow(2) + score_fake.pow(2)
    loss_dis = loss_dis.mean()
    loss_gen = (score_fake - 1).pow(2).mean()
    return loss_dis, loss_gen


@gin.register
def nonsaturating_gan(score_real, score_fake):
    score_real = torch.clamp(torch.sigmoid(score_real), 1e-7, 1 - 1e-7)
    score_fake = torch.clamp(torch.sigmoid(score_fake), 1e-7, 1 - 1e-7)
    loss_dis = -(torch.log(score_real) + torch.log(1 - score_fake)).mean()
    loss_gen = -torch.log(score_fake).mean()
    return loss_dis, loss_gen


@torch.enable_grad()
def get_rave_receptive_field(model):
    N = 2**15
    model.eval()
    while True:
        x = torch.randn(1, 1, N, requires_grad=True)
        y = model(x)
        y[0, 0, N // 2].backward()
        grad = x.grad.data.reshape(-1)
        left_grad, right_grad = grad.chunk(2, 0)
        large_enough = (left_grad[0] == 0) and right_grad[-1] == 0
        if large_enough:
            break
        else:
            N *= 2
    left_receptive_field = len(left_grad[left_grad != 0])
    right_receptive_field = len(right_grad[right_grad != 0])
    model.zero_grad()
    return left_receptive_field, right_receptive_field


def valid_signal_crop(x, left_rf, right_rf):
    dim = x.shape[1]
    x = x[..., left_rf // dim:]
    if right_rf:
        x = x[..., :-right_rf // dim]
    return x
