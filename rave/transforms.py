from random import choice, randint, random, randrange
import bisect
import torchaudio
import gin.torch
from typing import Tuple
import librosa as li
import numpy as np
import torch
import scipy.signal as signal
from udls.transforms import *


class Transform(object):
    def __call__(self, x: torch.Tensor):
        raise NotImplementedError


class RandomApply(Transform):
    """
    Apply transform with probability p
    """
    def __init__(self, transform, p=.5):
        self.transform = transform
        self.p = p

    def __call__(self, x: np.ndarray):
        if random() < self.p:
            x = self.transform(x)
        return x

class Resample(Transform):
    """
    Resample target signal to target sample rate.
    """
    def __init__(self, orig_sr: int, target_sr: int):
        self.orig_sr = orig_sr
        self.target_sr = target_sr

    def __call__(self, x: np.ndarray):
        return torchaudio.functional.resample(torch.from_numpy(x).float(), self.orig_sr, self.target_sr).numpy()


class Compose(Transform):
    """
    Apply a list of transform sequentially
    """
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, x: np.ndarray):
        for elm in self.transform_list:
            x = elm(x)
        return x


class RandomPitch(Transform):
    def __init__(self, n_signal, pitch_range = [0.7, 1.3], max_factor: int = 20, prob: float = 0.5):
        self.n_signal = n_signal
        self.pitch_range = pitch_range
        self.factor_list, self.ratio_list = self._get_factors(max_factor, pitch_range)
        self.prob = prob

    def _get_factors(self, factor_limit, pitch_range):
        factor_list = []
        ratio_list = []
        for x in range(1, factor_limit):
            for y in range(1, factor_limit):
                if (x==y):
                    continue
                factor = x / y
                if factor <= pitch_range[1] and factor >= pitch_range[0]:
                    i = bisect.bisect_left(factor_list, factor)
                    factor_list.insert(i, factor)
                    ratio_list.insert(i, (x, y))
        return factor_list, ratio_list

    def __call__(self, x: np.ndarray):
        perform_pitch = bool(torch.bernoulli(torch.tensor(self.prob)))
        if not perform_pitch:
            return x
        random_range = list(self.pitch_range)
        random_range[1] = min(random_range[1], x.shape[-1] / self.n_signal)
        random_pitch = random() * (random_range[1] - random_range[0]) + random_range[0]
        ratio_idx = bisect.bisect_left(self.factor_list, random_pitch)
        if ratio_idx == len(self.factor_list):
            ratio_idx -= 1
        up, down = self.ratio_list[ratio_idx]
        x_pitched = signal.resample_poly(x, up, down, padtype='mean', axis=-1)
        return x_pitched


class RandomCrop(Transform):
    """
    Randomly crops signal to fit n_signal samples
    """
    def __init__(self, n_signal):
        self.n_signal = n_signal

    def __call__(self, x: np.ndarray):
        in_point = randint(0, x.shape[-1] - self.n_signal)
        x = x[..., in_point:in_point + self.n_signal]
        return x


class Dequantize(Transform):
    def __init__(self, bit_depth):
        self.bit_depth = bit_depth

    def __call__(self, x: np.ndarray):
        x += np.random.rand(*x.shape) / 2**self.bit_depth
        return x


@gin.configurable
class Compress(Transform):
    def __init__(self, time="0.1,0.1", lookup="6:-70,-60,-20 ", gain="0", sr=44100):
        self.sox_args = ['compand', time, lookup, gain]
        self.sr = sr

    def __call__(self, x: torch.Tensor):
        x = torchaudio.sox_effects.apply_effects_tensor(torch.from_numpy(x).float(), self.sr, [self.sox_args])[0].numpy()
        return x

@gin.configurable
class RandomCompress(Transform):
    def __init__(self, threshold = -40, amp_range = [-60, 0], attack=0.1, release=0.1, prob=0.8, sr=44100):
        assert prob >= 0. and prob <= 1., "prob must be between 0. and 1."
        self.amp_range = amp_range
        self.threshold = threshold
        self.attack = attack
        self.release = release
        self.prob = prob
        self.sr = sr

    def __call__(self, x: torch.Tensor):
        perform = bool(torch.bernoulli(torch.full((1,), self.prob)))
        if perform:
            amp_factor = torch.rand((1,)) * (self.amp_range[1] - self.amp_range[0]) + self.amp_range[0]
            x_aug = torchaudio.sox_effects.apply_effects_tensor(torch.from_numpy(x).float(),
                                                            self.sr,
                                                             [['compand', f'{self.attack},{self.release}', f'6:-80,{self.threshold},{float(amp_factor)}']]
                                                            )[0].numpy()
            return x_aug
        else:
            return x

@gin.configurable
class RandomGain(Transform):
    def __init__(self, gain_range: Tuple[int, int] = [-6, 3], prob: float = 0.5, limit = True):
        assert prob >= 0. and prob <= 1., "prob must be between 0. and 1."
        self.gain_range = gain_range
        self.prob = prob
        self.limit = limit

    def __call__(self, x: torch.Tensor):
        perform = bool(torch.bernoulli(torch.full((1,), self.prob)))
        if perform:
            gain_factor = np.random.rand(1)[None, None][0] * (self.gain_range[1] - self.gain_range[0]) + self.gain_range[0]
            amp_factor = np.power(10, gain_factor / 20)
            x_amp = x * amp_factor
            if (self.limit) and (x_amp.abs().max() > 1): 
                x_amp = x_amp / x_amp.abs().max()
            return x
        else:
            return x


@gin.configurable
class RandomMute(Transform):
    def __init__(self, prob: torch.Tensor = 0.1):
        assert prob >= 0. and prob <= 1., "prob must be between 0. and 1."
        self.prob = prob

    def __call__(self, x: torch.Tensor):
        mask = torch.bernoulli(torch.full((x.shape[0],), 1 - self.prob))
        mask = np.random.binomial(1, 1-self.prob, size=1)
        return x * mask


@gin.configurable
class FrequencyMasking(Transform):
    def __init__(self, prob = 0.5, max_size: int = 80):
        self.prob = prob
        self.max_size = max_size

    def __call__(self, x: torch.Tensor):
        perform = bool(torch.bernoulli(torch.full((1,), self.prob)))
        if not perform:
            return x
        spectrogram = signal.stft(x, nperseg=4096)[2]
        mask_size = randrange(1, self.max_size)
        freq_idx = randrange(0, spectrogram.shape[-2] - mask_size)
        spectrogram[..., freq_idx:freq_idx+mask_size, :] = 0
        x_inv = signal.istft(spectrogram)[1]
        return x_inv
            


# Utilitary for GIN recording of augmentations


_augmentations = []

@gin.configurable()
def add_augmentation(aug):
    global _augmentations
    _augmentations.append(aug)

def get_augmentations():
    return _augmentations