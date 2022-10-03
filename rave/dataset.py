import gin
import torch
from udls.generated import AudioExample
import librosa as li
import numpy as np
import udls
from udls import transforms
from torch.utils import data
from random import random
import lmdb


@gin.configurable
def simple_audio_preprocess(sampling_rate, N, crop=False, trim_silence=False):

    def preprocess(name):
        try:
            x, sr = li.load(name, sr=sampling_rate)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            return None

        if trim_silence:
            try:
                x = np.concatenate(
                    [x[e[0]:e[1]] for e in li.effects.split(x, 50)],
                    -1,
                )
            except Exception as e:
                print(e)
                return None

        if crop:
            crop_size = len(x) % N
            if crop_size:
                x = x[:-crop_size]
        else:
            pad = (N - (len(x) % N)) % N
            x = np.pad(x, (0, pad))

        if not len(x):
            return None

        x = x.reshape(-1, N)
        return x.astype(np.float16)

    return preprocess


def get_dataset(data_dir, preprocess_dir, sr, n_signal):
    dataset = udls.SimpleDataset(
        preprocess_dir,
        data_dir,
        preprocess_function=simple_audio_preprocess(sr, 2 * n_signal),
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
    split1, split2 = data.random_split(
        dataset,
        [split1, split2],
        generator=torch.Generator().manual_seed(42),
    )
    return split1, split2


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
