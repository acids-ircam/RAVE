from typing import Dict, Optional, Sequence
import torch
from udls.generated import AudioExample
import numpy as np
from udls import transforms
from torch.utils import data
from random import random
import lmdb
from scipy.signal import lfilter


class AudioDataset(data.Dataset):

    @property
    def env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self._db_path)
        return self._env

    @property
    def keys(self) -> Sequence[str]:
        if self._keys is None:
            with self.env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        return self.keys

    def __init__(self,
                 db_path: str,
                 audio_key: str = 'waveform',
                 transforms: Optional[transforms.Transform] = None) -> None:
        super().__init__()
        self._db_path = db_path
        self._audio_key = audio_key
        self._env = None
        self._keys = None
        self._transforms = transforms

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, index):
        with self.env.begin() as txn:
            ae = AudioExample.FromString(txn.get(self._keys[index]))

        buffer = ae.buffers[self._audio_key]
        assert buffer.precision == AudioExample.Precision.INT16

        audio = np.frombuffer(buffer.data, dtype=np.int16)
        audio = audio.astype(np.float) / (2**15 - 1)

        if self._transforms is not None:
            audio = self._transforms(audio)

        return audio


def get_dataset(db_path, sr, n_signal):
    return AudioDataset(
        db_path,
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
