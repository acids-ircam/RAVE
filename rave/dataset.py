import math
import os
import subprocess
from random import random
from typing import Dict, Iterable, Optional, Sequence

import gin
import lmdb
import numpy as np
import torch
import yaml
from scipy.signal import lfilter
from torch.utils import data
from tqdm import tqdm
from udls import transforms
from udls.generated import AudioExample


def get_derivator_integrator(sr: int):
    alpha = 1 / (1 + 1 / sr * 2 * np.pi * 10)
    derivator = ([.5, -.5], [1])
    integrator = ([alpha**2, -alpha**2], [1, -2 * alpha, alpha**2])

    return lambda x: lfilter(*derivator, x), lambda x: lfilter(*integrator, x)


class AudioDataset(data.Dataset):

    @property
    def env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self._db_path, lock=False)
        return self._env

    @property
    def keys(self) -> Sequence[str]:
        if self._keys is None:
            with self.env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        return self._keys

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
        return len(self.keys)

    def __getitem__(self, index):
        with self.env.begin() as txn:
            ae = AudioExample.FromString(txn.get(self.keys[index]))

        buffer = ae.buffers[self._audio_key]
        assert buffer.precision == AudioExample.Precision.INT16

        audio = np.frombuffer(buffer.data, dtype=np.int16)
        audio = audio.astype(np.float) / (2**15 - 1)

        if self._transforms is not None:
            audio = self._transforms(audio)

        return audio


class LazyAudioDataset(data.Dataset):

    @property
    def env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self._db_path, lock=False)
        return self._env

    @property
    def keys(self) -> Sequence[str]:
        if self._keys is None:
            with self.env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        return self._keys

    def __init__(self,
                 db_path: str,
                 n_signal: int,
                 sampling_rate: int,
                 transforms: Optional[transforms.Transform] = None) -> None:
        super().__init__()
        self._db_path = db_path
        self._env = None
        self._keys = None
        self._transforms = transforms
        self._n_signal = n_signal
        self._sampling_rate = sampling_rate

        self.parse_dataset()

    def parse_dataset(self):
        items = []
        for key in tqdm(self.keys, desc='Discovering dataset'):
            with self.env.begin() as txn:
                ae = AudioExample.FromString(txn.get(key))
            length = float(ae.metadata['length'])
            n_signal = int(math.floor(length * self._sampling_rate))
            n_chunks = n_signal // self._n_signal
            items.append(n_chunks)
        items = np.asarray(items)
        items = np.cumsum(items)
        self.items = items

    def __len__(self):
        return self.items[-1]

    def __getitem__(self, index):
        audio_id = np.where(index < self.items)[0][0]
        if audio_id:
            index -= self.items[audio_id - 1]

        key = self.keys[audio_id]

        with self.env.begin() as txn:
            ae = AudioExample.FromString(txn.get(key))

        audio = extract_audio(
            ae.metadata['path'],
            self._n_signal,
            self._sampling_rate,
            index * self._n_signal,
        )

        if self._transforms is not None:
            audio = self._transforms(audio)

        return audio


def normalize_signal(x: np.ndarray, max_gain_db: int = 30):
    peak = np.max(abs(x))
    if peak == 0: return x

    log_peak = 20 * np.log10(peak)
    log_gain = min(max_gain_db, -log_peak)
    gain = 10**(log_gain / 20)

    return x * gain


def get_dataset(db_path,
                sr,
                n_signal,
                derivative: bool = False,
                normalize: bool = False):
    with open(os.path.join(db_path, 'metadata.yaml'), 'r') as metadata:
        metadata = yaml.safe_load(metadata)
    lazy = metadata['lazy']

    transform_list = [
        lambda x: x.astype(np.float32),
        transforms.RandomCrop(n_signal),
        transforms.RandomApply(
            lambda x: random_phase_mangle(x, 20, 2000, .99, sr),
            p=.8,
        ),
        transforms.Dequantize(16),
    ]

    if normalize:
        transform_list.append(normalize_signal)

    if derivative:
        transform_list.append(get_derivator_integrator(sr)[0])

    transform_list.append(lambda x: x.astype(np.float32))

    transform_list = transforms.Compose(transform_list)

    if lazy:
        return LazyAudioDataset(db_path, n_signal, sr, transform_list)
    else:
        return AudioDataset(
            db_path,
            transforms=transform_list,
        )


@gin.configurable
def split_dataset(dataset, percent, max_residual: Optional[int] = None):
    split1 = max((percent * len(dataset)) // 100, 1)
    split2 = len(dataset) - split1
    if max_residual is not None:
        split2 = min(max_residual, split2)
        split1 = len(dataset) - split2
    print(f'train set: {split1} examples')
    print(f'val set: {split2} examples')
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


def extract_audio(path: str, n_signal: int, sr: int,
                  start_sample: int) -> Iterable[np.ndarray]:
    start_sec = start_sample / sr
    length = n_signal / sr + 0.1
    process = subprocess.Popen(
        [
            'ffmpeg',
            '-v',
            'error',
            '-ss',
            str(start_sec),
            '-i',
            path,
            '-ar',
            str(sr),
            '-ac',
            '1',
            '-t',
            str(length),
            '-f',
            's16le',
            '-',
        ],
        stdout=subprocess.PIPE,
    )

    chunk = process.communicate()[0]

    chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 2**15
    chunk = np.concatenate([chunk, np.zeros(n_signal)], -1)
    return chunk[:n_signal]
