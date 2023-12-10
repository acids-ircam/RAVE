import base64
import logging
import math
import os
import subprocess
from random import random
from typing import Dict, Iterable, Optional, Sequence, Union, Callable

import gin
import lmdb
import numpy as np
import requests
import torch
import torchaudio
import yaml
from scipy.signal import lfilter
from torch.utils import data
from tqdm import tqdm
from . import transforms
from udls import AudioExample as AudioExampleWrapper
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
                 transforms: Optional[transforms.Transform] = None, 
                 n_channels: int = 1) -> None:
        super().__init__()
        self._db_path = db_path
        self._audio_key = audio_key
        self._env = None
        self._keys = None
        self._transforms = transforms
        self._n_channels = n_channels
        lens = []
        with self.env.begin() as txn:
            for k in self.keys:
               ae = AudioExample.FromString(txn.get(k)) 
               lens.append(np.frombuffer(ae.buffers['waveform'].data, dtype=np.int16).shape)


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with self.env.begin() as txn:
            ae = AudioExample.FromString(txn.get(self.keys[index]))

        buffer = ae.buffers[self._audio_key]
        assert buffer.precision == AudioExample.Precision.INT16

        audio = np.frombuffer(buffer.data, dtype=np.int16)
        audio = audio.astype(np.float32) / (2**15 - 1)
        audio = audio.reshape(self._n_channels, -1)

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
                 transforms: Optional[transforms.Transform] = None,
                 n_channels: int = 1) -> None:
        super().__init__()
        self._db_path = db_path
        self._env = None
        self._keys = None
        self._transforms = transforms
        self._n_signal = n_signal
        self._sampling_rate = sampling_rate
        self._n_channels = n_channels

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
            int(ae.metadata['channels']),
            self._n_channels
        )

        if self._transforms is not None:
            audio = self._transforms(audio)

        return audio

def get_channels_from_dataset(db_path):
    with open(os.path.join(db_path, 'metadata.yaml'), 'r') as metadata:
        metadata = yaml.safe_load(metadata)
    return metadata.get('channels')

def get_training_channels(db_path, target_channels):
    dataset_channels = get_channels_from_dataset(db_path)
    if dataset_channels is not None:
        if target_channels > dataset_channels:
            raise RuntimeError('[Error] Requested number of channels is %s, but dataset has %s channels')%(FLAGS.channels, dataset_channels)
    n_channels = target_channels or dataset_channels
    if n_channels is None:
        print('[Warning] channels not found in dataset, taking 1 by default')
        n_channels = 1
    return n_channels

class HTTPAudioDataset(data.Dataset):

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        logging.info("starting remote dataset session")
        self.length = int(requests.get("/".join([db_path, "len"])).text)
        logging.info("connection established !")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        example = requests.get("/".join([
            self.db_path,
            "get",
            f"{index}",
        ])).text
        example = AudioExampleWrapper(base64.b64decode(example)).get("audio")
        return example.copy()


def normalize_signal(x: np.ndarray, max_gain_db: int = 30):
    peak = np.max(abs(x))
    if peak == 0: return x

    log_peak = 20 * np.log10(peak)
    log_gain = min(max_gain_db, -log_peak)
    gain = 10**(log_gain / 20)

    return x * gain

@gin.configurable
def get_dataset(db_path,
                sr,
                n_signal,
                derivative: bool = False,
                normalize: bool = False,
                rand_pitch: bool = False,
                augmentations: Union[None, Iterable[Callable]] = None, 
                n_channels: int = 1):
    if db_path[:4] == "http":
        return HTTPAudioDataset(db_path=db_path)
    with open(os.path.join(db_path, 'metadata.yaml'), 'r') as metadata:
        metadata = yaml.safe_load(metadata)

    sr_dataset = metadata.get('sr', 44100)
    lazy = metadata['lazy']

    transform_list = [
        lambda x: x.astype(np.float32),
        transforms.RandomCrop(n_signal),
        transforms.RandomApply(
            lambda x: random_phase_mangle(x, 20, 2000, .99, sr_dataset),
            p=.8,
        ),
        transforms.Dequantize(16),
    ]

    if rand_pitch:
        rand_pitch = list(map(float, rand_pitch))
        assert len(rand_pitch) == 2, "rand_pitch must be given two floats"
        transform_list.insert(1, transforms.RandomPitch(n_signal, rand_pitch))

    if sr_dataset != sr:
        transform_list.append(transforms.Resample(sr_dataset, sr))

    if normalize:
        transform_list.append(normalize_signal)

    if derivative:
        transform_list.append(get_derivator_integrator(sr)[0])

    if augmentations:
        transform_list.extend(augmentations)

    transform_list.append(lambda x: x.astype(np.float32))

    transform_list = transforms.Compose(transform_list)

    if lazy:
        return LazyAudioDataset(db_path, n_signal, sr_dataset, transform_list, n_channels)
    else:
        return AudioDataset(
            db_path,
            transforms=transform_list,
            n_channels=n_channels
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
                  start_sample: int, input_channels: int, channels: int) -> Iterable[np.ndarray]:
    # channel mapping
    channel_map = range(channels)
    if input_channels < channels:
        channel_map = (math.ceil(channels / input_channels) * list(range(input_channels)))[:channels]
    # time information
    start_sec = start_sample / sr
    length = (n_signal * 2) / sr
    chunks = []
    for i in channel_map:
        process = subprocess.Popen(
            [
                'ffmpeg', '-v', 'error',
                '-ss',
                str(start_sec),
                '-i',
                path,
                '-ar',
                str(sr),
                '-filter_complex',
                'channelmap=%d-0'%i,
                '-t',
                str(length),
                '-f',
                's16le',
                '-'
            ],
            stdout=subprocess.PIPE,
        )

        chunk = process.communicate()[0]
        chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 2**15
        chunk = np.concatenate([chunk, np.zeros(n_signal)], -1)
        chunks.append(chunk)
    return np.stack(chunks)[:, :(n_signal*2)]
