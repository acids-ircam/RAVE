import functools
import multiprocessing
import pathlib
import subprocess
from datetime import timedelta
from functools import partial
from itertools import repeat
from typing import Callable, Iterable, Tuple

import lmdb
import numpy as np
import os
import torch
from effortless_config import Config
from tqdm import tqdm
from udls.generated import AudioExample

torch.set_grad_enabled(False)


def float_array_to_int16_bytes(x):
    return np.floor(x * (2**15 - 1)).astype(np.int16).tobytes()


class args(Config):
    INPUT_PATH = None
    OUTPUT_PATH = None
    NUM_SIGNAL = 131072
    SAMPLING_RATE = 48000
    MAX_DB_SIZE = 100
    EXT = ['wav', 'opus', 'mp3', 'aac', 'flac']


def load_audio_chunk(path: str, n_signal: int,
                     sr: int) -> Iterable[np.ndarray]:
    process = subprocess.Popen(
        [
            'ffmpeg', '-hide_banner', '-loglevel', 'panic', '-i', path, '-ac',
            '1', '-ar',
            str(sr), '-f', 's16le', '-'
        ],
        stdout=subprocess.PIPE,
    )
    chunk = process.stdout.read(n_signal * 2)

    while len(chunk) == n_signal * 2:
        yield chunk
        chunk = process.stdout.read(n_signal * 2)

    process.stdout.close()


def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm


def process_audio_array(audio: Tuple[int, bytes],
                        env: lmdb.Environment) -> int:
    audio_id, audio_samples = audio

    buffers = {}
    buffers['waveform'] = AudioExample.AudioBuffer(
        sampling_rate=args.SAMPLING_RATE,
        data=audio_samples,
        precision=AudioExample.Precision.INT16,
    )

    ae = AudioExample(buffers=buffers)
    key = f'{audio_id:08d}'
    with env.begin(write=True) as txn:
        txn.put(
            key.encode(),
            ae.SerializeToString(),
        )
    return audio_id


def flatmap(pool: multiprocessing.Pool,
            func: Callable,
            iterable: Iterable,
            chunksize=None):
    queue = multiprocessing.Manager().Queue(maxsize=os.cpu_count())
    pool.map_async(
        functools.partial(flat_mappper, func),
        zip(iterable, repeat(queue)),
        chunksize,
        lambda _: queue.put(None),
        lambda *e: print(e),
    )

    item = queue.get()
    while item is not None:
        yield item
        item = queue.get()


def flat_mappper(func, arg):
    data, queue = arg
    for item in func(data):
        queue.put(item)


def main():
    args.parse_args()

    assert args.OUTPUT_PATH is not None, "You must define an output path !"
    assert args.INPUT_PATH is not None, "You must define an input path !"

    chunk_load = partial(load_audio_chunk,
                         n_signal=args.NUM_SIGNAL,
                         sr=args.SAMPLING_RATE)

    # create database
    env = lmdb.open(args.OUTPUT_PATH, map_size=args.MAX_DB_SIZE * 1024**3)
    pool = multiprocessing.Pool()

    # search for audio files
    audios = flatten(
        map(
            lambda ext: pathlib.Path(args.INPUT_PATH).rglob(f'*.{ext}'),
            args.EXT,
        ))
    audios = list(map(str, audios))

    # load chunks
    chunks = flatmap(pool, chunk_load, audios)
    chunks = enumerate(chunks)

    # apply rave on dataset
    processed_samples = map(partial(process_audio_array, env=env), chunks)

    pbar = tqdm(processed_samples)
    for audio_id in pbar:
        n_seconds = args.NUM_SIGNAL / args.SAMPLING_RATE * audio_id

        pbar.set_description(f'dataset length: {timedelta(seconds=n_seconds)}')

    pool.close()