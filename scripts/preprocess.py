import functools
import multiprocessing
import os
import pathlib
import subprocess
from datetime import timedelta
from functools import partial
from itertools import repeat
from typing import Callable, Iterable, Tuple
from absl import flags, app

import lmdb
import numpy as np
import torch
from tqdm import tqdm
from udls.generated import AudioExample

torch.set_grad_enabled(False)

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path',
                    None,
                    help='Path to a directory containing audio files',
                    required=True)
flags.DEFINE_string('output_path',
                    None,
                    help='Output directory for the dataset',
                    required=True)
flags.DEFINE_integer('num_signal',
                     131072,
                     help='Number of audio samples to use during training')
flags.DEFINE_integer('sampling_rate',
                     48000,
                     help='Sampling rate to use during training')
flags.DEFINE_integer('max_db_size',
                     100,
                     help='Maximum size (in GB) of the dataset')
flags.DEFINE_multi_string(
    'ext',
    default=['wav', 'opus', 'mp3', 'aac', 'flac'],
    help='Extension to search for in the input directory')


def float_array_to_int16_bytes(x):
    return np.floor(x * (2**15 - 1)).astype(np.int16).tobytes()


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
        sampling_rate=FLAGS.sampling_rate,
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
    app.run(preprocess)


def preprocess(argv):
    chunk_load = partial(load_audio_chunk,
                         n_signal=FLAGS.num_signal,
                         sr=FLAGS.sampling_rate)

    # create database
    env = lmdb.open(FLAGS.output_path, map_size=FLAGS.max_db_size * 1024**3)
    pool = multiprocessing.Pool()

    # search for audio files
    audios = flatten(
        map(
            lambda ext: pathlib.Path(FLAGS.input_path).rglob(f'*.{ext}'),
            FLAGS.ext,
        ))
    audios = list(map(str, audios))

    # load chunks
    chunks = flatmap(pool, chunk_load, audios)
    chunks = enumerate(chunks)

    # apply rave on dataset
    processed_samples = map(partial(process_audio_array, env=env), chunks)

    pbar = tqdm(processed_samples)
    for audio_id in pbar:
        n_seconds = FLAGS.num_signal / FLAGS.sampling_rate * audio_id

        pbar.set_description(f'dataset length: {timedelta(seconds=n_seconds)}')

    pool.close()


if __name__ == '__main__':
    main()