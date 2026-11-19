"""Generate synthetic multichannel test datasets.

Creates <target>/{mono,stereo,4ch,mixed}, each with ~n_files wav files of
random lengths. Each channel gets a distinct signal so channels are
distinguishable after preprocessing.

usage: python tests/generate_test_datasets.py <target> [--n_files 50] [--sr 44100]
"""
import argparse
import os

import numpy as np
from scipy.io import wavfile

DATASETS = {"mono": [1], "stereo": [2], "4ch": [4], "mixed": [1, 2, 3, 4]}


def make_channel(n, sr, rng):
    t = np.arange(n) / sr
    kind = rng.choice(["sine", "sweep", "harmonic", "noise"])
    if kind == "sine":
        x = np.sin(2 * np.pi * rng.uniform(50, 2000) * t)
    elif kind == "sweep":
        f0, f1 = rng.uniform(50, 500), rng.uniform(1000, 8000)
        dur = max(t[-1], 1e-3) if n else 1.
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * dur))
        x = np.sin(phase)
    elif kind == "harmonic":
        f0 = rng.uniform(60, 400)
        x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 8))
    else:
        x = rng.standard_normal(n)
    # slow amplitude envelope
    env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 3) * t + rng.uniform(0, 2 * np.pi))
    x = x * env
    return 0.8 * x / (np.abs(x).max() + 1e-8)


def random_length(sr, rng):
    # mostly log-uniform 0.5s-30s, with a few very short files (< 1 chunk)
    if rng.random() < 0.1:
        return int(rng.uniform(0.05, 0.5) * sr)
    return int(np.exp(rng.uniform(np.log(0.5), np.log(30))) * sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--n_files", type=int, default=50)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for name, channel_choices in DATASETS.items():
        out_dir = os.path.join(args.target, name)
        os.makedirs(out_dir, exist_ok=True)
        total = 0.
        for i in range(args.n_files):
            n_ch = int(rng.choice(channel_choices))
            n = random_length(args.sr, rng)
            audio = np.stack([make_channel(n, args.sr, rng) for _ in range(n_ch)], -1)
            path = os.path.join(out_dir, f"{name}_{i:03d}_{n_ch}ch.wav")
            wavfile.write(path, args.sr, audio.astype(np.float32))
            total += n / args.sr
        print(f"{out_dir}: {args.n_files} files, {total / 60:.1f} min")


if __name__ == "__main__":
    main()
