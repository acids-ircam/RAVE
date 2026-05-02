"""
Extract RAVE latent representations from a speaker-organised dataset.

Expected directory structure:
  dataset_dir/
    speaker_001/
      utterance_001.wav
      utterance_002.wav
    speaker_002/
      utterance_001.wav   ← same stem = same content (controlled dataset)
      ...

Speaker label : parent directory name
Content label : filename stem  (assumes controlled dataset where the same
                utterance IDs appear across speakers; if your dataset is not
                controlled, content metrics in evaluate_disentanglement.py
                will not be meaningful)
Pitch         : per-utterance mean voiced F0 via librosa.pyin

Usage:
  python scripts/extract_latents.py \\
      --model  runs/my_run \\
      --dataset_dir  data/test \\
      --output  latents_test.npz \\
      --sr 44100
"""

import argparse
import os
import pathlib
import sys

import gin
import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import rave
import rave.core


AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg', '.aac', '.aif', '.aiff'}


def discover_files(dataset_dir):
    """Return list of (filepath, speaker_id, content_id) sorted deterministically."""
    items = []
    root = pathlib.Path(dataset_dir)
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        for audio_file in sorted(speaker_dir.iterdir()):
            if audio_file.suffix.lower() in AUDIO_EXTENSIONS:
                items.append((str(audio_file), speaker_id, audio_file.stem))
    return items


def extract_pitch(audio_np, sr):
    """Return mean voiced F0 (Hz). Returns 0.0 for fully unvoiced audio."""
    f0, voiced, _ = librosa.pyin(
        audio_np,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
    )
    if voiced is not None and voiced.any():
        return float(np.nanmean(f0[voiced]))
    return 0.0


def load_model(model_path, device):
    config_path = rave.core.search_for_config(model_path)
    if config_path is None:
        raise FileNotFoundError(f"config.gin not found under {model_path}")
    gin.parse_config_file(config_path)

    run = rave.core.search_for_run(model_path)
    if run is None:
        raise FileNotFoundError(f"No .ckpt found under {model_path}")

    model = rave.RAVE.load_from_checkpoint(run, map_location=device)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def encode_file(model, path, target_sr, device, use_pca):
    audio, sr = torchaudio.load(path)
    audio = audio.mean(0, keepdim=True)          # force mono [1, T]
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    audio_np = audio.squeeze(0).numpy()
    pitch = extract_pitch(audio_np, target_sr)

    x = audio.unsqueeze(0).to(device)            # [1, 1, T]
    z_raw = model.encode(x)                      # pre-reparameterisation
    z, _ = model.encoder.reparametrize(z_raw)[:2]  # [1, latent_size, T_latent]

    if use_pca:
        z = z - model.latent_mean.unsqueeze(-1)
        z = torch.einsum('bl,lt->bt', model.latent_pca, z.squeeze(0)).unsqueeze(0)

    z_mean = z.squeeze(0).mean(-1).cpu().numpy()  # [latent_size]
    return z_mean, pitch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='Path to RAVE training run directory (contains config.gin + checkpoints)')
    parser.add_argument('--dataset_dir', required=True,
                        help='Root of speaker-organised audio directory')
    parser.add_argument('--output', required=True,
                        help='Output .npz path')
    parser.add_argument('--sr', type=int, default=44100,
                        help='Target sample rate (must match training SR)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_pca', action='store_true',
                        help='Apply the model\'s stored PCA projection to z')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading model from {args.model} on {device} …")
    model = load_model(args.model, device)

    items = discover_files(args.dataset_dir)
    if not items:
        raise RuntimeError(f"No audio files found in {args.dataset_dir}")
    print(f"Found {len(items)} files across "
          f"{len({s for _, s, _ in items})} speakers, "
          f"{len({c for _, _, c in items})} unique content IDs")

    speakers = sorted({s for _, s, _ in items})
    contents = sorted({c for _, _, c in items})
    spk2idx = {s: i for i, s in enumerate(speakers)}
    cnt2idx = {c: i for i, c in enumerate(contents)}

    z_list, spk_list, cnt_list, pitch_list, file_list = [], [], [], [], []

    for path, speaker, content in tqdm(items, desc='Encoding'):
        try:
            z, pitch = encode_file(model, path, args.sr, device, args.use_pca)
        except Exception as exc:
            print(f"  [skip] {path}: {exc}")
            continue
        z_list.append(z)
        spk_list.append(spk2idx[speaker])
        cnt_list.append(cnt2idx[content])
        pitch_list.append(pitch)
        file_list.append(path)

    z_arr = np.stack(z_list)
    np.savez(
        args.output,
        z=z_arr,
        speaker=np.array(spk_list, dtype=np.int32),
        content=np.array(cnt_list, dtype=np.int32),
        pitch=np.array(pitch_list, dtype=np.float32),
        files=np.array(file_list),
        speaker_names=np.array(speakers),
        content_names=np.array(contents),
    )
    print(f"\nSaved {len(z_list)} samples → {args.output}")
    print(f"  z shape      : {z_arr.shape}")
    print(f"  Speakers     : {len(speakers)}")
    print(f"  Content IDs  : {len(contents)}")
    print(f"  Pitch range  : {np.array(pitch_list).min():.1f}–{np.array(pitch_list).max():.1f} Hz")


if __name__ == '__main__':
    main()
