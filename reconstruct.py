import torch

torch.set_grad_enabled(False)

from tqdm import tqdm

from rave import RAVE
from rave.core import search_for_run

from effortless_config import Config
from os import path, makedirs, environ
from pathlib import Path

import librosa as li

import GPUtil as gpu

import soundfile as sf


class args(Config):
    CKPT = None  # PATH TO YOUR PRETRAINED CHECKPOINT
    WAV_FOLDER = None  # PATH TO YOUR WAV FOLDER
    OUT = "./reconstruction/"


args.parse_args()

# GPU DISCOVERY
CUDA = gpu.getAvailable(maxMemory=.05)
if len(CUDA):
    environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])
    use_gpu = 1
elif torch.cuda.is_available():
    print("Cuda is available but no fully free GPU found.")
    print("Reconstruction may be slower due to concurrent processes.")
    use_gpu = 1
else:
    print("No GPU found.")
    use_gpu = 0

device = torch.device("cuda:0" if use_gpu else "cpu")

# LOAD RAVE
rave = RAVE.load_from_checkpoint(
    search_for_run(args.CKPT),
    strict=False,
).eval().to(device)

# COMPUTE LATENT COMPRESSION RATIO
x = torch.randn(1, 1, 2**14).to(device)
z = rave.encode(x)
ratio = x.shape[-1] // z.shape[-1]

# SEARCH FOR WAV FILES
audios = tqdm(list(Path(args.WAV_FOLDER).rglob("*.wav")))

# RECONSTRUCTION
makedirs(args.OUT, exist_ok=True)
for audio in audios:
    audio_name = path.splitext(path.basename(audio))[0]
    audios.set_description(audio_name)

    # LOAD AUDIO TO TENSOR
    x, sr = li.load(audio, sr=rave.sr)
    x = torch.from_numpy(x).reshape(1, 1, -1).float().to(device)

    # PAD AUDIO
    n_sample = x.shape[-1]
    pad = (ratio - (n_sample % ratio)) % ratio
    x = torch.nn.functional.pad(x, (0, pad))

    # ENCODE / DECODE
    y = rave.decode(rave.encode(x))
    y = y.reshape(-1).cpu().numpy()[:n_sample]

    # WRITE AUDIO
    sf.write(path.join(args.OUT, f"{audio_name}_reconstruction.wav"), y, sr)
