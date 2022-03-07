import torch
import torch.nn as nn

torch.set_grad_enabled(False)
from effortless_config import Config
from rave.core import search_for_run
from prior import Prior
import math

import soundfile as sf


class args(Config):
    PRIOR_CKPT = None  # path to prior .ckpt file
    EXPORTED_RAVE = None  # path to .ts file
    LENGTH = 10  # in second
    OUT_PATH = "unconditional.wav"


args.parse_args()

args.PRIOR_CKPT = search_for_run(args.PRIOR_CKPT)

prior = Prior.load_from_checkpoint(args.PRIOR_CKPT).eval()
rave = torch.jit.load(args.EXPORTED_RAVE).eval()

sr = int(rave.sampling_rate.item())
n_samples = math.ceil(sr * args.LENGTH)

x = torch.zeros(1, 1, n_samples)
z = rave.encode(x).zero_()
z = prior.quantized_normal.encode(z)
z = prior.generate(z)
z = prior.diagonal_shift.inverse(prior.quantized_normal.decode(z))

y = rave.decode(z).reshape(-1).numpy()
sf.write(args.OUT_PATH, y, sr)