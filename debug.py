from effortless_config import Config, setting
from rave import RAVE
import torch
from time import time
import numpy as np
import gin

torch.set_grad_enabled(False)


class args(Config):
    groups = ["small", "large"]

    DATA_SIZE = 16
    CAPACITY = setting(default=64, small=32, large=64)
    LATENT_SIZE = 128
    BIAS = True
    NO_LATENCY = False
    RATIOS = setting(
        default=[4, 4, 4, 2],
        small=[4, 4, 4, 2],
        large=[4, 4, 2, 2, 2],
    )

    MIN_KL = 1e-1
    MAX_KL = 1e-1
    CROPPED_LATENT_SIZE = 0
    FEATURE_MATCH = True

    LOUD_STRIDE = 1

    USE_NOISE = True
    NOISE_RATIOS = [4, 4, 4]
    NOISE_BANDS = 5

    D_CAPACITY = 16
    D_MULTIPLIER = 4
    D_N_LAYERS = 4

    WARMUP = setting(default=1000000, small=1000000, large=3000000)
    MODE = "hinge"
    CKPT = None

    PREPROCESSED = None
    WAV = None
    SR = 48000
    N_SIGNAL = 65536
    MAX_STEPS = setting(default=3000000, small=3000000, large=6000000)
    VAL_EVERY = 10000

    BATCH = 8

    NAME = None


args.parse_args()

gin.parse_config_file("original.gin")
model = RAVE().cuda()

x = torch.randn(1, 1, 2**16).cuda()

for i in range(3):
    model.decode(model.encode(x))

times = []
for i in range(40):
    st = time()
    model.decode(model.encode(x))
    st = time() - st
    times.append(st)

times = 1000 * np.asarray(times)
mean = times.mean()
std = times.std()

print(f"{mean:.2f}ms +- {std:.2f}ms")
