import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

from prior.model import Model
from effortless_config import Config
from os import environ, path

from udls import SimpleDataset, simple_audio_preprocess
import numpy as np

import math

import GPUtil as gpu


class args(Config):
    RESOLUTION = 32

    RES_SIZE = 512
    SKP_SIZE = 256
    KERNEL_SIZE = 3
    CYCLE_SIZE = 4
    N_LAYERS = 10
    PRETRAINED_VAE = None

    PREPROCESSED = None
    WAV = None
    SR = 48000
    N_SIGNAL = 65536

    BATCH = 8
    CKPT = None

    NAME = None


args.parse_args()
assert args.NAME is not None


def get_n_signal(a, m):
    k = a.KERNEL_SIZE
    cs = a.CYCLE_SIZE
    l = a.N_LAYERS

    rf = (k - 1) * sum(2**(np.arange(l) % cs)) + 1
    ratio = m.encode_params[-1].item()

    return 2**math.ceil(math.log2(rf * ratio))


model = Model(
    resolution=args.RESOLUTION,
    res_size=args.RES_SIZE,
    skp_size=args.SKP_SIZE,
    kernel_size=args.KERNEL_SIZE,
    cycle_size=args.CYCLE_SIZE,
    n_layers=args.N_LAYERS,
    pretrained_vae=args.PRETRAINED_VAE,
)

args.N_SIGNAL = max(args.N_SIGNAL, get_n_signal(args, model.synth))

dataset = SimpleDataset(
    args.PREPROCESSED,
    args.WAV,
    preprocess_function=simple_audio_preprocess(args.SR, args.N_SIGNAL),
    split_set="full",
    transforms=lambda x: x.reshape(1, -1),
)

val = (2 * len(dataset)) // 100
train = len(dataset) - val
train, val = random_split(dataset, [train, val])

train = DataLoader(train, args.BATCH, True, drop_last=True, num_workers=8)
val = DataLoader(val, args.BATCH, False, num_workers=8)

# CHECKPOINT CALLBACKS
validation_checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="validation",
    filename="best",
)
last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")

CUDA = gpu.getAvailable(maxMemory=.05)
assert len(CUDA)
environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])

trainer = pl.Trainer(
    logger=pl.loggers.TensorBoardLogger(path.join("runs", args.NAME),
                                        name="prior"),
    gpus=1,
    check_val_every_n_epoch=10,
    callbacks=[validation_checkpoint, last_checkpoint],
    resume_from_checkpoint=args.CKPT,
    max_epochs=100000,
)
trainer.fit(model, train, val)
