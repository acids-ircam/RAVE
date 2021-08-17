import torch
from torch.utils.data import DataLoader, random_split
from fd.parallel_model.model import ParallelModel
from fd.parallel_model.core import random_phase_mangle, EMAModelCheckPoint
from udls import SimpleDataset, simple_audio_preprocess
from effortless_config import Config
import pytorch_lightning as pl
from os import environ
import numpy as np

from udls.transforms import Compose, RandomApply, Dequantize, RandomCrop

if __name__ == "__main__":

    class args(Config):
        DATA_SIZE = 1
        CAPACITY = 64
        LATENT_SIZE = 16
        RATIOS = [8, 8, 4, 4]
        BIAS = True
        D_CAPACITY = 16
        D_MULTIPLIER = 4
        D_N_LAYERS = 4
        WARMUP = 100000
        MODE = "hinge"
        CKPT = None

        PREPROCESSED = None
        WAV = None
        SR = 24000
        N_SIGNAL = 16384

        BATCH = 8
        CUDA = 0

    args.parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = str(args.CUDA)

    model = ParallelModel(
        data_size=args.DATA_SIZE,
        capacity=args.CAPACITY,
        latent_size=args.LATENT_SIZE,
        ratios=args.RATIOS,
        bias=args.BIAS,
        d_capacity=args.D_CAPACITY,
        d_multiplier=args.D_MULTIPLIER,
        d_n_layers=args.D_N_LAYERS,
        warmup=args.WARMUP,
        mode=args.MODE,
        sr=args.SR,
    )

    dataset = SimpleDataset(
        args.PREPROCESSED,
        args.WAV,
        preprocess_function=simple_audio_preprocess(args.SR,
                                                    2 * args.N_SIGNAL),
        split_set="full",
        transforms=Compose([
            RandomCrop(args.N_SIGNAL),
            RandomApply(
                lambda x: random_phase_mangle(x, 20, 2000, .99, args.SR),
                p=.8,
            ),
            Dequantize(16),
            lambda x: x.astype(np.float32),
        ]),
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
    ema_checkpoint = EMAModelCheckPoint(model,
                                        filename="ema",
                                        monitor="validation")

    trainer = pl.Trainer(
        gpus=1,
        # val_check_interval=1,
        check_val_every_n_epoch=1,
        callbacks=[validation_checkpoint,
                   last_checkpoint],  #, ema_checkpoint],
        resume_from_checkpoint=args.CKPT,
        max_epochs=100000,
    )
    trainer.fit(model, train, val)