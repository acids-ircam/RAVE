import torch
from torch.utils.data import DataLoader

import rave

from effortless_config import Config
import pytorch_lightning as pl
import os
import rave.core

import gin

if __name__ == "__main__":

    class args(Config):
        GIN = "configs/xrave.gin"

        PREPROCESSED = None
        WAV = None
        N_SIGNAL = 131072
        MAX_STEPS = 6000000
        VAL_EVERY = 10000

        BATCH = 8
        CKPT = None

        NAME = None

    args.parse_args()

    assert args.NAME is not None, "You must enter a name for this run"

    gin_config = gin.parse_config_file(args.GIN)

    os.makedirs(os.path.join("runs", args.NAME, "rave"), exist_ok=True)

    rave.core.copy_config(
        gin_config.filename,
        os.path.join("runs", args.NAME, "rave", "config.gin"),
    )

    model = rave.RAVE()

    dataset = rave.core.get_dataset(
        args.WAV,
        args.PREPROCESSED,
        model.sr,
        args.N_SIGNAL,
    )
    train, val = rave.core.split_dataset(dataset, 98)
    train = DataLoader(train, args.BATCH, True, drop_last=True, num_workers=8)
    val = DataLoader(val, args.BATCH, False, num_workers=8)

    # CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(monitor="validation",
                                                         filename="best")
    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")

    val_check = {}
    if len(train) >= args.VAL_EVERY:
        val_check["val_check_interval"] = args.VAL_EVERY
    else:
        nepoch = args.VAL_EVERY // len(train)
        val_check["check_val_every_n_epoch"] = nepoch

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            os.path.join("runs", args.NAME),
            name="rave",
        ),
        gpus=rave.core.setup_gpu(),
        callbacks=[validation_checkpoint, last_checkpoint],
        max_epochs=100000,
        max_steps=args.MAX_STEPS,
        profiler="simple",
        **val_check,
    )

    run = rave.core.search_for_run(args.CKPT)
    if run is not None:
        step = torch.load(run, map_location='cpu')["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = step

    trainer.fit(model, train, val, ckpt_path=run)