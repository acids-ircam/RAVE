import hashlib
import os
import subprocess

import gin
import pytorch_lightning as pl
import torch
from effortless_config import Config
from torch.utils.data import DataLoader

import rave
import rave.core


def main():

    class args(Config):
        GIN = "configs/rave_v2.gin"

        WAV = None
        PREPROCESSED = None
        MAX_STEPS = 6000000
        VAL_EVERY = 10000
        N_SIGNAL = 131072

        BATCH = 8
        CKPT = None

        NAME = None

    args.parse_args()

    assert args.NAME is not None, "You must enter a name for this run"

    gin.parse_config_file(args.GIN)

    gin_hash = hashlib.md5(
        gin.operative_config_str().encode()).hexdigest()[:10]

    RUN_NAME = f'{args.NAME}_{gin_hash}'

    os.makedirs(os.path.join("runs", RUN_NAME), exist_ok=True)

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
            "runs",
            name=RUN_NAME,
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

    gin.finalize()
    with open(os.path.join("runs", RUN_NAME, "config.gin"), "w") as config_out:
        config_out.write(gin.operative_config_str())
    with open(os.path.join("runs", RUN_NAME, "commit"),
              "w") as training_commit:
        training_commit.write(
            subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode())

    trainer.fit(model, train, val, ckpt_path=run)


if __name__ == "__main__":
    main()