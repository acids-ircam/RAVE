import torch
from torch.utils.data import DataLoader, random_split
from fd.flows.ar_model import TeacherFlow
from udls import SimpleDataset, simple_audio_preprocess
from effortless_config import Config
import pytorch_lightning as pl
from os import environ

if __name__ == "__main__":

    class args(Config):
        IN_SIZE = 1
        RES_SIZE = 256
        SKP_SIZE = 256
        KERNEL_SIZE = 3
        N_BLOCK = 10
        DILATION_CYCLE = 5
        N_FLOW = 4

        PREPROCESSED = None
        WAV = None
        SR = 24000
        N_SIGNAL = 16384

        BATCH = 8
        CUDA = 0

    args.parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = str(args.CUDA)

    model = TeacherFlow(
        args.IN_SIZE,
        args.RES_SIZE,
        args.SKP_SIZE,
        args.KERNEL_SIZE,
        args.N_BLOCK,
        args.DILATION_CYCLE,
        args.N_FLOW,
    )

    dataset = SimpleDataset(
        args.PREPROCESSED,
        args.WAV,
        preprocess_function=simple_audio_preprocess(args.SR, args.N_SIGNAL),
        split_set="full",
    )

    val = (2 * len(dataset)) // 100
    train = len(dataset) - val
    train, val = random_split(dataset, [train, val])

    train = DataLoader(train, args.BATCH, True)
    val = DataLoader(val, args.BATCH, False)

    check_callback = pl.callbacks.ModelCheckpoint(monitor="val_logpx")
    trainer = pl.Trainer(gpus=1, callbacks=[check_callback])
    trainer.fit(model, train, val)