import torch

torch.set_grad_enabled(False)
from torch.utils.data import DataLoader

import rave

from effortless_config import Config
import rave.core

if __name__ == "__main__":

    class args(Config):
        WAV = None
        PREPROCESSED = None
        RAVE = None
        N_SIGNAL = 2**21
        BATCH_SIZE = 16
        OUT_PATH = "."

    args.parse_args()
    model = torch.jit.load(args.RAVE).eval().cuda()
    dataset = rave.core.get_dataset(
        args.WAV,
        args.PREPROCESSED,
        model.sr,
        args.N_SIGNAL,
    )
    loader = DataLoader(dataset, args.BATCH_SIZE, drop_last=True)
    rave.core.extract_codes(model, loader, args.OUT_PATH)
