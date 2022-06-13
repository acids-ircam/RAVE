import numpy as np
import torch
import yaml
import os


class CodeDataset(torch.utils.data.Dataset):

    def __init__(self, path) -> None:
        super().__init__()
        with open(os.path.join(path, "info.yaml"), "r") as info:
            shape = yaml.safe_load(info)["shape"]
        self.data = np.memmap(
            os.path.join(path, "data.npy"),
            dtype=np.uint16,
            mode="r",
        ).reshape(*shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].astype(np.int32)