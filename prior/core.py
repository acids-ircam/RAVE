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


def get_decode_function(model):

    @torch.no_grad()
    def decode_function(z):
        model.to(z.device)
        model.eval()
        y = model.decode(z).cpu()
        model.cpu()
        return y

    return decode_function