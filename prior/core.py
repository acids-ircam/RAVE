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


def get_decode_function(model_path):
    model = torch.jit.script(model_path)

    @torch.no_grad()
    def decode_function(z):
        model.to(z.device)
        model.eval()
        y = model.decode(z).cpu()
        model.cpu()
        return y

    return decode_function


@torch.enable_grad()
def get_prior_receptive_field(model):
    N = 2**11
    model.eval()
    device = next(iter(model.parameters())).device
    while True:
        x = torch.zeros(1,
                        model.codebook_dim,
                        N,
                        requires_grad=True,
                        device=device)
        y = model(x, one_hot_encoding=False)

        y[0, 0, N // 2].backward()
        assert x.grad is not None, "input has no grad"

        grad = x.grad.data[0, 0]
        left_grad, right_grad = grad.chunk(2, 0)
        large_enough = (left_grad[0] == 0) and right_grad[-1] == 0
        if large_enough:
            break
        else:
            N *= 2
    left_receptive_field = len(left_grad[left_grad != 0])
    right_receptive_field = len(right_grad[right_grad != 0])
    model.zero_grad()
    return left_receptive_field, right_receptive_field
