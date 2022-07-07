# %%
import torch
import torch.nn as nn

model = torch.jit.load("piano.ts")


def verify_model(model: nn.Module, buffer_size=8192):
    checked_methods = []
    for n, b in model.named_buffers():
        if "_params" in n:
            method = n[:-7]
            n_in, ratio_in, n_out, ratio_out = b.numpy()
            x = torch.randn(1, n_in, buffer_size // ratio_in)
            y = getattr(model, method)(x)
            assert y.shape[0] == x.shape[
                0], f"{method}: batch size inconsistent"
            assert y.shape[
                1] == n_out, f"{method}: wrong output channel number"
            assert y.shape[
                2] == buffer_size // ratio_out, f"{method}: out_buffer is {y.shape[-1].item()}, should be {2**14 // ratio_out}"
            checked_methods.append(method)

    print(f"The following methods have passed the tests "
          f"with buffer size {buffer_size}:")
    for m in checked_methods:
        print(f" - {m}")


verify_model(model)