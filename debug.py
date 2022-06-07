# %%
import gin
from rave import RAVE
import torch
import matplotlib.pyplot as plt

gin.parse_config_file("configs/rave_v2.gin")

rave = RAVE().eval()


def get_rave_receptive_field(model):
    N = 2**15
    model.eval()
    while True:
        x = torch.randn(1, 1, N, requires_grad=True)
        y = model(x)
        y[0, 0, N // 2].backward()
        grad = x.grad.data.reshape(-1)
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


print(get_rave_receptive_field(rave))

# %%
rf_v1 = 26157 + 22060
rf_v2 = 45933 + 41836

print(rf_v1 / 44100)
print(rf_v2 / 44100)

# %%
