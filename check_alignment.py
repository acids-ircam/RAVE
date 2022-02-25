# %%
import torch

torch.manual_seed(0)

torch.set_grad_enabled(False)
import cached_conv

from effortless_config import Config


class args(Config):
    CACHE = True


args.parse_args()

cached_conv.use_buffer_conv(args.CACHE)

from rave.blocks import Generator
import matplotlib.pyplot as plt

model = Generator(
    16,
    8,
    16,
    [2],
    2048,
    False,
    None,
    None,
    padding_mode="centered",
).net.blocks[1]

x = torch.randn(1, 16, 16)
z = torch.randn(1, 16, 16)

x, z, mean, scale = model(x, z)
x = x * scale + mean

plt.plot(x[0, 0])

# %%
