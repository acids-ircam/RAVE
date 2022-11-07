import matplotlib.pyplot as plt
import torch

from rave.balancer import EMA
from rave.core import mean_difference

ema = EMA()

inputs = {'x': torch.randn(10, 10), 'y': torch.randn(100)}
other_inputs = {'x': torch.randn(10, 10), 'y': torch.randn(100)}

ema(inputs)

for i in range(10000):
    out = ema(other_inputs)

    dist = 0

    for v1, v2 in zip(other_inputs.values(), out.values()):
        dist = dist + mean_difference(v1, v2)

    print(dist)