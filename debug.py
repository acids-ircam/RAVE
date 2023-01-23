import torch
from rave.quantization import VQ

vq = VQ(128, 1024, .99)
x = torch.randn(3, 128, 512, requires_grad=True)

y, diff = vq(x)

y.mean().backward()
print(x.grad)