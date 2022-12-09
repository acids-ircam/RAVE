import torch
from rave.discriminator import EncodecConvNet
import torchsummary


x = torch.randn(1, 2, 64, 256)
net = EncodecConvNet(32)


numel = 0

for p in net.parameters():
    if p.requires_grad:
        numel += p.numel()

print(numel/1e6)