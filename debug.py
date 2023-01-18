import torch
import torch.nn as nn
import cached_conv as cc

x = torch.randn(1, 1, 128)
conv = cc.Conv1d(1, 1, 4, 2, padding=cc.get_padding(4, 2))

print(cc.get_padding(3, mode="causal"))
print(cc.get_padding(3, 2, mode="causal"))
print(cc.get_padding(4, mode="causal"))
print(cc.get_padding(4, 2, mode="causal"))
print(cc.get_padding(1, mode="causal"))
