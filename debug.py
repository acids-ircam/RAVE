import torch
import torch.nn as nn

embd = nn.Embedding(4, 16)
x = torch.arange(10)
y = embd(x % 4).transpose(0, 1)
print(y.shape)
print(y)