import torch
import torch.nn as nn
import torch.nn.functional as F


def closest_code(x, embed):
    batch_size = x.shape[0]
    x = x.permute(0, 2, 1).reshape(-1, x.shape[1])
    embed = embed.transpose(0, 1)
    distance = (x.pow(2).sum(1, keepdim=True) - 2 * x @ embed +
                embed.pow(2).sum(0, keepdim=True))
    return torch.argmin(distance, -1).reshape(batch_size, -1)


def residual_quantize(x, embed_list):
    y = 0.
    index = []
    for embed in embed_list:
        embed_ind = closest_code(x, embed)
        q = F.embedding(embed_ind, embed).permute(0, 2, 1)
        x = x - q
        y = y + q
        index.append(embed_ind)
    return torch.stack(index, 1)
