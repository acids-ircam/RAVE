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


class SimpleQuantizer(nn.Module):

    def __init__(self, embed_list) -> None:
        super().__init__()
        self.register_buffer("embed", torch.stack(embed_list, 0))
        self.n_codes = self.embed.shape[1]

    def residual_quantize(self, x: torch.Tensor) -> torch.Tensor:
        index = []
        for embed in self.embed:
            embed_ind = closest_code(x, embed)
            q = F.embedding(embed_ind, embed).permute(0, 2, 1)
            x = x - q
            index.append(embed_ind)
        return torch.stack(index, 1)

    def residual_dequantize(self, index: torch.Tensor) -> torch.Tensor:
        y = torch.tensor(0.).to(index.device)
        for i, embed in zip(index.transpose(0, 1), self.embed):
            y = y + F.embedding(i, embed).permute(0, 2, 1)
        return y
