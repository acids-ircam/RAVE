import torch
import torch.nn as nn
import math


def softplus(x):
    return nn.functional.softplus(x) + 1e-6


def logsigmoid(x):
    return -softplus(-x)


def logit(x):
    return torch.log(x + 1e-6) - torch.log(1 - x + 1e-6)


def loggradlogit(x):
    return -torch.log(x + 1e-6) - torch.log(1 - x + 1e-6)


class NAF1d(nn.Module):
    def __init__(self, causal_fun, tol=1e-6):
        super().__init__()
        self.causal_fun = causal_fun
        self.tol = tol

    def apply_transform(self, x, out):
        loggrad = torch.zeros_like(x)

        weight, scale, bias = torch.split(out, out.shape[1] // 3, 1)

        scale = softplus(scale)
        weight = torch.log_softmax(weight, 1)

        loggrad = loggrad + torch.log(scale)
        x = scale * (x + bias)

        loggrad = loggrad + logsigmoid(x) + logsigmoid(-x)
        x = torch.sigmoid(x)

        loggrad = loggrad + weight
        x = torch.exp(weight) * x

        loggrad = loggrad.logsumexp(1, keepdim=True)
        x = x.sum(1, keepdim=True)

        loggrad = loggrad + loggradlogit(x)
        x = logit(x)

        return x, loggrad

    def forward(self, x, logdet):
        x, loggrad = self.apply_transform(
            x,
            self.causal_fun(x),
        )

        _logdet = loggrad.reshape(loggrad.shape[0], -1).sum(-1)
        return x, logdet + _logdet
