# Balancer - credit to https://github.com/facebookresearch/encodec

from typing import Callable, Dict

import torch


class EMA:

    def __init__(self, beta: float = .999) -> None:
        self.shadows = {}
        self.beta = beta

    def __call__(self, inputs: Dict[str, torch.Tensor]):
        outputs = {}
        for k, v in inputs.items():
            if not k in self.shadows:
                self.shadows[k] = v
                continue

            self.shadows[k] *= self.beta
            self.shadows[k] += (1 - self.beta) * v

            outputs[k] = self.shadows[k].clone()
        return outputs


class Balancer:

    def __init__(self,
                 ema_averager: Callable[[], EMA],
                 weights: Dict[str, float],
                 scale_gradients: bool = False) -> None:
        self.ema_averager = ema_averager()
        self.weights = weights
        self.scale_gradients = scale_gradients

    def backward(self, losses: Dict[str, torch.Tensor], x: torch.Tensor):
        grads = {}
        norms = {}
        for k, v in losses.items():
            grads[k] = torch.autograd.grad(v, [x], retain_graph=True)
            norms[k] = grads[k].norm()

        avg_norms = self.ema_averager(norms)

        sum_weights = sum([self.weights.get(k, 1) for k in avg_norms])

        for name, norm in avg_norms.items():
            if self.scale_gradients:
                ratio = self.weights.get(name, 1) / sum_weights
                grads[name] *= ratio
                grads[name] /= norm + 1e-6
            else:
                grads[name] *= self.weights.get(name, 1)

        full_grad = sum(grads.values())
        x.backward(full_grad)
