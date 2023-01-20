# Balancer - adapted from https://github.com/facebookresearch/encodec

from typing import Any, Callable, Dict, Optional, Sequence

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
            else:
                self.shadows[k] *= self.beta
                self.shadows[k] += (1 - self.beta) * v

            outputs[k] = self.shadows[k].clone()
        return outputs


class Balancer:

    def __init__(
        self,
        ema_averager: Callable[[], EMA],
        weights: Dict[str, float],
        scale_gradients: bool = False,
        deny_list: Optional[Sequence[str]] = None,
    ) -> None:
        self.ema_averager = ema_averager()
        self.weights = weights
        self.scale_gradients = scale_gradients
        self.deny_list = deny_list

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        model_output: torch.Tensor,
        logger: Optional[Callable[[str, float], None]] = None,
        profiler: Optional[Any] = None,
    ):
        grads = {}
        norms = {}

        for k, v in losses.items():
            if self.deny_list is not None:
                if k in self.deny_list: continue

            grads[k], = torch.autograd.grad(
                v,
                [model_output],
                retain_graph=True,
            )

            if (nans := torch.isnan(grads[k])).any():
                count = nans.float().mean()
                grads[k] = torch.where(nans, torch.zeros_like(nans), grads[k])
                if logger is not None:
                    logger(f"{k}_nan_ratio", count)

            norms[k] = grads[k].norm(
                dim=tuple(range(1, grads[k].dim()))).mean()

            if profiler is not None:
                profiler(f'partial backward {k}')

        avg_norms = self.ema_averager(norms)

        if profiler is not None:
            profiler('grad norm estimation')

        sum_weights = sum([self.weights.get(k, 1) for k in avg_norms])

        for name, norm in avg_norms.items():
            if self.scale_gradients:
                ratio = self.weights.get(name, 1) / sum_weights
                scale = ratio / (norm + 1e-6)
                grads[name] *= scale

                if logger is not None:
                    logger(f'scale_{name}', scale)
                    logger(f'grad_norm_{name}', grads[name].norm())
                    logger(f'target_norm_{name}', ratio)
            else:
                scale = self.weights.get(name, 1)
                grads[name] *= scale

                if logger is not None:
                    logger(f'scale_{name}', scale)
                    logger(f'grad_norm_{name}', grads[name].norm())

        if profiler is not None:
            profiler('norm scaling')

        full_grad = sum([grads[name] for name in avg_norms.keys()])
        model_output.backward(full_grad, retain_graph=True)

        if profiler is not None:
            profiler('scaled backward')

        if self.deny_list is not None:
            for k in self.deny_list:
                if k in losses:
                    loss = losses[k] * self.weights.get(k, 1)
                    if loss.requires_grad:
                        loss.backward(retain_graph=True)

        if profiler is not None:
            profiler('denied backward')
