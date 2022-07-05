import torch

torch.set_grad_enabled(False)
import gin
import rave
import os
from effortless_config import Config

import cached_conv as cc
import torch.nn as nn
import re


class args(Config):
    NAME = None


args.parse_args()

assert args.NAME is not None

root = os.path.join("runs", args.NAME, "rave")
gin.parse_config_file(os.path.join(root, "config.gin"))
checkpoint = rave.core.search_for_run(root)

print(f"using {checkpoint}")

pretrained = rave.RAVE()
pretrained.load_state_dict(torch.load(checkpoint)["state_dict"])
pretrained.eval()

for m in pretrained.modules():
    if hasattr(m, "weight_g"):
        nn.utils.remove_weight_norm(m)


def recursive_replace(model: nn.Module):
    for name, child in model.named_children():
        if isinstance(child, cc.convs.Conv1d):
            conv = nn.Conv1d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child._pad[0],
                child.dilation,
                child.groups,
                child.bias,
            )
            conv.weight.data.copy_(child.weight.data)
            if conv.bias is not None:
                conv.bias.data.copy_(child.bias.data)
            setattr(model, name, conv)
        elif isinstance(child, cc.convs.ConvTranspose1d):
            conv = nn.ConvTranspose1d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                child.output_padding,
                child.groups,
                child.bias,
                child.dilation,
                child.padding_mode,
            )
            conv.weight.data.copy_(child.weight.data)
            if conv.bias is not None:
                conv.bias.data.copy_(child.bias.data)
            setattr(model, name, conv)
        else:
            recursive_replace(child)


recursive_replace(pretrained)

x = torch.randn(1, 1, 2**15)
pretrained(x)

torch.onnx.export(
    pretrained,
    x,
    f"{args.NAME}.onnx",
    export_params=True,
    opset_version=12,
    input_names=["audio_in"],
    output_names=["audio_out"],
    dynamic_axes={
        "audio_in": {
            2: "audio_length"
        },
        "audio_out": [0],
    },
    do_constant_folding=False,
)