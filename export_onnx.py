import torch

torch.set_grad_enabled(False)
import gin
import rave
import os
from effortless_config import Config


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

x = torch.randn(1, 1, 2**15)
pretrained(x).shape

torch.onnx.export(
    pretrained,
    x,
    f"{args.NAME}.onnx",
    export_params=True,
    opset_version=15,
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
