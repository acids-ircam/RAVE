import torch
import torch.nn as nn
from effortless_config import Config


class args(Config):
    PRIOR = None
    RAVE = None
    NAME = "combined"


args.parse_args()


class Combined(nn.Module):
    def __init__(self, prior, rave):
        super().__init__()
        self._prior = torch.jit.load(prior)
        self._rave = torch.jit.load(rave)

        self.register_buffer("encode_params", self._rave.encode_params)
        self.register_buffer("decode_params", self._rave.decode_params)
        self.register_buffer("forward_params", self._rave.forward_params)
        self.register_buffer("prior_params", self._prior.forward_params)

    @torch.jit.export
    def encode(self, x):
        return self._rave.encode(x)

    @torch.jit.export
    def encode_amortized(self, x):
        return self._rave.encode_amortized(x)

    @torch.jit.export
    def decode(self, x):
        return self._rave.decode(x)

    @torch.jit.export
    def prior(self, x):
        return self._prior(x)

    @torch.jit.export
    def forward(self, x):
        return self._rave(x)


model = torch.jit.script(Combined(args.PRIOR, args.RAVE))
torch.jit.save(model, f"{args.NAME}.ts")
