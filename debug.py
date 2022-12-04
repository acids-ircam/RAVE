import torch
import gin
from rave.model import RAVE

gin.parse_config_file('discrete.gin')

model = RAVE()


x = torch.randn(1,1,2**16)

z = model.encode(x)

print(z.shape)

y = model.decode(z)

print(y.shape)