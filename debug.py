# %%
import rave
import gin
import torch

gin.enter_interactive_mode()
gin.parse_config_file('encodec.gin')

model = rave.RAVE()
# %%
print(model)
# %%

x  =torch.randn(1,1,131072)

z = model.encode(x)

y = model.decode(z)

print(x.shape, z.shape, y.shape)
# %%
131072//256
# %%
44100/512
# %%
