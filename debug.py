import torch
import gin

@gin.configurable
def f(v):
    return v

gin.parse_config_file('rave/configs/transformer.gin')
gin.parse_config('''
LATENT_SIZE = 256
f.v = @Transformer()
''')

m = f()

x = torch.randn(1,256, 128)
print(m(x).shape)