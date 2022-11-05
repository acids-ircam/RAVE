import pathlib

import gin

from rave import RAVE

configs = pathlib.Path('rave/configs').glob('*.gin')
configs = map(str, configs)

for config in configs:
    print(f'testing {config}')
    gin.clear_config()
    gin.parse_config_file(config)
    model = RAVE()
    numel = 0
    for p in model.parameters():
        if p.requires_grad:
            numel += p.numel()
    print(numel / 1000000)
