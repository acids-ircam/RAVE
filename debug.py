# %%
import gin
import rave
from glob import glob

configs = glob("configs/*.gin")

for config in configs:
    if "prior" in config or "general" in config:
        continue
    print(f"testing {config}")
    gin.parse_config_file(config)
    model = rave.RAVE()
    nel = 0
    for p in model.parameters():
        if p.requires_grad:
            nel += p.numel()
    print(f"{nel/1e6:.2f}M parameters")
