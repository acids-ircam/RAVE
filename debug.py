# %%
import torch

torch.set_grad_enabled(False)
import cached_conv as cc
import prior
import rave.core
import gin
import soundfile as sf
import torch.nn.functional as F
from time import time

device = torch.device(f"cuda:{rave.core.setup_gpu()[0]}")

gin.parse_config_file("configs/prior.gin")

cc.use_cached_conv(True)

synth = torch.jit.load("piano.ts").eval().to(device)
# model = prior.Prior.load_from_checkpoint(
#     rave.core.search_for_run("runs/prior_piano/prior"))
model = prior.Prior().to(device)
model.eval()
x = torch.zeros(1, 1024).to(device)
model(x)
model = torch.jit.script(model)
print("scripted !")

st = time()
z = model.generate(torch.zeros(1, 2**10).to(device), sample=True)
print(time() - st)

z = z.reshape(1, -1, 4).transpose(1, 2)

y = synth.decode(z).cpu()
sf.write("out.wav", y.reshape(-1).numpy(), 44100)

# %%
