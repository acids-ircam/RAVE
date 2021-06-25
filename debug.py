# %%
import torch

torch.set_grad_enabled(False)
from fd.flows.ar_model import TeacherFlow

model = TeacherFlow.load_from_checkpoint(
    "lightning_logs/version_0/checkpoints/epoch=0-step=1625.ckpt").eval()
x = torch.randn(1, 1, 1024)

y = model.flows.inverse(x)
