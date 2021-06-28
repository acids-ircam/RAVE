# %%
import torch
import librosa as li
from random import choice
from glob import glob
import numpy as np

torch.set_grad_enabled(False)
from fd.parallel_model.model import ParallelModel
from fd.flows.ar_model import TeacherFlow
import matplotlib.pyplot as plt

ckpt = glob("lightning_logs/version_4/checkpoints/*.ckpt")[0]
student = ParallelModel.load_from_checkpoint(ckpt)
ckpt = glob("lightning_logs/version_3/checkpoints/*.ckpt")[0]
teacher = TeacherFlow.load_from_checkpoint(ckpt)
# %%

name = glob("/slow-2/antoine/dataset/ljspeech/LJSpeech-1.1/out_24k/*.wav")
name = choice(name)

x, sr = li.load(name, None)

x = x[:2**15].reshape(1, 1, -1)
plt.plot(x.reshape(-1))
plt.show()
x = torch.from_numpy(x).float()

y, logdet = teacher.flows(x)
# %%

plt.plot(y.reshape(-1))
print(logdet)

#%% SPEED TEST
from time import time

device = torch.device("cuda:4")
model.to(device)


def bench(func):
    mean = 0
    nel = 0

    for i in range(20):
        st = time()
        func()
        st = time() - st
        nel += 1
        mean += (st - mean) / nel

    return mean


N = 32768
sr = 24000
x = torch.randn(1, 1, N).to(device)
z = model.encoder(x)[0]

encode_time = bench(lambda: model.encoder(x))
decode_time = bench(lambda: model.decoder(z))

encode_rtf = (N / sr) / encode_time
decode_rtf = (N / sr) / decode_time

print(encode_rtf, decode_rtf)
# %%
