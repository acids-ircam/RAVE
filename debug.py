# %%
import subprocess

export_rave = subprocess.Popen(
    "python export_rave.py --name darbouka_onnx".split(" "),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
export_onnx = subprocess.Popen(
    "python export_onnx.py --name darbouka_onnx".split(" "),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

print(export_rave.wait())
print(export_onnx.wait())
# %%
import onnxruntime as ort
import librosa as li
import torch
import numpy as np
import soundfile as sf

torch.set_grad_enabled(False)

onnx = ort.InferenceSession("darbouka_onnx.onnx")
torchscript = torch.jit.load("darbouka_onnx.ts")

x, sr = li.load("cantina.wav", 44100)
x = x[:2 * sr]
x_numpy = x.reshape(1, 1, -1).astype(np.float32)
x_torch = torch.from_numpy(x_numpy)

y_onnx = onnx.run(None, {"audio_in": x_numpy})[0]
y_torchscript = torchscript(x_torch).numpy()

sf.write("onnx.wav", y_onnx.reshape(-1), 44100)
sf.write("torchscript.wav", y_torchscript.reshape(-1), 44100)
# %%
