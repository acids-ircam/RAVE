![rave_logo](docs/rave.png)

# RAVE: Realtime Audio Variational autoEncoder

Official implementation of _RAVE: A variational autoencoder for fast and high-quality neural audio synthesis_ ([article link](https://arxiv.org/abs/2111.05011))

## Installation

RAVE needs `python 3.9`. Install the dependencies using

```bash
pip install -r requirements.txt
```

## Training

Both RAVE and the prior model are available in this repo. For most users we recommand to use the `cli_helper.py` script, since it will generate a set of instructions allowing the training and export of both RAVE and the prior model on a specific dataset.

```bash
python cli_helper.py
```

However, if you want to customize even more your training, you can use the provided `train_{rave, prior}.py` and `export_{rave, prior}.py` scripts manually.

## Offline usage

Once trained, you can evaluate RAVE and the prior model using

```python
import torch

torch.set_grad_enabled(False)
from rave import RAVE
from prior import Prior

import librosa as li
import soundfile as sf

################ LOADING PRETRAINED MODELS ################
rave = RAVE.load_from_checkpoint("/path/to/checkpoint.ckpt", strict=False).eval()
prior = Prior.load_from_checkpoint("/path/to/checkpoint.ckpt", strict=False).eval()

################ RECONSTRUCTION ################

# STEP 1: LOAD INPUT AUDIO
x, sr = li.load("input_audio.wav", sr=rave.sr)

# STEP 2: ENCODE DECODE AUDIO
x = torch.from_numpy(x).reshape(1, 1, -1).float()
latent = rave.encode(x)
y = rave.decode(latent)

# STEP 3: EXPORT
sf.write("output_audio.wav", y.reshape(-1).numpy(), sr)

################ PRIOR GENERATION ################

# STEP 1: CREATE DUMMY INPUT TENSOR
generation_length = 2**18  # approximately 6s at 48kHz
x = torch.randn(1, 1, generation_length)  # dummy input
z = rave.encode(x)  # dummy latent representation
z = torch.zeros_like(z)

# STEP 2: AUTOREGRESSIVE GENERATION
z = prior.quantized_normal.encode(prior.diagonal_shift(z))
z = prior.generate(z)
z = prior.diagonal_shift.inverse(prior.quantized_normal.decode(z))

# STEP 3: SYNTHESIS AND EXPORT
y = rave.decode(z)
sf.write("output_audio.wav", y.reshape(-1).numpy(), sr)

```

## Online usage

RAVE exported as a realtime torchscript file can be used like this

```python
import torch

model = torch.jit.load("pretrained.ts")

# DUMMY INPUT
x = torch.randn(1, 1, 16384)

# ENCODE DECODE
z = model.encode(x)
y = model.decode(z)
y = model(x)

# PRIOR GENERATION
# we give to the prior method a tensor containing the temperature of the generation
# here prior will generate 2048 latent points with temperature 0.5
# temperature must be a real-valued number
z = model.prior(torch.ones(1,1,2048) * .5)
y = model.decode(z)-

```

## MAX / MSP - PureData usage

**[NOT AVAILABLE YET]**

RAVE and the prior model can be used in realtime inside [max/msp](https://cycling74.com/), allowing creative interactions with both models. Code and details about this part of the project are not available yet, we are currently working on the corresponding article !

![max_msp_screenshot](docs/maxmsp_screenshot.png)

An audio example of the prior sampling patch is available in the `docs/` folder.
