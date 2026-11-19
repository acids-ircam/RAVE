# Tutorial 4 — Use a RAVE model from the command line

You don't need Max to use RAVE: `rave generate` processes whole files or folders, and a few lines
of Python give you full control over the latent space. This is the way to batch-process a sound
library, render long files, or prepare material for a piece.

**You need:** RAVE installed, a model (an exported `.ts` or a training run folder). A GPU is
optional.

## 1. Transform files with `rave generate`

```bash
rave generate --model percussion.ts --input voice.wav --out_path generations/
```

The result is written to `generations/darbouka_streaming/stream/voice_loudness=0.0.wav`.

`--input` accepts files and folders (searched recursively), and can be repeated. `--model` can
also be repeated, to render the same inputs through several models in one go:

```bash
rave generate \
    --model models/percussion.ts \
    --model models/isis.ts \
    --input ~/sounds/voices/ \
    --input ~/sounds/field_recording.wav \
    --out_path generations/
```

The output tree mirrors the input tree under `<out_path>/<model>/<mode>/`.

### Input gain

RAVE models react strongly to input level. `--loudness` applies a gain in dB to the input; give
several values to render several variations:

```bash
rave generate --model percussion.ts --input voice.wav --loudness -12 --loudness -6 --loudness 0
```

### Streaming vs. full mode

| `--mode`           | What happens                                                                                | Use it when                                                   |
| ------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `stream` (default) | The file is processed in successive chunks with cached convolutions, exactly like in `nn~`. | You want to hear what the model sounds like live; long files. |
| `full`             | The whole file is processed at once with regular convolutions.                              | Short files; maximum quality; non-streaming models.           |

With `stream`, `--chunk_size` sets the block size in samples (it must exceed the model's
receptive field; the script warns you otherwise). Without it, the whole file is one block.

```bash
rave generate --model percussion.ts --input long_take.wav --mode stream --chunk_size 65536
```

### Device

```bash
rave generate --model percussion.ts --input folder/ --device cuda:0   # or mps on Apple silicon
```

### Using a training run directly

`--model` also accepts a run folder or a `.ckpt`, which avoids an export when you just want to
listen to a checkpoint:

```bash
rave generate --model runs/percussion_abc123 --input voice.wav --mode full
```

## 2. Go further with Python

`rave generate` always does encode → decode. To manipulate the latent space offline, load the
exported model with PyTorch. A **non-streaming** export is best here (`rave export` without
`--streaming`), since each call processes a whole file.

```python
import torch, torchaudio

model = torch.jit.load("percussion.ts").eval()

def load(path):
    x, sr = torchaudio.load(path)
    x = torchaudio.functional.resample(x, sr, model.sr)
    return x[:model.n_channels][None]            # (1, channels, time)

with torch.no_grad():
    x = load("voice.wav")
    z = model.encode(x)                          # (1, latent_size, frames)
    ratio = x.shape[-1] // z.shape[-1]           # audio samples per latent frame (2048 for v2)
    print(z.shape, f"one latent frame every {ratio} samples")
```

### Bias and scale latent dimensions

```python
with torch.no_grad():
    z = model.encode(load("voice.wav"))
    z[:, 0] += 1.5          # shift the first (most important) dimension
    z[:, 1] *= 2.0          # exaggerate the second
    y = model.decode(z)
torchaudio.save("voice_shifted.wav", y[0], model.sr)
```

### Interpolate between two sounds

```python
with torch.no_grad():
    za = model.encode(load("a.wav"))
    zb = model.encode(load("b.wav"))
    n = min(za.shape[-1], zb.shape[-1])
    alpha = torch.linspace(0, 1, n)               # morph from a to b over time
    z = za[..., :n] * (1 - alpha) + zb[..., :n] * alpha
    torchaudio.save("morph.wav", model.decode(z)[0], model.sr)
```

### Generate from random trajectories

```python
import math
with torch.no_grad():
    frames = 400
    t = torch.linspace(0, 1, frames)
    z = torch.zeros(1, model.latent_size, frames)
    for d in range(model.latent_size):
        z[0, d] = torch.sin(2 * math.pi * (d + 1) * 0.5 * t + d)   # slow sines
    model.set_temperature(0.5)
    torchaudio.save("generated.wav", model.decode(z)[0], model.sr)
```

### Use another projection

If the model has custom projections ([Tutorial 5](05_custom_representations.md)):

```python
print(model.get_projection_names())     # e.g. "default|raw|sfa_corpus"
model.set_projection("sfa_corpus")
z = model.encode(load("voice.wav"))     # now in SFA coordinates
```

### Use a checkpoint instead of an export

```python
import rave
model, _ = rave.load_rave_checkpoint("runs/percussion_abc123")
model.eval()
with torch.no_grad():
    y = model(load("voice.wav"))        # forward = reconstruction
```

Note that a checkpoint's `encode` returns the _raw_ encoder output (for variational models: mean
and scale, stacked), not the PCA coordinates of an exported model. See the
[Python API](../reference/python_api.md).

## 3. Batch recipes

Render a whole library through every model of a folder:

```bash
rave generate $(for m in models/*.ts; do echo --model "$m"; done) --input library/ --out_path renders/
```

Render a dry/wet gain sweep for comparison:

```bash
rave generate --model percussion.ts --input voice.wav --mode full \
    --loudness -18 --loudness -12 --loudness -6 --loudness 0 --loudness 6
```
