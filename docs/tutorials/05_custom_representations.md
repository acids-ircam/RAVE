# Tutorial 5 — Add custom representations

The latent space a RAVE model learns is not organized for humans: by default you get the
**principal components** of the latents seen during training, sorted by variance. That is a good
start, but often you want axes that mean something for *your* use — "brightness", "pitch",
"the slow evolutions of the sound", "violin vs. voice"...

RAVE lets you store any number of alternative coordinate systems — **projections** — inside a model,
and switch between them at runtime with the `projection` attribute, without retraining anything.

```text
           encode                    projection P          your patch           P⁻¹                decode
 audio ──► encoder ──► raw latent ──────────────────► controls ──────► controls ─────► raw latent ──► decoder ──► audio
```

**You need:** a trained **continuous** model (variational — `v1`, `v2`, `v3`... — or Wasserstein),
as a run folder or an exported `.ts`, and some audio. Discrete and spherical models do not support
projections.

## 1. See what a model already has

```bash
rave dimred show --model my_model_streaming.ts
```

prints something like:

```text
exported latent size: 8 (full: 128)

name           type  dims (in -> out)  notes
-------------  ----  ----------------  -----
default        PCA   128 -> 128        dims for fidelity 0.9: 5, 0.95: 8, 0.99: 23; selected
raw            PCA   128 -> 128
pca_0          PCA   128 -> 128        ...
pca_full_mean  PCA   128 -> 128        ...
```

- `default` — the PCA computed during training (the axes you use when you do nothing).
- `raw` — the raw latent space, no rotation.
- `pca_0`, `pca_full_mean` — PCAs recorded at the last validation.

The model's **latent size** (8 here, number of inlets/outlets of `encode`/`decode`) is fixed at export.
Projections change what those 8 dimensions *mean*; dimensions beyond them are filled with noise
(scaled by `temperature`) when decoding.

## 2. A PCA on your own sounds

The default PCA describes the training data as a whole. If you mostly feed the model with one kind
of sound — say your own voice — a PCA computed on that sound gives axes that vary the most *for
that input*:

```bash
rave dimred add --model runs/violin_abc123 --name voice --audio ~/sounds/my_voice/ --type pca
```

This encodes the audio, fits the projection and writes it into the checkpoint as `pca_voice`.
Export as usual, then select it:

```bash
rave export --run runs/violin_abc123 --streaming
```

In Max: `set projection pca_voice`. In Python: `model.set_projection("pca_voice")`.

You can also edit an **exported** model directly. The `.ts` is uncompiled, edited and re-exported
with the same settings (streaming, latent size, sampling rate):

```bash
rave dimred add --model violin_streaming.ts --name voice --audio ~/sounds/my_voice/ --out violin_voice_streaming.ts
```

Without `--out`, the `.ts` is replaced after confirmation (`--yes` to skip it). A prior embedded in
the `.ts` cannot be kept.

## 3. Choosing a projection type

Several types can be fitted at once with repeated `--type`; they are stored as `<type>_<name>`.

### Unsupervised

```bash
rave dimred add --model runs/violin_abc123 --name corpus --audio ~/datasets/violin_raw/ \
    --type pca --type wpca --type ica --type sfa --dims 8
```

| Type   | Axes are...                                   | Good for |
| ------ | --------------------------------------------- | -------- |
| `pca`  | sorted by variance                            | The "biggest" changes first. |
| `wpca` | the PCA axes, rescaled to unit variance       | Controllers with comparable ranges on every axis (faders, LFOs between −1 and 1). |
| `ica`  | statistically independent                     | Axes that act more separately from each other. RAVE tests whether ICA is relevant (non-gaussian latents) and warns you if not. |
| `sfa`  | sorted from the slowest to the fastest varying| Separating *slow* aspects (timbre, register, dynamics) from fast ones (texture, articulation). Great for gestural control: drive only the first SFA dims. |

`--dims` (or `--fidelity`, default 0.99) sets how many dimensions are actually fitted by `ica` /
`sfa`; the remaining ones are completed with principal components so that the projection stays
exactly invertible.

### Aligned with audio descriptors

`pls` and `cca` look for axes that follow **audio descriptors** computed on the same audio:
`loudness` (dB), `centroid` (brightness), `bandwidth`, `flatness` (noisiness), `f0` (pitch, in
semitones).

```bash
rave dimred add --model runs/violin_abc123 --name desc --audio ~/datasets/violin_raw/ \
    --type pls --descriptors loudness --descriptors centroid --descriptors f0
```

The log prints the correlation of each new axis with each descriptor, e.g.:

```text
pls_desc: correlation of projected dimensions with descriptors
         loudness   centroid         f0
    0       0.93       0.21       0.05
    1       0.12       0.81       0.10
    2       0.03       0.14       0.77
```

Here axis 0 is (mostly) loudness, axis 1 brightness, axis 2 pitch. Decoding with those axes gives
you "descriptor-like" knobs — keep in mind that the model can only produce what it has learned:
pushing the pitch axis outside the range of the training data will not give clean notes.

- `pls` (partial least squares) favors axes that *predict* descriptors well, scaled to unit variance.
- `cca` (canonical correlation analysis) favors axes maximally *correlated* with them.
- `--pitch_mode` chooses the F0 tracker (`pyin` default, `yin`, or `crepe`/`pesto`/`world`/`praat`
  if installed). Unvoiced frames are ignored.

### Separating classes

`lda` (linear discriminant analysis) finds the axes that best separate **classes** of sounds. Each
`--audio` input is a class:

```bash
rave dimred add --model runs/orchestra_abc123 --name families \
    --audio ~/sounds/strings/ --audio ~/sounds/brass/ --audio ~/sounds/woodwinds/ --type lda
```

or classes are read from file paths with a regular expression (first group = label):

```bash
rave dimred add --model runs/orchestra_abc123 --name instr --audio ~/sounds/orchestra/ \
    --type lda --label_regex "orchestra/([^/]+)/"
```

With 3 classes you get 2 discriminant axes; moving along them morphs the sound between families.

### Non-linear: the mapper

The `mapper` is a small neural auto-encoder trained on the latent space, compressing it to
`--latent_size` dimensions non-linearly. It is computed at export time:

```bash
rave export --run runs/violin_abc123 --streaming --db_path ~/datasets/violin --mapper --latent_size 4
```

## 4. Selecting the audio

`--audio` accepts folders, single files and parts of a preprocessed dataset:

```bash
# only the files matching a pattern
rave dimred add ... --audio ~/sounds/ --filter "*pizz*" --exclude "*_noisy.wav"

# the validation split of the training dataset
rave dimred add ... --audio "~/datasets/violin->val"
```

Encoding can be slow on CPU: use `--device cuda`, limit the data with `--max_chunks 2000`, and add
`--cache` to keep the encoded latents, so that fitting other projection types on the same audio is
instant. `rave dimred purge_cache` clears the cache.

## 5. Projections at export time

The same machinery is available directly from `rave export` with `--db_path`, which also
**recomputes the default PCA** on the given dataset (necessary for Wasserstein models, which do
not compute one during training):

```bash
rave export --run runs/violin_abc123 --streaming \
    --db_path ~/datasets/violin --db_partition val \
    --ica --sfa --pls --descriptors loudness --descriptors f0 --fidelity 0.95
```

Projections computed this way are named after their type (`ica`, `sfa`, `pls`, `cca`, `mapper`).

## 6. Removing projections

```bash
rave dimred remove --model runs/violin_abc123 --name corpus --type ica   # only ica_corpus
rave dimred remove --model runs/violin_abc123 --name corpus              # every *_corpus
```

## 7. Using projections live

In Max (see [Tutorial 1](01_max_msp.md)):

```text
[set projection sfa_corpus(       [set projection default(
          \                         /
          [nn~ violin_streaming encode]   ...   [nn~ violin_streaming decode]
```

Send the message to **both** `encode` and `decode` objects if they are separate instances: each
`nn~` object holds its own copy of the model and its own attribute values. Switching projections
on the fly is click-free but the meaning of every latent signal changes instantly — crossfade if
needed.

## Writing your own projection

Any **invertible linear map** of the latent space can be added by hand. A linear projection is
stored in the checkpoint as a `(4, c, c)` tensor `[mean, components, mixing, info]`, where `c` is
the raw latent size:

```text
projected = (z - mean) @ components.T          z = projected @ mixing.T + mean
```

```python
import torch
import rave
from rave.core import search_for_run

run = "runs/violin_abc123"
model, ckpt_path = rave.load_rave_checkpoint(run)
c = model.latent_size

# your directions: here, a random rotation of the default PCA as an example
q, _ = torch.linalg.qr(torch.randn(c, c))
components = q @ model.latent_pca                    # (c, c), rows = new axes
mixing = torch.linalg.inv(components)
mean = model.latent_mean[None].repeat(c, 1)          # mean, repeated on every row
info = torch.zeros(c, c); info[0, 0] = c             # info[0, 0]: number of fitted dims

buffer = torch.stack([mean, components, mixing, info]).float()

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
ckpt["state_dict"]["projections.wpca_myrotation"] = buffer
torch.save(ckpt, ckpt_path)
```

The key must start with a known prefix to be exported: `pca` for buffers in the PCA format
`[mean, cumulative variance, components...]`, or one of the linear prefixes `ica`, `wpca`, `sfa`,
`pls`, `cca`, `lda` for the format above (the prefix only changes the label shown by
`rave dimred show`). To register a new type properly, add its prefix to
`LINEAR_PROJECTION_PREFIXES` in [`rave/projections.py`](https://github.com/acids-ircam/RAVE/blob/master/rave/projections.py) and to
`PROJECTION_TYPES` in [`rave/scripting.py`](https://github.com/acids-ircam/RAVE/blob/master/rave/scripting.py), and its fitting function to
`fit_projections` — it then becomes available as `rave dimred add --type <yourtype>`.

For variational models, use the posterior **means** to fit your directions: `model.encode(x)`
returns mean and scale stacked along the channel axis, so take `model.encode(x).chunk(2, 1)[0]`.
