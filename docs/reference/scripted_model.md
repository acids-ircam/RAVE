# Exported models (`.ts`)

`rave export` produces a TorchScript file built on [`nn_tilde.Module`](https://github.com/acids-ircam/nn_tilde).
The same file is used by `nn~` / `mcs.nn~` in Max/MSP and Pure Data, by the RAVE VST, by
`rave generate`, and from Python with `torch.jit.load`.

Tensors are shaped `(batch, channels, time)`. Audio is at the model's sampling rate (or at the rate
given by `rave export --sr`). Latent signals run at `sr / ratio`, where `ratio` is the number of audio
samples per latent frame (2048 for `v2`).

## Methods

| Method        | Available for            | Input → output | Description |
| ------------- | ------------------------ | -------------- | ----------- |
| `forward`     | all                      | audio (`n_channels`) → audio | Encode then decode. For continuous models this goes through the full latent space (no projection, no cropping), so it is the highest-fidelity reconstruction. |
| `encode`      | all                      | audio → latent (`latent_size`) | Encode and apply the selected **projection**, cropped to `latent_size` dimensions. |
| `decode`      | all                      | latent (`latent_size`) → audio | Inverse projection, then decode. Missing dimensions are filled with noise scaled by `temperature`. |
| `encode_full` | variational, Wasserstein | audio → latent (`full_latent_size`) | Raw latent space, no projection. |
| `decode_full` | variational, Wasserstein | latent (`full_latent_size`) → audio | Decode a raw latent. |
| `encode_dist` | variational              | audio → `2 × latent_size` | Posterior mean (PCA space) followed by its standard deviation. |
| `prior`       | if exported with `--prior` | temperature (1 channel) → latent | Generate latent trajectories; feed them to `decode`. |

What the latent dimensions mean depends on the model type:

| Model type   | `encode` output |
| ------------ | --------------- |
| variational (`v1`, `v2`, `v3`...) | Coordinates in the selected projection. With the default PCA, dimension 1 carries the most variance, dimension 2 the next, etc. Values are roughly standard-normal. |
| Wasserstein  | Same, but the default projection is the raw latent space unless the model was exported with `--db_path`. |
| discrete     | Code indices (one per quantizer), as floats in `[0, codebook_size - 1]`. |
| spherical    | Angles on the hypersphere (`latent_size - 1` dimensions). |

## Attributes

Attributes are set from Max/Pd with a `set <name> <value>` message to `nn~`
(e.g. `set temperature 0.5`), and from Python with `model.set_<name>(value)` /
`model.get_<name>()`.

| Attribute          | Type        | Default     | Description |
| ------------------ | ----------- | ----------- | ----------- |
| `projection`       | string      | `default`   | Active latent projection (continuous models). `default` is the model's PCA (or the raw space for `_nopca` exports), `raw` bypasses the PCA, other names come from `rave dimred` or `rave export --ica/--sfa/--pls/--cca/--mapper`. |
| `projection_names` | string, read-only | — | Available projections, separated by `|`. |
| `temperature`      | float ≥ 0   | `1.0`       | Scale of the noise used for the dimensions not driven by `decode`'s input, and of the posterior sampling in `encode` (variational). `0` gives deterministic output. |
| `receptive_field`  | (int, int), read-only | — | Receptive field of the model (left, right), in samples. |
| `learn_source` / `learn_target` | bool | `false` | AdaIN models (`v3`, `adain`): while on, the model accumulates the statistics of the incoming (`source`) or reference (`target`) sound. |
| `reset_source` / `reset_target` | bool | `false` | AdaIN models: reset the accumulated statistics (switches back off automatically). |

Changing `projection` does not change the number of inlets / outlets of `encode`/`decode`, which is
fixed at export by `latent_size`.

## Other members

| Member              | Description |
| ------------------- | ----------- |
| `sr`                | Sampling rate. |
| `n_channels`        | Number of audio channels of the model. |
| `latent_size`       | Number of latent dimensions exposed by `encode` / `decode`. |
| `full_latent_size`  | Size of the raw latent space. |
| `fidelity`          | Cumulative explained variance of the default PCA (index `i` = fraction explained by the first `i+1` dims). |
| `gin_config_str`    | Configuration of the model (used by `rave uncompile`). |
| `set_stereo_mode(bool)` | Used by the VST: processes a stereo signal with a mono model (two batch items) or a stereo model. |

## Using a `.ts` from Python

```python
import torch, torchaudio

model = torch.jit.load("my_model_streaming.ts").eval()
x, sr = torchaudio.load("input.wav")                 # (channels, time)
x = torchaudio.functional.resample(x, sr, model.sr)[None, :model.n_channels]

with torch.no_grad():
    z = model.encode(x)                              # (1, latent_size, time / ratio)
    z[:, 0] += 1.0                                   # move along the first latent axis
    y = model.decode(z)                              # (1, n_channels, time)

torchaudio.save("output.wav", y[0], model.sr)
```

A **streaming** export keeps internal buffers between calls (it expects consecutive blocks of one
signal). For offline processing of independent files, prefer a non-streaming export, or call
`rave generate --mode stream --chunk_size N`, which handles the chunking for you.

```python
model.set_projection("pca_strings")   # returns 0 on success, -1 if the name is unknown
print(model.get_projection_names())   # "default|raw|pca_strings"
model.set_temperature(0.)
```
