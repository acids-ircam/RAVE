# Python API

Everything the command line does is available from Python. This page lists the entry points
that are stable enough to script against.

## Loading a trained model

```python
import rave

model, ckpt_path = rave.load_rave_checkpoint("runs/my_model_abc123")
model.eval()
```

`rave.load_rave_checkpoint(model_path, ema=False, name="last.ckpt", remove_keys=None, configs=[], overrides=[], keep_config=False)`

| Argument      | Description |
| ------------- | ----------- |
| `model_path`  | Run folder, version folder or `.ckpt`. A `.ts` path returns `torch.jit.load(model_path)` instead. |
| `ema`         | Load the EMA weights if the run was trained with `--ema`. |
| `name`        | Checkpoint file name to search for; `None` takes the most recently written checkpoint. Raises `FileNotFoundError` if not found. |
| `remove_keys` | Regular expression(s) of weights not to load (e.g. `"discriminator.*"`). |
| `configs`, `overrides` | Extra gin config files / bindings applied on top of the run's `config.gin`. |
| `keep_config` | Leave the run's configuration parsed in gin after loading (needed to build related objects). |

Returns `(rave.RAVE, path_of_the_checkpoint)`.

The returned `rave.RAVE` is a PyTorch Lightning module:

| Method / attribute     | Description |
| ---------------------- | ----------- |
| `encode(x)`            | Audio `(batch, channels, time)` → raw encoder output. For variational models this is mean and scale stacked on the channel axis: use `model.encoder.reparametrize(z)[0]` to sample. |
| `decode(z)`            | Latent → audio. |
| `forward(x)`           | Full reconstruction. |
| `sr`, `n_channels`, `latent_size` | Sampling rate, channels, raw latent size. |
| `latent_pca`, `latent_mean`, `fidelity` | Default PCA computed during validation. |
| `projections`          | `nn.ParameterDict` of custom projections (see `rave dimred`). |
| `import_weights(weight_dict, strict=False)` | Copy weights from a state dict (used by `--init_from`). |
| `state_dict().keys()`  | Weight names, useful to write `--train` / `--freeze` / `--init_from` patterns. |

## Exporting

```python
from rave.scripting import export

ts_path = export("runs/my_model_abc123", streaming=True, out_path="exports/")
```

`rave.scripting.export(run, out_path=None, name=None, streaming=False, fidelity=.95, latent_size=None, no_pca=False, ema_weights=False, sr=None, prior=None, use_causal_stft=True, ckpt_name="last.ckpt", db_path=None, db_partition="all", ica=False, mapper=False, max_chunks=None, device="cpu", mapper_steps=5000, sfa=False, pls=False, cca=False, descriptors=None, pitch_mode="pyin") -> str`

Same arguments as [`rave export`](cli.md#rave-export). Returns the path of the written `.ts`.

## Custom projections

All functions live in `rave.scripting` and accept a run folder, a `.ckpt` or a `.ts`.

```python
from rave.scripting import add_projections, remove_projections, list_projections, describe_projections, format_projections

add_projections("runs/my_model_abc123", name="strings", audio=["/audio/strings"], types=["pca", "sfa"])
print(list_projections("runs/my_model_abc123"))        # ['pca_strings', 'sfa_strings']
print(format_projections(describe_projections("my_model_streaming.ts")))
remove_projections("runs/my_model_abc123", name="strings", types=["sfa"])
```

| Function | Description |
| -------- | ----------- |
| `add_projections(model_path, name, audio, types=("pca",), out=None, overwrite=False, dims=None, fidelity=.99, descriptors=None, label_regex=None, pitch_mode="pyin", filters=None, exclude=None, cache=False, force=False, chunk_length=2**17, batch_size=8, max_chunks=None, device="cpu", ckpt_name="last.ckpt")` | Encode `audio` and fit projections; returns the path of the written model. A `.ts` is uncompiled, edited and re-exported; it is only overwritten if `overwrite=True` (or written to `out`). |
| `remove_projections(model_path, name, types=None, out=None, overwrite=False)` | Remove projections; returns `(path, removed_keys)`. |
| `list_projections(model_path)` | Names of the stored projections. |
| `describe_projections(model_path)` / `format_projections(desc)` | Type and dimensions of every projection, as a dict / a printable table. |
| `purge_latents_cache(model_path=None)` | Delete cached latents. |
| `fit_projections(model, data, name, types, dims=None, fidelity=.99, descriptors=None, labels=None)` | Lower level: fit projections on already-computed latents `(n, latent, time)` and register them in `model.projections`. |
| `collect_latents(model, audio, ...)` / `encode_dataset(model, dataset, ...)` | Encode audio files / an acids-dataset into a latent tensor. |

The projection classes themselves are in `rave.projections`:

| Class / function | Description |
| ---------------- | ----------- |
| `RAVEProjection` | Base class: `forward(z, latent_size=None)` projects, `inverse(z, latent_size=None, temperature=None)` goes back to the raw latent space, filling missing dims with `noise × temperature`. |
| `PCAProjection`  | Stored as a `(2 + c, c)` buffer `[mean, cumulative variance, components]`. |
| `LinearProjection` / `ICAProjection` | Any invertible linear map, stored as a `(4, c, c)` buffer `[mean, components, mixing, info]`: `projected = (z - mean) @ components.T`, `z = projected @ mixing.T + mean`. Used by `ica`, `wpca`, `sfa`, `pls`, `cca`, `lda`. |
| `MapperProjection`, `train_mapper(z, output_size, ...)` | Non-linear MLP auto-encoder projection. |
| `compute_ica`, `compute_whitened_pca`, `compute_sfa`, `compute_pls`, `compute_cca`, `compute_lda` | Return the buffers above from latents `(n, c, t)`. |
| `compute_descriptors(x, sr, n_frames, pitch_mode)` / `DESCRIPTORS` | Audio descriptors (`loudness`, `centroid`, `bandwidth`, `flatness`, `f0`) aligned with latent frames. |

See [Tutorial 5](../tutorials/05_custom_representations.md#writing-your-own-projection) for writing
your own projection type.

## Uncompiling

```python
from rave.uncompiling import uncompile, save_uncompiled

model, config_str, metadata = uncompile("my_model.ts", out_dir="my_model_uncompiled")
```

`uncompile(model_path, out_dir=..., mode="auto", ...)` rebuilds a trainable `rave.RAVE` from a
`.ts`; `metadata` describes the export (streaming, causal, sampling rate, latent size, use of the
PCA, presence of a prior). See [`rave uncompile`](cli.md#rave-uncompile).

## Lower-level helpers

| Function | Description |
| -------- | ----------- |
| `rave.core.search_for_run(path, name="last.ckpt")` | Resolve a checkpoint inside a run folder (`name=None` for the most recent). |
| `rave.core.search_for_config(path)` | Find the `config.gin` of a run. |
| `rave.core.get_minimum_size(model)` | Minimum input length (receptive field) of a model, in samples. |
| `rave.get_run_name(ckpt_path)` | Run name from a checkpoint path. |
| `rave.dataset.get_training_channels(db_path, channels, stereo, midside, ambisonics)` | Channel logic used by `rave train`. |
| `rave.parse_augmentations(names, sr)` | Build acids-dataset transforms from augmentation config names. |
