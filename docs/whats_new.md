# What's new in 2.4

This page lists what changed between RAVE **2.3** and **2.4**, and what to change in existing
scripts and patches. Each item links to the page that documents it in full.

## At a glance

- **New commands:** `rave finetune`, `rave dimred`, `rave uncompile`, `rave adapt`, `rave train_after`.
- **Data:** preprocessing and loading now use [acids-dataset](https://github.com/acids-ircam/acids-dataset).
  You can also train directly on a folder of audio files, without preprocessing.
- **Multichannel:** `--stereo`, `--midside` and `--ambisonics` training modes.
- **Export:** the latent PCA can be recomputed on a dataset (`--db_path`), the latent size can be
  set explicitly (`--latent_size`, `--no_pca`), and custom **projections** can be embedded and
  switched live from Max.
- **Exported models:** new `encode_full`, `decode_full` and `encode_dist` methods, and new
  `projection`, `projection_names`, `temperature` and `receptive_field` attributes. `forward` now
  goes through the full latent space.
- **Architectures:** experimental `v4` (complex-STFT, 2D convolutions) and a spectrogram
  discriminator.
- **Packaging:** `pyproject.toml`, Python ≥ 3.11, scripts moved into the `rave` package, and these
  docs.

---

## Training

### Datasets with acids-dataset

`rave preprocess` is now built on acids-dataset. It still writes a database to `--output_path`, and
gains `--filter` / `--exclude` (file selection), `--meta_regexp` (metadata parsed from file paths)
and `--feature` (extra features computed at preprocessing). See [`rave preprocess`](reference/cli.md#rave-preprocess).

**Training on raw audio.** `--db_path` can now point to a plain folder of audio files. Files are
decoded and resampled on the fly. This is the quickest way to start. For long trainings,
preprocessing is still the faster option. See the
[quick start of Tutorial 2](tutorials/02_train.md#quick-start-train-directly-on-your-audio-files).

**Data augmentations** now come from acids-dataset, with many more transforms (pitch shift,
room simulation, EQ, distortion, noise...). Pass them with `--augment <name>`. The `configs/augmentations/`
folder of RAVE has been removed. See [Data augmentations](reference/configurations.md#data-augmentations).

New data flags:

| Flag | Description |
| ---- | ----------- |
| `--normalize none\|each\|dataset\|dataset_compress` | Was a boolean in 2.3; now selects a normalization strategy. |
| `--one_batch_per_file` | Balances datasets with very uneven file lengths. |

### Multichannel models

| Flag | Description |
| ---- | ----------- |
| `--channels N` | Number of training channels (`0`: from the dataset). |
| `--stereo` | 2-channel L/R training. |
| `--midside` | 2-channel training in mid/side; the exported model still takes and returns L/R. |
| `--ambisonics` | For A-format recordings: trains on their B-format encoding. |

Mid/side and ambisonics are implemented as **output transforms** (`configs/output_transforms/`).
The conversion is embedded in the exported model. See the
[cookbook](cookbook.md#stereo) and [`rave train`](reference/cli.md#channels).

### Devices, resuming, transfer

| 2.3 | 2.4 |
| --- | --- |
| `--gpu 0` | `--device cuda:0` (multi: `--device cuda:0 --device cuda:1` for multi-GPU; `mps`, `cpu`, `auto`) |
| `--ckpt path` | `--resume path` (run folder, version folder or `.ckpt`) |
| — | `--allow_partial_resume`: resume from a checkpoint whose weights do not fully match |
| — | `--init_from`: initialize some or all weights from another run, by regular expression |
| — | `--beta`: target KL / MMD weight, without writing a gin override |

When resuming, the configuration is read from the run. You can still add the *resumable*
configurations `no_encoder_freeze` (keep training the encoder in phase 2) and `adv_at_start`.
See [Run folders and checkpoints](reference/runs.md).

### New configurations

| Config | Description |
| ------ | ----------- |
| `v4` | *(experimental)* Spectrogram-domain model: 2D encoder / decoder on a complex STFT, no PQMF, streamable through causal STFT and cached 2D convolutions. |
| `no_encoder_freeze` | Keep training the encoder during the adversarial phase. |
| `adv_at_start` | *(experimental)* Adversarial losses from the first step. |
| `output_transforms/midside`, `output_transforms/ambisonics` | Selected through `--midside` / `--ambisonics`. |

`v2_small`, `v2_nopqmf` and `v2_nopqmf_small` were updated. See [Configurations](reference/configurations.md).

---

## New commands

| Command | What it does | Docs |
| ------- | ------------ | ---- |
| `rave finetune` | Continues training an existing model on new data, in `encoder`, `decoder`, `full` or `alternating` mode, with per-module learning rates and selectable trainable weights. | [Tutorial 7](tutorials/07_finetune.md) |
| `rave dimred` | Adds, removes or lists custom latent projections (`pca`, `wpca`, `ica`, `sfa`, `pls`, `cca`, `lda`) on a run or directly on a `.ts`. | [Tutorial 5](tutorials/05_custom_representations.md) |
| `rave uncompile` | Turns an exported `.ts` back into a trainable checkpoint (to fine-tune it, or to add projections to it). | [CLI](reference/cli.md#rave-uncompile) |
| `rave adapt` | Fits a linear adapter between the latent spaces of two exported models. | [CLI](reference/cli.md#rave-adapt) |
| `rave train_after` | Trains an [AFTER](https://github.com/acids-ircam/AFTER) diffusion model on a RAVE latent space (`pip install "acids-rave[after]"`). | [CLI](reference/cli.md#rave-train_after) |

`rave generate` was rewritten. It accepts several `--model` and `--input` values, runs in
`--mode stream` and/or `full`, applies input gains (`--loudness`), and runs on any `--device`. It
also accepts training runs as well as `.ts` files. See [Tutorial 4](tutorials/04_command_line.md).

---

## Export

`rave export` gained most of its options in 2.4. See [Tutorial 3](tutorials/03_export_max.md).

| Flag | Description |
| ---- | ----------- |
| `--db_path`, `--db_partition`, `--max_chunks`, `--device` | Recompute the latent PCA, its mean and the fidelity curve on a whole dataset. In 2.3, the PCA came only from training-time validation batches. |
| `--latent_size` | Explicit number of latent dimensions (overrides `--fidelity`). |
| `--no_pca` | Export the raw latent space, without PCA rotation. |
| `--ica`, `--sfa`, `--pls`, `--cca`, `--mapper` | Embed custom projections computed on `--db_path`. |
| `--ckpt_name` | Choose the checkpoint (`last.ckpt`, `best.ckpt`...). |
| `--name`, `--out_path` | Model name and output folder / file (`--output` in 2.3). |
| `--use_causal_stft` | For STFT-based models (`v4`) in streaming mode. |

**Wasserstein models** are now exported like variational ones, through projections. They compute
no PCA during training, so without `--db_path` they are exported in their raw latent space (with a
`_nopca` suffix).

The exported file embeds the **gin configuration** of the model, which is what makes
`rave uncompile` possible.

---

## Exported models (`.ts`)

See [Exported models](reference/scripted_model.md) for the complete reference.

### New methods

| Method | Description |
| ------ | ----------- |
| `encode_full` / `decode_full` | Raw latent space (`full_latent_size`), with no projection and no cropping. |
| `encode_dist` | Posterior mean and standard deviation (variational models). |

They are explained in [Tutorial 3, step 9](tutorials/03_export_max.md#9-beyond-encode-decode-encode_full-decode_full-encode_dist).

### New attributes

| Attribute | Description |
| --------- | ----------- |
| `projection` | Active latent projection: `default` (PCA), `raw`, or any custom projection. Switch live with `set projection <name>`. |
| `projection_names` | Read-only list of available projections, separated by `\|`. |
| `temperature` | Scales the posterior sampling and the noise used for the latent dimensions not driven by `decode`. `0` = deterministic. |
| `receptive_field` | Read-only receptive field (left, right) in samples. |

### Behaviour changes

- **`forward` uses the full latent space.** In 2.3, `forward` was `decode(encode(x))`, so it was
  cropped to the exported latent size and the remaining dimensions were filled with noise. In 2.4,
  `forward` is `decode_full(encode_full(x))`, the best reconstruction available, whatever the
  `--fidelity`. Patches using `forward` will sound closer to the input than before.
- **Latent dimensions may have changed** if you re-export with `--db_path`: the PCA is recomputed
  on the whole dataset. Re-tune patches that depend on specific latent axes.
- **Decoded noise is scaled by `temperature`.** In 2.3, the dimensions not driven by `decode`'s
  inlets were filled with unit noise. They are now multiplied by `temperature` (default `1`, so the
  default sound is unchanged).

---

## Python API

- `rave.load_rave_checkpoint(path)` loads a run, a version folder or a `.ckpt` (with its
  configuration) and returns `(model, ckpt_path)`.
- `rave.scripting.export(...)` exports from Python, with the same arguments as `rave export`.
- `rave.scripting.add_projections`, `remove_projections`, `list_projections`... manage projections
  from Python.
- `rave.uncompiling` holds the uncompiling tools.

See [Python API](reference/python_api.md).

---

## Packaging

- Installation is driven by `pyproject.toml` (`setup.py` was removed). Python ≥ 3.11 is required.
- Command-line scripts moved from `scripts/` to `rave/scripts/`. The `rave` entry point is
  unchanged.
- Optional extra: `pip install "acids-rave[after]"` for `rave train_after`.
- `rave remote_dataset` and `rave export_onnx` are kept as **legacy** commands. `remote_dataset`
  does not support acids-dataset databases.

---

## Migrating from 2.3

| You used to... | Now |
| -------------- | --- |
| `rave train --gpu 0` | `rave train --device cuda:0` |
| `rave train --ckpt runs/my_run` | `rave train --resume runs/my_run` |
| `rave train --normalize` | `rave train --normalize each` (or `dataset`, `dataset_compress`) |
| `rave train --augment mute` (RAVE's own `compress` / `gain` / `mute` configs) | Same syntax, but names now refer to acids-dataset transforms; see [Data augmentations](reference/configurations.md#data-augmentations). |
| `rave export --output folder/` | `rave export --out_path folder/` |
| `rave generate --gpu 0 --stream` | `rave generate --device cuda:0 --mode stream` |
| Datasets preprocessed with RAVE 2.3 (`udls`) | Preprocess again with `rave preprocess`, or train directly on the audio folder. |
| `.ts` exported with 2.3 | Still load in `nn~`. Re-export them to get projections and the new methods, or use `rave uncompile` on them. |
