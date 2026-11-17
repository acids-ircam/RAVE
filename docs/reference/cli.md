# Command-line reference

All functionality is exposed through a single entry point:

```bash
rave <command> [--flag value ...]
```

| Command                           | Purpose                                                                 |
| --------------------------------- | ----------------------------------------------------------------------- |
| [`preprocess`](#rave-preprocess)  | Turn a folder of audio files into a training dataset                    |
| [`train`](#rave-train)            | Train a RAVE model                                                      |
| [`finetune`](#rave-finetune)      | Continue training an existing model, possibly on new data / with frozen parts |
| [`export`](#rave-export)          | Script a trained model into a `.ts` file for `nn~`, the VST or Python   |
| [`generate`](#rave-generate)      | Process audio files offline with a model                                |
| [`dimred`](#rave-dimred)          | Add / remove / list custom latent projections                           |
| [`train_prior`](#rave-train_prior)| Train an autoregressive prior on a model's latent space                 |
| [`train_after`](#rave-train_after)| Train an AFTER diffusion model on a RAVE latent space (optional dependency) |
| [`uncompile`](#rave-uncompile)    | Turn an exported `.ts` back into a trainable checkpoint                 |
| [`adapt`](#rave-adapt)            | Fit a linear adapter between the latent spaces of two exported models   |
| [`export_onnx`](#legacy-commands), [`remote_dataset`](#legacy-commands) | Legacy commands (see below)                 |

Flags follow the [absl](https://abseil.io/docs/python/guides/flags) conventions:

- boolean flags are set with `--flag` and unset with `--noflag` (e.g. `--streaming`, `--noprogress`);
- *multi* flags are repeated: `--config v2 --config causal`;
- *list* flags take a comma-separated value: `--rand_pitch 0.8,1.2`;
- `rave <command> --helpfull` prints every flag with its default.

Paths given to `--run`, `--model` or `--resume` can be a run folder, a version folder or a
`.ckpt` file: see [Run folders and checkpoints](runs.md) for how RAVE resolves them.

---

## `rave preprocess`

Builds a dataset from audio files with [acids-dataset](https://github.com/acids-ircam/acids-dataset).
Audio is decoded, resampled, cut into chunks of `2 × num_signal` samples, and stored in a database
in `--output_path`.

```bash
rave preprocess --input_path /audio/folder --output_path /data/my_dataset --channels 1
```

| Flag             | Default  | Description |
| ---------------- | -------- | ----------- |
| `--input_path`   | required | Folder containing the audio files (searched recursively). |
| `--output_path`  | required | Output folder of the dataset. |
| `--num_signal`   | `131072` | Training chunk length in samples. Stored chunks are twice this length. Must match `--n_signal` at training time if you change it. |
| `--channels`     | `1`      | Number of audio channels stored. Use `2` for stereo models. |
| `--sampling_rate`| `44100`  | Sampling rate of the stored audio. Must match the `SAMPLING_RATE` of the configuration you train with. |
| `--max_db_size`  | `100`    | Maximum size of the database, in GB. |
| `--filter`       | —        | (multi) Wildcard selecting files to include, e.g. `--filter "*violin*"`. |
| `--exclude`      | —        | (multi) Wildcard of files to exclude. |
| `--meta_regexp`  | —        | (multi) Regular expression to parse metadata from file paths. |
| `--feature`      | —        | (multi) Additional acids-dataset feature configs computed at preprocessing. |

Notes:

- Chunks overlap; the processed dataset is often **larger** than the original audio.
- If preprocessing seems stuck at `0it`, your files are shorter than a chunk: lower
  `--num_signal` (and pass the same value as `--n_signal` to `train`).

---

## `rave train`

Trains a model from scratch (or resumes a training).

```bash
rave train --name my_model --db_path /data/my_dataset --out_path runs/ --config v2
```

### Required

| Flag        | Description |
| ----------- | ----------- |
| `--name`    | Run name. The run folder is `<out_path>/<name>_<hash>`, where `<hash>` is computed from the configuration. |
| `--db_path` | Preprocessed dataset (output of `rave preprocess`). |

### Model configuration

| Flag          | Default  | Description |
| ------------- | -------- | ----------- |
| `--config`    | `v2`     | (multi) Gin configuration(s), by name (`v2`, `causal`, `wasserstein`...) or by path. Applied in order. **Passing any `--config` replaces the default**: write `--config v2 --config causal`, not just `--config causal`. See [Configurations](configurations.md). |
| `--override`  | —        | (multi) Extra gin binding, e.g. `--override "CAPACITY=64"`. |
| `--augment`   | —        | (multi) Data augmentation config from acids-dataset, e.g. `--augment mute --augment compress`. See [Configurations](configurations.md#data-augmentations). |
| `--beta`      | config   | Target regularization weight (KL weight for variational models, MMD weight for Wasserstein models). Sets `BetaWarmupCallback.target_value`. |

### Channels

| Flag            | Default | Description |
| --------------- | ------- | ----------- |
| `--channels`    | `0`     | Number of channels to train on. `0` uses the number of channels of the dataset. |
| `--stereo`      | off     | Shortcut for 2-channel (left/right) training. |
| `--midside`     | off     | 2-channel training in mid/side representation (the model encodes M/S and decodes back to L/R). |
| `--ambisonics`  | off     | **Only for A-format signals** (raw capsules of an ambisonic microphone): trains on their B-format encoding, and the exported model takes and returns A-format. The dataset must contain `(order+1)²` channels (4, 9, 16...). For B-format recordings, do not use it: `--channels 4` (9, 16...) is enough. |

Only one of `--channels`, `--stereo`, `--midside`, `--ambisonics` may be given. The dataset must
have been preprocessed with at least as many channels.

### Data

| Flag                   | Default  | Description |
| ---------------------- | -------- | ----------- |
| `--n_signal`           | `131072` | Training chunk length (samples). Must be ≤ the `--num_signal` used at preprocessing. |
| `--batch`              | `8`      | Batch size. |
| `--workers`            | `-1`     | Data-loading workers. `-1` = number of CPUs (capped at 32). Set `0` if you hit multiprocessing issues (Windows, macOS). |
| `--normalize`          | `none`   | Input normalization: `none`, `each` (normalize each chunk), `dataset` (by the peak of the whole dataset), `dataset_compress` (compress into a target loudness range computed on the dataset). |
| `--one_batch_per_file` | off      | Sample one chunk per file per batch instead of flattening all chunks (balances datasets with very uneven file lengths). |
| `--derivative`         | off      | Train on the derivative of the signal. |
| `--rand_pitch`         | —        | (list) Random time-stretch range, e.g. `0.8,1.2`. |

### Training schedule

| Flag            | Default   | Description |
| --------------- | --------- | ----------- |
| `--max_steps`   | `6000000` | Total number of training steps. |
| `--val_every`   | `10000`   | Run validation every N steps (also logs audio examples and updates the latent PCA). |
| `--save_every`  | `200000`  | Write a numbered checkpoint every N training batches. `last.ckpt` is refreshed every 25,000 batches. |
| `--seed`        | `0`       | Random seed. |
| `--ema`         | —         | Keep an exponential moving average of the weights with this factor (e.g. `0.999`). Export them with `rave export --ema_weights`. |
| `--device`      | `auto`    | (multi) Training device: `auto`, `cpu`, `cuda`, `cuda:0`, `mps`... Several `--device cuda:0 --device cuda:1` enable multi-GPU training. |
| `--progress`    | on        | Show the progress bar (`--noprogress` to hide it, e.g. in log files). |
| `--smoke_test`  | off       | Run one training and one validation batch per epoch — useful to check that a configuration builds and runs. |

### Resuming and transfer

| Flag                     | Default | Description |
| ------------------------ | ------- | ----------- |
| `--resume`               | —       | Run folder or `.ckpt` to resume from. The configuration is read from the run; only "resumable" configs (`no_encoder_freeze`, `adv_at_start`) can be added with `--config`. |
| `--allow_partial_resume` | off     | Allow resuming from a checkpoint whose keys do not fully match. |
| `--init_from`            | —       | (multi) Initialize weights from another run before training: `path/to/run` (all weights) or `path/to/run->encoder,decoder` / `path/to/run->^decoder\.net\.[0-3].*` (shortcuts `all`, `encoder`, `decoder`, `discriminator`, or regular expressions on weight names). |
| `--strict_init`          | on      | Fail if `--init_from` weights do not match the new model (`--nostrict_init` to allow partial transfer). |

### Training phases

RAVE trains in two phases:

1. **Representation learning** (`phase_1_duration` steps, 1M for `v1`/`v2`, 200k for
   `discrete`/`wasserstein`/`spherical`): encoder and decoder are trained with a spectral
   reconstruction loss and the latent regularization.
2. **Adversarial fine-tuning**: the encoder is frozen (unless `no_encoder_freeze`) and the decoder
   is trained against a discriminator. Reconstruction losses usually go *up* when phase 2 starts —
   this is expected (see the [Tensorboard guide](../tensorboard_guide.md)).

---

## `rave finetune`

Loads an existing run and continues training it, usually on a different dataset, with fresh
optimizers and a selectable set of trainable weights. See [Tutorial 7](../tutorials/07_finetune.md) for when to use each mode.

```bash
rave finetune --name voice_ft --run runs/flute_abc123 --db_path /data/voice --mode encoder
```

| Flag                      | Default    | Description |
| ------------------------- | ---------- | ----------- |
| `--name`                  | required   | Name of the fine-tuning run. Written to `<out_path>/<name>/version_N/`. |
| `--run`                   | required   | Source run (folder or `.ckpt`). Its configuration is reused. |
| `--db_path`               | required   | Preprocessed dataset. |
| `--out_path`              | `runs/`    | Output folder. |
| `--mode`                  | `encoder`  | Objective, see below: `encoder`, `decoder`, `full`, `alternating`. |
| `--train`                 | `all`      | (multi) Weights to train: `all`, `encoder`, `decoder`, `discriminator`, or a regular expression on weight names. Intersected with `--mode`. |
| `--freeze`                | —          | (multi) Weights to freeze (same syntax), removed from the `--train` selection. |
| `--lr_encoder`            | `1e-5`     | Encoder learning rate. |
| `--lr_decoder`            | `1e-4`     | Decoder learning rate. |
| `--lr_discriminator`      | `1e-4`     | Discriminator learning rate. |
| `--encoder_start_step`    | `0`        | Number of fine-tuning batches during which the encoder stays fixed. |
| `--max_steps`             | `6000000`  | Total number of fine-tuning batches (including those already done when resuming). |
| `--beta`                  | source / 4 | KL weight of a variational encoder; an explicit value overrides the default (`0` disables KL). |
| `--update_pca`            | on         | Recompute the latent PCA at validation (`--noupdate_pca` keeps the source PCA, so the exported latent axes stay comparable with the source model). |
| `--reset_discriminator`   | off        | Re-initialize the discriminator (and train it). |
| `--ema_weights`           | off        | Start from the EMA weights of the source run, if available. |
| `--resume`                | —          | Resume an interrupted fine-tuning (same mode / learning rates / settings). |
| `--override`              | —          | (multi) Gin bindings applied on top of the source configuration. |
| `--augment`, `--n_signal`, `--batch`, `--workers`, `--device`, `--normalize`, `--derivative`, `--rand_pitch`, `--one_batch_per_file`, `--val_every`, `--save_every`, `--seed`, `--progress`, `--smoke_test`, `--allow_partial_resume` | | As in `rave train`. |

Modes:

- **`encoder`** — only encoder weights are updated, with reconstruction and latent regularization
  losses; the decoder is fixed. Use it to make a model "understand" a new kind of input (e.g. timbre
  transfer from voice to a flute model).
- **`decoder`** — only decoder weights are updated, with reconstruction, adversarial and feature-matching
  losses against the source discriminator, which stays fixed (do not combine with `--reset_discriminator`).
- **`full`** — encoder and decoder are trained with reconstruction, adversarial, feature-matching and
  regularization losses; the discriminator is updated every `update_discriminator_every` batches.
- **`alternating`** — even batches update decoder + discriminator (reconstruction + adversarial), odd
  batches update the encoder (reconstruction + regularization).

---

## `rave export`

Scripts a trained run into a TorchScript `.ts` file, usable with `nn~` (Max/MSP, Pure Data), the
RAVE VST, `rave generate` and `torch.jit.load`. See [Tutorial 3](../tutorials/03_export_max.md) for a walkthrough.

```bash
rave export --run runs/my_model_abc123 --streaming
```

| Flag                | Default      | Description |
| ------------------- | ------------ | ----------- |
| `--run`             | required     | Run folder or `.ckpt`. |
| `--streaming`       | off          | Use cached convolutions for real-time, block-based processing. **Required for `nn~` and the VST** — without it you will hear clicks. |
| `--out_path`        | run's parent | Output folder, or full path ending in `.ts`. |
| `--name`            | run name     | Model name. `_nopca` and `_streaming` are appended when relevant. |
| `--ckpt_name`       | `last.ckpt`  | Checkpoint to export; falls back to the most recently written one. Use `best.ckpt` for the best validation checkpoint. |
| `--ema_weights`     | off          | Export the EMA weights (if trained with `--ema`). |
| `--sr`              | model sr     | Target sampling rate; the model resamples internally. Must be a multiple of the model's rate (e.g. 88200 for a 44100 model). |
| `--use_causal_stft` | on           | For STFT-based models (`v4`), replace the streamable STFT with a causal one in streaming mode. |
| `--prior`           | —            | Path of a prior run (`rave train_prior`) to embed; adds a `prior` method. |

Latent size (continuous models: variational and Wasserstein):

| Flag             | Default | Description |
| ---------------- | ------- | ----------- |
| `--fidelity`     | `0.95`  | Keep the number of PCA dimensions explaining this fraction of the latent variance (rounded up to a power of two). |
| `--latent_size`  | —       | Explicit number of latent dimensions (overrides `--fidelity`). |
| `--no_pca`       | off     | Expose the raw latent space (no PCA rotation), with the full latent size. Automatic for Wasserstein models exported without `--db_path`. |

Recomputing statistics and adding projections at export (see [Tutorial 5](../tutorials/05_custom_representations.md)):

| Flag             | Default | Description |
| ---------------- | ------- | ----------- |
| `--db_path`      | —       | Dataset (preprocessed or raw folder) used to recompute the default PCA, its mean and the fidelity curve from its latents with the final model. Recommended: during training the PCA is only estimated on one validation batch. Required for the projections below, and for a PCA on Wasserstein models. Do not combine with `--prior`. |
| `--db_partition` | `all`   | `all`, `train` or `val` (the split written during training). |
| `--max_chunks`   | all     | Encode at most N randomly chosen chunks. |
| `--device`       | `cpu`   | Device used to encode the dataset / train the mapper. |
| `--ica`          | off     | Add an `ica` projection (independent components). |
| `--sfa`          | off     | Add an `sfa` projection (slow feature analysis). |
| `--pls`, `--cca` | off     | Add `pls` / `cca` projections aligned with audio descriptors. |
| `--descriptors`  | all     | (multi) Descriptors for `pls`/`cca`: `loudness`, `centroid`, `bandwidth`, `flatness`, `f0`. |
| `--pitch_mode`   | `pyin`  | F0 tracker for the `f0` descriptor (`pyin`, `yin`, and `crepe`/`pesto`/`world`/`praat` if installed). |
| `--mapper`       | off     | Add a `mapper` projection (small MLP auto-encoder on the latent space). |
| `--mapper_steps` | `5000`  | Training steps of the mapper. |

After exporting, the script checks whether the model works in stereo mode and tells you whether it
is compatible with the RAVE VST.

---

## `rave generate`

Processes audio files (or folders) with one or several models and writes the results.

```bash
rave generate --model my_model_streaming.ts --input sounds/ --out_path generations/
```

| Flag           | Default       | Description |
| -------------- | ------------- | ----------- |
| `--model`      | required      | (multi) Exported `.ts` or run folder / `.ckpt`. |
| `--input`      | required      | (multi) Audio file or folder (`.wav`, `.aif`, `.aiff`, `.opus`, `.mp3`, `.aac`; searched recursively). |
| `--out_path`   | `generations` | Output folder. Files are written to `<out_path>/<model name>/<mode>/<file>_loudness=<gain>.wav`. |
| `--mode`       | `stream`      | (multi) `stream` (cached convolutions, chunk by chunk) or `full` (whole file at once). |
| `--chunk_size` | whole file    | Chunk size in samples for `stream` mode. Must be larger than the model's receptive field. |
| `--loudness`   | `0`           | (multi) Gain in dB applied to the input before processing. Several values produce several outputs. |
| `--device`     | `cpu`         | `cpu`, `cuda`, `cuda:0`, `mps`... |

Inputs are resampled to the model's sampling rate. Multichannel inputs are truncated to the
model's number of channels.

---

## `rave dimred`

Manages *custom latent projections* (dimensionality reductions) of a model: alternative coordinate
systems on the latent space, selectable at runtime through the `projection` attribute. Works on
run folders / `.ckpt` (edited in place) and on exported `.ts` files (uncompiled, edited and
re-exported with the same settings). See [Tutorial 5](../tutorials/05_custom_representations.md).

```bash
rave dimred [add|remove|show|purge_cache] --model M [--name N] [--audio A ...] [options]
```

| Mode          | Description |
| ------------- | ----------- |
| `add` (default) | Encode `--audio` with the model and fit projections named `--name`. |
| `remove`      | Remove projections `--name` (of the given `--type`, default: all types). |
| `show`        | Print the projections stored in a model, with their dimensions. |
| `purge_cache` | Delete cached latents (all, or of `--model` only). |

| Flag            | Default      | Description |
| --------------- | ------------ | ----------- |
| `--model`       | required     | Run folder, `.ckpt` or `.ts`. |
| `--name`        | required for add/remove | Projection name. Stored as `<type>_<name>`, e.g. `pca_strings`. |
| `--audio`       | required for add | (multi) Audio folders / files, or part of a preprocessed dataset split: `/data/db->train`, `/data/db->val`, `/data/db->partition_file:part`. |
| `--type`        | `pca`        | (multi) Projection type(s), see table below. |
| `--filter`, `--exclude` | —    | (multi) Glob patterns on file paths relative to each `--audio` folder. |
| `--dims`        | —            | Number of fitted dimensions (overrides `--fidelity`). |
| `--fidelity`    | `0.99`       | Explained variance defining the number of dimensions when `--dims` is not given. |
| `--descriptors` | all          | (multi) For `pls` / `cca`: `loudness`, `centroid`, `bandwidth`, `flatness`, `f0`. |
| `--pitch_mode`  | `pyin`       | F0 tracker for the `f0` descriptor. |
| `--label_regex` | —            | For `lda`: regular expression on file paths whose first group is the class label (default: one class per `--audio` input). |
| `--out`         | —            | Write to a new file instead of editing the model in place. |
| `--yes`         | off          | Do not ask before overwriting a `.ts`. |
| `--chunk_length`| `131072`     | Length of encoded audio chunks. |
| `--batch_size`  | `8`          | Encoding batch size. |
| `--max_chunks`  | all          | Encode at most N random chunks. |
| `--device`      | `cpu`        | Encoding device. |
| `--cache`       | off          | Cache encoded latents in `rave/.dimred_cache/` and reuse them. |
| `--force`       | off          | Re-encode even if a valid cache exists. |
| `--ckpt_name`   | `last.ckpt`  | Checkpoint name when `--model` is a run folder. |

| Type   | Name                              | Supervision      | What you get |
| ------ | --------------------------------- | ---------------- | ------------ |
| `pca`  | Principal component analysis      | —                | Axes sorted by variance on *your* audio. |
| `wpca` | Whitened PCA                      | —                | Same axes, each scaled to unit variance. |
| `ica`  | Independent component analysis    | —                | Statistically independent axes (on the whitened PCA subspace). |
| `sfa`  | Slow feature analysis             | —                | Axes sorted from slowest- to fastest-varying in time. |
| `pls`  | Partial least squares             | audio descriptors| Axes that predict loudness, brightness, pitch... |
| `cca`  | Canonical correlation analysis    | audio descriptors| Axes maximally correlated with the descriptors. |
| `lda`  | Linear discriminant analysis      | class labels     | Axes that separate classes (one class per `--audio` input, or `--label_regex`). |

Projections are only used by continuous models (variational and Wasserstein). Discrete models are
not supported.

---

## `rave train_prior`

Trains an autoregressive prior on the latent space of a trained model, so that the model can
generate audio without input. See [Tutorial 6](../tutorials/06_prior.md).

```bash
rave train_prior --name my_prior --model runs/my_model_abc123 --db_path /data/my_dataset --out_path runs/
```

| Flag            | Default          | Description |
| --------------- | ---------------- | ----------- |
| `--model`       | required         | Trained RAVE run (folder or `.ckpt`). |
| `--db_path`     | required         | Preprocessed dataset (usually the one the model was trained on). |
| `--name`        | —                | Run name; the run folder is `<out_path>/<name>_<hash>`. Always give one. |
| `--out_path`    | `runs/`          | Output folder. |
| `--config`      | `prior/prior_v1` | (multi) Prior configuration. |
| `--override`    | —                | (multi) Gin bindings. |
| `--fidelity`    | `0.99`           | Number of modeled latent dimensions, from the model's PCA (rounded up to a power of two). |
| `--latent_size` | —                | Explicit number of modeled dimensions (overrides `--fidelity`). |
| `--n_signal`    | from config      | Chunk length; at least the prior's receptive field. |
| `--batch`       | `8`              | Batch size. |
| `--max_steps`   | `1000000`        | Training steps. |
| `--val_every`, `--save_every`, `--device`, `--workers`, `--resume`, `--seed`, `--progress`, `--smoke_test`, `--normalize`, `--derivative`, `--rand_pitch` | | As in `rave train`. |

The prior's latent size must match the latent size of the export: export with the same
`--fidelity` (or `--latent_size`) as used here — note that the export default (`0.95`) differs from
the prior default (`0.99`).

---

## `rave train_after`

Trains an [AFTER](https://github.com/acids-ircam/AFTER) diffusion model on top of a RAVE latent
space. Requires `pip install "acids-rave[after]"`.

```bash
rave train_after --name my_after --model runs/my_model_abc123 --db_path /data/my_dataset --device cuda:0
```

`--model` accepts an exported `.ts`, or a run folder, version folder or `.ckpt`. A run or checkpoint is
first exported (non-streaming) to `<out_path>/<name>/rave/`, with its latent size chosen by
`--fidelity` (0.99) or `--latent_size`; `--ckpt_name` and `--ema_weights` select the weights as in
`rave export`. When resuming, that exported model is reused (`--force` re-exports it). For `.ts`
models, the exported latent size is used.

Main flags: `--name`, `--db_path` (multi; several datasets can be mixed with
`--db_weights`), `--arch` (`rectified` or `edm`), `--config`, `--out_path`, `--device`, `--batch`
(32), `--n_signal`, `--adv_weight` (0.05), data augmentation (`--stretch`,
`--stretch_range 0.7,1.6`, `--shift`, `--shift_range -6,6`, `--n_augmentations 4`), structure
conditioning (`--structure`, `--midi`, `--beat`), and `--resume` (AFTER run folder or checkpoint;
`--ckpt` is kept as a deprecated alias). Use `--helpfull` for the full list.

---

## `rave uncompile`

Converts an exported `.ts` back into a trainable checkpoint (e.g. to fine-tune a model you only have
as a `.ts`, or to add projections to it).

```bash
rave uncompile --model my_model.ts --out my_model_uncompiled --check
```

| Flag           | Default                  | Description |
| -------------- | ------------------------ | ----------- |
| `--model`      | required                 | Exported `.ts`. |
| `--out`        | `<model>_uncompiled/`    | Output folder; receives `last.ckpt`, `config.gin` and `metadata.json`. |
| `--mode`       | `auto`                   | How the configuration is recovered: `embedded` (config stored at export), `file` (a `config.gin` next to the `.ts`, or `--config`), `infer` (rebuilt from the architecture), `shell` (inference only, nothing written), `auto` (first that works). |
| `--config`     | —                        | (multi) Gin config(s) to rebuild the model with. |
| `--override`   | —                        | (multi) Extra gin bindings. |
| `--ckpt_name`  | `last.ckpt`              | Name of the written checkpoint. |
| `--reference`  | —                        | Original run / ckpt to compare weights against. |
| `--check`      | off                      | Run round-trip and functional checks. |

The discriminator is not part of the export: it is freshly initialized in the uncompiled checkpoint.
A prior embedded in the `.ts` cannot be recovered.

---

## `rave adapt`

Fits a linear map between the latent spaces of two exported models, and exports it as an `nn~`
model, so that `to_model.decode(adapter(from_model.encode(x)))` reconstructs `x`.

```bash
rave adapt --from_model a.ts --to_model b.ts --db_path /data/dataset --name a_to_b --out_path adapters/
```

Flags: `--from_model`, `--to_model`, `--db_path`, `--name` (required); `--out_path` (`runs/`),
`--n_signal` (126976), `--seed`, `--data_gain`, `--bias`, `--clip`, `--data_normalize`,
`--data_derivative`.

---

## Legacy commands

- `rave export_onnx --run RUN` — ONNX export of the `onnx`/`raspberry` (v1-based) configurations.
  Relies on the old dependency `effortless_config`; prefer `rave export` + `nn~` for Raspberry Pi.
- `rave remote_dataset --db_path DB [--port 5000]` — serves a dataset over HTTP. Relies on the
  pre-2.4 dataset format (`udls`) and is not compatible with acids-dataset databases.
