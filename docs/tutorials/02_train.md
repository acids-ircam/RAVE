# Tutorial 2 — Train a RAVE model

This tutorial takes you from a folder of audio files to a model you can play in Max.

**You need:** a machine with an NVIDIA GPU (16 GB of memory for the default `v2` configuration),
RAVE installed (see [Installation](../index.md#installation)), and some audio.

Training a full model takes a few days. Plan for it: run it in `tmux`/`screen` or as a background
job, and check on it with tensorboard.

## Quick start: train directly on your audio files

You can skip preprocessing entirely and point `--db_path` at a plain folder of audio files:

```bash
rave train \
    --config v2 \
    --db_path ~/audio/my_corpus \
    --out_path ~/runs \
    --name my_corpus \
    --channels 1 \
    --augment mute \
    --device cpu
```

If you train on GPU, replace `--device cpu` by `--device cuda` (or `cuda:X` if you have several GPUs, replacing `X` by the GPU ID). You can add several devices for multi-GPU training.

Any folder that is not a preprocessed dataset is read **on the fly**: files are decoded,
resampled to the model's sampling rate (44.1 kHz) and converted to `--channels` channels as
training goes. Then export the result as in [step 6](#6-export-the-model):

```bash
rave export --run ~/runs/my_corpus_<hash> --streaming --db_path ~/audio/my_corpus
```

This is the fastest way to get started, with a few things to know:

- **Higher CPU load during training.** Every chunk is decoded and resampled again at each step. With
  compressed formats (mp3, ogg) or many files, data loading can become the bottleneck. If the GPU
  sits idle, preprocess instead.
- **Hidden files are written into your audio folder:** a small index of the files (`.ad`), the
  detected channel count (`.ad_channels`) and the train/validation split. They make later runs start
  faster; you can delete them at any time.
- **Pass `--channels` explicitly** when your files mix mono and stereo. Without it, RAVE trains on
  the largest channel count it finds.

The rest of this tutorial uses the preprocessed workflow, which is the recommended one for long
trainings. Every `rave train` option shown below works the same with a raw folder.

## 1. Gather a dataset

- **Quantity:** at least 1–3 hours of audio; more for varied material (mixtures, many
  instruments). With less than an hour, [data augmentations](#data-augmentations) are mandatory.
- **Homogeneity:** RAVE learns _one_ sound world. A model trained on a single instrument or a
  coherent style sounds much better than one trained on "everything".
- **Format:** any format readable by ffmpeg (wav, aiff, flac, mp3, ogg...), any sampling rate, in a
  single folder (subfolders are fine).
- **Cleanliness:** remove long silences, clicks and unwanted noise — the model will reproduce them.

!!! tip "Check your dataset's health before training"
    [acids-dataset](https://github.com/acids-ircam/acids-dataset), installed with RAVE, can scan your
    files and flag the usual problems: **clipping**, **DC offset**, **silent** or **mostly silent**
    files, and **very quiet** files.

    ```bash
    acids-dataset info --path ~/audio/my_corpus --health --nowrite_index
    ```

    After a description of the dataset, the `health` section prints the number of files per problem,
    then each flagged file with its problems, peak level, clipping ratio, DC offset and ratio of
    silent frames:

    ```text
    flagged files :
    /…/my_corpus/take_03.wav: clipping (peak 0.0 dB, clipping 3.20e-03, dc 0.001, silence 0.04)
    /…/my_corpus/room_tone.wav: mostly silent, very quiet (peak -52.3 dB, clipping 0.00e+00, dc 0.000, silence 0.91)
    ```

    Fix or remove these files before training: silent and clipped material in particular wastes
    training time and leaks into the model's sound.

    - It works on raw audio folders and on preprocessed datasets (`--path ~/datasets/my_corpus`).
    - `--health_max_files 200` checks only the first files of a large corpus; `--filter` / `--exclude`
      select files in a raw folder.
    - `--nowrite_index` avoids writing the hidden `.ad` index into the folder; drop it if you plan to
      [train directly on the folder](#quick-start-train-directly-on-your-audio-files) — the index
      then speeds up the first training.
    - To listen to flagged files and browse the dataset interactively, use the dataset browser:
      `acids-dataset ui --path ~/audio/my_corpus`.

    A file is flagged when more than 0.01 % of its samples clip, its DC offset exceeds 0.01, more than
    half of it is silent (below −60 dB), or its peak stays below −40 dB.

## 2. Preprocess

```bash
rave preprocess \
    --input_path ~/audio/my_corpus \
    --output_path ~/datasets/my_corpus \
    --channels 1
```

This resamples everything to 44.1 kHz, cuts it into chunks and stores it in a database. Use
`--channels 2` if you want to train a stereo model later (see the [Cookbook](../cookbook.md#stereo)).

## 3. Choose a configuration

| If you want...                                      | Use                                  |
| --------------------------------------------------- | ------------------------------------ |
| A good general-purpose model (default)              | `--config v2`                        |
| Low latency for live use                            | `--config v2 --config causal`        |
| A smoother, more controllable, smaller latent space | `--config v2 --config wasserstein`   |
| A lighter / faster model, or a GPU with 8 GB        | `--config v2_small`                  |
| A model for a Raspberry Pi                          | `--config raspberry --config causal` |
| Real-time style transfer with AdaIN (32 GB GPU)     | `--config v3`                        |
| Discrete tokens (for token-based priors)            | `--config discrete`                  |

See [Configurations](../reference/configurations.md) for everything else. Remember that passing
`--config` replaces the default: always include an architecture first.

## 4. Train

```bash
rave train \
    --config v2 \
    --db_path ~/datasets/my_corpus \
    --out_path ~/runs \
    --name my_corpus \
    --channels 1 \
    --augment mute
```

(`--augment mute` teaches the model silence: see [Data augmentations](#data-augmentations).)

The run is written to `~/runs/my_corpus_<hash>/`. Before launching days of training, you can check
that everything builds and runs with `--smoke_test`:

```bash
rave train --config v2 --db_path ~/datasets/my_corpus --out_path /tmp/test --name test --smoke_test --max_steps 10
```

Useful options:

- `--device cuda:1` to pick a GPU, `--device cuda:0 --device cuda:1` for two GPUs.
- `--batch 16` if you have memory to spare; lower it (`--batch 4`) if you run out of memory.
- `--workers 0` if data loading crashes (frequent on Windows / macOS).
- `--save_every 100000` to keep more intermediate checkpoints.

### Data augmentations

Augmentations apply random transformations (gain, compression, pitch shift, EQ, reverb...) to the
training chunks on the fly, with transforms from acids-dataset:

```bash
rave train --config v2 --augment mute --augment compress --augment gain ...
```

- **Small corpora (less than about an hour): augmentations are mandatory.** Without them the model
  overfits the few recordings it sees.
- **Any corpus:** an augmentation trains the latent representation on *transformed* versions of your
  data, so the model learns to encode and decode the whole range covered by the transformation, not
  only the original recordings. For instance `pitchshift` widens the pitch range the model can follow,
  `gain` / `compress` make it robust to the input level, and `roomsimulator` lets it handle
  reverberant input. Choose them according to what you will feed the model with — e.g. a live
  microphone in a room for timbre transfer.

!!! warning "Always add `--augment mute`"
    `mute` replaces about 10 % of the training chunks with silence (a tiny noise floor). Training
    chunks are taken from your recordings, which rarely contain silence, so **without `mute` the
    model never learns to produce silence**: when the input stops, it keeps outputting noise, hum or
    "ghost" sounds. Use it in every training, whatever the size of the corpus.

The full list is in [Configurations](../reference/configurations.md#data-augmentations).

## 5. Monitor the training

```bash
tensorboard --logdir ~/runs
```

Open http://localhost:6006. If that port is taken (another tensorboard, or several trainings
monitored separately), choose another one with `--port`:

```bash
tensorboard --logdir ~/runs/my_corpus_<hash> --port 6007    # then open http://localhost:6007
```

On a remote GPU machine, add `--bind_all` to reach it from your network
(`http://<machine address>:6007`), or keep it local and forward the port through ssh:

```bash
ssh -L 6007:localhost:6007 user@gpu-server    # then open http://localhost:6007 on your computer
```

What to look at (details in the [Tensorboard guide](../tensorboard_guide.md)):

- **Phase 1** (first 1M steps for `v2`): `validation` and the `audio_distance` losses should
  decrease steadily. Listen to the `audio_val` examples in the _Audio_ tab: reconstructions are
  blurry but should follow the input.
- **Phase 2** (after `phase_1_duration`): the discriminator kicks in. Reconstruction losses jump up —
  **this is normal** — and the audio becomes much crisper over the following hundreds of thousands
  of steps. `loss_dis`, `loss_gen`, `pred_real`, `pred_fake` appear.
- **Latent size**: the fidelity curves show how many latent dimensions are needed to reach 80, 90, 95
  and 99 % of the latent variance. This is what `--fidelity` uses at export time.

A `v2` model is usually good after 1.5–3M steps. You can stop and export at any time.

If the training stops (crash, reboot), resume it:

```bash
rave train --resume ~/runs/my_corpus_<hash> --db_path ~/datasets/my_corpus --out_path ~/runs --name my_corpus
```

## 6. Export the model

```bash
rave export --run ~/runs/my_corpus_<hash> --streaming --db_path ~/datasets/my_corpus --out_path ~/models
```

This writes `~/models/my_corpus_<hash>_streaming.ts`, ready for `nn~`
([Tutorial 1](01_max_msp.md)), the RAVE VST, or the command line ([Tutorial 4](04_command_line.md)).
`--streaming` is required for real-time use, and `--db_path` recomputes the latent PCA on your data
(recommended). Everything else about exporting — latent size, latency, sampling rate, checking the
model before a concert — is in [Tutorial 3 — Export a model for Max/MSP](03_export_max.md).

## 7. Going further

- **Fine-tune** your model on another dataset ([Tutorial 7](07_finetune.md)), e.g. to improve timbre transfer from voice:

  ```bash
  rave finetune --name my_corpus_voice --run ~/runs/my_corpus_<hash> \
      --db_path ~/datasets/voice --mode encoder --out_path ~/runs
  ```

- Add **custom latent controls** ([Tutorial 5](05_custom_representations.md)).
- Train a **prior** to generate without input, or an **AFTER** model for timbre transfer ([Tutorial 6](06_prior.md)).
- More recipes in the [Cookbook](../cookbook.md).

## Common problems

| Problem                                                             | Solution                                                                                                                                                   |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Preprocessing shows `0it`                                           | Files are shorter than a chunk: `rave preprocess --num_signal 65536 ...` then `rave train --n_signal 65536 ...`.                                           |
| `n_components=128 must be between 0 and min(n_samples, n_features)` | The dataset is too small to compute the latent PCA (needs at least 128 batches). Add data, or lower `--n_signal` at preprocessing time to get more chunks. |
| CUDA out of memory                                                  | Lower `--batch`, or use a smaller configuration (`v2_small`, `--override "CAPACITY=64"`).                                                                  |
| Data loader crashes / hangs                                         | `--workers 0`.                                                                                                                                             |
| `Requested number of channels is 2, but dataset has 1 channels`     | Re-run `rave preprocess` with `--channels 2`.                                                                                                              |
| Model sounds noisy after phase 2 starts                             | Normal for the first 100–200k steps of phase 2; keep training.                                                                                             |
