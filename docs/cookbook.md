# Cookbook

Copy-paste recipes for common tasks. Paths are placeholders: `~/audio/X` is a folder of audio
files, `~/datasets/X` a preprocessed dataset, `~/runs` the training output folder, and
`~/runs/X_<hash>` a run folder created by `rave train`. See the
[command-line reference](reference/cli.md) for every flag.

- [Preprocessing](#preprocessing)
- [Training](#training): [mono](#mono-default), [stereo](#stereo), [mid/side](#midside), [ambisonics](#ambisonics), [fast and lightweight](#fast-and-lightweight-model), [Raspberry Pi](#raspberry-pi), [Wasserstein](#wasserstein), [causal](#low-latency-causal), [discrete](#discrete-tokens), [style transfer (v3)](#style-transfer-v3), [data augmentations](#data-augmentations), [other sampling rates](#other-sampling-rates)
- [Training infrastructure](#training-infrastructure): GPUs, resuming, transfer, EMA
- [Export](#export): [Max/MSP & Pd](#for-maxmsp-and-pure-data), [VST](#for-the-rave-vst), latent size, Wasserstein, prior
- [Fine-tuning](#fine-tuning)
- [Offline processing](#offline-processing)
- [Custom projections](#custom-projections)
- [Priors](#priors)
- [Recovering and inspecting models](#recovering-and-inspecting-models)

---

## Preprocessing

```bash
# mono, 44.1 kHz (default)
rave preprocess --input_path ~/audio/strings --output_path ~/datasets/strings --channels 1

# stereo
rave preprocess --input_path ~/audio/strings --output_path ~/datasets/strings_stereo --channels 2

# 48 kHz
rave preprocess --input_path ~/audio/strings --output_path ~/datasets/strings_48k --sampling_rate 48000

# short files (one-shots, samples): smaller chunks — use the same value for --n_signal at training
rave preprocess --input_path ~/audio/oneshots --output_path ~/datasets/oneshots --num_signal 65536

# only some files
rave preprocess --input_path ~/audio/orchestra --output_path ~/datasets/violins \
    --filter "*violin*" --exclude "*tuning*"

# large corpus: cap the database size (GB)
rave preprocess --input_path ~/audio/huge --output_path ~/datasets/huge --max_db_size 300
```

---

## Training

!!! warning "Every training below uses `--augment mute`"
    `mute` replaces about 10 % of the training chunks with silence. Recordings almost never contain
    silence, so **without it the model never learns to produce silence**: when the input stops, it
    keeps outputting noise, hum or ghost sounds. Add it to every training, including fine-tuning,
    whatever the size of your dataset.

### Mono (default)

```bash
rave train --config v2 --augment mute --db_path ~/datasets/strings --out_path ~/runs --name strings --channels 1
```

Without preprocessing: give the raw audio folder directly (decoded on the fly; more CPU load, writes
hidden index files in the folder):

```bash
rave train --config v2 --augment mute --db_path ~/audio/strings --out_path ~/runs --name strings --channels 1
```

### Stereo

Preprocess with `--channels 2`, then:

```bash
rave train --config v2 --augment mute --db_path ~/datasets/strings_stereo --out_path ~/runs --name strings_stereo --stereo
# equivalent: --channels 2
```

The model encodes both channels jointly and outputs a true stereo signal. Stereo models need more
memory: lower `--batch` if needed.

> To use a **mono** model on a stereo signal, you don't need a stereo model: the VST handles it, and
> in Max you can simply run one `nn~` per channel (or `mcs.nn~`).

### Mid/side

Stereo trained in mid/side representation (often better for wide, ambient material):

```bash
rave train --config v2 --augment mute --db_path ~/datasets/ambient_stereo --out_path ~/runs --name ambient_ms --midside
```

Input and output of the exported model are still left/right.

### Ambisonics

Ambisonic datasets have `(order+1)²` channels (4 for first order, 9 for second order...).

**B-format recordings (already encoded: W, X, Y, Z...): `--ambisonics` is not needed.** A plain
multichannel training is sufficient — the model simply learns the 4 (or 9, 16...) channels:

```bash
rave preprocess --input_path ~/audio/foa_bformat --output_path ~/datasets/foa_b --channels 4
rave train --config v2 --augment mute --db_path ~/datasets/foa_b --out_path ~/runs --name foa --channels 4
```

**`--ambisonics` is needed ONLY to train on A-format signals** (raw capsule signals of an ambisonic
microphone). It adds an ambisonic encoder: the model is trained on the B-format conversion of the
signals, and the exported model takes and returns A-format:

```bash
rave preprocess --input_path ~/audio/foa_aformat --output_path ~/datasets/foa_a --channels 4
rave train --config v2 --augment mute --db_path ~/datasets/foa_a --out_path ~/runs --name foa_a --ambisonics
```

### Fast and lightweight model

The quickest model to train and run, with ~8 GB of GPU memory:

```bash
rave train --config v2_small --augment mute --db_path ~/datasets/strings --out_path ~/runs --name strings_small
```

Make it even cheaper:

```bash
rave train --config v2_small --config causal --augment mute \
    --override "CAPACITY=32" \
    --override "RAVE.phase_1_duration=300000" \
    --max_steps 1000000 \
    --db_path ~/datasets/strings --out_path ~/runs --name strings_tiny
```

Check that a configuration runs before committing to a long training:

```bash
rave train --config v2_small --augment mute --db_path ~/datasets/strings --out_path /tmp/rave_test --name test \
    --smoke_test --max_steps 20 --workers 0
```

### Raspberry Pi

The `raspberry` configuration (v1 architecture, capacity 16, no noise generator) runs in real time
on a Raspberry Pi 4 with `nn~` for Pure Data. Make it causal for lower latency:

```bash
rave train --config raspberry --config causal --augment mute \
    --db_path ~/datasets/strings --out_path ~/runs --name strings_rpi
rave export --run ~/runs/strings_rpi_<hash> --streaming --fidelity 0.9
```

Copy the `.ts` to the Pi and load it in Pd with `[nn~ strings_rpi_<hash>_streaming forward 4096]`
(use a large buffer size on the Pi). A lower `--fidelity` (fewer latent dimensions) does not make the
decoder cheaper, but makes the patch simpler.

### Wasserstein

A Wasserstein auto-encoder gives a compact (16 dims), smooth and well-used latent space. **Always
stack it on an architecture:**

```bash
rave train --config v2 --config wasserstein --augment mute --db_path ~/datasets/strings --out_path ~/runs --name strings_wae
```

Bigger latent space, or stronger regularization:

```bash
rave train --config v2 --config wasserstein --augment mute --override "LATENT_SIZE=32" --beta 200 ...
```

Export with the dataset, so that a PCA is computed (otherwise the raw latent space is exported):

```bash
rave export --run ~/runs/strings_wae_<hash> --streaming --db_path ~/datasets/strings
```

### Low latency (causal)

```bash
rave train --config v2 --config causal --augment mute --db_path ~/datasets/strings --out_path ~/runs --name strings_causal
```

Causality is decided at training time; combine it with any architecture (`v2_small`, `wasserstein`...).

### Discrete tokens

```bash
rave train --config discrete --augment mute --db_path ~/datasets/strings --out_path ~/runs --name strings_discrete
```

For use with token-based priors such as [msprior](https://github.com/caillonantoine/msprior).
`--config discrete --config causal` for real-time priors.

### Style transfer (v3)

AdaIN-based real-time style transfer (needs ~32 GB of GPU memory):

```bash
rave train --config v3 --augment mute --db_path ~/datasets/strings --out_path ~/runs --name strings_v3
```

### Noisy / percussive material

```bash
rave train --config v2 --config noise --augment mute --db_path ~/datasets/drums --out_path ~/runs --name drums
```

### Data augmentations

Mandatory for small datasets (less than about an hour). For any dataset, they train the latent
space on transformed versions of the data (here: level changes, compression, transposition), so the
model also handles inputs within the range of these transformations:

```bash
rave train --config v2 --augment mute --augment compress --augment gain --augment pitchshift \
    --db_path ~/datasets/small --out_path ~/runs --name small
```

For very uneven file lengths, sample one chunk per file per batch:

```bash
rave train --config v2 --augment mute --one_batch_per_file ...
```

### Loudness normalization

```bash
rave train --config v2 --augment mute --normalize each ...              # normalize every chunk
rave train --config v2 --augment mute --normalize dataset ...           # scale by the dataset peak
rave train --config v2 --augment mute --normalize dataset_compress ...  # compress to the dataset's loudness range
```

### Other sampling rates

The sampling rate is part of the configuration; preprocess at the same rate:

```bash
rave preprocess --input_path ~/audio/strings --output_path ~/datasets/strings_48k --sampling_rate 48000
rave train --config v2 --augment mute --override "SAMPLING_RATE=48000" --db_path ~/datasets/strings_48k --out_path ~/runs --name strings48
```

### Keep training the encoder in phase 2

```bash
rave train --config v2 --config no_encoder_freeze --augment mute ...
```

---

## Training infrastructure

```bash
# choose a GPU
rave train ... --device cuda:1

# several GPUs
rave train ... --device cuda:0 --device cuda:1

# Apple silicon (small experiments only)
rave train ... --device mps --workers 0

# bigger batches if memory allows
rave train ... --batch 16

# keep a checkpoint every 100k steps
rave train ... --save_every 100000

# no progress bar (log files, clusters)
rave train ... --noprogress

# run in the background and keep a log
nohup rave train --config v2 --augment mute --db_path ~/datasets/strings --out_path ~/runs --name strings --noprogress > strings.log 2>&1 &
```

Resume an interrupted training:

```bash
rave train --augment mute --resume ~/runs/strings_<hash> --db_path ~/datasets/strings --out_path ~/runs --name strings
```

Resume and switch on encoder training / early adversarial training (only these configs can be
added when resuming):

```bash
rave train --augment mute --resume ~/runs/strings_<hash> --config no_encoder_freeze --db_path ~/datasets/strings --out_path ~/runs --name strings
```

Start a new model from the weights of an existing one (transfer learning):

```bash
# all compatible weights
rave train --config v2 --augment mute --init_from ~/runs/strings_<hash> --db_path ~/datasets/winds --out_path ~/runs --name winds

# only the decoder
rave train --config v2 --augment mute --init_from "$HOME/runs/strings_<hash>->decoder" ...

# by regular expression, allowing partial transfer
rave train --config v2 --augment mute --init_from "$HOME/runs/strings_<hash>->^decoder\.net\.[0-3].*" --nostrict_init ...
```

Weight averaging (often smoother results), exported with `--ema_weights`:

```bash
rave train --config v2 --augment mute --ema 0.999 ...
rave export --run ~/runs/strings_<hash> --streaming --ema_weights
```

Monitor:

```bash
tensorboard --logdir ~/runs
# custom port (default 6006 already taken, several runs side by side...)
tensorboard --logdir ~/runs/strings_<hash> --port 6007
# on a remote machine: forward the port...
ssh -L 6007:localhost:6007 user@gpu-server   # then open http://localhost:6007
# ...or expose tensorboard on the network
tensorboard --logdir ~/runs --port 6007 --bind_all
```

---

## Export

### For Max/MSP and Pure Data

```bash
rave export --run ~/runs/strings_<hash> --streaming --db_path ~/datasets/strings \
    --out_path ~/Documents/Max\ 8/Packages/nn_tilde/models
```

### For the RAVE VST

Mono or stereo models exported in streaming mode work in the VST; the export log tells you
whether the model is compatible.

```bash
rave export --run ~/runs/strings_<hash> --streaming --name strings_vst
```

### Recompute the PCA on the data (recommended)

During training the latent PCA is estimated on a single validation batch; `--db_path` recomputes it
(and the fidelity curve) on the whole dataset with the final model:

```bash
rave export --run ~/runs/strings_<hash> --streaming --db_path ~/datasets/strings --device cuda
rave export --run ~/runs/strings_<hash> --streaming --db_path ~/datasets/strings --db_partition val --max_chunks 5000
# axes describing another material (e.g. what you will play into the model)
rave export --run ~/runs/strings_<hash> --streaming --db_path ~/audio/my_voice
```

Not with `--prior` (the prior is trained on the training-time PCA).

### Choose the latent size

```bash
rave export --run ~/runs/strings_<hash> --streaming --fidelity 0.99    # more dimensions
rave export --run ~/runs/strings_<hash> --streaming --latent_size 4    # exactly 4 dimensions
rave export --run ~/runs/strings_<hash> --streaming --no_pca           # raw latent space, all dims
```

### Other exports

```bash
# best validation checkpoint instead of last
rave export --run ~/runs/strings_<hash> --streaming --ckpt_name best.ckpt

# a specific checkpoint
rave export --run ~/runs/strings_<hash>/version_0/checkpoints/epoch_1000000.ckpt --streaming

# run at 88.2 kHz (internal resampling; must be a multiple of the model rate)
rave export --run ~/runs/strings_<hash> --streaming --sr 88200

# offline / Python use (no streaming caches)
rave export --run ~/runs/strings_<hash> --out_path ~/models/strings_offline.ts

# recompute the PCA on the validation split, and add projections
rave export --run ~/runs/strings_<hash> --streaming --db_path ~/datasets/strings --db_partition val --sfa --pls
```

---

## Fine-tuning

The modes and options are explained in [Tutorial 7](tutorials/07_finetune.md).

```bash
# make a model respond better to a new kind of input (e.g. voice into a strings model)
rave finetune --augment mute --name strings_voice --run ~/runs/strings_<hash> --db_path ~/datasets/voice \
    --mode encoder --out_path ~/runs

# adapt the whole model to new material, adversarially
rave finetune --augment mute --name strings_v2data --run ~/runs/strings_<hash> --db_path ~/datasets/strings_new \
    --mode full --max_steps 300000 --out_path ~/runs

# decoder then encoder, 25,000 batches each
rave finetune --augment mute --name adaptation --run ~/runs/strings_<hash> --db_path ~/datasets/new \
    --mode full --lr_encoder 1e-5 --lr_decoder 1e-4 --lr_discriminator 1e-4 \
    --max_steps 50000

# only the first decoder layers
rave finetune --augment mute --name partial --run ~/runs/strings_<hash> --db_path ~/datasets/new \
    --mode decoder --train "decoder.net.[0-3].*"

# keep the source PCA so that exported latent axes keep their meaning
rave finetune --augment mute ... --noupdate_pca

# fresh discriminator
rave finetune --augment mute ... --mode full --reset_discriminator

# resume a fine-tuning (same settings)
rave finetune --augment mute --name strings_voice --run ~/runs/strings_<hash> --db_path ~/datasets/voice \
    --mode encoder --resume ~/runs/strings_voice
```

Export a fine-tuned model like any run:

```bash
rave export --run ~/runs/strings_voice --streaming
```

List weight names for `--train` / `--freeze`:

```python
import rave
model, _ = rave.load_rave_checkpoint("/home/me/runs/strings_<hash>")   # full path: "~" is not expanded
print("\n".join(model.state_dict().keys()))
```

---

## Offline processing

```bash
# one file
rave generate --model strings_streaming.ts --input voice.wav

# folders, several models, on GPU
rave generate --model strings.ts --model drums.ts --input ~/sounds/a --input ~/sounds/b \
    --out_path renders/ --device cuda

# input gain sweep
rave generate --model strings.ts --input voice.wav --loudness -12 --loudness -6 --loudness 0

# whole-file processing (no streaming)
rave generate --model strings.ts --input voice.wav --mode full

# listen to a training checkpoint without exporting
rave generate --model ~/runs/strings_<hash> --input voice.wav --mode full
```

---

## Custom projections

```bash
# what is in a model?
rave dimred show --model strings_streaming.ts

# PCA + slow features on your own sounds, into a new .ts
rave dimred add --model strings_streaming.ts --name voice --audio ~/sounds/voice \
    --type pca --type sfa --dims 8 --out strings_voice_streaming.ts

# descriptor-aligned axes (loudness, brightness, pitch)
rave dimred add --model ~/runs/strings_<hash> --name desc --audio ~/datasets/strings \
    --type pls --descriptors loudness --descriptors centroid --descriptors f0 --device cuda --cache

# axes separating classes (one class per --audio)
rave dimred add --model ~/runs/orch_<hash> --name families --type lda \
    --audio ~/sounds/strings --audio ~/sounds/brass --audio ~/sounds/winds

# on the validation split of the training dataset
rave dimred add --model ~/runs/strings_<hash> --name val --audio "~/datasets/strings->val" --type ica

# remove
rave dimred remove --model ~/runs/strings_<hash> --name desc

# clear cached latents
rave dimred purge_cache
```

In Max: `set projection sfa_voice`. In Python: `model.set_projection("sfa_voice")`.

---

## Priors

```bash
rave train_prior --name strings_prior --model ~/runs/strings_<hash> --db_path ~/datasets/strings \
    --out_path ~/runs --fidelity 0.95

# export with the SAME fidelity
rave export --run ~/runs/strings_<hash> --prior ~/runs/strings_prior_<hash> --streaming --fidelity 0.95

# AFTER diffusion model on top of RAVE (pip install "acids-rave[after]")
rave train_after --name strings_after --model ~/runs/strings_<hash> --db_path ~/datasets/strings --device cuda:0
```

---

## Recovering and inspecting models

```bash
# turn a .ts back into a trainable checkpoint
rave uncompile --model strings_streaming.ts --out strings_uncompiled --check

# ...and fine-tune it
rave finetune --augment mute --name strings_ft --run strings_uncompiled --db_path ~/datasets/new --mode full --reset_discriminator

# fit a latent adapter between two models
rave adapt --from_model a.ts --to_model b.ts --db_path ~/datasets/strings --name a_to_b --out_path adapters/
```

```python
# inspect an exported model
import torch
m = torch.jit.load("strings_streaming.ts")
print(m.sr, m.n_channels, m.latent_size, m.full_latent_size)
print(m.get_projection_names())
print(m.gin_config_str)                       # configuration used for training
```
