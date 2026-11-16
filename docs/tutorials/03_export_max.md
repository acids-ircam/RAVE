# Tutorial 3 — Export a model for Max/MSP

Training produces checkpoints (`.ckpt`) that only Python can read. To play a model in Max/MSP or
Pure Data with [`nn~`](https://acids-ircam.github.io/nn_tilde/), or in the RAVE VST, you **export** it
to a single TorchScript file (`.ts`). This tutorial covers the choices made at export time and how to
check the result before using it on stage.

**You need:** a trained run (see [Tutorial 2](02_train.md)), the dataset it was trained on, and RAVE
installed. A GPU is optional (it only speeds up the PCA computation).

## 1. The command

```bash
rave export \
    --run ~/runs/my_corpus_<hash> \
    --streaming \
    --db_path ~/datasets/my_corpus \
    --out_path ~/models
```

This writes `~/models/my_corpus_<hash>_streaming.ts`. The two flags that matter most:

- **`--streaming`** — makes the model process audio block by block, keeping the state of its
  convolutions between blocks, as `nn~` does. **Without it you will hear clicks at every block.**
  Always use it for Max/MSP, Pure Data and the VST.
- **`--db_path`** — recomputes the latent PCA on your data (see [step 3](#3-recompute-the-latent-axes-with-db_path)).

`--run` accepts the run folder, a version folder or a `.ckpt` file. By default the `last.ckpt`
checkpoint is used; since it only appears after `--save_every` steps, the most recent checkpoint is
taken otherwise. To choose another one:

```bash
rave export --run ~/runs/my_corpus_<hash> --ckpt_name best.ckpt --streaming ...      # best validation loss
rave export --run ~/runs/my_corpus_<hash>/version_0/checkpoints/epoch_1000000.ckpt --streaming ...
```

## 2. Name it and put it where Max finds it

```bash
rave export --run ~/runs/my_corpus_<hash> --streaming --db_path ~/datasets/my_corpus \
    --name strings \
    --out_path ~/Documents/Max\ 8/Packages/nn_tilde/models
```

- `--name strings` writes `strings_streaming.ts` instead of `my_corpus_<hash>_streaming.ts`.
- `nn~` finds models by name (`[nn~ strings_streaming encode]`) in Max's search path: the
  `models/` folder of the `nn_tilde` package, the folder of your patch, or any folder added in
  *Options → File Preferences*. You can also give `nn~` a full path.
- `--out_path` may also be a full file name ending in `.ts`.

## 3. Recompute the latent axes with `--db_path`

The inlets and outlets of `nn~ ... encode` / `decode` are the **principal components** (PCA) of the
latent space: dimension 1 is the direction along which your sounds vary the most, dimension 2 the
next one, and so on. The PCA also decides how many dimensions are exported (step 4).

During training, this PCA is only estimated on **a single validation batch** (8 chunks), at each
validation of the first phase. It is a rough estimate: axes can be poorly ordered, and the number
of dimensions chosen by `--fidelity` can be off. With `--db_path`, the dataset is encoded with the
final model and the PCA, its mean and the fidelity curve are recomputed on all your data. The log
prints how many dimensions reach 80, 90, 95 and 99 % fidelity.

```bash
rave export --run ~/runs/my_corpus_<hash> --streaming --db_path ~/datasets/my_corpus --device cuda
```

- `--device cuda` (or `mps`) speeds up the encoding; `--max_chunks 5000` encodes a random subset of
  a large dataset.
- `--db_partition val` (or `train`) uses only the validation (or training) split written during
  training; the default `all` uses everything.
- `--db_path` also accepts **another** dataset or a raw audio folder — for instance recordings of
  what you will play into the model. The axes then describe how *that* material varies in the
  latent space, which often gives more useful controls for timbre transfer.
- **Wasserstein models** do not compute a PCA during training: without `--db_path` they are exported
  with their raw latent space (the file name gets a `_nopca` suffix).

Two cautions:

- Recomputing the PCA changes what each latent dimension means: a patch built on an export *without*
  `--db_path` behaves differently with an export *with* it. Re-export, then re-tune your patch.
- Do not combine `--db_path` with `--prior`: the prior was trained on the training-time PCA and
  would no longer match the recomputed axes (see [Tutorial 6](06_prior.md)).

## 4. Choose the number of latent dimensions

The number of latent dimensions is fixed at export time: it is the number of outlets of
`encode` and inlets of `decode`.

| Option              | Effect |
| ------------------- | ------ |
| `--fidelity 0.95` (default) | Keep enough dimensions to explain 95 % of the latent variance, rounded up to a power of two. |
| `--fidelity 0.99`   | More dimensions: more detailed control and better `encode → decode` reconstructions, bigger patches. |
| `--fidelity 0.8`    | Fewer dimensions: each one has a strong, audible effect. Good for performance with a few faders. |
| `--latent_size 8`   | Exactly 8 dimensions (overrides `--fidelity`). |
| `--no_pca`          | The raw latent space, all dimensions, no PCA rotation. |

Dimensions that are not exported are not lost: when decoding, `decode` fills them with random noise,
scaled by the `temperature` attribute. And `forward` always uses the full latent space, whatever the
exported size.

## 5. Latency

The latency you get in Max adds up from:

1. **The model's architecture.** Models trained with `--config causal` only look at the past and
   have low latency; regular models also look slightly ahead, which adds latency. This is decided at
   training time — you cannot make a model causal at export.
2. **The latent rate.** A `v2` model produces one latent frame every 2048 samples (~46 ms at
   44.1 kHz); `v2_small` every 512.
3. **The `nn~` buffer size** (third argument, e.g. `[nn~ strings_streaming forward 2048]`). Smaller
   buffers lower the latency but raise the CPU load. Use a multiple of the model's latent frame
   (2048 samples for `v2`).

The read-only attribute `receptive_field` of the exported model gives its receptive field (left,
right) in samples.

## 6. Sampling rate

`nn~` does not resample: the model expects audio at its own rate (44.1 kHz unless trained
otherwise). Run Max at the model's rate. If you must run Max at a multiple of it, the model can
resample internally:

```bash
rave export --run ~/runs/my_corpus_<hash> --streaming --sr 88200      # for a 44.1 kHz model
```

Only integer multiples of the model's rate are accepted (88200 or 176400 for a 44.1 kHz model,
96000 for a 48 kHz one). Running Max at 48 kHz with a 44.1 kHz model plays everything slightly too
high and too fast; there is no conversion between 44.1 and 48 kHz at export.

## 7. Extras: projections and prior

The export can embed more than the plain model:

- **Custom projections** — alternative latent axes (independent components, slow features, axes
  aligned with loudness / brightness / pitch...), selected live in Max with
  `set projection <name>`. They can be computed at export with `--db_path` and `--ica`, `--sfa`,
  `--pls`, `--cca`, `--mapper`, or added later with `rave dimred`. See
  [Tutorial 5](05_custom_representations.md).
- **A prior** — `--prior ~/runs/my_prior_<hash>` adds a `prior` method that generates latent
  trajectories without input. Use the same `--fidelity` as for training the prior. See
  [Tutorial 6](06_prior.md).
- **EMA weights** — if you trained with `--ema`, `--ema_weights` exports the averaged weights, often
  slightly smoother.

## 8. Check the model before using it

**Read the export log.** It reports the checkpoint used, the latent size chosen from the fidelity,
the PCA fidelities if `--db_path` was given, and ends with:

```text
this model seems compatible with the RAVE vst.
all good ! model exported to /…/models/strings_streaming.ts
```

If the stereo test fails, the log says the model will not work in the VST; it can still work in
`nn~`.

**Inspect it in Python** (a few seconds, no Max needed):

```python
import torch

model = torch.jit.load("strings_streaming.ts")
print("sr:", model.sr, "| channels:", model.n_channels)
print("latent:", model.latent_size, "of", model.full_latent_size)
print("projections:", model.get_projection_names())

x = torch.zeros(1, model.n_channels, 2048 * 8)
with torch.no_grad():
    z = model.encode(x)
    y = model.decode(z)
print("encode:", tuple(z.shape), "-> decode:", tuple(y.shape))
```

Or list everything stored in it:

```bash
rave dimred show --model strings_streaming.ts
```

**Listen to it offline** with `rave generate` (see [Tutorial 4](04_command_line.md)), which processes
files block by block exactly like `nn~`:

```bash
rave generate --model strings_streaming.ts --input test_sounds/ --mode stream --chunk_size 8192
```

Then load it in Max as in [Tutorial 1](01_max_msp.md):

```text
[ezadc~]
 |
[nn~ strings_streaming forward]
 |
[ezdac~]
```

## Troubleshooting

| Symptom | Cause and fix |
| ------- | ------------- |
| Clicks or crackles in Max | Exported without `--streaming`; re-export with it. |
| `nn~` says the model cannot be found | The `.ts` is not in Max's search path: move it next to the patch, into `nn_tilde/models/`, or give a full path. |
| Pitch and speed are wrong | Max runs at a different sampling rate than the model. Change Max's rate, or export with `--sr <multiple>`. |
| Too many / too few inlets and outlets | Re-export with another `--fidelity` or `--latent_size`. |
| Latent axes changed after re-exporting | `--db_path` was added or removed, or another dataset was used: the PCA was recomputed. |
| `prior` method missing after export | The log shows the prior could not be loaded: use the same `--fidelity` / `--latent_size` as `train_prior`, and no `--db_path`. |
| Export fails on `--sr` | The target rate is not an integer multiple of the model's rate. |
| `no checkpoint found` | Check `--run`; early in training `last.ckpt` does not exist yet, but any other checkpoint is used. |
