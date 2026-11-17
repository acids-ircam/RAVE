# Tutorial 7 — Fine-tune a model

Training a RAVE model from scratch takes days. **Fine-tuning** starts from a model that already
works and continues training it, usually on other data, often updating only part of it. It is the
way to:

- make a model **react better to a new kind of input** (e.g. a voice played into a violin model);
- **adapt a model to a new sound world** in hours instead of days, even with little data;
- **refine the output** of a model without changing its latent space (so your patches keep working).

**You need:** a trained run (see [Tutorial 2](02_train.md)) and a dataset — preprocessed, or a raw
audio folder — for the material you fine-tune on. A GPU.

## 1. The command

```bash
rave finetune \
    --name violin_voice \
    --run ~/runs/violin_<hash> \
    --db_path ~/datasets/voice \
    --out_path ~/runs \
    --mode encoder \
    --augment mute \
    --max_steps 200000
```

What happens:

- The configuration and weights of `--run` are loaded. Options of the source run (architecture,
  losses, loss weights) are kept; `--override` can change gin bindings on top of them.
- Training starts at fine-tuning step 0 with **fresh optimizers** and the learning rates given here.
  The model is considered fully trained: there is no "phase 1" and the adversarial losses are
  available from the first step.
- `--mode` decides **which parts are updated, and with which losses** (next section).
- The result is written to `~/runs/violin_voice/version_N/`, with its own `config.gin`. Export it like
  any run (see [Tutorial 3](03_export_max.md)):

  ```bash
  rave export --run ~/runs/violin_voice --streaming
  ```

> `--max_steps` defaults to 50,000 fine-tuning batches. In `full`, each phase receives
> half of that total. You can also stop and export the latest checkpoint.

As for any training, add `--augment mute` so the model keeps producing silence when the input stops,
plus any augmentation useful for your material (see
[Tutorial 2](02_train.md#data-augmentations)).

## 2. The three modes

A RAVE model has three parts: the **encoder** (audio → latent), the **decoder** (latent → audio) and
the **discriminator** (judges whether audio sounds real; only used for training).

| `--mode` | Updated | Losses | Fixed |
| -------- | ------- | ------ | ----- |
| `encoder` *(default)* | encoder | reconstruction + latent regularization | decoder, discriminator |
| `decoder` | decoder and discriminator on separate batches | reconstruction + adversarial + feature matching | encoder |
| `full` | decoder phase, then encoder phase (equal lengths) | losses of the active mode | - |

### `encoder` — a new input, the same sound

Only the encoder learns. The decoder is frozen, so **the model keeps sounding exactly the same**: it
learns to map the new material into the latent space it already knows.

This is the mode for **timbre transfer**. A violin model trained on violin has never "heard" a
voice: when you sing into it, the latents can land in odd places and the result is unstable.
Fine-tune its encoder on the material you will *play into* the model:

```bash
rave finetune --name violin_voice --run ~/runs/violin_<hash> --db_path ~/datasets/voice \
    --mode encoder --augment mute --max_steps 200000
```

The encoder learns to encode voice so that the (unchanged) violin decoder reconstructs it as well as
it can, i.e. the closest violin sound. It is the cheapest mode (no discriminator involved), and the
safest: it cannot degrade the output quality of the model.

### `decoder` — the same latent space, a new output

The decoder learns with reconstruction, adversarial and feature-matching losses. The discriminator
is updated on separate batches at the configured cadence. The encoder stays fixed.

Use it to **refine or shift the output** while keeping the controls: improve the audio quality of a
model on a cleaner dataset, or move it towards a related sound, while patches built on the original
model still drive it the same way. Add `--noupdate_pca` to keep the exported latent axes identical
(see [section 5](#5-the-latent-axes-pca)):

```bash
rave finetune --name violin_clean --run ~/runs/violin_<hash> --db_path ~/datasets/violin_studio \
    --mode decoder --augment mute --noupdate_pca --max_steps 300000
```

> Decoder mode trains the discriminator at its configured cadence.
> `--reset_discriminator` optionally initializes it afresh.

### `full` — A new model

`full` runs decoder mode for half of the requested batches, followed by encoder mode
for the other half. The default is 25,000 batches per phase. Choose the total with
`--max_steps` (a positive even integer); do not use `--encoder_start_step` with `full`. Combining both steps enables to learn to generate samples in the new domain while shaping the latent space for the new features. 

```bash
rave finetune --name winds --run ~/runs/strings_<hash> --db_path ~/datasets/winds \
    --mode full --augment mute --max_steps 50000
```

The decoder phase retains decoder mode's discriminator cadence and losses. In the
encoder phase, decoder and discriminator are fixed and the encoder learns from
reconstruction and latent regularization. Checkpoints retain the phase boundary;
resume with the same `--max_steps`.

### Which mode?

| You want to... | Mode |
| -------------- | ---- |
| Play a new kind of input into an existing model (timbre transfer) | `encoder`, on the input material |
| Improve / change the output, keep the latent controls | `decoder` (+ `--noupdate_pca`) |
| Get a model for a new sound world quickly, or with little data | `full` |

## 3. Choosing exactly which weights are trained

`--train` and `--freeze` refine the selection made by `--mode`. They accept the shortcuts `all`,
`encoder`, `decoder`, `discriminator`, or regular expressions on weight names, and can be repeated.
The weights actually trained are: `--train` selection, minus `--freeze` selection, restricted to the
encoder (in `encoder` mode) or decoder/discriminator (in `decoder` mode).
`full` requires trainable weights in both encoder and decoder.

```bash
# only the first layers of the decoder
rave finetune ... --mode decoder --train "decoder.net.[0-3].*"

# everything but the discriminator
rave finetune ... --mode full --freeze discriminator

# decoder, without its noise generator
rave finetune ... --mode decoder --freeze ".*noise.*"
```

To see the weight names of a model:

```python
import rave

model, _ = rave.load_rave_checkpoint("/full/path/to/runs/violin_<hash>")
print("\n".join(model.state_dict().keys()))
```

The log prints the number of fine-tuned weights tensors at start; an empty selection is an error.

## 4. Learning rates and schedule

| Flag | Default | Notes |
| ---- | ------- | ----- |
| `--lr_encoder` | `1e-5` | Low on purpose: the latent space should move slowly. |
| `--lr_decoder` | `1e-4` | |
| `--lr_discriminator` | `1e-4` | |
| `--encoder_start_step` | `0` | Encoder-mode delay; omit for `full`. |
| `--max_steps` | `50000` | Total fine-tuning batches; full splits them equally. Keep the same total when resuming full. |

If the output degrades quickly (noise, artifacts), lower the decoder and discriminator learning rates
(e.g. `3e-5`). If nothing seems to change, raise them or train longer.

### Latent regularization

The regularization keeps the latent space well-behaved (for variational models, close to a
Gaussian). Fine-tuning uses the source encoder's beta divided by 4 unless you provide an
explicit value, for example `--beta 0.01`. Beta must be finite and nonnegative;
`--beta 0` disables the KL term. There is no additional regularization multiplier.
For non-variational encoders, the source regularization is preserved.

## 5. The latent axes (PCA)

By default the latent PCA — the axes exposed in `nn~` — is recomputed at each validation
(`--update_pca`). If the encoder changes (`encoder` and `full` modes), that is usually
what you want. To **keep the axes of the source model** — so that patches and automation keep the same
meaning — use `--noupdate_pca`, and export *without* `--db_path` (which would recompute them).

## 6. Monitoring and resuming

```bash
tensorboard --logdir ~/runs/violin_voice --port 6007
```

The usual curves and audio examples are logged (see [Tutorial 2](02_train.md#5-monitor-the-training)),
plus `finetune_steps`. Listen to the `audio_val` examples: they are reconstructions of the *new*
dataset.

To resume an interrupted fine-tuning, give the same settings (mode, learning rates,
`--encoder_start_step`, beta) plus `--resume`. Keep the same source `--run` and
`--beta` setting so the effective beta is unchanged:

```bash
rave finetune --name violin_voice --run ~/runs/violin_<hash> --db_path ~/datasets/voice \
    --mode encoder --augment mute --max_steps 200000 --resume ~/runs/violin_voice
```

## 7. Fine-tuning a model you only have as a `.ts`

Exported models (for instance downloaded pretrained models) can be turned back into a checkpoint with
`rave uncompile`. The discriminator is not part of an export: it comes back **randomly initialized**,
so train it with `decoder` or `full`:

```bash
rave uncompile --model darbouka_streaming.ts --out ~/runs/darbouka_uncompiled --check
rave finetune --name darbouka_tabla --run ~/runs/darbouka_uncompiled --db_path ~/datasets/tabla \
    --mode full --augment mute --max_steps 300000
```

`encoder` mode works too, since it does not use the discriminator at all.

## 8. Fine-tuning vs. `rave train --init_from`

`rave train --init_from` also reuses the weights of a run, but starts a **regular training** of a new
configuration: phase 1, then phase 2, with its full duration. Use it when you change the architecture
(e.g. reuse a decoder in a model with another encoder), or want a complete training with a better
starting point. Use `rave finetune` to adapt an existing model with its own configuration.

## Troubleshooting

| Problem | Solution |
| ------- | -------- |
| `No training keys found` | `--train` / `--freeze` / `--mode` select no weight: check the names, and that the mode matches (`--mode encoder` keeps only `encoder.*`). |
| `--beta requires a variational encoder` | `--beta` only exists for variational models; omit it for other encoders. |
| `max_steps must exceed the resumed fine-tuning step` | When resuming, `--max_steps` counts the steps already done: increase it. |
| Output quality degrades quickly | Lower `--lr_decoder` / `--lr_discriminator`, or switch to `encoder` mode. |
| The model no longer produces silence | Add `--augment mute`. |
| Latent controls changed meaning | Use `--noupdate_pca`, and export without `--db_path`. |
