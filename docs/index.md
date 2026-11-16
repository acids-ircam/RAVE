<img src="assets/rave_logo_black.svg#gh-light-mode-only" alt="RAVE" width="420">
<img src="assets/rave_logo_white.svg#gh-dark-mode-only" alt="RAVE" width="420">

# RAVE documentation

RAVE (Realtime Audio Variational autoEncoder) is a neural audio codec / synthesizer that runs
in real time. You train it on a corpus of sounds; it learns a compact *latent representation*
of that corpus, which you can then use to resynthesize, transform or generate audio — offline
from the command line, or live in Max/MSP, Pure Data or the RAVE VST through
[`nn~`](https://acids-ircam.github.io/nn_tilde/).

This documentation covers version **2.4**.

## Where to start

| I want to...                                   | Read                                                                 |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| Play with a pretrained model in Max            | [Tutorial 1 — Use a model in Max/MSP](tutorials/01_max_msp.md)       |
| Train a model on my own sounds                 | [Tutorial 2 — Train a model](tutorials/02_train.md)                  |
| Export my model for Max/MSP, Pd or the VST     | [Tutorial 3 — Export a model for Max/MSP](tutorials/03_export_max.md) |
| Process audio files without Max                | [Tutorial 4 — Use a model from the command line](tutorials/04_command_line.md) |
| Get more musical / meaningful latent controls  | [Tutorial 5 — Add custom representations](tutorials/05_custom_representations.md) |
| Generate audio without any input               | [Tutorial 6 — Train a prior](tutorials/06_prior.md)                  |
| Adapt an existing model to new sounds or inputs | [Tutorial 7 — Fine-tune a model](tutorials/07_finetune.md) |
| Copy-paste a command for a common case         | [Cookbook](cookbook.md)                                              |
| Look up a flag, method or config               | [Reference](#reference)                                              |

## Reference

- [Command-line reference](reference/cli.md) — every `rave <command>` and its flags.
- [Configurations](reference/configurations.md) — architectures, regularizations, gin overrides, augmentations.
- [Exported models (`.ts`)](reference/scripted_model.md) — methods and attributes available in `nn~` and in Python.
- [Python API](reference/python_api.md) — loading checkpoints, exporting, projections, uncompiling.
- [Run folders and checkpoints](reference/runs.md) — what training writes to disk and how RAVE finds it.
- [Tensorboard guide](tensorboard_guide.md) — reading the training curves.

## The RAVE workflow in one picture

```text
 audio files ──► rave preprocess ──► dataset ──► rave train ──► run folder (.ckpt + config.gin)
                                                      │                  │
                                                      │                  ├─► rave dimred      (custom projections)
                                                      │                  ├─► rave train_prior (generation without input)
                                                      │                  └─► rave finetune    (adapt to new data)
                                                      ▼
                                              rave export (--streaming) ──► model.ts ──► nn~ (Max / Pd), RAVE VST,
                                                                                         rave generate, torch.jit.load
```

## Installation

Install PyTorch first (pick the build matching your GPU on [pytorch.org](https://pytorch.org)),
then RAVE:

```bash
conda create -n rave python=3.11
conda activate rave
conda install -c conda-forge pytorch torchaudio ffmpeg sox   # or pip, see pytorch.org
pip install acids-rave
```

RAVE requires Python ≥ 3.11. Training needs a GPU (CUDA, or Apple `mps` for small experiments);
using an exported model only needs a CPU. Optional extras:

```bash
pip install "acids-rave[after]"     # for `rave train_after` (AFTER diffusion models)
```

Check the install with:

```bash
rave            # prints the list of available commands
```

## Getting help

- `rave <command> --helpfull` prints every flag of a command.
- [Discord](https://discord.gg/dhX73sPTBb) and [GitHub discussions](https://github.com/acids-ircam/RAVE/discussions).
- The FAQ at the end of the [README](https://github.com/acids-ircam/RAVE/blob/master/README.md#frequently-asked-question-faq).
