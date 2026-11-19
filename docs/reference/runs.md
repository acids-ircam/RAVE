# Run folders and checkpoints

## What `rave train` writes

```text
<out_path>/
└── <name>_<hash>/                 ← the "run folder"
    ├── config.gin                 ← full configuration of the model
    └── version_0/                 ← one per training session (resuming creates version_1, ...)
        ├── events.out.tfevents... ← tensorboard logs
        ├── hparams.yaml
        └── checkpoints/
            ├── best.ckpt          ← best validation loss so far
            ├── step_200000.ckpt   ← numbered checkpoint every --save_every batches
            ├── last.ckpt          ← refreshed every 25,000 batches and at normal training end
            └── interrupt.ckpt     ← saved on a handled Ctrl+C
```

`<hash>` is derived from the configuration, so two trainings with the same name but different
configs do not overwrite each other.

`rave finetune` writes to `<out_path>/<name>/version_N/`, with the `config.gin` inside the version
folder.

`rave train_prior` writes to `<out_path>/<name>_<hash>/` with the same layout.

## How paths are resolved

Every command that takes a model (`--run`, `--model`, `--resume`) accepts:

- a `.ckpt` file — used directly;
- a run or version folder — RAVE searches it recursively for `last.ckpt` (or the name given by
  `--ckpt_name` in `export` / `dimred`). If none is found, `export`, `generate`, `finetune`,
  `train_prior` and `dimred` fall back to the **most recently written** checkpoint.

The configuration is searched as `config.gin` in the given folder and its subfolders, then in its
parent and grand-parent folders. Keep `config.gin` next to (or above) your checkpoints when moving
runs around.

> `last.ckpt` is refreshed every 25,000 batches and when training finishes. To export a model
> earlier, export the run folder anyway (the most recent checkpoint is used) or pass
> `--ckpt_name best.ckpt`.

## Monitoring

```bash
tensorboard --logdir runs/
```

See the [Tensorboard guide](../tensorboard_guide.md) for how to read the curves. Validation also
logs audio reconstructions (`audio_val`) and latent statistics (`latent_active_dims`, fidelity
curves) used to choose the export latent size.

## What `rave export` writes

`<out_path>/<name>[_nopca][_streaming].ts` — a single self-contained TorchScript file holding the
encoder, decoder, PQMF, latent projections, optional prior and the gin configuration. It does not
contain the discriminator, so it cannot be trained further directly; use
[`rave uncompile`](cli.md#rave-uncompile) to get a trainable checkpoint back.
