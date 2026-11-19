"""
Converts an exported TorchScript RAVE model (.ts) back into a trainable `rave.RAVE`
checkpoint (see `rave.uncompiling.uncompile` for details on how the config is recovered).

Writes into --out (default: <model>_uncompiled next to the .ts):
    last.ckpt       state dict, loadable with rave.load_rave_checkpoint(out)
    config.gin      gin config of the rebuilt model
    metadata.json   export information (causal, streaming, sr, latent size, ...)

Usage:
    rave uncompile --model model.ts [--out out_dir] [--mode auto|embedded|file|infer|shell]
                   [--config config.gin] [--override BINDING] [--reference run_or_ckpt] [--check]
"""
import json
import logging
from pathlib import Path

from absl import flags

import rave
from rave.uncompiling import (UNCOMPILE_MODES, uncompile, save_uncompiled, compare_state_dicts,
                              effective_state_dict)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("uncompile")

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, help="exported .ts model", required=True)
flags.DEFINE_string("out", None, help="output folder (default: <model>_uncompiled next to the .ts)")
flags.DEFINE_enum("mode", "auto", list(UNCOMPILE_MODES),
                  help="config source: auto (config embedded in the .ts, else inferred from the architecture), "
                       "embedded, file (config.gin next to the .ts), infer, or shell (inference-only, nothing written)")
flags.DEFINE_multi_string("config", None, help="gin config(s) to rebuild the model with (overrides the embedded one)")
flags.DEFINE_multi_string("override", [], help="additional gin bindings")
flags.DEFINE_string("ckpt_name", "last.ckpt", help="name of the written checkpoint")
flags.DEFINE_string("reference", None, help="original run / ckpt to compare weights against")
flags.DEFINE_bool("check", False, help="run round-trip and functional checks")


def main(argv):
    model, config_str, metadata = uncompile(FLAGS.model, config=FLAGS.config, overrides=FLAGS.override,
                                            mode=FLAGS.mode, check=FLAGS.check)
    print(f"\nuncompiled model: encoder: {type(model.encoder).__name__} | decoder: {type(model.decoder).__name__}"
          f" | sr={model.sr} | n_channels={model.n_channels} | latent_size={model.latent_size}")
    print(json.dumps(metadata, indent=2) + "\n")

    if FLAGS.reference is not None:
        ref, _ = rave.load_rave_checkpoint(FLAGS.reference)
        compare_state_dicts(model.state_dict(), ref.state_dict(), name_a="uncompiled", name_b="reference")
        compare_state_dicts(effective_state_dict(model), effective_state_dict(ref),
                            name_a="uncompiled(folded)", name_b="reference(folded)")
        for k in ("latent_pca", "latent_mean", "fidelity"):
            log.info(f"[uncompiled vs reference] {k} max abs diff = "
                     f"{(getattr(model, k) - getattr(ref, k)).abs().max().item():.3e}")

    if config_str is None:
        log.warning("shell mode: no config available, nothing written")
        return
    out = FLAGS.out or Path(FLAGS.model).with_name(Path(FLAGS.model).stem + "_uncompiled")
    save_uncompiled(model, out, config_str, metadata, ckpt_name=FLAGS.ckpt_name)


if __name__ == "__main__":
    from absl import app
    app.run(main)
