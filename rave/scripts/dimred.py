"""
Manages custom latent projections (dimension reductions) of a RAVE model.

    rave dimred [add] --model M --name N --audio A [--audio B] [--filter '*pattern*'] [--exclude ...]
                      [--type pca --type ica] [--out O]
    rave dimred remove --model M --name N [--type pca] [--out O]
    rave dimred show --model M
    rave dimred purge_cache [--model M]

modes:
  add          fits projections on the latents of the given audio folders / files
  remove       removes the projections N (of the given --type, default: all types)
  show         prints the name, type and dimensionality of the projections of a model
  purge_cache  removes the latent caches (of --model only, if given), including those
               written by `rave export --db_path` (--cache, default)

audio inputs (--audio): folders / files, or parts of partitions of pre-processed acids
datasets, as /path/to/dataset->part (e.g. ->train, ->val) or /path/to/dataset->file:part
when several partition files contain the part. Each input is a class for lda.

models (--model):
  * run folder / .ckpt: the checkpoint is edited in place (or written to --out).
  * .ts: the model is uncompiled to a temporary folder, edited, and re-exported with its
    original settings to --out, or replaces the original .ts after confirmation.

projection types (stored as projections.<type>_<name> in the model):
    pca    PCA
    ica    ICA on the PCA-whitened subspace of dimension --dims (or reaching --fidelity)
    wpca   whitened PCA (unit variance on every axis)
    sfa    slow feature analysis: axes sorted from the slowest to the fastest, on the
           PCA-whitened subspace of dimension --dims (or reaching --fidelity)
    pls    partial least squares with audio --descriptors (default: all)
    cca    canonical correlation analysis with audio --descriptors
    lda    linear discriminant analysis between classes: one per --audio input, or the
           first group of --label_regex on file paths
  descriptors: loudness (dB), centroid / bandwidth (log Hz), flatness, f0 (continuous, log
  scale in semitones, acids_dataset F0 tracker --pitch_mode; unvoiced frames are ignored
  by pls / cca).
  Linear projections are completed to invertible maps (remaining dims: principal
  components); pls / lda axes are scaled to unit variance.

--fidelity / --dims set the ica / sfa subspace, the number of pls / cca / lda components
(default: number of descriptors / classes - 1), and the number of dimensions reported for
pca / wpca (stored full-size: their latent size is chosen at runtime).

latent cache (add mode): with --cache, latents are stored in the package, in
    rave/.dimred_cache/<model_name>_<path_hash>/<version>/<name>/<id_hash>/
where path_hash identifies the model file, and id_hash identifies the audio inputs (paths, sizes, modification times) and encoding
parameters, and reloaded instead of encoding audio again. --force re-encodes the audio
(and re-writes the cache).
"""
import logging
import sys
from pathlib import Path

from absl import flags

from rave.projections import DESCRIPTORS
from rave.scripting import (PROJECTION_TYPES, add_projections, remove_projections, purge_latents_cache,
                            projection_key, latents_cache_size, format_size, LATENTS_CACHE_ROOT,
                            describe_projections, format_projections)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

MODES = ("add", "remove", "show", "purge_cache")

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, help="run folder, .ckpt or .ts model")
flags.DEFINE_string("name", None, help="name of the projections")
flags.DEFINE_multi_string("audio", None, help="add: audio folder(s) / file(s) to fit projections on, or a part of a "
                          "partition of a pre-processed dataset: /path/to/dataset->train (or ->file:part)")
flags.DEFINE_multi_string("filter", [], help="add: glob-like pattern(s) selecting audio files, matched against paths "
                          "relative to each --audio folder (or file names), e.g. '*violin*', 'takes/*_close.wav'")
flags.DEFINE_multi_string("exclude", [], help="add: glob-like pattern(s) of audio files to exclude")
flags.DEFINE_multi_enum("type", ["pca"], list(PROJECTION_TYPES),
                        help="projection type(s): " + ", ".join(f"{k} ({v[1]})" for k, v in PROJECTION_TYPES.items())
                        + " (add default: pca; remove default: all types)")
flags.DEFINE_multi_enum("descriptors", None, list(DESCRIPTORS),
                        help="pls / cca: audio descriptors to align with (default: all)")
flags.DEFINE_string("pitch_mode", "pyin", help="pls / cca: acids_dataset F0 tracker used for the f0 descriptor "
                    "(pyin, yin, and crepe / pesto / world / praat if installed)")
flags.DEFINE_string("label_regex", None, help="lda: regular expression on audio file paths whose first group "
                    "is the class label (default: one class per --audio input)")
flags.DEFINE_string("out", None, help="output path (default: overwrite the input model)")
flags.DEFINE_bool("yes", False, help="do not ask for confirmation before replacing a .ts")
flags.DEFINE_integer("chunk_length", 2**17, help="length (samples) of encoded audio chunks")
flags.DEFINE_integer("batch_size", 8, help="encoding batch size")
flags.DEFINE_integer("max_chunks", None, help="maximum number of encoded chunks (random subset)")
flags.DEFINE_string("device", "cpu", help="encoding device")
flags.DEFINE_float("fidelity", .99, help="explained variance defining the number of dimensions: whitened subspace "
                   "for ica, reported dimension for pca")
flags.DEFINE_integer("dims", None, help="number of dimensions (overrides --fidelity)")
flags.DEFINE_bool("cache", False, help="cache computed latents in rave/.dimred_cache, "
                  "and load them instead of encoding audio when valid")
flags.DEFINE_bool("force", False, help="re-encode audio even if a valid cache exists (and re-write the cache)")
flags.DEFINE_string("ckpt_name", "last.ckpt", help="checkpoint name, if --model is a run folder")


def _confirm(message) -> bool:
    if not sys.stdin.isatty():
        return False
    return input(f"{message} [y/N] ").strip().lower() in ("y", "yes")


def _parse_mode(argv) -> str:
    args = [a for a in argv[1:] if a != "dimred"]  # "rave dimred <mode>" keeps "dimred" in argv
    if len(args) > 1:
        raise flags.Error(f"too many positional arguments: {args}")
    mode = args[0] if args else "add"
    if mode not in MODES:
        raise flags.Error(f"unknown mode {mode}; available: {MODES}")
    return mode


def _require(*names):
    missing = [f"--{n}" for n in names if not getattr(FLAGS, n)]
    if missing:
        raise flags.Error(f"missing required flag(s): {', '.join(missing)}")


def _overwrite_ok(model: Path, action: str) -> bool:
    if model.suffix != ".ts" or FLAGS.out is not None:
        return True
    if FLAGS.yes or _confirm(f"{model} will be replaced by the model with {action}. Are you sure?"):
        return True
    print("aborted (use --out to write a new model, or --yes to replace it)")
    return False


def main(argv):
    mode = _parse_mode(argv)

    if mode == "purge_cache":
        removed = purge_latents_cache(model_path=FLAGS.model, ckpt_name=FLAGS.ckpt_name)
        print(f"\nremoved {len(removed)} cache folder(s)" + "".join(f"\n  {r}" for r in removed))
        return

    if mode == "show":
        _require("model")
        print(f"\nprojections of {FLAGS.model}\n")
        print(format_projections(describe_projections(FLAGS.model, ckpt_name=FLAGS.ckpt_name)))
        return

    _require("model", "name", *(["audio"] if mode == "add" else []))
    model = Path(FLAGS.model)

    if mode == "add":
        if not _overwrite_ok(model, "new projections"):
            return
        keys = [projection_key(FLAGS.name, t) for t in FLAGS.type]
        out = add_projections(model, FLAGS.name, FLAGS.audio, types=FLAGS.type, out=FLAGS.out,
                              overwrite=True, chunk_length=FLAGS.chunk_length, batch_size=FLAGS.batch_size,
                              max_chunks=FLAGS.max_chunks, device=FLAGS.device, ckpt_name=FLAGS.ckpt_name,
                              dims=FLAGS.dims, fidelity=FLAGS.fidelity, cache=FLAGS.cache, force=FLAGS.force,
                              filters=FLAGS.filter, exclude=FLAGS.exclude, descriptors=FLAGS.descriptors,
                              label_regex=FLAGS.label_regex, pitch_mode=FLAGS.pitch_mode)
        print(f"\nprojections {keys} added to {out}")
        if FLAGS.cache:
            print(f"cache size: {format_size(latents_cache_size(model, ckpt_name=FLAGS.ckpt_name))} for this model, "
                  f"{format_size(latents_cache_size())} in total ({LATENTS_CACHE_ROOT})")

    elif mode == "remove":
        if not _overwrite_ok(model, "projections removed"):
            return
        types = FLAGS.type if FLAGS["type"].present else None
        out, removed = remove_projections(model, FLAGS.name, types=types, out=FLAGS.out, overwrite=True,
                                          ckpt_name=FLAGS.ckpt_name)
        print(f"\nprojections {removed} removed from {out}" if removed else f"\nno projection removed")


if __name__ == "__main__":
    from absl import app
    app.run(main)
