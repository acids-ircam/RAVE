"""
Uncompiling: scripted RAVE models (.ts) -> rave.RAVE objects / checkpoints.

Main entry points: uncompile, save_uncompiled, load_uncompile_metadata.
"""
import os
import re

import cached_conv as cc
import gin
import torch
import torch.nn as nn
from typing import Optional

from . import blocks
from .model import RAVE
from .projections import LINEAR_PROJECTION_PREFIXES

# --------------------------------------------------------------------------- #
#  Uncompiling: scripted model (.ts) -> rave.RAVE checkpoint
# --------------------------------------------------------------------------- #
#
#  A RAVE object is rebuilt from a gin configuration, then every weight / buffer
#  of the scripted model is transferred into it. Weight-norm (removed at export)
#  is re-applied by recomputing (weight_g, weight_v) from the merged weight, so
#  the resulting model is a regular, trainable RAVE (discriminator and losses are
#  freshly initialised, since they are not exported).
#
#  The configuration is taken from (in "auto" mode, first available wins):
#    1. the config embedded in the scripted model at export   ("embedded"),
#       written verbatim as config.gin
#    2. inference from the scripted architecture              ("inferred"):
#       architecture macros (capacity, ratios, dilations, latent size, causal,
#       snake / adain...) are read from the scripted modules, candidate base
#       configs from rave/configs are built with them, and the one whose
#       module structure matches the scripted model is kept.
#    3. a "shell" model wrapping the scripted sub-modules     ("shell"):
#       inference only, no config / checkpoint can be written.
#  An explicit config file (or a config.gin next to the .ts) is only used in
#  "file" mode, or when given through the `config` argument.

import datetime
import gc
import json
import logging
from pathlib import Path
from torch.nn.utils.weight_norm import WeightNorm

_log = logging.getLogger(__name__)

UNCOMPILE_METADATA_FILE = "metadata.json"
UNCOMPILE_MODES = ("auto", "embedded", "file", "infer", "shell")

# buffers that only exist in the scripted wrapper / streaming caches
_SKIP_PATTERNS = (".cache.pad", ".downsampling_delay.pad", ".pad", ".cache")
_ARCH_PREFIXES = ("encoder", "decoder", "pqmf", "spectrogram")
_IGNORED_SUBMODULES = (".cache", ".downsampling_delay", ".paddings", "rvq.")
_ENCODER_ADDONS = {"WasserteinEncoder": "wasserstein", "SphericalEncoder": "spherical"}
_MODEL_PREFIXES = ("encoder.", "decoder.", "pqmf.", "spectrogram.")
_STATE_BUFFERS = ("warmed_up",)


def _is_streaming_buffer(key: str) -> bool:
    return key.endswith(_SKIP_PATTERNS)


_MISSING = object()

# legacy exports (RAVE v1 `TraceModel`): attribute name -> legacy attribute name.
# There, `latent_size` is the full latent size and `cropped_latent_size` the exported one.
_LEGACY_ATTRS = {"sr": "sampling_rate", "full_latent_size": "latent_size", "latent_size": "cropped_latent_size"}


def _is_legacy_export(ts) -> bool:
    return isinstance(ts, torch.jit.ScriptModule) and hasattr(ts, "cropped_latent_size")


def _ts_attr(ts, name, default=None):
    if ts is None:
        return default
    if _is_legacy_export(ts):
        if name == "n_channels" and not hasattr(ts, "n_channels") and hasattr(ts, "encode_params"):
            return int(ts.encode_params[0])  # legacy exports: input channels of the encode method
        name = _LEGACY_ATTRS.get(name, name)
    try:
        value = getattr(ts, name)
    except Exception:
        return default
    if isinstance(value, torch.Tensor) and value.ndim == 0:  # legacy exports store scalars as buffers
        value = value.item()
    return value


def is_scripted_rave(ts) -> bool:
    """True if ts is a model exported by RAVE (`rave export`: *ScriptedRAVE classes, RAVE + prior
    `Combined` wrappers, legacy v1 TraceModel), not any other scripted model."""
    if not isinstance(ts, torch.jit.ScriptModule):
        return False
    if _raw_class_name(ts) == "Combined" and _ts_attr(ts, "_rave") is not None:
        return is_scripted_rave(ts._rave)
    if _is_legacy_export(ts):
        return _raw_class_name(ts) == "TraceModel"
    return _raw_class_name(ts).endswith("ScriptedRAVE")


def unwrap_scripted(ts):
    """returns the scripted RAVE inside wrappers (e.g. `Combined` RAVE + prior exports)."""
    while _raw_class_name(ts) == "Combined" and _ts_attr(ts, "_rave") is not None:
        _log.info("scripted model is a RAVE + prior wrapper: uncompiling its RAVE")
        ts = ts._rave
    if _ts_attr(ts, "encoder") is None or _ts_attr(ts, "decoder") is None:
        children = [n for n, _ in ts.named_children()]
        raise NotImplementedError(
            f"unsupported scripted model {_raw_class_name(ts)} (no encoder / decoder sub-modules, "
            f"children: {children}); graph-based exports (e.g. torchbend) cannot be uncompiled")
    return ts


def _raw_class_name(mod) -> str:
    name = getattr(mod, "original_name", None) if isinstance(mod, torch.jit.ScriptModule) else None
    return name or type(mod).__name__


def _module_class(mod) -> str:
    name = _raw_class_name(mod)
    return name[len("Cached"):] if name.startswith("Cached") else name


def _weight_norm_hooks(model: nn.Module):
    """yields (module_name, module, WeightNorm hook) for every weight-normed module."""
    for mname, m in model.named_modules():
        for hook in m._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm):
                yield mname, m, hook


def _jsonable(v):
    if isinstance(v, torch.Tensor):
        return v.tolist()
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, Path):
        return str(v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


# ---- scripted model inspection --------------------------------------------- #

def _convs(root, prefix):
    out = []
    for name, mod in root.named_modules():
        if name.startswith(prefix) and _module_class(mod) in ("Conv1d", "ConvTranspose1d") \
                and not any(p in name + "." for p in _IGNORED_SUBMODULES):
            out.append((name, mod, tuple(mod.weight.shape)))
    return out


def is_streaming(root: nn.Module) -> bool:
    """True if the (scripted or python) model contains cached convolutions."""
    for name, mod in root.named_modules():
        # CachedPQMF / CachedSequential exist in non-streaming models too
        if name.startswith(_ARCH_PREFIXES) and _raw_class_name(mod).startswith(("CachedConv", "CachedPadding")):
            return True
    return any(k.endswith(".cache.pad") for k in root.state_dict())


def is_causal(root: nn.Module) -> bool:
    """True if no encoder / decoder convolution looks at the future (works on scripted
    or python models, streaming or not)."""
    for _, mod, shape in _convs(root, "encoder") + _convs(root, "decoder"):
        pad = _ts_attr(mod, "_pad")
        if isinstance(_ts_attr(mod, "cache"), nn.Module):
            # streaming: non-causal convs have right padding, which shows up as delay
            if shape[-1] > 1 and int(mod.stride[0]) == 1 and int(_ts_attr(mod, "cumulative_delay", 0)) > 0:
                return False
        elif pad is not None and shape[-1] > 1 and pad[1] > 0:
            return False
    return True


def get_scripted_metadata(ts: torch.jit.ScriptModule) -> dict:
    """export-time information that can be read from a scripted RAVE."""
    resampler = _ts_attr(ts, "resampler")
    model_sr = _ts_attr(resampler, "model_sr") if resampler is not None else None
    sr = _ts_attr(ts, "sr")
    projections = _ts_attr(ts, "projections_names")
    return _jsonable({
        "script_class": _raw_class_name(ts),
        "streaming": is_streaming(ts),
        "causal": is_causal(ts),
        "sr": sr,
        "model_sr": model_sr if model_sr is not None else sr,
        "resampled": resampler is not None,
        "n_channels": _ts_attr(ts, "n_channels"),
        "target_channels": _ts_attr(ts, "target_channels"),
        "latent_size": _ts_attr(ts, "latent_size"),
        "full_latent_size": _ts_attr(ts, "full_latent_size"),
        "input_mode": _ts_attr(ts, "input_mode"),
        "output_mode": _ts_attr(ts, "output_mode"),
        "use_pca": _ts_attr(ts, "use_pca"),
        "ratio_encode": _ts_attr(ts, "ratio_encode"),
        "receptive_field": _ts_attr(ts, "receptive_field"),
        "projections": list(projections) if projections is not None else None,
        "has_prior": bool(_ts_attr(ts, "_has_prior", False)),
        "has_embedded_config": bool(_ts_attr(ts, "gin_config_str")),
    })


# ---- building a RAVE from a configuration ---------------------------------- #

def build_rave_from_config(configs=(), overrides=(), config_str=None):
    """builds a non-cached RAVE from gin files and / or a gin config string.
    Returns (model, operative config string)."""
    from . import core
    configs = [str(c) for c in (configs if isinstance(configs, (list, tuple)) else [configs]) if c]
    cached_state = cc.use_cached_conv.__globals__["USE_CACHED_CONV"]
    cc.use_cached_conv(False)
    try:
        with core.GinEnv():
            if config_str is not None:
                gin.parse_config(config_str)
            gin.parse_config_files_and_bindings(configs, list(overrides))
            model = RAVE()
            out_config_str = gin.config_str()
    finally:
        cc.use_cached_conv(cached_state)
    return model, out_config_str


def architecture_signature(root: nn.Module) -> dict:
    """{module name: structural description} for encoder / decoder / pqmf modules.
    Works on both scripted and python modules, cached or not, weight-normed or not."""
    sig = {}
    for name, mod in root.named_modules():
        if not name.startswith(_ARCH_PREFIXES) or any(p in name + "." for p in _IGNORED_SUBMODULES):
            continue
        own = {}
        for pname, p in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
            # biases: older models only had some of them; handled in transfer_weights
            if pname.endswith("_g") or pname in ("pad", "cache", "bias"):
                continue
            own[pname[:-2] if pname.endswith("_v") else pname] = tuple(p.shape)
        is_leaf = len(list(mod.children())) == 0
        if (not own and not is_leaf) or _module_class(mod) in ("DumbSpectrogram", "DumbPQMF"):
            continue
        sig[name] = (_module_class(mod), tuple(sorted(own.items())),
                     _ts_attr(mod, "stride"), _ts_attr(mod, "dilation"), _total_pad(mod))
    return sig


def _total_pad(mod):
    if _module_class(mod) == "ConvTranspose1d":
        # not stored consistently across cached_conv versions; set by kernel / stride in RAVE blocks
        return None
    cache = _ts_attr(mod, "cache")
    if isinstance(cache, nn.Module):  # cached (streaming) conv: padding lives in the cache
        pad = dict(cache.named_buffers()).get("pad")
        return int(pad.shape[-1]) if pad is not None else None
    pad = _ts_attr(mod, "_pad")
    return int(sum(pad)) if pad is not None else None


def _signature_groups(sig: dict) -> dict:
    """{root ("encoder", "decoder"...): [(module name, description), ...]} in module order."""
    groups = {}
    for k, v in sig.items():
        if v[1] and all(name in _STATE_BUFFERS for name, _ in v[1]):
            continue  # containers holding only training state (absent from older exports)
        groups.setdefault(k.split(".")[0], []).append((k, v))
    return groups


def module_name_map(sig_a: dict, sig_b: dict) -> dict:
    """maps module names of `sig_a` to those of `sig_b` for every root (encoder, decoder...)
    whose modules have the same descriptions in the same order, but different names
    (e.g. legacy exports, where containers were nested differently)."""
    mapping = {}
    groups_a, groups_b = _signature_groups(sig_a), _signature_groups(sig_b)
    for root, mods_a in groups_a.items():
        mods_b = groups_b.get(root, [])
        if [n for n, _ in mods_a] == [n for n, _ in mods_b]:
            continue
        if [v for _, v in mods_a] == [v for _, v in mods_b]:
            mapping.update({na: nb for (na, _), (nb, _) in zip(mods_a, mods_b)})
    return mapping


def signature_distance(sig_a: dict, sig_b: dict):
    """(number of differing modules, [(name, a, b), ...]). Modules are compared by name,
    or by order for roots whose module names differ (see `module_name_map`)."""
    mapping = module_name_map(sig_a, sig_b)
    if mapping:
        sig_a = {mapping.get(k, k): v for k, v in sig_a.items()}
    is_state = lambda v: bool(v[1]) and all(name in _STATE_BUFFERS for name, _ in v[1])
    sig_a = {k: v for k, v in sig_a.items() if not is_state(v)}
    sig_b = {k: v for k, v in sig_b.items() if not is_state(v)}
    diffs = [(k, sig_a.get(k), sig_b.get(k)) for k in sorted(set(sig_a) | set(sig_b))
             if sig_a.get(k) != sig_b.get(k)]
    return len(diffs), diffs


# ---- configuration inference ----------------------------------------------- #

def infer_hyperparameters(ts) -> dict:
    """reads architecture macros from the scripted model."""
    hp = {}
    n_channels = int(_ts_attr(ts, "n_channels", 1))
    hp["n_channels"] = n_channels
    resampler = _ts_attr(ts, "resampler")
    hp["SAMPLING_RATE"] = int(_ts_attr(resampler, "model_sr", None) or _ts_attr(ts, "sr"))
    hp["LATENT_SIZE"] = int(_ts_attr(ts, "full_latent_size"))

    pqmf = _ts_attr(ts, "pqmf")
    pqmf_bands = int(_ts_attr(pqmf, "n_band")) if pqmf is not None and _ts_attr(pqmf, "n_band") else None
    hp["attenuation"] = _ts_attr(pqmf, "attenuation") if pqmf is not None else None

    enc_root = "encoder.encoder" if _ts_attr(ts.encoder, "encoder") is not None else "encoder"
    enc_convs = _convs(ts, enc_root)
    hp["CAPACITY"] = enc_convs[0][2][0]
    hp["N_ENCODER_BAND"] = enc_convs[0][2][1] // n_channels
    hp["N_BAND"] = 1 if _ts_attr(ts, "output_mode") == "raw" else (pqmf_bands or hp["N_ENCODER_BAND"])

    # ratios (strided convs) and dilations (dilated convs between them)
    ratios, dilations, stage, kernel = [], [], [], None
    for name, mod, shape in enc_convs:
        stride, dil = int(mod.stride[0]), int(mod.dilation[0])
        if stride > 1:
            # older cached_conv: strided convs padded with (stride - 1) samples less
            pad, span = _total_pad(mod), (shape[-1] - 1) * dil
            if pad is not None and pad != span and pad == max(span + 1 - stride, 0):
                hp["legacy_stride_padding"] = True
            ratios.append(stride)
            dilations.append(stage)
            stage = []
        elif ".branches." in name and shape[-1] > 1:
            stage.append(dil)
            kernel = kernel or shape[-1]
    hp["RATIOS"] = ratios
    # decoder upsampling stages (noise generators excluded), in config (encoder) order
    dec_ratios, dec_dilations, dec_stage = [], [], None
    for name, mod, shape in _convs(ts, "decoder.net."):
        if _module_class(mod) == "ConvTranspose1d" or int(mod.stride[0]) > 1:
            dec_ratios.append(int(mod.stride[0]))
            dec_stage = []
            dec_dilations.append(dec_stage)
        elif dec_stage is not None and ".branches." in name and shape[-1] > 1:
            dec_stage.append(int(mod.dilation[0]))
    hp["DEC_RATIOS"] = dec_ratios[::-1]
    if any(dilations):
        hp["DILATIONS"] = dilations
    if _ts_attr(ts, "input_mode") == "mel" and dec_ratios:
        # spectrogram encoders: RATIOS / DILATIONS describe the decoder, ENCODER_RATIOS the encoder
        hp["ENCODER_RATIOS"], hp["RATIOS"] = ratios, hp["DEC_RATIOS"]
        if any(dec_dilations):
            hp["DILATIONS"] = dec_dilations[::-1]
    if kernel:
        hp["KERNEL_SIZE"] = kernel

    dec_convs = _convs(ts, "decoder")
    hp["NOISE_AUGMENTATION"] = max(dec_convs[0][2][1] - hp["LATENT_SIZE"], 0)

    # discrete models
    sd = ts.state_dict()
    layers = {k.split("rvq.layers.")[1].split(".")[0] for k in sd if "rvq.layers." in k}
    if layers:
        hp["NUM_QUANTIZERS"] = len(layers)
        embed = [v for k, v in sd.items() if "rvq.layers.0." in k and v.ndim == 2]
        if embed:
            hp["CODEBOOK_SIZE"] = int(embed[0].shape[0])

    # v1 decoders: residual stacks (dilations of every residual layer of the first stack)
    stacks = [n for n, m in ts.named_modules() if n.startswith("decoder") and _module_class(m) == "ResidualStack"]
    if stacks:
        layers, kernels = {}, {}  # residual layer -> dilations / kernel size
        for name, mod, shape in _convs(ts, stacks[0] + "."):
            if ".aligned." in name and shape[-1] > 1:
                layer = name.rsplit(".aligned.", 1)[0]
                layers.setdefault(layer, []).append(int(mod.dilation[0]))
                kernels.setdefault(layer, shape[-1])
        if layers:
            # one branch of residual layers per kernel size (same dilations for every branch)
            hp["RES_KERNEL_SIZES"] = list(dict.fromkeys(kernels.values()))
            hp["RES_DILATIONS"] = [d for l, d in layers.items() if kernels[l] == hp["RES_KERNEL_SIZES"][0]]

    # v1 encoders: (repeat_layers - 1) stride-1 convs after each strided conv
    if _module_class(_ts_attr(ts.encoder, "encoder") or ts.encoder) == "Encoder" and len(ratios) > 1:
        strided = [i for i, (_, m, _) in enumerate(enc_convs) if int(m.stride[0]) > 1]
        hp["REPEAT_LAYERS"] = strided[1] - strided[0]

    # v2 noise generator (decoder.noise_module)
    noise_convs = _convs(ts, "decoder.noise_module.")
    if noise_convs:
        hp["NOISE_RATIOS"] = [int(m.stride[0]) for _, m, _ in noise_convs]
        if len(noise_convs) > 1:
            hp["NOISE_HIDDEN"] = noise_convs[0][2][0]
        hp["NOISE_BANDS"] = noise_convs[-1][2][0] // (hp["N_BAND"] * n_channels)

    # v2 decoders: amplitude modulation doubles the output channels of the waveform conv
    if _module_class(ts.decoder) == "GeneratorV2":
        wave = _convs(ts, "decoder.waveform_module") or [c for c in _convs(ts, "decoder.net.")]
        if wave:
            ratio = wave[-1][2][0] / (hp["N_BAND"] * n_channels)
            if ratio in (1, 2):
                hp["AMPLITUDE_MODULATION"] = ratio == 2

    # biases (older configs, e.g. v1, disable them globally)
    for cls in ("Conv1d", "ConvTranspose1d"):
        convs = [m for _, m, _ in _convs(ts, "encoder") + _convs(ts, "decoder") if _module_class(m) == cls]
        hp[f"{cls}_bias"] = any(_ts_attr(m, "bias") is not None for m in convs)

    hp["classes"] = {_module_class(m) for n, m in ts.named_modules() if n.startswith(("encoder", "decoder"))}
    hp["causal"] = is_causal(ts)
    return hp


def _config_text(files) -> str:
    """concatenated text of gin files, with their includes resolved."""
    base_path = Path(__file__).parent.resolve()
    text, seen, stack = "", set(), [Path(f) for f in files]
    while stack:
        f = stack.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        content = f.read_text()
        text += content
        for inc in re.findall(r'^include\s+["\'](.+?)["\']', content, flags=re.M):
            stack.append(base_path / inc)
    return text


def _hp_bindings(hp, files) -> list:
    """gin bindings for inferred macros (only macros actually used by the config files)."""
    text = _config_text(files)
    values = dict(hp, ENC_RATIOS=hp.get("ENCODER_RATIOS", hp["RATIOS"]))
    b = [f"{k} = {values[k]}" for k in ("SAMPLING_RATE", "LATENT_SIZE", "CAPACITY", "N_BAND", "N_ENCODER_BAND",
                                         "KERNEL_SIZE", "DILATIONS", "NOISE_AUGMENTATION", "NUM_QUANTIZERS",
                                         "CODEBOOK_SIZE", "RATIOS", "ENC_RATIOS", "ENCODER_RATIOS", "DEC_RATIOS")
         if k in values and f"%{k}" in text and not (k == "RATIOS" and "%ENC_RATIOS" in text)]
    b.append(f"RAVE.n_channels = {hp['n_channels']}")
    if hp.get("attenuation") is not None:
        b.append(f"pqmf.CachedPQMF.attenuation = {hp['attenuation']}")
    if "AMPLITUDE_MODULATION" in hp and "blocks.GeneratorV2" in text:
        b.append(f"blocks.GeneratorV2.amplitude_modulation = {hp['AMPLITUDE_MODULATION']}")
    if "NOISE_BANDS" in hp and "blocks.NoiseGeneratorV2" in text:
        b.append(f"blocks.NoiseGeneratorV2.ratios = {hp['NOISE_RATIOS']}")
        b.append(f"blocks.NoiseGeneratorV2.noise_bands = {hp['NOISE_BANDS']}")
        if "NOISE_HIDDEN" in hp:
            b.append(f"blocks.NoiseGeneratorV2.hidden_size = {hp['NOISE_HIDDEN']}")
    if "RES_DILATIONS" in hp and "blocks.ResidualStack" in text:
        b.append(f"blocks.ResidualStack.dilations_list = {hp['RES_DILATIONS']}")
        b.append(f"blocks.ResidualStack.kernel_sizes = {hp['RES_KERNEL_SIZES']}")
    if "REPEAT_LAYERS" in hp and "blocks.Encoder" in text:
        b.append(f"blocks.Encoder.repeat_layers = {hp['REPEAT_LAYERS']}")
    for cls in ("Conv1d", "ConvTranspose1d"):
        if hp.get(f"{cls}_bias") and re.search(rf"cc\.{cls}\.bias\s*=\s*False", text):
            b.append(f"cc.{cls}.bias = True")
    if hp.get("legacy_stride_padding"):
        b.append("cc.get_padding.legacy_stride = True")
    return b


def _candidate_configs(ts, hp):
    """yields lists of config files, most likely first."""
    config_dir = Path(__file__).parent.resolve() / "configs"
    enc_cls = _module_class(ts.encoder)
    dec_cls = _module_class(ts.decoder)
    classes = hp["classes"]
    snake, adain = "Snake" in classes, "AdaptiveInstanceNormalization" in classes
    noise = "NoiseGeneratorV2" in classes

    if dec_cls == "Generator":
        bases = ["v1"]
    elif _ts_attr(ts, "input_mode") == "mel":
        bases = ["v2_with_augs", "hybrid"]
    elif _ts_attr(ts, "output_mode") == "raw":
        bases = ["v2_nopqmf", "v2_nopqmf_small"]
    elif enc_cls == "DiscreteEncoder":
        bases = ["discrete_v3", "discrete"] if snake else ["discrete"]
    else:
        bases = ["v3", "v2"] if (snake and adain) else ["v2"]
    for b in ["v2", "v1", "discrete", "v3", "v2_nopqmf", "v2_nopqmf_small", "v2_with_augs", "hybrid", "v4"]:
        if b not in bases:
            bases.append(b)

    addons = []
    if enc_cls in _ENCODER_ADDONS:
        addons.append(_ENCODER_ADDONS[enc_cls])
    if snake:
        addons.append("snake")
    if adain:
        addons.append("adain")
    if noise:
        addons.append("noise")

    for causal in (hp["causal"], not hp["causal"]):
        for base in bases:
            base_text = _config_text([config_dir / f"{base}.gin"])
            names = [base] + [a for a in addons + (["causal"] if causal else [])
                              if f"configs/{a}.gin" not in base_text]
            yield [str(config_dir / f"{n}.gin") for n in names]


def infer_config(ts, overrides=(), max_candidates=12):
    """builds the RAVE model whose architecture best matches the scripted model.
    Returns (model, config_str, info) where info holds the candidate files, bindings
    and the architecture distance (0 = exact match).

    PQMF filter sizes are not taken into account to rank candidates (the scripted
    filters are loaded as is, whatever the attenuation of the config). Candidates are
    built one at a time and released, the best one being rebuilt at the end, so that
    only one candidate model is in memory at once."""
    hp = infer_hyperparameters(ts)
    _log.info("inferred hyper-parameters: " + ", ".join(f"{k}={v}" for k, v in hp.items() if k != "classes"))
    ts_sig = architecture_signature(ts)
    best = None  # ((core distance, distance), diffs, files, bindings)
    for i, files in enumerate(_candidate_configs(ts, hp)):
        if i >= max_candidates:
            break
        bindings = _hp_bindings(hp, files) + list(overrides)
        try:
            model, _ = build_rave_from_config(files, bindings)
            dist, diffs = signature_distance(ts_sig, architecture_signature(model))
        except Exception as e:
            _log.debug(f"candidate {files} failed: {e}")
            continue
        finally:
            model = None
            gc.collect()
        core_dist = sum(1 for k, _, _ in diffs if not k.startswith("pqmf"))
        _log.info(f"candidate {[Path(f).stem for f in files]}: {dist} structural differences"
                  + (f" ({dist - core_dist} in pqmf filters)" if dist != core_dist else ""))
        if best is None or (core_dist, dist) < best[0]:
            best = ((core_dist, dist), diffs, files, bindings)
        if core_dist == 0:
            break
    if best is None:
        raise RuntimeError("could not build any candidate configuration")
    (core_dist, dist), diffs, files, bindings = best
    model, config_str = build_rave_from_config(files, bindings)
    if core_dist > 0:
        _log.warning(f"no exact architecture match; best candidate {files} has {dist} differences, e.g.:")
        for k, a, b in [d for d in diffs if not d[0].startswith("pqmf")][:8]:
            _log.warning(f"  {k}: scripted={a} / rebuilt={b}")
    else:
        _log.info(f"architecture match with {[Path(f).stem for f in files]}"
                  + (" (pqmf filters loaded from the scripted model)" if dist else ""))
    header = (f"# config inferred from a scripted model\n"
              f"# candidate: {[Path(f).stem for f in files]} + bindings {bindings}\n"
              f"# architecture distance: {dist}\n\n")
    info = {"candidate": [Path(f).stem for f in files], "bindings": bindings}
    return model, header + config_str, info


# ---- weight transfer -------------------------------------------------------- #

def transfer_weights(ts: torch.jit.ScriptModule, model: RAVE, report: Optional[dict] = None) -> RAVE:
    """copies weights / buffers of a scripted model into a (weight-normed) RAVE.
    If given, `report` is filled with the number of missing / unused / mismatched tensors."""
    ts_sd = {k: v.detach().clone() for k, v in ts.state_dict().items()}
    target_sd = model.state_dict()

    # 0. legacy exports: modules nested differently -> rename scripted keys by module order
    mapping = module_name_map(architecture_signature(ts), architecture_signature(model))
    if mapping:
        _log.info(f"scripted module names differ from the rebuilt model: matching {len(mapping)} modules by order")
        renamed = {}
        for k, v in ts_sd.items():
            mod, _, attr = k.rpartition(".")
            while mod and mod not in mapping and "." in mod:  # buffers of sub-modules (e.g. cache.pad)
                mod, _, sub = mod.rpartition(".")
                attr = f"{sub}.{attr}"
            renamed[f"{mapping[mod]}.{attr}" if mod in mapping else k] = v
        ts_sd = renamed
    loaded, mismatched, pqmf_resized = set(), [], []

    # 1. weight-normed modules: recompute g / v from merged weight
    for mname, m, hook in _weight_norm_hooks(model):
        prefix = f"{mname}." if mname else ""
        w_key, g_key, v_key = (prefix + hook.name, prefix + hook.name + "_g", prefix + hook.name + "_v")
        if g_key in ts_sd and v_key in ts_sd:  # weight norm was not removed
            w_g, w_v = ts_sd[g_key], ts_sd[v_key]
        elif w_key in ts_sd:
            w_v = ts_sd[w_key]
            # w = g * v / ||v||  ->  choosing v = w gives g = ||w||
            w_g = torch.norm_except_dim(w_v, 2, hook.dim)
        else:
            continue
        if w_v.shape != target_sd[v_key].shape:
            mismatched.append((w_key, tuple(w_v.shape), tuple(target_sd[v_key].shape)))
            continue
        getattr(m, hook.name + "_g").data.copy_(w_g)
        getattr(m, hook.name + "_v").data.copy_(w_v)
        loaded.update({w_key, g_key, v_key})

    # 2. every other matching parameter / buffer
    new_sd = {}
    for k, v in ts_sd.items():
        if k in loaded or k not in target_sd:
            continue
        if target_sd[k].shape != v.shape:
            if k.startswith("pqmf.") and not _is_streaming_buffer(k):
                # PQMF rebuilds its filters / convs from the loaded tensors (e.g. other attenuation)
                pqmf_resized.append(k)
            else:
                mismatched.append((k, tuple(v.shape), tuple(target_sd[k].shape)))
                continue
        new_sd[k] = v
        loaded.add(k)
    if pqmf_resized:
        _log.info(f"pqmf filters differ in size from config ({pqmf_resized}): using the scripted ones")
        for k in ("pqmf.h", "pqmf.hk"):  # PQMF._load_from_state_dict requires both
            if k in ts_sd and k not in new_sd:
                new_sd[k] = ts_sd[k]
    model.load_state_dict(new_sd, strict=False)

    # 3. attributes stored outside the state dict
    rf = _ts_attr(ts, "receptive_field")
    if rf is not None:
        model.receptive_field.copy_(torch.tensor([int(rf[0]), int(rf[1])]))

    names = list(_ts_attr(ts, "projections_names", ["default"]))
    for i, name in enumerate(names):
        if i == 0:
            continue  # default projection == latent_pca / latent_mean
        proj = ts.projections.projection_list.__getattr__(str(i))
        psd = proj.state_dict()
        if name.startswith("pca"):
            buf = torch.cat([psd["latent_mean"][None], psd["latent_var"][None], psd["components"]], 0)
        elif name.startswith(tuple(LINEAR_PROJECTION_PREFIXES)):
            try:
                shape = psd["components"].shape
                mean = psd["latent_mean"]
                mean = mean[None].expand(shape) if mean.ndim == 1 else mean.reshape(shape)
                buf = torch.stack([mean] + [psd[k] for k in ("components", "mixing", "whitening")])
            except RuntimeError:
                _log.warning(f"could not rebuild linear projection {name}; skipped")
                continue
        else:
            _log.warning(f"projection {name} cannot be inverted; skipped")
            continue
        model.projections[name] = nn.Parameter(buf, requires_grad=False)

    # biases of the rebuilt model absent from the scripted one (older models had fewer
    # biases): zeroed, which is functionally identical
    zeroed = [k for k in target_sd if k.endswith(".bias") and k not in loaded and k.startswith(_MODEL_PREFIXES)
              and k[:-len("bias")] + "weight" in loaded | {kk[:-2] for kk in loaded if kk.endswith("_v")}]
    with torch.no_grad():
        params = dict(model.named_parameters())
        for k in zeroed:
            params[k].zero_()
            loaded.add(k)
    if zeroed:
        _log.info(f"{len(zeroed)} biases absent from the scripted model set to zero")

    # report
    missing = [k for k in target_sd if k.startswith(_MODEL_PREFIXES) and k not in loaded
               and k.rsplit(".", 1)[-1] not in _STATE_BUFFERS]
    unexpected = [k for k in ts_sd if k not in loaded and k.startswith(_MODEL_PREFIXES)
                  and not _is_streaming_buffer(k)]
    _log.info(f"transferred {len(loaded)} tensors")
    for k, s1, s2 in mismatched:
        _log.warning(f"shape mismatch for {k}: scripted {s1} / model {s2}")
    if missing:
        _log.warning(f"{len(missing)} encoder/decoder tensors not found in scripted model: {missing[:10]}...")
    if unexpected:
        _log.warning(f"{len(unexpected)} scripted tensors unused: {unexpected[:10]}...")
    if report is not None:
        report.update(missing_tensors=len(missing), unused_tensors=len(unexpected),
                      mismatched_tensors=len(mismatched), zeroed_biases=len(zeroed))
    return model


# ---- shell mode ------------------------------------------------------------- #

def build_rave_shell(ts: torch.jit.ScriptModule) -> RAVE:
    """inference-only RAVE wrapping the scripted sub-modules (no gin config needed)."""
    _const = lambda m: (lambda **kwargs: m)
    pqmf = _ts_attr(ts, "pqmf")
    model = RAVE(
        latent_size=int(_ts_attr(ts, "full_latent_size")),
        sampling_rate=int(_ts_attr(ts, "sr")),
        encoder=_const(ts.encoder),
        decoder=_const(ts.decoder),
        discriminator=lambda **kwargs: nn.Identity(),
        phase_1_duration=0,
        gan_loss=None,
        valid_signal_crop=False,
        feature_matching_fun=None,
        num_skipped_features=0,
        audio_distance=nn.Identity,
        multiband_audio_distance=None,
        weights={},
        pqmf=_const(pqmf) if pqmf is not None else None,
        n_channels=int(_ts_attr(ts, "n_channels", 1)),
        input_mode=_ts_attr(ts, "input_mode", "pqmf"),
        output_mode=_ts_attr(ts, "output_mode", "pqmf"),
    )
    for k in ("latent_pca", "latent_mean", "fidelity"):
        getattr(model, k).copy_(getattr(ts, k))
    if is_streaming(ts):
        _log.warning("scripted model is a streaming export: wrapped modules are stateful (cached convs)")
    return model


# ---- checks ----------------------------------------------------------------- #

def effective_state_dict(model: nn.Module) -> dict:
    """state dict with weight norm folded, i.e. what export would script."""
    sd = dict(model.state_dict())
    for mname, mod, hook in _weight_norm_hooks(model):
        prefix = f"{mname}." if mname else ""
        g = sd.pop(prefix + hook.name + "_g")
        v = sd.pop(prefix + hook.name + "_v")
        sd[prefix + hook.name] = torch._weight_norm(v, g, hook.dim).detach()
    return sd


def compare_state_dicts(sd_a, sd_b, prefixes=("encoder.", "decoder.", "pqmf."), name_a="a", name_b="b") -> float:
    """max absolute difference between common floating tensors of two state dicts."""
    keys = [k for k in sd_a if k.startswith(prefixes) and k in sd_b and not _is_streaming_buffer(k)]
    max_err, worst = 0., None
    for k in keys:
        if sd_a[k].shape != sd_b[k].shape:
            _log.warning(f"[{name_a} vs {name_b}] shape mismatch {k}")
            continue
        if not sd_a[k].is_floating_point():
            continue
        err = (sd_a[k].float() - sd_b[k].float()).abs().max().item() if sd_a[k].numel() else 0.
        if err > max_err:
            max_err, worst = err, k
    only_a = [k for k in sd_a if k.startswith(prefixes) and k not in sd_b and not _is_streaming_buffer(k)]
    _log.info(f"[{name_a} vs {name_b}] {len(keys)} common tensors, max abs diff = {max_err:.3e} ({worst})")
    if only_a:
        _log.info(f"[{name_a} vs {name_b}] {len(only_a)} tensors only in {name_a}: {only_a[:5]}")
    return max_err


@torch.no_grad()
def functional_check(ts, model, n_samples=2**15):
    """compares deterministic encode / decode of the scripted and the uncompiled model.
    Returns (encoder max abs diff, decoder max abs diff), or None if not applicable."""
    from .model import _pqmf_encode
    if model.input_mode not in ("pqmf", "raw"):
        _log.info("functional check only implemented for pqmf/raw inputs")
        return None
    if is_streaming(ts):
        _log.info("scripted model is a streaming export: skipping functional check (cached convs add delay)")
        return None
    model.eval()
    x = torch.randn(1, model.n_channels, n_samples) * .1
    x_mb = _pqmf_encode(model.pqmf, x) if model.input_mode == "pqmf" else x
    z_ts = ts.encoder(x_mb.clone())
    z_inv = model.encoder(x_mb.clone())
    err_z = (z_ts - z_inv).abs().max().item()
    mean = z_inv.chunk(2, 1)[0] if isinstance(model.encoder, blocks.VariationalEncoder) else z_inv
    y_ts = ts.decoder(mean.clone())
    y_inv = model.decoder(mean.clone())
    err_y = (y_ts - y_inv).abs().max().item()
    _log.info(f"[functional] encoder max abs diff = {err_z:.3e} | decoder max abs diff = {err_y:.3e}"
              f" (|y| max = {y_ts.abs().max().item():.3e})")
    return err_z, err_y


# ---- main entry points ------------------------------------------------------ #

def get_embedded_config(ts) -> Optional[str]:
    """gin config stored in the scripted model at export time (None for older exports)."""
    for attr in ("gin_config_str", "gin_operative_config_str"):
        config_str = _ts_attr(ts, attr)
        if config_str:
            return config_str
    return None


def _resolve_config(ts, model_path, config, mode):
    """returns (config source, config files, config string), or None if the config
    has to be inferred."""
    from . import core
    if config:
        return "file", list(config) if isinstance(config, (list, tuple)) else [config], None
    if mode in ("auto", "embedded"):
        embedded = get_embedded_config(ts)
        if embedded:
            return "embedded", [], embedded
        if mode == "embedded":
            raise RuntimeError("scripted model has no embedded gin configuration")
        _log.info("no gin config embedded in the scripted model (older export)")
        return None
    if mode == "file" and model_path is not None:
        found = core.search_for_config(str(model_path))
        if found is not None and os.path.exists(found):
            return "file", [found], None
    if mode == "file":
        raise FileNotFoundError(f"no config.gin found next to {model_path}")
    return None


def uncompile(model, out_dir=None, config=None, overrides=(), mode="auto",
              check=False, ckpt_name="last.ckpt"):
    """Converts a scripted RAVE back into a `rave.RAVE` object.

    Args:
        model: path to a .ts file, or an already loaded scripted model.
        out_dir: if given, writes {out_dir}/config.gin, {out_dir}/{ckpt_name} and
            {out_dir}/metadata.json; the folder is loadable with rave.load_rave_checkpoint.
        config: gin config file(s) to rebuild the model with (overrides the embedded one).
        overrides: additional gin bindings.
        mode: "auto" (embedded config if any, else inferred), "embedded" (config stored
            in the .ts), "file" (config.gin next to the .ts), "infer" (from the
            architecture) or "shell" (wrap scripted modules).
        check: run weight round-trip and functional checks (stored in metadata).
        ckpt_name: checkpoint file name.

    Returns:
        (model, config_str, metadata). config_str is None in shell mode.
    """
    if mode not in UNCOMPILE_MODES:
        raise ValueError(f"mode must be one of {UNCOMPILE_MODES}, got {mode}")
    if isinstance(model, torch.jit.ScriptModule):
        ts, model_path = model, None
    else:
        model_path = Path(model).resolve()
        ts = torch.jit.load(str(model_path), map_location="cpu")
    ts = unwrap_scripted(ts)
    ts.eval()

    from .version import __version__
    metadata = {
        "source": str(model_path) if model_path is not None else None,
        "rave_version": __version__,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        **get_scripted_metadata(ts),
        "config_source": None,
        "config_files": None,
        "architecture_distance": None,
    }

    if mode == "shell":
        rave_model, config_str = build_rave_shell(ts), None
        metadata["config_source"] = "shell"
    else:
        resolved = None if mode == "infer" else _resolve_config(ts, model_path, config, mode)
        if resolved is not None:
            source, files, embedded_str = resolved
            _log.info(f"using {source} config" + (f" {files}" if files else ""))
            rave_model, config_str = build_rave_from_config(files, overrides, config_str=embedded_str)
            if embedded_str is not None and not overrides:
                config_str = embedded_str  # keep the exported config as is
            metadata.update(config_source=source, config_files=files or None)
        else:
            if mode == "auto":
                _log.warning("no config found for the scripted model, inferring it from the architecture")
            try:
                rave_model, config_str, info = infer_config(ts, overrides)
                metadata.update(config_source="inferred", config_files=info["candidate"],
                                config_bindings=info["bindings"])
            except Exception as e:
                if mode == "infer":
                    raise
                _log.error(f"config inference failed ({e}), falling back to shell mode")
                rave_model, config_str = build_rave_shell(ts), None
                metadata["config_source"] = "shell"

        if config_str is not None:
            # sanity checks against what the scripted model reports
            for attr in ("n_channels", "input_mode", "output_mode"):
                a, b = _ts_attr(ts, attr), getattr(rave_model, attr)
                if a is not None and a != b:
                    _log.warning(f"{attr} differs: scripted={a}, config={b}")
            if metadata["model_sr"] is not None and metadata["model_sr"] != rave_model.sr:
                _log.warning(f"sampling rate differs: scripted={metadata['model_sr']}, config={rave_model.sr}")
            rebuilt_causal = is_causal(rave_model)
            if rebuilt_causal != metadata["causal"]:
                _log.warning(f"causality differs: scripted={metadata['causal']}, rebuilt={rebuilt_causal}; "
                             "keeping the one of the rebuilt model")
                metadata["causal"] = rebuilt_causal
            transfer_report = {}
            rave_model = transfer_weights(ts, rave_model, report=transfer_report)
            metadata.update(transfer_report)
            metadata["architecture_distance"] = signature_distance(
                architecture_signature(ts), architecture_signature(rave_model))[0]

    rave_model.eval()

    if check and config_str is not None:
        metadata["checks"] = {
            "max_weight_diff": compare_state_dicts(effective_state_dict(rave_model), ts.state_dict(),
                                                   name_a="uncompiled(folded)", name_b="scripted"),
            "functional": functional_check(ts, rave_model),
        }

    metadata = _jsonable(metadata)
    if out_dir is not None:
        save_uncompiled(rave_model, out_dir, config_str, metadata, ckpt_name=ckpt_name)
    return rave_model, config_str, metadata


def save_uncompiled(model, out_dir, config_str=None, metadata=None, ckpt_name="last.ckpt"):
    """writes {out_dir}/config.gin, {out_dir}/{ckpt_name} and {out_dir}/metadata.json,
    loadable with rave.load_rave_checkpoint(out_dir)."""
    if config_str is None:
        raise ValueError("no gin config available (shell mode): cannot write a loadable checkpoint")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "callbacks": {},
                "uncompile_metadata": metadata or {}}, out_dir / ckpt_name)
    (out_dir / "config.gin").write_text(config_str)
    if metadata is not None:
        (out_dir / UNCOMPILE_METADATA_FILE).write_text(json.dumps(_jsonable(metadata), indent=2))
    _log.info(f"uncompiled model written to {out_dir} ({ckpt_name}, config.gin, {UNCOMPILE_METADATA_FILE})")
    return out_dir


def load_uncompile_metadata(path) -> dict:
    """reads the metadata written by `uncompile` (from its folder, metadata.json or checkpoint)."""
    path = Path(path)
    if path.is_dir():
        path = path / UNCOMPILE_METADATA_FILE
    if path.suffix == ".ckpt":
        return torch.load(path, map_location="cpu", weights_only=False).get("uncompile_metadata", {})
    return json.loads(path.read_text())
