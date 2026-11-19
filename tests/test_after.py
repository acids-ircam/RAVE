"""Tests of `rave train_after` (AFTER diffusion models on a RAVE latent space).

    * find_after_model (checkpoint resolution of --resume) and check_rave_model (--model must be
      a RAVE run / checkpoint / export): always run, without AFTER;
    * end-to-end runs of `rave train_after`, in subprocesses, on a small untrained RAVE and a
      synthetic pre-processed dataset:
        - --smoke_test: RAVE export, dataset encoding with augmentations, AFTER init;
        - a short training (--save_every / --val_every), then a resume from its checkpoint;
      they require a version of AFTER compatible with train_after (the `rave` branch, installed
      by `pip install "acids-rave[after]"`), and are skipped otherwise.
    * --smoke_test runs on your own models, if AFTER_DB_PATH and AFTER_CKPT_PATH / AFTER_TS_PATH
      (comma-separated) are given.

Environment variables:
    RAVE_TEST_AFTER_PATH   folder prepended to PYTHONPATH of the train_after runs, e.g. a clone of
                           the `rave` branch of AFTER (git clone -b rave https://github.com/acids-ircam/AFTER)
                           when the installed AFTER is another version
    RAVE_TEST_AFTER_DEVICE device of the train_after runs (default: cpu)
    AFTER_DB_PATH          pre-processed dataset for the smoke tests on your models
    AFTER_CKPT_PATH        RAVE runs / checkpoints (comma-separated)
    AFTER_TS_PATH          exported RAVE models (comma-separated)
"""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TESTS_DIR = Path(__file__).parent
ROOT = TESTS_DIR.parent
AFTER_PATH = os.environ.get("RAVE_TEST_AFTER_PATH")
DEVICE = os.environ.get("RAVE_TEST_AFTER_DEVICE", "cpu")
SR = 44100
NUM_SIGNAL = 2**17                  # preprocessing chunk (samples)
LATENT_SIZE = 4


# --------------------------------------------------------------------------- #
#  find_after_model (loaded from the source: importing train_after exits without AFTER)
# --------------------------------------------------------------------------- #

_source = ast.parse((ROOT / "rave/scripts/train_after.py").read_text())
_nodes = [n for n in _source.body
          if (isinstance(n, ast.FunctionDef) and n.name in ("find_after_model", "check_rave_model"))
          or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "after_regexp" for t in n.targets))]
import rave  # noqa: E402

_ns = {"os": os, "re": re, "torch": torch, "Path": Path, "rave": rave}
exec(compile(ast.Module(body=_nodes, type_ignores=[]), "rave/scripts/train_after.py", "exec"), _ns)
find_after_model = _ns["find_after_model"]
check_rave_model = _ns["check_rave_model"]


def _touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"")


def test_find_after_model_latest_step_in_folder(tmp_path):
    # numerical (not lexicographic) order; non-checkpoint files are ignored
    _touch(tmp_path, "checkpoint9_EMA.pt", "checkpoint100000_EMA.pt", "checkpoint20000_EMA.pt",
           "config.gin", "other.pt")
    name, folder, step = find_after_model(str(tmp_path))
    assert (name, folder, step) == ("checkpoint100000_EMA.pt", str(tmp_path), 100000)


def test_find_after_model_checkpoint_file(tmp_path):
    _touch(tmp_path, "checkpoint200_EMA.pt", "checkpoint300_EMA.pt")
    name, folder, step = find_after_model(str(tmp_path / "checkpoint200_EMA.pt"))
    assert (name, step) == ("checkpoint200_EMA.pt", 200)
    assert Path(folder) == tmp_path


def test_find_after_model_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_after_model(str(tmp_path / "missing"))
    _touch(tmp_path / "empty", "config.gin")
    with pytest.raises(FileNotFoundError):
        find_after_model(str(tmp_path / "empty"))
    _touch(tmp_path, "model.pt")
    with pytest.raises(AssertionError):
        find_after_model(str(tmp_path / "model.pt"))


# --------------------------------------------------------------------------- #
#  --model: RAVE models only
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def rave_models(tmp_path_factory):
    """untrained RAVE run, its export, and a scripted model that is not a RAVE."""
    sys.path.insert(0, str(TESTS_DIR))
    from test_dimred_export import _untrained_run
    from rave.scripting import export
    tmp = tmp_path_factory.mktemp("rave_models")
    run = _untrained_run(tmp / "run")
    ts = export(run, out_path=tmp / "export", latent_size=LATENT_SIZE)

    class Codec(torch.nn.Module):
        def forward(self, x):
            return x

        @torch.jit.export
        def encode(self, x):
            return x

        @torch.jit.export
        def decode(self, z):
            return z

    other = tmp / "other_codec.ts"
    torch.jit.script(Codec()).save(str(other))
    return {"run": run, "ckpt": run / "last.ckpt", "ts": Path(ts), "other_ts": other, "tmp": tmp}


@pytest.mark.parametrize("kind", ["run", "ckpt", "ts"])
def test_check_rave_model_accepts_rave(rave_models, kind):
    check_rave_model(str(rave_models[kind]))


def test_is_scripted_rave(rave_models):
    from rave.uncompiling import is_scripted_rave
    assert is_scripted_rave(torch.jit.load(str(rave_models["ts"])))
    assert not is_scripted_rave(torch.jit.load(str(rave_models["other_ts"])))
    assert not is_scripted_rave(torch.nn.Linear(2, 2))


def test_check_rave_model_rejects_other_models(rave_models, tmp_path):
    with pytest.raises(ValueError, match="not a RAVE export"):
        check_rave_model(str(rave_models["other_ts"]))
    (tmp_path / "broken.ts").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="could not load"):
        check_rave_model(str(tmp_path / "broken.ts"))
    (tmp_path / "weights.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="neither a RAVE run"):
        check_rave_model(str(tmp_path / "weights.pt"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no RAVE checkpoint"):
        check_rave_model(str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        check_rave_model(str(tmp_path / "missing"))
    # a checkpoint whose config is not a RAVE configuration
    other_run = tmp_path / "other_run"
    other_run.mkdir()
    (other_run / "last.ckpt").write_bytes(b"")
    (other_run / "config.gin").write_text("Autoencoder.latent_size = 16\n")
    with pytest.raises(ValueError, match="no RAVE configuration"):
        check_rave_model(str(other_run))


# --------------------------------------------------------------------------- #
#  end-to-end runs of rave train_after
# --------------------------------------------------------------------------- #

def _env():
    env = dict(os.environ)
    if AFTER_PATH:
        env["PYTHONPATH"] = os.pathsep.join([AFTER_PATH, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def _after_compatible() -> bool:
    """AFTER is importable with the API used by train_after (in a subprocess, with _env())."""
    code = "from after.diffusion import RectifiedFlow, EDM; from after.diffusion.utils import collate_fn"
    return subprocess.run([sys.executable, "-c", code], env=_env(), cwd=TESTS_DIR,
                          capture_output=True).returncode == 0


HAS_AFTER = _after_compatible()
needs_after = pytest.mark.skipif(not HAS_AFTER, reason="no AFTER version compatible with train_after "
                                 "(pip install \"acids-rave[after]\", or set RAVE_TEST_AFTER_PATH)")


def _rave(args, cwd, timeout=3600):
    """runs `rave <args>` in a subprocess; returns (returncode, output)."""
    cmd = [sys.executable, "-c", "import sys; from rave.scripts.main_cli import main; sys.exit(main())", *args]
    proc = subprocess.run(cmd, cwd=cwd, env=_env(), capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout + proc.stderr


def _check(rc, out):
    assert rc == 0, out[-5000:]
    return out


def _write_audio(folder: Path, n_files=6, seconds=24, seed=0):
    from scipy.io import wavfile
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    for i in range(n_files):
        f0 = rng.uniform(80, 800)
        x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))
        x *= 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.2, 2) * t)        # amplitude envelope
        x += 0.05 * rng.standard_normal(len(t))
        wavfile.write(folder / f"file_{i}.wav", SR, (0.8 * x / np.abs(x).max()).astype(np.float32))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """untrained RAVE run + pre-processed synthetic dataset (~100 chunks: the 2% validation
    split must hold at least one batch)."""
    if not HAS_AFTER:
        pytest.skip("AFTER not available")
    sys.path.insert(0, str(TESTS_DIR))
    from test_dimred_export import _untrained_run
    tmp = tmp_path_factory.mktemp("after")
    run = _untrained_run(tmp / "rave_run")
    _write_audio(tmp / "audio", n_files=10, seconds=40)
    _check(*_rave(["preprocess", "--input_path", str(tmp / "audio"), "--output_path", str(tmp / "db"),
                   "--num_signal", str(NUM_SIGNAL), "--channels", "1", "--sampling_rate", str(SR)], cwd=tmp))
    return {"tmp": tmp, "run": run, "db": tmp / "db"}


def _train_after_args(name, model, db, out, *extra):
    return ["train_after", "--name", name, "--model", str(model), "--db_path", str(db), "--out_path", str(out),
            "--device", DEVICE, "--batch", "2", "--workers", "0", "--n_augmentations", "1", *extra]


@needs_after
def test_smoke(setup):
    import acids_dataset as ad
    tmp = setup["tmp"]
    out = _check(*_rave(_train_after_args("smoke", setup["run"], setup["db"], tmp / "runs",
                                          "--latent_size", str(LATENT_SIZE), "--smoke_test"), cwd=tmp))
    assert "smoke_test succeeded" in out
    run_dir = tmp / "runs" / "smoke"
    # RAVE run exported (non-streaming) into the AFTER run, with the requested latent size
    exported = sorted((run_dir / "rave").glob("*.ts"))
    assert len(exported) == 1, exported
    ts = torch.jit.load(str(exported[0]))
    assert int(ts.encode_params[2]) == LATENT_SIZE
    assert "streaming" not in exported[0].name
    # latents (+ augmentations) stored in the dataset, under the embedding name
    embedding_name = exported[0].stem
    assert embedding_name in ad.get_feature_names_from_path(str(setup["db"]))
    # AFTER configured with the RAVE latent size and sample rate
    config = next((tmp / "logs").glob("log_config_*.txt")).read_text()
    assert re.search(rf"IN_SIZE\s*=\s*{LATENT_SIZE}\b", config), config
    assert re.search(rf"\bSR\s*=\s*{ts.sr}\b", config), config


@needs_after
def test_smoke_reuses_dataset_latents(setup):
    """a second run with the same embedding does not need --force_augmentation."""
    tmp = setup["tmp"]
    _check(*_rave(_train_after_args("smoke_again", setup["run"], setup["db"], tmp / "runs",
                                    "--latent_size", str(LATENT_SIZE), "--smoke_test"), cwd=tmp))


@needs_after
def test_smoke_rejects_other_augmentation_count(setup):
    tmp = setup["tmp"]
    args = _train_after_args("smoke_aug", setup["run"], setup["db"], tmp / "runs",
                             "--latent_size", str(LATENT_SIZE), "--smoke_test")
    args[args.index("--n_augmentations") + 1] = "2"
    rc, out = _rave(args, cwd=tmp)
    assert rc != 0 and "augmentations" in out, out[-3000:]


@needs_after
def test_rejects_non_rave_model(setup, rave_models):
    """other codecs are refused before any dataset encoding."""
    tmp = setup["tmp"]
    rc, out = _rave(_train_after_args("other", rave_models["other_ts"], setup["db"], tmp / "runs", "--smoke_test"),
                    cwd=tmp)
    assert rc != 0 and "only works on RAVE models" in out, out[-3000:]
    assert not (tmp / "runs" / "other" / "rave").exists()


@needs_after
def test_requires_name(setup):
    tmp = setup["tmp"]
    args = _train_after_args("x", setup["run"], setup["db"], tmp / "runs", "--smoke_test")
    del args[args.index("--name"):args.index("--name") + 2]
    rc, out = _rave(args, cwd=tmp)
    assert rc != 0 and "--name is required" in out, out[-3000:]


@needs_after
def test_train_and_resume(setup):
    """one epoch of training (tiny config) with checkpoints and validation, then a resume from
    the run folder, reusing the RAVE model exported at the first launch."""
    tmp = setup["tmp"]
    run_dir = tmp / "runs" / "train"
    common = ["--config", "tiny", "--override", "MAX_STEPS=1", "--save_every", "10", "--val_every", "10"]
    _check(*_rave(_train_after_args("train", setup["run"], setup["db"], tmp / "runs",
                                    "--latent_size", str(LATENT_SIZE), *common), cwd=tmp))
    ckpts = sorted(run_dir.glob("checkpoint*_EMA.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    assert ckpts, list(run_dir.iterdir())
    assert (run_dir / "config.gin").exists()
    assert any((run_dir / "logs").iterdir())
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    assert "model_state" in state and not any("emb_model" in k for k in state["model_state"])
    last_step = int(re.findall(r"\d+", ckpts[-1].name)[0])

    exported = sorted((run_dir / "rave").glob("*.ts"))
    out = _check(*_rave(_train_after_args("train", setup["run"], setup["db"], tmp / "runs",
                                          "--resume", str(run_dir), *common), cwd=tmp))
    assert re.search(rf"Restarting from step\s+{last_step + 1}\b", out), out[-3000:]
    assert "Reusing exported RAVE model" in out
    assert sorted((run_dir / "rave").glob("*.ts")) == exported


# --------------------------------------------------------------------------- #
#  smoke tests on your own models
# --------------------------------------------------------------------------- #

USER_DB = os.environ.get("AFTER_DB_PATH")
USER_MODELS = [m for v in ("AFTER_CKPT_PATH", "AFTER_TS_PATH") for m in (os.environ.get(v) or "").split(",") if m]


@needs_after
@pytest.mark.skipif(not (USER_DB and USER_MODELS), reason="AFTER_DB_PATH and AFTER_CKPT_PATH / AFTER_TS_PATH not set")
@pytest.mark.parametrize("model", USER_MODELS or [None])
def test_smoke_user_models(model, tmp_path):
    out = _check(*_rave(_train_after_args(f"smoke_{Path(model).stem}", model, USER_DB, tmp_path / "runs",
                                          "--smoke_test"), cwd=tmp_path))
    assert "smoke_test succeeded" in out
