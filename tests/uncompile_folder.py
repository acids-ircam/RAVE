"""
Batch uncompile check: uncompiles every scripted model (.ts) of a folder, reloads the
written checkpoint, and reports which ones succeeded / failed.

Every model is uncompiled in its own temporary folder (under --tmp, default: system
temp dir), deleted as soon as the model has been checked: nothing stays on disk but
the report.

Models are processed in parallel with processes, not threads (gin's configuration is
global, so concurrent uncompiles in one process would interfere). Each worker handles a
single model and then exits, so that no gin / torch state leaks between models.

Memory: a new worker is only started when at least --min-free-gb of RAM is available,
and a worker whose memory exceeds --max-worker-gb is killed (the model is then reported
as an error), so that the machine is never saturated.

Usage:
    python tests/uncompile_folder.py ~/Dropbox/Models/RAVE [-j 2] [-r] [--report report.json]
        [--mode auto] [--tmp /some/tmp] [--no-reload] [--check]
        [--min-free-gb 3] [--max-worker-gb 4]

Status per model:
    ok        exact architecture match, checkpoint reloads with the same architecture / weights
    mismatch  uncompiled, but structural differences remain (architecture_distance > 0)
    shell     no config could be recovered (inference-only shell model, nothing written)
    error     exception (see report for the traceback)
    skipped   empty file (e.g. Dropbox online-only placeholder)
"""
import argparse
import csv
import io
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings
import queue as queue_lib
from pathlib import Path

import psutil
from tqdm import tqdm


def _check_model(path: str, mode: str, tmp_dir: str, reload: bool, check: bool) -> dict:
    """uncompiles a single model in a temporary folder (worker process)."""
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)

    # capture warnings / errors of this model only
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.WARNING)

    result = {"model": path, "status": "error", "config_source": None, "config_files": None,
              "architecture_distance": None, "reload_distance": None, "reload_max_diff": None,
              "max_weight_diff": None, "time": None, "error": None, "log": None}
    t0 = time.time()
    if os.path.getsize(path) == 0:
        result.update(status="skipped", error="empty file (Dropbox online-only placeholder?)", time=0.)
        return result
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        import rave
        from rave.uncompiling import uncompile, architecture_signature, signature_distance, unwrap_scripted
        model, config_str, metadata = uncompile(path, mode=mode, check=check)
        result.update(config_source=metadata.get("config_source"),
                      config_files=metadata.get("config_files"),
                      architecture_distance=metadata.get("architecture_distance"))
        checks = metadata.get("checks")
        if check and isinstance(checks, dict):
            result["max_weight_diff"] = checks.get("max_weight_diff")

        if config_str is None:
            result["status"] = "shell"
        else:
            if reload:
                from rave.uncompiling import save_uncompiled
                save_uncompiled(model, tmp_dir, config_str, metadata)
                reloaded, _ = rave.load_rave_checkpoint(tmp_dir)
                ts = unwrap_scripted(torch.jit.load(path, map_location="cpu"))
                result["reload_distance"] = signature_distance(architecture_signature(ts),
                                                               architecture_signature(reloaded))[0]
                sd_a, sd_b = model.state_dict(), reloaded.state_dict()
                diffs = [(sd_a[k].float() - sd_b[k].float()).abs().max().item()
                         for k in sd_a if k in sd_b and sd_a[k].shape == sd_b[k].shape
                         and sd_a[k].numel() and sd_a[k].is_floating_point()]
                result["reload_max_diff"] = max(diffs) if diffs else None
                del reloaded, ts
            exact = result["architecture_distance"] == 0 and \
                (not reload or (result["reload_distance"] == 0 and (result["reload_max_diff"] or 0.) == 0.))
            result["status"] = "ok" if exact else "mismatch"
    except BaseException as e:  # noqa: also catches SystemExit raised by absl / gin
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        result["time"] = round(time.time() - t0, 1)
        result["log"] = log_stream.getvalue().strip() or None
    return result


def _worker(results_queue, path, mode, tmp_dir, reload, check):
    try:
        res = _check_model(path, mode, tmp_dir, reload, check)
    except BaseException as e:
        res = {"model": path, "status": "error", "error": f"{type(e).__name__}: {e}"}
    results_queue.put(res)


def _tree_rss(proc: psutil.Process) -> int:
    """resident memory of a process and its children, in bytes."""
    total = 0
    try:
        for p in [proc] + proc.children(recursive=True):
            try:
                total += p.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=str, help="folder containing .ts models")
    parser.add_argument("-r", "--recursive", action="store_true", help="search sub-folders too")
    parser.add_argument("-j", "--jobs", type=int, default=2, help="maximum number of parallel worker processes")
    parser.add_argument("--min-free-gb", type=float, default=3.,
                        help="only start a new model if at least this much RAM is available")
    parser.add_argument("--max-worker-gb", type=float, default=4.,
                        help="kill a worker (model reported as error) if its memory exceeds this")
    parser.add_argument("--mode", type=str, default="auto", help="uncompile mode (auto, embedded, file, infer, shell)")
    parser.add_argument("--tmp", type=str, default=None,
                        help="root of the temporary folders (default: system temp dir)")
    parser.add_argument("--report", type=str, default=None,
                        help="report path (.json, a .csv is written next to it). Default: no file, stdout only")
    parser.add_argument("--no-reload", action="store_true",
                        help="do not write / reload the checkpoint (faster, nothing written at all)")
    parser.add_argument("--check", action="store_true", help="also run uncompile round-trip checks")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    models = sorted(folder.rglob("*.ts") if args.recursive else folder.glob("*.ts"))
    if not models:
        print(f"no .ts model found in {folder}")
        return 1
    if args.tmp is not None:
        Path(args.tmp).mkdir(parents=True, exist_ok=True)
    print(f"uncompiling {len(models)} models from {folder} with {args.jobs} workers")

    results, counts = [], {}
    tmp_root = tempfile.mkdtemp(prefix="rave_uncompile_", dir=args.tmp)
    # spawn: clean gin / torch state for every model (one process per model)
    ctx = mp.get_context("spawn")
    results_queue = ctx.Queue()
    pending = [str(m) for m in models]
    running = {}  # model path -> (process, tmp dir, psutil handle)
    min_free, max_worker = args.min_free_gb * 2 ** 30, args.max_worker_gb * 2 ** 30

    def collect(res, bar):
        if res["model"] not in running:  # late result of a killed worker
            return
        proc, tmp_dir, _ = running.pop(res["model"])
        proc.join(timeout=10)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        results.append(res)
        counts[res["status"]] = counts.get(res["status"], 0) + 1
        extra = res.get("error") or (f"config={res.get('config_files') or res.get('config_source')}, "
                                     f"distance={res.get('architecture_distance')}")
        bar.write(f"{res['status']:<8} {Path(res['model']).name} ({res.get('time')}s) {extra}")
        bar.set_postfix({k: counts.get(k, 0) for k in ("ok", "mismatch", "shell", "error", "skipped")}, refresh=False)
        bar.update(1)

    try:
        with tqdm(total=len(models), unit="model", dynamic_ncols=True) as bar:
            while pending or running:
                # start new workers while there is room (always at least one running)
                while pending and len(running) < args.jobs and \
                        (not running or psutil.virtual_memory().available >= min_free):
                    path = pending.pop(0)
                    tmp_dir = os.path.join(tmp_root, f"{len(results) + len(running):04d}_{Path(path).stem}")
                    proc = ctx.Process(target=_worker, daemon=True,
                                       args=(results_queue, path, args.mode, tmp_dir, not args.no_reload, args.check))
                    proc.start()
                    running[path] = (proc, tmp_dir, psutil.Process(proc.pid))
                # results
                try:
                    while True:
                        collect(results_queue.get(timeout=.5), bar)
                except queue_lib.Empty:
                    pass
                # memory watchdog / crashed workers
                for path, (proc, tmp_dir, ps) in list(running.items()):
                    if path not in running:
                        continue
                    if proc.is_alive():
                        rss = _tree_rss(ps)
                        if rss > max_worker:
                            proc.kill()
                            collect({"model": path, "status": "error", "error":
                                     f"killed: worker memory {rss / 2 ** 30:.1f} GB > --max-worker-gb "
                                     f"{args.max_worker_gb}"}, bar)
                    else:
                        try:  # result may have been put right before exiting
                            collect(results_queue.get(timeout=1), bar)
                        except queue_lib.Empty:
                            collect({"model": path, "status": "error",
                                     "error": f"worker crashed (exit code {proc.exitcode})"}, bar)
    finally:
        for proc, _, _ in running.values():
            proc.kill()
        shutil.rmtree(tmp_root, ignore_errors=True)

    results.sort(key=lambda r: r["model"])
    print("\nsummary: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    for status in ("error", "mismatch", "shell", "skipped"):
        failed = [r for r in results if r["status"] == status]
        if failed:
            print(f"\n{status}:")
            for r in failed:
                print(f"  {Path(r['model']).name}: {r.get('error') or 'distance=' + str(r.get('architecture_distance'))}")

    if args.report:
        report = Path(args.report).expanduser()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.with_suffix(".json").write_text(json.dumps(results, indent=2, default=str))
        fields = ["model", "status", "config_source", "config_files", "architecture_distance",
                  "reload_distance", "reload_max_diff", "max_weight_diff", "time", "error"]
        with open(report.with_suffix(".csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nreport written to {report.with_suffix('.json')} / {report.with_suffix('.csv')}")
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
