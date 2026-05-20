"""GPU memory curve benchmark.

For every model under benchmark/configs/online + benchmark/configs/end2end,
sweep multiple batch_size values on a fixed 7scenes dense scene and record:
  - peak GPU memory (MiB, torch.cuda.max_memory_allocated)
  - inference time (s)
  - status: ok | oom | oom_skipped | timeout | timeout_skipped | error

No metrics are computed.

Scheduling strategy:
  - driver mode: multi-GPU pool scheduling. At most 1 worker per GPU at any time.
  - worker mode: full single-model pipeline, sweeping all N serially, flushing JSON after each N.
  - The hard 30 min timeout per inference is enforced by the driver via heartbeat monitoring + SIGKILL of the process group.

Usage:
  # driver
  python benchmark/evaluation/bench_memory_curve.py --gpus 4,5,6,7 --outdir results/memory_curve_bf16
  # worker (normally launched by the driver, not invoked manually)
  python benchmark/evaluation/bench_memory_curve.py --worker --config X.yaml --out-json out.json
"""
import argparse
import gc
import glob
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from copy import copy

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ---- Shared constants ----
# Default (main) run: 7scenes chess seq-05/0 dense, 1000 frames
# 13 buckets: small-N segment (<=500) densified, mid segment 800/1000, large-N segment 1250-2000 for memcurve
BATCH_SIZES = [10, 50, 100, 200, 300, 400, 500, 800, 1000, 1250, 1500, 1750, 2000]
FIXTURE_SCENE_ID = "7scenes_chess_seq-05_0_dense"
FIXTURE_SOURCE = "7scenes"
FIXTURE_PATH = "chess_seq-05/0"
FIXTURE_NUM_FRAMES = 1000
FIXTURE_TAGS = {"environment": "indoor", "dynamics": "static",
                "view_type": "normal", "data_type": "real",
                "view_density": "dense"}

# Extended (--extend-from) run: use TUM rgbd_dataset_freiburg3_long_office_household
# (2485 frames, taking the first 2000); only continue appending larger N to models
# that were still ok at N=1000 in the main run, until OOM/timeout.
# TUM is chosen because it is also an indoor hand-held RGBD dataset like 7scenes,
# original 640x480, with reader default resolution 518x392 -> per-frame GPU memory
# footprint is identical, so the model memory curves can be stitched together.
EXTENDED_BATCH_SIZES = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
EXTENDED_FIXTURE_SCENE_ID = "tum_freiburg3_long_office_household_ext"
EXTENDED_FIXTURE_SOURCE = "tum"
EXTENDED_FIXTURE_PATH = "rgbd_dataset_freiburg3_long_office_household/0"
EXTENDED_FIXTURE_NUM_FRAMES = 2000
EXTENDED_FIXTURE_TAGS = {"environment": "indoor", "dynamics": "static",
                         "view_type": "normal", "data_type": "real",
                         "view_density": "dense"}

TIMEOUT_S = 30 * 60
END2END_DIR = "benchmark/configs/end2end"
ONLINE_DIR = "benchmark/configs/online"
STD_KEYS = {
    # "Standard" YAML keys handled by run_benchmark.py; do not forward to adapter.configure
    "model", "checkpoint", "device", "checkpoints_dir",
    "scene_index", "tags", "max_scenes",
    "eval_metrics", "conf_threshold", "depth_alignment", "resolution",
    "shuffle_seed",
    "output_dir", "output", "visualize", "vis_conf_percent",
    "resolution_override",  # handled separately inside the worker
}


# ====================================================================
# Worker mode
# ====================================================================

def _import_all_adapters():
    """Import your registered adapters here, e.g.:
        import benchmark.evaluation.model_adapters.my_model_adapter  # noqa
    """
    import benchmark.evaluation.model_adapters.vggt_adapter  # noqa: F401
    import benchmark.evaluation.model_adapters.da3_adapter  # noqa: F401


def _make_temp_scene_index(out_path, fixture):
    """Build a temporary scene_index.json that contains only the fixture scene.

    fixture: dict with keys {scene_id, source, path, num_frames, tags,
                             [optional] frame_indices}.
    If fixture provides frame_indices, use them directly (for datasets like
    ropedia that need explicit raw frame indices); otherwise fall back to
    range(num_frames).
    """
    if "frame_indices" in fixture and fixture["frame_indices"] is not None:
        frame_indices = [int(i) for i in fixture["frame_indices"]]
    else:
        frame_indices = list(range(int(fixture["num_frames"])))
    scene = {
        "scene_id": fixture["scene_id"],
        "source_dataset": fixture["source"],
        "scene_path": fixture["path"],
        "tags": fixture.get("tags", {}),
        "frame_indices": frame_indices,
        "num_frames_total": int(fixture["num_frames"]),
    }
    with open(out_path, "w") as f:
        json.dump([scene], f)


def _default_fixture():
    return {"scene_id": FIXTURE_SCENE_ID, "source": FIXTURE_SOURCE,
            "path": FIXTURE_PATH, "num_frames": FIXTURE_NUM_FRAMES,
            "tags": FIXTURE_TAGS}


def _extended_fixture():
    return {"scene_id": EXTENDED_FIXTURE_SCENE_ID, "source": EXTENDED_FIXTURE_SOURCE,
            "path": EXTENDED_FIXTURE_PATH, "num_frames": EXTENDED_FIXTURE_NUM_FRAMES,
            "tags": EXTENDED_FIXTURE_TAGS}


def _slice_scene(scene, N):
    """Slice the first N frames of the scene dict returned by BenchmarkDataset. Does not deep-copy image data."""
    out = copy(scene)  # shallow-copy the top-level dict
    out['images']        = scene['images'][:N]
    out['images_raw']    = scene['images_raw'][:N]
    out['depth']         = scene['depth'][:N]
    out['extrinsic']     = scene['extrinsic'][:N]
    out['intrinsic']     = scene['intrinsic'][:N]
    out['valid_mask']    = scene['valid_mask'][:N]
    out['world_points']  = scene['world_points'][:N]
    if scene.get('sky_mask') is not None:
        out['sky_mask']  = scene['sky_mask'][:N]
    out['frame_indices'] = list(scene['frame_indices'])[:N]
    return out


def _measure_one(adapter, scene_n, N, with_metrics=False):
    """Run one inference for a single (model, N) pair and return a dict.

    When with_metrics=True, after inference call run_benchmark.evaluate_scene()
    to compute depth / pose / trajectory metrics (skipping pointcloud / tgm).
    GPU memory and timing still only record the pure inference segment; metric
    computation is not counted toward peak_mem_MiB / time_s.

    Time semantics: time_s = duration of adapter.predict() (forward + post-process).
    adapter.prepare() (e.g. scal3r's PNG writing + load_data) is called before
    timing starts, so it is not counted in time_s and does not pollute peak_mem_MiB.
    """
    # --- Data preparation phase (NOT timed; peak mem is only recorded after reset) ---
    try:
        adapter.prepare(scene_n)
    except torch.cuda.OutOfMemoryError as e:
        return {
            "N": N, "status": "oom",
            "peak_mem_MiB": None, "time_s": None,
            "error": f"prepare: {str(e)[:240]}",
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {
                "N": N, "status": "oom",
                "peak_mem_MiB": None, "time_s": None,
                "error": f"prepare: {str(e)[:240]}",
            }
        return {
            "N": N, "status": "error",
            "peak_mem_MiB": None, "time_s": None,
            "error": f"prepare: {type(e).__name__}: {e}"[:300],
        }
    except Exception as e:
        return {
            "N": N, "status": "error",
            "peak_mem_MiB": None, "time_s": None,
            "error": f"prepare: {type(e).__name__}: {e}"[:300],
        }

    # --- Timed segment begins ---
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    status = "ok"
    err_msg = None
    predictions = None
    t0 = time.perf_counter()
    try:
        predictions = adapter.predict(scene_n)
    except torch.cuda.OutOfMemoryError as e:
        status = "oom"
        err_msg = str(e)[:300]
    except RuntimeError as e:
        # Older PyTorch raises OOM as RuntimeError("CUDA out of memory")
        if "out of memory" in str(e).lower():
            status = "oom"
        else:
            status = "error"
        err_msg = str(e)[:300]
    except Exception as e:
        status = "error"
        err_msg = f"{type(e).__name__}: {e}"[:300]

    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    elapsed = time.perf_counter() - t0
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2

    out = {
        "N": N,
        "status": status,
        "peak_mem_MiB": round(peak_mib, 1),
        "time_s": round(elapsed, 3),
        "error": err_msg,
    }

    if with_metrics and status == "ok" and predictions is not None:
        # Lazy import: metrics dependencies are only needed when --with-metrics is enabled.
        from benchmark.evaluation.run_benchmark import evaluate_scene
        metric_status = "ok"
        metric_err = None
        try:
            mres, _mtiming = evaluate_scene(
                scene_n, predictions, adapter,
                depth_alignment="median",
                eval_metrics=["depth", "pose", "trajectory"],
                gt_mesh_path=None,
            )
            for k in ("depth", "depth_metric", "camera", "trajectory"):
                if k in mres:
                    out[k] = mres[k]
        except torch.cuda.OutOfMemoryError as e:
            metric_status = "oom"
            metric_err = str(e)[:300]
        except Exception as e:
            metric_status = "error"
            metric_err = f"{type(e).__name__}: {e}"[:300]
        out["metric_status"] = metric_status
        if metric_err:
            out["metric_error"] = metric_err

    if predictions is not None:
        del predictions

    return out


def _kind_from_path(cfg_path):
    if "/end2end/" in cfg_path.replace("\\", "/"):
        return "end2end"
    if "/online/" in cfg_path.replace("\\", "/"):
        return "online"
    return "unknown"


def _flush(out_json, payload):
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_json)


def _load_existing_results(out_json):
    """Read an existing per-model JSON and return (results_dict, completed_flag).
    Returns ({}, False) if the file is missing or unreadable.
    """
    if not os.path.isfile(out_json):
        return {}, False
    try:
        with open(out_json, "r") as f:
            payload = json.load(f)
        results = payload.get("results", {}) or {}
        # Normalize keys to str (JSON keys are already strings)
        results = {str(k): v for k, v in results.items()}
        return results, bool(payload.get("completed", False))
    except Exception:
        return {}, False


def run_worker(args):
    print(f"[worker] config={args.config} out={args.out_json}", flush=True)

    # ---- Thread throttling: the driver passes cpus-per-worker via env var; here we
    # additionally use torch/cv2's own APIs as a fallback to keep OpenCV / PyTorch
    # subthreads from saturating the entire machine's CPUs (problem amplified when
    # running multiple GPUs in parallel). OMP/MKL/OPENBLAS/NUMEXPR are already set
    # in the child process env; this only patches the bits that must be set via API.
    _n_threads = int(os.environ.get("BENCH_WORKER_THREADS", "0") or 0)
    if _n_threads > 0:
        try:
            torch.set_num_threads(_n_threads)
            torch.set_num_interop_threads(max(1, _n_threads // 2))
        except Exception:
            pass
        try:
            import cv2
            cv2.setNumThreads(_n_threads)
        except Exception:
            pass
        print(f"[worker] cpu thread cap = {_n_threads} "
              f"(OMP/MKL/OPENBLAS/NUMEXPR via env, torch/cv2 via API)",
              flush=True)

    _import_all_adapters()
    from benchmark.datasets.benchmark_dataset import BenchmarkDataset
    from benchmark.evaluation.model_adapters import get_adapter

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    model_name = cfg.get("model")
    checkpoint = cfg.get("checkpoint")
    resolution_override = cfg.get("resolution_override")
    extra = {k: v for k, v in cfg.items() if k not in STD_KEYS}

    # ---- Resolve fixture / batch_sizes (supports both main run and extended run) ----
    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    else:
        batch_sizes = list(BATCH_SIZES)
    fixture = json.loads(args.fixture) if args.fixture else _default_fixture()

    out_json = args.out_json
    heartbeat_path = out_json.replace(".json", ".heartbeat")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    # Temporary scene_index (using this run's fixture)
    tmp_index = out_json.replace(".json", "_scene_index.json")
    _make_temp_scene_index(tmp_index, fixture)

    # ---- Resume: read existing results, skip any N that has already been run ----
    existing_results, _ = _load_existing_results(out_json)
    if existing_results:
        print(f"[worker] resume: {len(existing_results)} N already done -> "
              f"{sorted(int(k) for k in existing_results.keys())}", flush=True)

    payload = {
        "model": model_name,
        "config": args.config,
        "kind": _kind_from_path(args.config),
        "fixture_scene": fixture["scene_id"],
        "fixture_source": fixture["source"],
        "fixture_path": fixture["path"],
        "fixture_num_frames": fixture["num_frames"],
        "batch_sizes": batch_sizes,
        "weights_MiB": None,
        "results": dict(existing_results),
        "completed": False,
    }
    _flush(out_json, payload)

    # If every N has already been run -> exit early (the driver should have skipped this already; this is a fallback)
    if all(str(N) in existing_results for N in batch_sizes):
        print("[worker] all N already done, exiting.", flush=True)
        payload["completed"] = True
        _flush(out_json, payload)
        return

    # ---- Stage 1: Load the dataset (once, 1000 frames) ----
    print(f"[worker] loading dataset (resolution_override={resolution_override})", flush=True)
    try:
        dataset = BenchmarkDataset(
            scene_index_path=tmp_index,
            resolution_override=resolution_override,
        )
        scene_full = dataset[0]
        del dataset
    except Exception as e:
        traceback.print_exc()
        for N in BATCH_SIZES:
            payload["results"][str(N)] = {
                "N": N, "status": "error",
                "peak_mem_MiB": None, "time_s": None,
                "error": f"dataset_load: {type(e).__name__}: {e}"[:300],
            }
        payload["completed"] = True
        _flush(out_json, payload)
        return

    # ---- Stage 2: Load the model ----
    print(f"[worker] loading model '{model_name}' (ckpt={checkpoint})", flush=True)
    try:
        adapter = get_adapter(model_name)
        if extra:
            adapter.configure(**extra)
        adapter.load_model(checkpoint=checkpoint, device="cuda")
    except torch.cuda.OutOfMemoryError as e:
        for N in batch_sizes:
            payload["results"][str(N)] = {
                "N": N, "status": "oom_skipped",
                "peak_mem_MiB": None, "time_s": None,
                "error": f"load_model: {e}"[:300],
            }
        payload["completed"] = True
        _flush(out_json, payload)
        return
    except Exception as e:
        traceback.print_exc()
        for N in batch_sizes:
            payload["results"][str(N)] = {
                "N": N, "status": "error",
                "peak_mem_MiB": None, "time_s": None,
                "error": f"load_model: {type(e).__name__}: {e}"[:300],
            }
        payload["completed"] = True
        _flush(out_json, payload)
        return

    payload["weights_MiB"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
    print(f"[worker] weights_MiB={payload['weights_MiB']}", flush=True)
    _flush(out_json, payload)

    # ---- Stage 3: Sweep batch_sizes ----
    # Slow-N threshold: if some N has time_s >= this value, all subsequent larger N will inevitably time out, so skip them directly.
    SLOW_SKIP_THRESHOLD_S = TIMEOUT_S  # 1800s, matches the driver hard timeout

    # Resume state restoration:
    #   If some historical N is already oom -> mark all subsequent ones as oom_skipped
    #   If some historical N is already timeout / timeout_skipped or time_s >= threshold -> mark subsequent ones as timeout_skipped
    oom_seen = any(
        existing_results.get(str(N), {}).get("status") == "oom"
        for N in batch_sizes
    )

    def _is_slow(r):
        if r is None:
            return False
        if r.get("status") in ("timeout", "timeout_skipped"):
            return True
        ts = r.get("time_s")
        return ts is not None and ts >= SLOW_SKIP_THRESHOLD_S

    slow_seen = any(_is_slow(existing_results.get(str(N))) for N in batch_sizes)

    for N in batch_sizes:
        # Resume: keep already-completed N as-is and do not re-measure
        if str(N) in existing_results:
            r = existing_results[str(N)]
            print(f"[worker] N={N}  [SKIP-RESUME] status={r.get('status')}  "
                  f"peak={r.get('peak_mem_MiB')}MiB  time={r.get('time_s')}s",
                  flush=True)
            if r.get("status") == "oom":
                oom_seen = True
            if _is_slow(r):
                slow_seen = True
            continue

        if oom_seen:
            payload["results"][str(N)] = {
                "N": N, "status": "oom_skipped",
                "peak_mem_MiB": None, "time_s": None, "error": None,
            }
            _flush(out_json, payload)
            continue

        if slow_seen:
            payload["results"][str(N)] = {
                "N": N, "status": "timeout_skipped",
                "peak_mem_MiB": None, "time_s": None,
                "error": f"prev N time_s >= {SLOW_SKIP_THRESHOLD_S}s",
            }
            _flush(out_json, payload)
            print(f"[worker] N={N}  status=timeout_skipped  "
                  f"(prev N time >= {SLOW_SKIP_THRESHOLD_S}s)", flush=True)
            continue

        # heartbeat for driver-side timeout watchdog
        try:
            with open(heartbeat_path, "w") as f:
                f.write(f"{N} {time.time()}\n")
        except Exception:
            pass

        try:
            scene_n = _slice_scene(scene_full, N)
        except Exception as e:
            payload["results"][str(N)] = {
                "N": N, "status": "error",
                "peak_mem_MiB": None, "time_s": None,
                "error": f"slice: {type(e).__name__}: {e}"[:300],
            }
            _flush(out_json, payload)
            continue

        r = _measure_one(adapter, scene_n, N, with_metrics=args.with_metrics)
        payload["results"][str(N)] = r
        _flush(out_json, payload)

        del scene_n
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

        print(f"[worker] N={N}  status={r['status']}  "
              f"peak={r['peak_mem_MiB']}MiB  time={r['time_s']}s", flush=True)

        if r["status"] == "oom":
            oom_seen = True
        if _is_slow(r):
            slow_seen = True

    payload["completed"] = True
    _flush(out_json, payload)
    # Clean up the heartbeat / temporary scene_index
    for p in (heartbeat_path, tmp_index):
        try:
            if os.path.isfile(p):
                os.unlink(p)
        except Exception:
            pass
    print("[worker] done", flush=True)


# ====================================================================
# Driver mode
# ====================================================================

def _name_from_cfg(cfg_path):
    return os.path.basename(cfg_path).replace("_eval.yaml", "")


def _collect_configs(whitelist=None):
    """Return ordered list of config yaml paths.

    whitelist: if given (list of str), use these paths in this order and skip auto-discovery.
    """
    if whitelist:
        out = []
        for p in whitelist:
            p = p.strip()
            if not p:
                continue
            if not os.path.isfile(p):
                print(f"[driver] [WARN] config not found: {p}", file=sys.stderr)
                continue
            out.append(p)
        return out
    paths = []
    for d in (END2END_DIR, ONLINE_DIR):
        paths.extend(sorted(glob.glob(os.path.join(d, "*_eval.yaml"))))
    return paths


def _finalize_one(out_json, cfg_path, batch_sizes, fixture):
    """Called after the worker exits. Mark any unfilled N as timeout/error."""
    if not os.path.isfile(out_json):
        # Worker never even started
        payload = {
            "model": _name_from_cfg(cfg_path),
            "config": cfg_path,
            "kind": _kind_from_path(cfg_path),
            "fixture_scene": fixture["scene_id"],
            "fixture_source": fixture["source"],
            "fixture_path": fixture["path"],
            "fixture_num_frames": fixture["num_frames"],
            "batch_sizes": list(batch_sizes),
            "weights_MiB": None,
            "results": {str(N): {"N": N, "status": "error",
                                 "peak_mem_MiB": None, "time_s": None,
                                 "error": "worker_did_not_start"}
                        for N in batch_sizes},
            "completed": True,
        }
        _flush(out_json, payload)
        return

    try:
        with open(out_json, "r") as f:
            payload = json.load(f)
    except Exception:
        return

    if payload.get("completed"):
        return

    done_Ns = set(int(k) for k in payload.get("results", {}).keys())
    first_missing = next((N for N in batch_sizes if N not in done_Ns), None)
    if first_missing is not None:
        payload["results"][str(first_missing)] = {
            "N": first_missing, "status": "timeout",
            "peak_mem_MiB": None, "time_s": None,
            "error": f"killed_after_{TIMEOUT_S}s",
        }
        for N in batch_sizes:
            if N > first_missing and str(N) not in payload["results"]:
                payload["results"][str(N)] = {
                    "N": N, "status": "timeout_skipped",
                    "peak_mem_MiB": None, "time_s": None, "error": None,
                }
    payload["completed"] = True
    _flush(out_json, payload)


def _aggregate_summary(outdir, batch_sizes):
    """Read all per_model/*.json files and write summary.json + summary.csv."""
    per_model_dir = os.path.join(outdir, "per_model")
    files = sorted(glob.glob(os.path.join(per_model_dir, "*.json")))
    summary = []
    for fp in files:
        try:
            with open(fp) as f:
                summary.append(json.load(f))
        except Exception:
            pass

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Check whether there is metric data (produced by --with-metrics); decide the CSV column set.
    has_metrics = any(
        isinstance(r, dict) and any(k in r for k in ("depth", "camera", "trajectory"))
        for entry in summary
        for r in (entry.get("results", {}) or {}).values()
    )

    def _fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 1e4 else f"{v:.2f}"
        return str(v)

    # CSV: rows=models, columns=per-N status / mem / time [+ metric_status / abs_rel / auc_5 / ate]
    csv_path = os.path.join(outdir, "summary.csv")
    headers = ["model", "kind", "fixture_scene", "weights_MiB"]
    for N in batch_sizes:
        headers += [f"N={N}_status", f"N={N}_mem_MiB", f"N={N}_time_s"]
        if has_metrics:
            headers += [f"N={N}_metric_status",
                        f"N={N}_abs_rel", f"N={N}_auc_5", f"N={N}_ate"]
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for entry in summary:
            row = [entry.get("model", "?"),
                   entry.get("kind", "?"),
                   entry.get("fixture_scene", "?"),
                   str(entry.get("weights_MiB") or "")]
            results = entry.get("results", {})
            for N in batch_sizes:
                r = results.get(str(N), {}) or {}
                row += [r.get("status", ""),
                        str(r.get("peak_mem_MiB") or ""),
                        str(r.get("time_s") or "")]
                if has_metrics:
                    depth = r.get("depth") or {}
                    cam = r.get("camera") or {}
                    traj = r.get("trajectory") or {}
                    row += [r.get("metric_status", ""),
                            _fmt(depth.get("abs_rel")),
                            _fmt(cam.get("auc_5")),
                            _fmt(traj.get("ate"))]
            # Simple CSV safety: the model/kind fields contain no commas
            f.write(",".join(row) + "\n")

    print(f"[driver] summary: {csv_path}")


def _is_model_completed(out_json, batch_sizes):
    """Has this model been fully completed (no dispatch needed)?"""
    if not os.path.isfile(out_json):
        return False
    try:
        with open(out_json, "r") as f:
            payload = json.load(f)
    except Exception:
        return False
    if not payload.get("completed"):
        return False
    results = payload.get("results", {}) or {}
    return all(str(N) in results for N in batch_sizes)


def _last_status_in_run(prev_outdir, model_name, batch_sizes):
    """Read a model's per_model JSON from the previous run's outdir and return the
    status of the largest N in batch_sizes. Returns None if it cannot be obtained.
    """
    fp = os.path.join(prev_outdir, "per_model", f"{model_name}.json")
    if not os.path.isfile(fp):
        return None
    try:
        with open(fp) as f:
            payload = json.load(f)
    except Exception:
        return None
    results = payload.get("results", {}) or {}
    last_N = max(batch_sizes)
    r = results.get(str(last_N), {})
    return r.get("status")


def run_driver(args):
    # ---- CLI overrides: --configs (whitelist + order) / --fixture-json / --batch-sizes ----
    whitelist = None
    if args.configs:
        whitelist = [x for x in args.configs.split(",") if x.strip()]
    configs_all = _collect_configs(whitelist=whitelist)
    if not configs_all:
        print("[driver] no configs found.", file=sys.stderr)
        sys.exit(1)

    # ---- Decide this run's fixture / batch_sizes / candidate model set ----
    if args.extend_from:
        # Extended mode: only run larger N for models that were still ok at N=max(BATCH_SIZES)=1000 in the previous run.
        fixture = _extended_fixture()
        batch_sizes = list(EXTENDED_BATCH_SIZES)
        prev_outdir = args.extend_from
        print(f"[driver] EXTEND mode: prev_outdir={prev_outdir}")
        print(f"[driver]   fixture: {fixture['scene_id']} "
              f"(source={fixture['source']}, path={fixture['path']}, "
              f"num_frames={fixture['num_frames']})")
        print(f"[driver]   extended batch_sizes: {batch_sizes}")
        configs = []
        skip_log = []
        for c in configs_all:
            name = _name_from_cfg(c)
            last = _last_status_in_run(prev_outdir, name, BATCH_SIZES)
            if last == "ok":
                configs.append(c)
            else:
                skip_log.append((name, last))
        if skip_log:
            print(f"[driver]   skip {len(skip_log)} models (last status != ok in prev run):")
            for n, s in skip_log:
                print(f"    · {n:30s} status={s}")
    else:
        if args.fixture_json:
            with open(args.fixture_json, "r") as f:
                fixture = json.load(f)
            print(f"[driver] custom fixture from {args.fixture_json}: "
                  f"scene_id={fixture['scene_id']}, source={fixture['source']}, "
                  f"path={fixture['path']}, num_frames={fixture['num_frames']}")
        else:
            fixture = _default_fixture()
        if args.batch_sizes:
            batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
            print(f"[driver] custom batch_sizes: {batch_sizes}")
        else:
            batch_sizes = list(BATCH_SIZES)
        configs = configs_all
        if args.with_metrics:
            print("[driver] --with-metrics ON: will call evaluate_scene() after "
                  "each N's inference to compute depth/pose/trajectory metrics")

    os.makedirs(os.path.join(args.outdir, "per_model"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "log"), exist_ok=True)

    # ---- Resume: skip fully-completed models, split the list ----
    todo, already_done = [], []
    for cfg_path in configs:
        out_json = os.path.join(args.outdir, "per_model",
                                f"{_name_from_cfg(cfg_path)}.json")
        (already_done if _is_model_completed(out_json, batch_sizes)
         else todo).append(cfg_path)

    print(f"[driver] gpus={args.gpus}  candidates={len(configs)}  "
          f"todo={len(todo)}  done={len(already_done)}  "
          f"outdir={args.outdir}  timeout={TIMEOUT_S}s")
    if already_done:
        print("[driver] [SKIP-DONE] (already fully completed, not re-measured):")
        for c in already_done:
            print(f"  · {_name_from_cfg(c)}")
    if todo:
        print("[driver] todo:")
        for c in todo:
            existing = os.path.join(args.outdir, "per_model",
                                    f"{_name_from_cfg(c)}.json")
            done_Ns, _ = _load_existing_results(existing)
            tag = (f"  (resume: {len(done_Ns)}/{len(batch_sizes)} N done)"
                   if done_Ns else "")
            print(f"  · {_name_from_cfg(c)}{tag}")

    queue = list(todo)
    running = {}  # gpu_id -> {"proc", "config", "out_json", "heartbeat", "log_fh", "started"}

    fixture_arg = json.dumps(fixture)
    batch_sizes_arg = ",".join(str(x) for x in batch_sizes)

    # ---- Signal handling: on Ctrl-C / SIGTERM, KILL all worker process groups ----
    def _shutdown(sig, _frame):
        print(f"\n[driver] received signal {sig}, killing {len(running)} workers...",
              flush=True)
        for gpu_id, info in list(running.items()):
            try:
                pgid = os.getpgid(info["proc"].pid)
                os.killpg(pgid, signal.SIGKILL)
                print(f"  killed {info['name']} (GPU {gpu_id}, pgid={pgid})")
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"  kill {info['name']}: {e}")
        # Do not call _aggregate_summary; let the user see partial results preserved as-is so a restart can resume
        sys.exit(130)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    def dispatch(gpu_id, cfg_path):
        name = _name_from_cfg(cfg_path)
        out_json = os.path.join(args.outdir, "per_model", f"{name}.json")
        log_path = os.path.join(args.outdir, "log", f"{name}.log")
        heartbeat = out_json.replace(".json", ".heartbeat")

        # Remove the old heartbeat (if this is a resume)
        try:
            if os.path.isfile(heartbeat):
                os.unlink(heartbeat)
        except Exception:
            pass

        env = {**os.environ,
               "CUDA_VISIBLE_DEVICES": str(gpu_id),
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        # Thread throttling: with multiple workers in parallel, prevent OpenMP / MKL /
        # OpenBLAS / NumExpr / OpenCV from saturating the whole machine's CPUs. The
        # driver reads args.cpus_per_worker for the cap, and it takes effect before
        # the worker process starts (these env vars are read by the corresponding
        # libraries at child-process import time).
        if args.cpus_per_worker > 0:
            n = str(args.cpus_per_worker)
            env.update({
                "OMP_NUM_THREADS": n,
                "MKL_NUM_THREADS": n,
                "OPENBLAS_NUM_THREADS": n,
                "NUMEXPR_NUM_THREADS": n,
                "NUMEXPR_MAX_THREADS": n,
                "VECLIB_MAXIMUM_THREADS": n,
                "BENCH_WORKER_THREADS": n,  # the worker reads this internally and then calls the torch/cv2 APIs
            })
        log_fh = open(log_path, "w")
        worker_cmd = [
            sys.executable, os.path.abspath(__file__),
            "--worker", "--config", cfg_path, "--out-json", out_json,
            "--batch-sizes", batch_sizes_arg, "--fixture", fixture_arg,
        ]
        if args.with_metrics:
            worker_cmd.append("--with-metrics")
        proc = subprocess.Popen(
            worker_cmd,
            env=env, stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        running[gpu_id] = {
            "proc": proc, "config": cfg_path, "out_json": out_json,
            "heartbeat": heartbeat, "log_fh": log_fh,
            "started": time.time(), "name": name,
        }
        print(f"[{time.strftime('%H:%M:%S')}] [START] {name} -> GPU {gpu_id}  "
              f"pid={proc.pid}  log={log_path}")

    def kill_timed_out():
        now = time.time()
        for gpu_id, info in list(running.items()):
            try:
                if not os.path.isfile(info["heartbeat"]):
                    continue
                with open(info["heartbeat"], "r") as f:
                    parts = f.read().split()
                if len(parts) < 2:
                    continue
                hb_ts = float(parts[1])
                hb_N = parts[0]
                if now - hb_ts > TIMEOUT_S:
                    pgid = os.getpgid(info["proc"].pid)
                    print(f"[{time.strftime('%H:%M:%S')}] [TIMEOUT] {info['name']} "
                          f"(GPU {gpu_id})  N={hb_N}  killing pgid={pgid}")
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            except Exception:
                pass

    # Main scheduling loop
    while queue or running:
        # Dispatch
        for gpu_id in args.gpus:
            if gpu_id not in running and queue:
                dispatch(gpu_id, queue.pop(0))

        # Wait a moment if there is nothing to do
        if not running:
            break
        time.sleep(2)

        # Timeout check
        kill_timed_out()

        # Reap finished workers
        for gpu_id, info in list(running.items()):
            ec = info["proc"].poll()
            if ec is None:
                continue
            info["log_fh"].close()
            elapsed = time.time() - info["started"]
            tag = "[DONE]" if ec == 0 else f"[FAIL ec={ec}]"
            print(f"[{time.strftime('%H:%M:%S')}] {tag} {info['name']} "
                  f"(GPU {gpu_id}, {elapsed:.1f}s)")
            try:
                _finalize_one(info["out_json"], info["config"],
                              batch_sizes, fixture)
            except Exception as e:
                print(f"  finalize error: {e}", file=sys.stderr)
            # Clean up heartbeat
            try:
                if os.path.isfile(info["heartbeat"]):
                    os.unlink(info["heartbeat"])
            except Exception:
                pass
            del running[gpu_id]

    _aggregate_summary(args.outdir, batch_sizes)
    print("[driver] all done.")


# ====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true",
                    help="worker mode (used internally by the driver)")
    ap.add_argument("--config", type=str, default=None,
                    help="worker: yaml path")
    ap.add_argument("--out-json", type=str, default=None,
                    help="worker: per-model output json path")
    ap.add_argument("--batch-sizes", type=str, default=None,
                    help="worker: comma-separated list of N, overrides the default BATCH_SIZES")
    ap.add_argument("--fixture", type=str, default=None,
                    help="worker: JSON string {scene_id, source, path, num_frames, tags}")
    ap.add_argument("--gpus", type=str, default=None,
                    help="driver: comma-separated GPU ids, e.g. '4' or '4,5,6,7'")
    ap.add_argument("--outdir", type=str, default=None,
                    help="driver: output root directory")
    ap.add_argument("--extend-from", type=str, default=None,
                    help="driver: previous run's outdir; enters extended mode (KITTI 02, N>=1500), "
                         "appending tests only for models that had status=ok at N=1000 in the previous run.")
    ap.add_argument("--configs", type=str, default=None,
                    help="driver: comma-separated whitelist of yaml paths, queued in list order "
                         "(overrides the default end2end+online full scan).")
    ap.add_argument("--fixture-json", type=str, default=None,
                    help="driver: fixture JSON file path, content "
                         "{scene_id, source, path, num_frames, [frame_indices], tags}; "
                         "overrides the default 7scenes fixture. Only effective when not in --extend-from mode.")
    ap.add_argument("--with-metrics", action="store_true",
                    help="After each (model, N) inference completes, reuse run_benchmark.evaluate_scene "
                         "to compute depth/pose/trajectory metrics, written into the per-model JSON and summary.csv.")
    ap.add_argument("--cpus-per-worker", type=int, default=8,
                    help="driver: maximum CPU threads each worker subprocess may use "
                         "(caps OMP/MKL/OpenBLAS/NumExpr/torch/cv2). Default 8; "
                         "pass 0 for no limit. Recommended 8-16 when running multiple GPUs in parallel, to keep the machine's CPUs from being saturated.")
    args = ap.parse_args()

    if args.worker:
        if not args.config or not args.out_json:
            ap.error("--worker requires --config and --out-json")
        run_worker(args)
    else:
        if not args.gpus or not args.outdir:
            ap.error("driver mode requires --gpus and --outdir")
        args.gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
        run_driver(args)


if __name__ == "__main__":
    main()
