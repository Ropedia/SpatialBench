#!/usr/bin/env python3
"""One-command R3 demo: run inference, save outputs, then open Viser.

Usage:
    conda activate r3
    python demo.py

    python demo.py --seq_path examples/indoor --max_frames 4
    python demo.py --seq_path path/to/images_or_video --output_dir scratch/demo/my_run
    python demo.py --no_viewer

The demo intentionally uses the saved-output flow:

1. infer.py writes depth/color/conf/camera files to --output_dir
2. view.py opens that output directory in Viser

This keeps rerunning the viewer cheap and makes the generated artifacts easy to
inspect or share.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = "ckpt/r3.safetensors"
LONG_CKPT = "ckpt/r3_long.safetensors"


MODE_ALIASES = {
    "short": "local",
    "sampled": "strided",
    "sparse": "strided",
}


MODE_PRESETS = {
    "test": dict(
        kv_cache_mode="all",
        enable_fallback=False,
        max_segment_frames=0,
        metric_scale=False,
    ),
    "local": dict(
        kv_cache_mode="dynamic",
        enable_fallback=False,
        max_segment_frames=0,
        metric_scale=False,
    ),
    "long": dict(
        kv_cache_mode="dynamic",
        enable_fallback=True,
        max_segment_frames=300,
        fallback_drought_threshold_pct=45.0,
        metric_scale=True,
        metric_bootstrap_frames=5,
    ),
    "strided": dict(
        kv_cache_mode="all",
        enable_fallback=True,
        max_segment_frames=100,
        fallback_drought_threshold_pct=45.0,
        metric_scale=True,
        metric_bootstrap_frames=5,
    ),
}


def _default_output_dir(seq_path: str, kv_backend: str, kv_cache_mode: str) -> Path:
    scene = Path(seq_path.rstrip("/")).stem or "sequence"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "scratch" / "demo" / f"{scene}_{kv_backend}_{kv_cache_mode}_{timestamp}"


def _default_ckpt_for_mode(mode: str) -> str:
    return LONG_CKPT if mode in {"long", "strided"} else DEFAULT_CKPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seq_path", default="examples/indoor", help="Image directory or video path.")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Where to save run outputs. Defaults to scratch/demo/<scene>_<backend>_<mode>_<timestamp>.",
    )
    parser.add_argument("--config_name", default="r3-large", help="Config stem or YAML path.")
    parser.add_argument(
        "--ckpt",
        default=None,
        help=(
            "Checkpoint path. Defaults to ckpt/r3.safetensors, or "
            "ckpt/r3_long.safetensors for --mode long/strided. "
            "Pass an empty string to auto-resolve from --config_name."
        ),
    )
    parser.add_argument("--device", default="cuda", help="Inference device.")
    parser.add_argument("--size", type=int, default=504, help="Image resize target.")
    parser.add_argument("--max_frames", type=int, default=0, help="Limit frames for the demo; 0 means all.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Use every N-th input frame.")
    parser.add_argument("--kv_backend", choices=["dense", "paged"], default="dense", help="Online KV backend.")
    parser.add_argument(
        "--kv_cache_mode",
        choices=["all", "dynamic"],
        default="dynamic",
        help="Online KV retention mode.",
    )
    parser.add_argument("--recent_frames", type=int, default=0, help="Recent frames retained by dynamic mode.")
    parser.add_argument(
        "--bootstrap_full_attention_frames",
        type=int,
        default=0,
        help="Run a full-attention bootstrap on the first N frames before online inference.",
    )
    parser.add_argument(
        "--rel_pose_method",
        choices=["greedy", "pgo"],
        default="greedy",
        help="Relative-pose reconstruction method.",
    )
    parser.add_argument(
        "--mode",
        default="test",
        metavar="{test,local,long,strided}",
        help=(
            "Preset bundle. "
            "'test' = quick test run, all kv, no fallback, no metric. "
            "'local' = small-coverage scenes, dynamic kv, no fallback, no metric. "
            "'long' = long trajectories / large outdoor scenes, dynamic kv + fallback + metric. "
            "'strided' = temporally strided video sequences, all kv + fallback + metric. "
            "Legacy aliases: 'short' -> 'local', 'sampled'/'sparse' -> 'strided'."
        ),
    )
    parser.add_argument("--enable_fallback", action="store_true", help="Enable confidence fallback re-anchoring.")
    parser.add_argument(
        "--max_segment_frames",
        type=int,
        default=300,
        help="Cap segment length when fallback is enabled (0 disables the cap).",
    )
    parser.add_argument(
        "--fallback_drought_threshold_pct",
        type=float,
        default=50.0,
        help="Fallback drought threshold as %% of warmup-mean confidence.",
    )
    parser.add_argument(
        "--metric_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DA3-metric scale anchoring. This may need a cached/downloadable metric model.",
    )
    parser.add_argument("--metric_bootstrap_frames", type=int, default=3, help="Metric-scale bootstrap frames.")
    parser.add_argument(
        "--compute_sky_mask",
        action="store_true",
        help="Export sky/non-sky masks when the selected model emits a sky tensor.",
    )
    parser.add_argument("--sky_mask_threshold", type=float, default=0.3, help="Sky mask threshold.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--load_workers", type=int, default=32, help="Viewer frame-loading workers.")
    parser.add_argument("--no_viewer", action="store_true", help="Only run inference and save outputs.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output_dir if it already exists. Only applies when --output_dir is explicit.",
    )
    args = parser.parse_args()
    args.requested_mode = args.mode
    if args.mode:
        args.mode = MODE_ALIASES.get(args.mode, args.mode)
        if args.mode not in MODE_PRESETS:
            parser.error("--mode must be one of: test, local, long, strided")
    return args


def main() -> int:
    args = parse_args()

    if args.mode:
        preset = MODE_PRESETS[args.mode]
        for key, value in preset.items():
            setattr(args, key, value)
        suffix = f" (alias for {args.mode})" if args.requested_mode != args.mode else ""
        print(f"=== R3 demo: applying preset --mode {args.requested_mode}{suffix} ===")
        for key, value in preset.items():
            print(f"  {key} = {value}")

    if args.ckpt is None:
        args.ckpt = _default_ckpt_for_mode(args.mode)

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(
        args.seq_path, args.kv_backend, args.kv_cache_mode
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            print(f"Output dir already exists and is not empty: {output_dir}")
            print("Pass --overwrite or choose a different --output_dir.")
            return 2
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    inference_cmd = [
        sys.executable,
        str(REPO_ROOT / "infer.py"),
        "--seq_path",
        args.seq_path,
        "--output_dir",
        str(output_dir),
        "--config_name",
        args.config_name,
        "--ckpt",
        args.ckpt,
        "--device",
        args.device,
        "--size",
        str(args.size),
        "--max_frames",
        str(args.max_frames),
        "--frame_stride",
        str(args.frame_stride),
        "--online_kv_backend",
        args.kv_backend,
        "--online_kv_cache_mode",
        args.kv_cache_mode,
        "--online_recent_frames",
        str(args.recent_frames),
        "--bootstrap_full_attention_frames",
        str(args.bootstrap_full_attention_frames),
        "--keyframe_mode",
        "novelty",
        "--keyframe_novelty_threshold",
        "0.985",
        "--keyframe_max_interval",
        "30",
        "--keyframe_max_keyframes",
        "100",
        "--rel_pose_reconstruction_method",
        args.rel_pose_method,
        "--online_verbose",
    ]
    if args.compute_sky_mask:
        inference_cmd.extend(
            [
                "--compute_sky_mask",
                "--sky_mask_threshold",
                str(args.sky_mask_threshold),
            ]
        )
    if args.enable_fallback:
        inference_cmd.extend(
            [
                "--online_fallback_enabled",
                "--fallback_drought_length",
                "3",
                "--fallback_drought_threshold",
                "0",
                "--fallback_drought_threshold_pct",
                str(args.fallback_drought_threshold_pct),
                "--fallback_num_bridge_frames",
                "10",
                "--evict_low_conf_threshold",
                "0",
                "--fallback_ref_mode",
                "bridge",
                "--min_segment_frames",
                "16",
                "--max_segment_frames",
                str(args.max_segment_frames),
                "--fallback_replay_attention",
                "full",
                "--disable_segment_pgo",
            ]
        )
    if args.metric_scale:
        inference_cmd.extend(
            [
                "--metric_scale_enabled",
                "--metric_bootstrap_frames",
                str(args.metric_bootstrap_frames),
            ]
        )

    print("=== R3 demo: inference ===")
    print(f"sequence: {args.seq_path}")
    print(f"output:   {output_dir}")
    subprocess.run(inference_cmd, cwd=REPO_ROOT, env=env, check=True)

    if args.no_viewer:
        print(f"\nSaved run output to {output_dir}")
        print(f"Open later with: {sys.executable} view.py --data_dir {output_dir}")
        return 0

    viewer_cmd = [
        sys.executable,
        str(REPO_ROOT / "view.py"),
        "--data_dir",
        str(output_dir),
        "--port",
        str(args.port),
        "--load_workers",
        str(args.load_workers),
    ]
    print("\n=== R3 demo: viewer ===")
    print(f"Open http://localhost:{args.port}")
    subprocess.run(viewer_cmd, cwd=REPO_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
