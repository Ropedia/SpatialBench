"""
SpatialBenchBenchmark evaluation script with GT prior injection.

Supports injecting GT camera pose / depth / intrinsics during inference,
to evaluate the reconstruction capability of a model when assisted by
partial prior information.

Usage:
    # Use a config file
    python benchmark/evaluation/run_benchmark_with_prior.py \
        --config benchmark/configs/prior/omnivggt_prior_eval.yaml

    # CLI override: inject GT pose + depth on 50% of frames
    python benchmark/evaluation/run_benchmark_with_prior.py \
        --config benchmark/configs/prior/omnivggt_prior_eval.yaml \
        --use-gt-pose --use-gt-depth --gt-ratio 0.5

    # Inject GT pose on all frames (DA3, all-or-nothing)
    python benchmark/evaluation/run_benchmark_with_prior.py \
        --config benchmark/configs/prior/da3_giant_prior_eval.yaml \
        --use-gt-pose --gt-ratio 1.0
"""
import argparse
import gc
import os
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.datasets.benchmark_dataset import BenchmarkDataset
from benchmark.evaluation.metrics import (
    get_gt_mesh_path, should_run_pointcloud_eval,
)
from benchmark.evaluation.run_benchmark import (
    load_config, merge_config_and_args, evaluate_scene, ALL_EVAL_METRICS,
)
from benchmark.evaluation.report import (
    build_report, write_json_report, write_overall,
    print_summary, generate_output_name,
)
from benchmark.utils.visualization import visualize_scene, save_scene_inputs
from benchmark.utils.paths import DEFAULT_CHECKPOINTS_DIR

# Import your own adapters here to trigger @register_adapter, e.g.:
#     import benchmark.evaluation.model_adapters.my_model_adapter
import benchmark.evaluation.model_adapters.vggt_adapter  # noqa: F401
import benchmark.evaluation.model_adapters.da3_adapter  # noqa: F401
from benchmark.evaluation.model_adapters import get_adapter


def compute_gt_frame_indices(scene_id, n_frames, gt_ratio, gt_seed, partial_support,
                             salt=0):
    """Compute which frames in a scene receive the GT prior.

    Args:
        scene_id: scene ID (used for deterministic randomness)
        n_frames: total number of frames
        gt_ratio: ratio of GT frames (0.0 - 1.0)
        gt_seed: random seed
        partial_support: whether the model supports per-frame injection
        salt: extra salt so different prior types produce different random frame selections

    Returns:
        list[int]: selected frame indices (sorted)
    """
    if gt_ratio <= 0:
        return []

    if not partial_support or gt_ratio >= 1.0:
        # all-or-nothing or all frames
        return list(range(n_frames))

    n_gt = max(1, round(n_frames * gt_ratio))
    n_gt = min(n_gt, n_frames)

    # Deterministic random: seed + scene_id hash + salt
    seed = gt_seed + (hash(scene_id) % (2**31)) + salt
    rng = np.random.RandomState(seed)
    indices = sorted(rng.choice(n_frames, n_gt, replace=False).tolist())
    return indices


def generate_prior_output_name(model_name, scene_index_path, tags, gt_config):
    """Generate an output name that includes prior info."""
    base = generate_output_name(model_name, scene_index_path, tags)

    prior_parts = []
    if gt_config.get('use_pose'):
        r = int(gt_config.get('gt_camera_ratio', 1.0) * 100)
        prior_parts.append(f"pose{r}")
    if gt_config.get('use_depth'):
        r = int(gt_config.get('gt_depth_ratio', 1.0) * 100)
        prior_parts.append(f"depth{r}")
    if gt_config.get('use_intrinsic'):
        r = int(gt_config.get('gt_camera_ratio', 1.0) * 100)
        prior_parts.append(f"intr{r}")

    if prior_parts:
        return f"{base}_prior_{'_'.join(prior_parts)}"
    return base


def main():
    parser = argparse.ArgumentParser(
        description="SpatialBenchBenchmark with GT Prior Injection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file (benchmark/configs/prior/*.yaml)')

    # Model parameters
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoints-dir', type=str, default=DEFAULT_CHECKPOINTS_DIR)

    # Data parameters
    parser.add_argument('--scene-index', type=str, default=None)

    # Filter parameters
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--max-scenes', type=int, default=None)
    parser.add_argument('--scene-id', type=str, default=None,
                        help='Run only this single scene_id (overrides tags/max-scenes; '
                             'YAML key: scene_id).')

    # Evaluation parameters
    parser.add_argument('--eval-metrics', nargs='+',
                        choices=ALL_EVAL_METRICS, default=None)
    parser.add_argument('--conf-threshold', type=float, default=0.5)
    parser.add_argument('--depth-alignment', choices=['median', 'lstsq'],
                        default='median')

    # Shuffle mode: only applies to end2end / prior models
    parser.add_argument('--shuffle-seed', type=int, default=None,
                        help='Base seed for shuffling frame order within each scene. None=no shuffle (default). '
                             'Only applies to end2end models (benchmark/configs/end2end/, '
                             'benchmark/configs/prior/). When set, each scene derives a deterministic permutation '
                             'from (seed + hash(scene_id)); GT and images stay aligned.')
    # GT Prior parameters
    # --use-gt-camera: inject GT camera (pose + intrinsics coupled, e.g. OmniVGGT)
    # --use-gt-depth: inject GT depth
    # --use-gt-intrinsic-only: inject intrinsics only, no pose (MapAnything/WorldMirror)
    parser.add_argument('--use-gt-camera', action='store_true',
                        help='Inject GT camera (pose + intrinsics). '
                             'For OmniVGGT: camera_gt_index controls both at once; '
                             'For other models: enables both pose and intrinsics')
    parser.add_argument('--use-gt-depth', action='store_true',
                        help='Inject GT depth')
    parser.add_argument('--use-gt-intrinsic-only', action='store_true',
                        help='Inject GT intrinsics only (no pose). '
                             'Only effective for models that support standalone intrinsics such as MapAnything/WorldMirror. '
                             'Ignored if --use-gt-camera is also set')
    parser.add_argument('--gt-camera-ratio', type=float, default=1.0,
                        help='Ratio of GT camera frames (0.0-1.0); controls pose+intrinsics')
    parser.add_argument('--gt-depth-ratio', type=float, default=1.0,
                        help='Ratio of GT depth frames (0.0-1.0); independent of camera ratio')
    parser.add_argument('--gt-seed', type=int, default=42,
                        help='Random seed for GT frame selection (ensures reproducibility)')

    # Output
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--output', type=str, default=None)

    # Visualization
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vis-conf-percent', type=float, default=20.0)

    args = parser.parse_args()

    # ---- Load config file and merge ----
    cfg = {}
    if args.config:
        print(f"Loading config: {args.config}")
        cfg = load_config(args.config)
        args = merge_config_and_args(cfg, args, parser)

    # Validate required arguments
    if not args.model:
        parser.error("--model is required (or set 'model' in config)")
    if not args.scene_index:
        parser.error("--scene-index is required (or set 'scene_index' in config)")

    # Shuffle scope guard: not allowed on online/streaming models
    if args.shuffle_seed is not None and args.config and \
            '/online/' in args.config.replace('\\', '/'):
        parser.error(
            f"--shuffle-seed is not supported for online/streaming configs "
            f"({args.config}). Online models are designed for sequential input; "
            f"shuffling is only meaningful for end2end / prior models."
        )

    # ---- Resolve GT prior switches ----
    # use_gt_camera -> use_pose=True, use_intrinsic=True
    # use_gt_intrinsic_only -> use_pose=False, use_intrinsic=True (only when use_gt_camera is not set)
    args.use_gt_pose = args.use_gt_camera
    args.use_gt_intrinsic = args.use_gt_camera or args.use_gt_intrinsic_only

    use_any_prior = args.use_gt_pose or args.use_gt_depth or args.use_gt_intrinsic
    if not use_any_prior:
        print("\n[WARNING] No GT prior enabled (--use-gt-camera/depth/intrinsic-only).")
        print("         Running in standard mode (equivalent to run_benchmark.py).\n")

    # Print configuration
    print(f"\n[Config] model={args.model}, checkpoint={args.checkpoint}, "
          f"device={args.device}")
    print(f"[Config] scene_index={args.scene_index}, tags={args.tags}, "
          f"max_scenes={args.max_scenes}, scene_id={args.scene_id}")
    print(f"[Config] eval_metrics={args.eval_metrics or 'all'}, "
          f"depth_alignment={args.depth_alignment}")
    print(f"[Config] GT Prior: camera(pose+intr)={args.use_gt_camera} ratio={args.gt_camera_ratio}, "
          f"depth={args.use_gt_depth} ratio={args.gt_depth_ratio}, "
          f"intrinsic_only={args.use_gt_intrinsic_only}, seed={args.gt_seed}")
    if args.shuffle_seed is not None:
        print(f"[Config] shuffle_seed={args.shuffle_seed} (frame order randomized per scene)")

    # ---- 1. Load model ----
    print(f"\n[1/3] Loading model '{args.model}'...")
    t_model_load = time.time()
    adapter = get_adapter(args.model)
    adapter.load_model(checkpoint=args.checkpoint, device=args.device)
    model_load_time = time.time() - t_model_load
    print(f"  Model loaded in {model_load_time:.2f}s")

    # Count model parameters
    model_params = adapter.get_model_params()
    if model_params:
        print(f"  Parameters: {model_params['total_params_M']:.2f}M total")

    # Validate the model's GT prior support
    prior_caps = adapter.supports_gt_prior()
    if use_any_prior:
        print(f"  GT prior capabilities: {prior_caps}")
        warnings = []
        if args.use_gt_pose and not prior_caps['pose']:
            warnings.append("pose (not supported, will be ignored)")
            args.use_gt_pose = False
        if args.use_gt_depth and not prior_caps['depth']:
            warnings.append("depth (not supported, will be ignored)")
            args.use_gt_depth = False
        if args.use_gt_intrinsic and not prior_caps['intrinsic']:
            warnings.append("intrinsic (not supported, will be ignored)")
            args.use_gt_intrinsic = False
        if warnings:
            print(f"  WARNING: Ignoring unsupported priors: {', '.join(warnings)}")
        if not prior_caps['partial']:
            if 0 < args.gt_camera_ratio < 1.0:
                print(f"  WARNING: {adapter.name()} does not support partial GT; "
                      f"gt_camera_ratio={args.gt_camera_ratio} will be treated as 1.0")
            if 0 < args.gt_depth_ratio < 1.0:
                print(f"  WARNING: {adapter.name()} does not support partial GT; "
                      f"gt_depth_ratio={args.gt_depth_ratio} will be treated as 1.0")

    # Build the global gt_config
    gt_config_base = {
        'use_pose': args.use_gt_pose,
        'use_depth': args.use_gt_depth,
        'use_intrinsic': args.use_gt_intrinsic,
        'gt_camera_ratio': args.gt_camera_ratio,
        'gt_depth_ratio': args.gt_depth_ratio,
        'gt_seed': args.gt_seed,
    }

    # ---- 2. Build the dataset ----
    print(f"\n[2/3] Loading benchmark dataset...")

    tags = None
    tag_operator = "AND"
    if args.tags:
        if '+' in args.tags:
            tags = args.tags.split('+')
            tag_operator = "AND"
        elif '|' in args.tags:
            tags = args.tags.split('|')
            tag_operator = "OR"
        else:
            tags = [args.tags]

    if args.tags and "single" in args.tags:
        args.eval_metrics = ["depth"]
        print(f"[Config] Single-frame mode detected, eval_metrics forced to ['depth']")

    resolution_override = cfg.get('resolution_override', None)

    dataset = BenchmarkDataset(
        scene_index_path=args.scene_index,
        tags=tags,
        tag_operator=tag_operator,
        max_scenes=args.max_scenes,
        conf_threshold=args.conf_threshold,
        resolution_override=resolution_override,
        shuffle_seed=args.shuffle_seed,
    )

    # Single-scene mode: pick one entry by scene_id directly from the full scene index, ignoring tags/max_scenes
    if args.scene_id:
        matched = [s for s in dataset.registry.scenes
                   if s.get("scene_id") == args.scene_id]
        if not matched:
            parser.error(
                f"--scene-id '{args.scene_id}' not found in scene index "
                f"'{args.scene_index}'."
            )
        dataset.scenes = matched
        print(f"[Config] scene_id filter active: running 1 scene '{args.scene_id}' "
              f"(tags/max_scenes ignored)")

    pointcloud_root = os.path.join(dataset.benchmark_root, "pointcloud")
    if not os.path.isdir(pointcloud_root):
        print(f"[pointcloud] GT root not found: {pointcloud_root}; "
              f"pointcloud eval will be skipped")

    # ---- Determine output path ----
    if args.output:
        output_path = args.output
        run_dir = os.path.dirname(output_path) or args.output_dir
    else:
        auto_name = generate_prior_output_name(
            adapter.name(), args.scene_index, args.tags, gt_config_base
        )
        if args.shuffle_seed is not None:
            auto_name = f"{auto_name}_shuffle{args.shuffle_seed}"
        run_dir = os.path.join(args.output_dir, auto_name)
        output_path = os.path.join(run_dir, f"{auto_name}.json")

    os.makedirs(run_dir, exist_ok=True)
    output_dir = run_dir

    # Only save input images in single-scene debug mode (--scene-id); otherwise large-scale
    # sweeps would write a massive number of PNGs (N scenes x 1000+ frames x M models)
    # and blow up the disk.
    save_inputs = bool(args.scene_id)
    inputs_dir = os.path.join(run_dir, "inputs") if save_inputs else None
    if save_inputs:
        os.makedirs(inputs_dir, exist_ok=True)

    if args.visualize:
        vis_gt_dir = os.path.join(run_dir, "visualization", "gt")
        vis_pred_dir = os.path.join(run_dir, "visualization", "pred")
        os.makedirs(vis_gt_dir, exist_ok=True)
        os.makedirs(vis_pred_dir, exist_ok=True)

    # ---- 3. Evaluation loop ----
    print(f"\n[3/3] Running evaluation on {len(dataset)} scenes "
          f"(with GT prior: {gt_config_base})...")
    all_results = []

    timing_accum = {
        "data_loading": [], "inference": [], "eval_total": [],
        "depth_eval": [], "pose_eval": [], "pointcloud_eval": [],
        "visualization": [],
    }

    benchmark_start = time.time()

    for idx in range(len(dataset)):
        t_data = time.time()
        scene = dataset[idx]
        data_load_time = time.time() - t_data

        scene_id = scene["scene_id"]
        n_frames = len(scene['frame_indices'])
        print(f"\n  [{idx+1}/{len(dataset)}] {scene_id} "
              f"({n_frames} frames)")

        if save_inputs:
            save_scene_inputs(scene, inputs_dir)

        # ---- Resolve pointcloud GT mesh path (only whitelisted datasets + medium/dense) ----
        raw_scene = dataset.scenes[idx]
        source_ds = raw_scene.get("source_dataset")
        gt_mesh_path = None
        if should_run_pointcloud_eval(source_ds, raw_scene.get("tags")):
            gt_mesh_path = get_gt_mesh_path(
                source_ds,
                pointcloud_root,
                raw_scene.get("scene_path"),
            )
            if gt_mesh_path is None:
                print(f"    [pointcloud] no GT mesh found for {source_ds}/"
                      f"{raw_scene.get('scene_path')}, skipping pointcloud eval")

        try:
            # ---- Compute the GT frame indices for this scene (camera and depth independent) ----
            camera_gt_indices = []
            depth_gt_indices = []

            if args.use_gt_pose or args.use_gt_intrinsic:
                camera_gt_indices = compute_gt_frame_indices(
                    scene_id=scene_id,
                    n_frames=n_frames,
                    gt_ratio=args.gt_camera_ratio,
                    gt_seed=args.gt_seed,
                    partial_support=prior_caps['partial'],
                    salt=0,  # camera uses salt=0
                )
                # OmniVGGT/MapAnything require frame 0 to be in the camera GT
                if camera_gt_indices and 0 not in camera_gt_indices:
                    if args.model in ('mapanything', 'omnivggt'):
                        camera_gt_indices = sorted([0] + camera_gt_indices)

            if args.use_gt_depth:
                depth_gt_indices = compute_gt_frame_indices(
                    scene_id=scene_id,
                    n_frames=n_frames,
                    gt_ratio=args.gt_depth_ratio,
                    gt_seed=args.gt_seed,
                    partial_support=prior_caps['partial'],
                    salt=10000,  # depth uses a different salt to produce a different random set of frames
                )

            gt_config = {
                **gt_config_base,
                'gt_frame_indices': sorted(set(camera_gt_indices + depth_gt_indices)),  # backward-compatible interface
                'camera_gt_indices': camera_gt_indices,
                'depth_gt_indices': depth_gt_indices,
            }

            if use_any_prior and (camera_gt_indices or depth_gt_indices):
                parts = []
                if camera_gt_indices:
                    parts.append(f"camera={camera_gt_indices} "
                                 f"({len(camera_gt_indices)}/{n_frames})")
                if depth_gt_indices:
                    parts.append(f"depth={depth_gt_indices} "
                                 f"({len(depth_gt_indices)}/{n_frames})")
                print(f"    GT frames: {', '.join(parts)}")

            # ---- Model inference ----
            t0 = time.time()
            if use_any_prior:
                predictions = adapter.predict(scene, gt_config=gt_config)
            else:
                predictions = adapter.predict(scene)
            inference_time = time.time() - t0

            # ---- Evaluation ----
            t_eval = time.time()
            scene_result, eval_timing = evaluate_scene(
                scene, predictions, adapter,
                depth_alignment=args.depth_alignment,
                eval_metrics=args.eval_metrics,
                gt_mesh_path=gt_mesh_path,
            )
            total_eval_time = time.time() - t_eval

            # Record GT prior info
            if use_any_prior:
                scene_result["gt_prior"] = {
                    "camera_gt_indices": camera_gt_indices,
                    "depth_gt_indices": depth_gt_indices,
                    "n_camera_gt": len(camera_gt_indices),
                    "n_depth_gt": len(depth_gt_indices),
                    "n_total_frames": n_frames,
                    "use_pose": args.use_gt_pose,
                    "use_depth": args.use_gt_depth,
                    "use_intrinsic": args.use_gt_intrinsic,
                    "gt_camera_ratio": args.gt_camera_ratio,
                    "gt_depth_ratio": args.gt_depth_ratio,
                }

            # ---- Visualization ----
            vis_time = 0.0
            if args.visualize:
                t_vis = time.time()
                visualize_scene(scene, predictions, adapter,
                                gt_dir=vis_gt_dir, pred_dir=vis_pred_dir,
                                vis_conf_percent=args.vis_conf_percent)
                vis_time = time.time() - t_vis

            # Aggregate timing
            scene_total = data_load_time + inference_time + total_eval_time + vis_time
            scene_timing = {
                "data_loading": round(data_load_time, 3),
                "inference": round(inference_time, 3),
                "depth_eval": eval_timing.get("depth_eval", 0),
                "pose_eval": eval_timing.get("pose_eval", 0),
                "pointcloud_eval": eval_timing.get("pointcloud_eval", 0),
                "eval_total": round(total_eval_time, 3),
                "visualization": round(vis_time, 3),
                "scene_total": round(scene_total, 3),
            }
            scene_result["timing"] = scene_timing
            all_results.append(scene_result)

            # Accumulate
            timing_accum["data_loading"].append(data_load_time)
            timing_accum["inference"].append(inference_time)
            timing_accum["eval_total"].append(total_eval_time)
            timing_accum["depth_eval"].append(eval_timing.get("depth_eval", 0))
            timing_accum["pose_eval"].append(eval_timing.get("pose_eval", 0))
            timing_accum["pointcloud_eval"].append(eval_timing.get("pointcloud_eval", 0))
            timing_accum["visualization"].append(vis_time)

            # Print timing
            print(f"    ⏱ data={data_load_time:.2f}s | infer={inference_time:.2f}s | "
                  f"eval={total_eval_time:.2f}s "
                  f"[depth={eval_timing.get('depth_eval',0):.2f} "
                  f"pose={eval_timing.get('pose_eval',0):.2f} "
                  f"pc={eval_timing.get('pointcloud_eval',0):.2f}]"
                  f"{f' | vis={vis_time:.2f}s' if vis_time > 0 else ''}"
                  f" | total={scene_total:.2f}s")

            # Print metrics
            parts = []
            if "depth_metric" in scene_result:
                parts.append(f"abs_rel_metric={scene_result['depth_metric'].get('abs_rel', float('nan')):.4f}")
            if "depth" in scene_result:
                parts.append(f"abs_rel={scene_result['depth'].get('abs_rel', float('nan')):.4f}")
            if "camera" in scene_result:
                parts.append(f"racc_5={scene_result['camera'].get('racc_5', float('nan')):.4f}")
                parts.append(f"tacc_5={scene_result['camera'].get('tacc_5', float('nan')):.4f}")
            if "trajectory" in scene_result:
                parts.append(f"ate={scene_result['trajectory'].get('ate', float('nan')):.4f}")
            if "pointcloud_gt_pose" in scene_result:
                pg = scene_result['pointcloud_gt_pose']
                parts.append(f"f_gt={pg.get('f_score', float('nan')):.4f}")
                parts.append(f"ov_gt={pg.get('overall', float('nan')):.4f}")
            if "pointcloud_pred_pose" in scene_result:
                pp = scene_result['pointcloud_pred_pose']
                parts.append(f"f_pred={pp.get('f_score', float('nan')):.4f}")
                parts.append(f"ov_pred={pp.get('overall', float('nan')):.4f}")
            if "pointcloud" in scene_result:
                pd = scene_result['pointcloud']
                parts.append(f"f_pc={pd.get('f_score', float('nan')):.4f}")
                parts.append(f"ov_pc={pd.get('overall', float('nan')):.4f}")
            if parts:
                print(f"    📊 {' | '.join(parts)}")

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "scene_id": scene_id,
                "source_dataset": scene["source_dataset"],
                "tags": scene["tags"],
                "error": str(e),
            })
        finally:
            scene = None
            predictions = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    benchmark_total = time.time() - benchmark_start

    # ---- Print global timing summary ----
    print(f"\n{'='*70}")
    print(f"  TIMING SUMMARY  ({len(all_results)} scenes, with GT prior)")
    print(f"{'='*70}")
    print(f"  Model loading:      {model_load_time:>8.2f}s")
    for key, label in [
        ("data_loading", "Data loading"),
        ("inference", "Inference"),
        ("eval_total", "Evaluation total"),
        ("depth_eval", "  - Depth eval"),
        ("pose_eval", "  - Pose eval"),
        ("pointcloud_eval", "  - Pointcloud eval"),
        ("visualization", "Visualization"),
    ]:
        vals = timing_accum[key]
        if not vals:
            continue
        total = sum(vals)
        avg = total / len(vals)
        pct = total / benchmark_total * 100 if benchmark_total > 0 else 0
        print(f"  {label:<21s} {total:>8.2f}s  (avg {avg:.2f}s/scene, {pct:5.1f}%)")
    print(f"  {'─'*50}")
    print(f"  Benchmark total:    {benchmark_total:>8.2f}s")
    print(f"{'='*70}")

    # ---- 4. Generate report ----
    report = build_report(
        model_name=adapter.name(),
        query_tags=args.tags,
        per_scene_results=all_results,
        checkpoint=args.checkpoint,
        model_params=model_params,
    )

    # Record GT prior config in report meta
    if "meta" in report:
        report["meta"]["gt_prior"] = {
            "use_pose": args.use_gt_pose,
            "use_depth": args.use_gt_depth,
            "use_intrinsic": args.use_gt_intrinsic,
            "gt_camera_ratio": args.gt_camera_ratio,
            "gt_depth_ratio": args.gt_depth_ratio,
            "gt_seed": args.gt_seed,
            "model_prior_caps": prior_caps,
        }
        if args.shuffle_seed is not None:
            report["meta"]["shuffle_seed"] = args.shuffle_seed

    write_json_report(report, output_path)
    write_overall(report, output_dir)
    print_summary(report)

    print(f"\n[GT Prior Config] camera(pose+intr)={args.use_gt_camera} ratio={args.gt_camera_ratio}, "
          f"depth={args.use_gt_depth} ratio={args.gt_depth_ratio}, "
          f"intrinsic_only={args.use_gt_intrinsic_only}")


if __name__ == '__main__':
    main()
