"""
SpatialBenchBenchmark main evaluation script.

Usage:
    # Use a config file
    python benchmark/evaluation/run_benchmark.py \
        --config benchmark/configs/vggt_eval.yaml

    # Config file + CLI overrides
    python benchmark/evaluation/run_benchmark.py \
        --config benchmark/configs/vggt_eval.yaml \
        --tags "sparse+indoor" --max-scenes 5 --visualize

    # Pure CLI (without a config file)
    python benchmark/evaluation/run_benchmark.py \
        --model vggt \
        --scene-index benchmark/scene_indices/all_scenes.json
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.datasets.benchmark_dataset import BenchmarkDataset
from benchmark.evaluation.alignment import (
    median_scale_alignment,
    lstsq_alignment,
    procrustes_alignment,
)
from benchmark.evaluation.metrics import (
    compute_depth_metrics,
    compute_tgm_metric,
    compute_pose_metrics,
    compute_pose_metrics_c2w,
    compute_trajectory_metrics,
    compute_pointcloud_metrics,
    unproject_to_pointcloud,
    align_pointcloud_procrustes,
    fuse_depth_to_pointcloud,
    get_gt_mesh_path,
    load_gt_pointcloud_from_mesh,
    save_pointcloud_ply,
    should_run_pointcloud_eval,
    get_pointcloud_eval_params,
)
from benchmark.evaluation.report import (
    build_report, write_json_report, write_overall,
    print_summary, generate_output_name,
)
from benchmark.utils.visualization import visualize_scene, save_scene_inputs
from benchmark.utils.paths import DEFAULT_CHECKPOINTS_DIR, DEFAULT_DROID_ROOT

# Import all adapters (triggers @register_adapter registration)
# Import your own adapters here to trigger @register_adapter, e.g.:
#     import benchmark.evaluation.model_adapters.my_model_adapter
import benchmark.evaluation.model_adapters.vggt_adapter  # noqa: F401
import benchmark.evaluation.model_adapters.da3_adapter  # noqa: F401
# ---- Optimization-based adapters ----
import benchmark.evaluation.model_adapters.dust3r_adapter            # noqa: F401
import benchmark.evaluation.model_adapters.mast3r_adapter            # noqa: F401
# ---- End-to-end feed-forward adapters ----
import benchmark.evaluation.model_adapters.amb3r_adapter             # noqa: F401
import benchmark.evaluation.model_adapters.da3nested_adapter         # noqa: F401
import benchmark.evaluation.model_adapters.fastvggt_adapter          # noqa: F401
import benchmark.evaluation.model_adapters.mapanything_adapter       # noqa: F401
import benchmark.evaluation.model_adapters.omnivggt_adapter          # noqa: F401
import benchmark.evaluation.model_adapters.pi3_adapter               # noqa: F401
import benchmark.evaluation.model_adapters.pi3x_adapter              # noqa: F401
import benchmark.evaluation.model_adapters.vggt_omega_adapter        # noqa: F401
import benchmark.evaluation.model_adapters.worldmirror_adapter       # noqa: F401
# ---- Online / streaming adapters ----
import benchmark.evaluation.model_adapters.infinitevggt_adapter      # noqa: F401
import benchmark.evaluation.model_adapters.lingbot_map_adapter       # noqa: F401
import benchmark.evaluation.model_adapters.lingbot_map_stream_adapter  # noqa: F401
import benchmark.evaluation.model_adapters.stream3r_adapter          # noqa: F401
import benchmark.evaluation.model_adapters.page4d_adapter            # noqa: F401
import benchmark.evaluation.model_adapters.streamvggt_adapter        # noqa: F401
# ---- Chunk-wise adapters ----
import benchmark.evaluation.model_adapters.vggt_long_adapter         # noqa: F401
import benchmark.evaluation.model_adapters.pi_long_adapter           # noqa: F401
import benchmark.evaluation.model_adapters.da3_streaming_adapter     # noqa: F401
# ---- Test-time training adapters ----
import benchmark.evaluation.model_adapters.scal3r_adapter            # noqa: F401
import benchmark.evaluation.model_adapters.loger_adapter             # noqa: F401
import benchmark.evaluation.model_adapters.zipmap_adapter            # noqa: F401
import benchmark.evaluation.model_adapters.vgg_ttt_adapter           # noqa: F401
from benchmark.evaluation.model_adapters import get_adapter


def load_config(config_path):
    """Load a YAML config file and return a dict."""
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def merge_config_and_args(cfg, args, parser):
    """Use config file values as defaults; CLI args take precedence.

    Logic: for each argument, if the CLI explicitly passed it use the CLI value,
           otherwise use the config file value, otherwise fall back to the
           argparse default.
    """
    # Find the arguments the user explicitly passed on the command line
    # Method: parse once more with no args to see which are non-default
    defaults = vars(parser.parse_args([]))

    model_extra_params = {}
    for key, val in cfg.items():
        # YAML keys use underscores; argparse uses underscores (internal)
        arg_key = key.replace('-', '_')
        if arg_key not in defaults:
            # Non-standard argparse field -> collect as model-specific param
            model_extra_params[arg_key] = val
            continue
        # If the CLI did not explicitly pass this arg (value equals the default), use the config file value
        cli_val = getattr(args, arg_key, None)
        default_val = defaults.get(arg_key)
        if cli_val == default_val and val is not None:
            setattr(args, arg_key, val)

    # Store model-specific params on args for later forwarding to adapter.configure()
    args.model_extra_params = model_extra_params
    return args


ALL_EVAL_METRICS = ["depth", "tgm", "pose", "trajectory", "pointcloud"]


def evaluate_scene(scene, predictions, adapter, depth_alignment="median",
                   eval_metrics=None, gt_mesh_path=None):
    """Evaluate a single scene.

    Args:
        eval_metrics: list of metric categories to evaluate. Allowed values:
            - "depth": depth metrics (abs_rel, rmse, delta, inlier, etc.)
            - "pose": camera pose metrics (racc, tacc, auc, etc., based on DA3 pairwise protocol)
            - "trajectory": trajectory metrics (ATE, RPE_t, RPE_r, based on evo Sim(3) alignment)
            - "pointcloud": point cloud metrics (F-score, Overall)
            None means evaluate all categories.
        gt_mesh_path: if provided, use this .ply as the GT point cloud (only for
            the whitelisted datasets + medium/dense view scene settings); when
            None, the pointcloud evaluation is skipped.

    Returns:
        (result_dict, timing_dict)
    """
    if eval_metrics is None:
        eval_metrics = ALL_EVAL_METRICS
    eval_metrics = set(eval_metrics)

    result = {
        "scene_id": scene["scene_id"],
        "source_dataset": scene["source_dataset"],
        "tags": scene["tags"],
        "num_frames": len(scene["frame_indices"]),
    }
    eval_timing = {}

    gt_depth = scene["depth"]
    gt_poses = scene["extrinsic"]
    gt_intrinsic = scene["intrinsic"]
    valid_mask = scene["valid_mask"]
    N = len(gt_depth)

    # ---- Depth evaluation ----
    t_depth = time.time()
    need_depth = "depth" in eval_metrics
    need_tgm = "tgm" in eval_metrics
    if (need_depth or need_tgm) and "pred_depth" in predictions:
        pred_depth = predictions["pred_depth"]

        # Collect valid pred / gt pixels across all frames
        all_pred_valid = []
        all_gt_valid = []
        for i in range(N):
            frame_mask = valid_mask[i]
            if not frame_mask.any():
                continue
            all_pred_valid.append(pred_depth[i][frame_mask])
            all_gt_valid.append(gt_depth[i][frame_mask])

        if all_pred_valid:
            cat_pred = np.concatenate(all_pred_valid)
            cat_gt = np.concatenate(all_gt_valid)
            global_mask = np.ones(len(cat_gt), dtype=bool)

            # (1) If the model supports metric depth, compute metric metrics first (no alignment, direct comparison)
            if need_depth and adapter.supports_metric_depth():
                result["depth_metric"] = compute_depth_metrics(
                    cat_pred, cat_gt, global_mask)

            # (2) Alignment: lstsq -> y = s*x + t; median -> y = scale*x
            if depth_alignment == "lstsq":
                A = np.stack([cat_pred, np.ones_like(cat_pred)], axis=1)
                res = np.linalg.lstsq(A, cat_gt, rcond=None)
                s_align, t_align = float(res[0][0]), float(res[0][1])
                aligned_pred = cat_pred * s_align + t_align
                pred_depth_aligned = pred_depth * s_align + t_align
            else:
                s_align = float(np.median(cat_gt) / (np.median(cat_pred) + 1e-8))
                aligned_pred = cat_pred * s_align
                pred_depth_aligned = pred_depth * s_align

            if need_depth:
                result["depth"] = compute_depth_metrics(
                    aligned_pred, cat_gt, global_mask)

            # ---- TGM: consistency of depth change between adjacent frames (using aligned pred depth) ----
            if need_tgm:
                result["tgm"] = compute_tgm_metric(
                    pred_depth_aligned, gt_depth, valid_mask)

    eval_timing["depth_eval"] = round(time.time() - t_depth, 4)

    # ---- Pose evaluation (pairwise: racc, tacc, auc) ----
    t_pose = time.time()
    if "pred_pose" in predictions:
        pred_w2c = predictions["w2c_extrinsics"]  # adapter returns c2w
        gt_w2c_norm = adapter.normalize_gt_poses(scene)  # adapter decides normalization method
        pose_metrics = compute_pose_metrics(pred_w2c, gt_w2c_norm)
        result["camera"] = pose_metrics

        # ---- Trajectory evaluation (Sim(3) alignment: ATE, RPE_t, RPE_r) ----
        if "trajectory" in eval_metrics:
            pred_c2w = predictions["pred_pose"]  # (N, 3, 4) c2w
            traj_metrics = compute_trajectory_metrics(pred_c2w, gt_poses)
            result["trajectory"] = traj_metrics

    eval_timing["pose_eval"] = round(time.time() - t_pose, 4)

    # ---- Pointcloud evaluation (only enabled for whitelisted datasets + medium/dense; GT comes from a mesh file) ----
    t_pc = time.time()
    if "pointcloud" in eval_metrics and gt_mesh_path is not None:
        images_raw = scene["images_raw"]  # (N, 3, H, W) Tensor [0,1]

        # GT point cloud: load directly from the mesh file (sample if triangles; use as-is if already a point cloud)
        gt_pc = load_gt_pointcloud_from_mesh(gt_mesh_path)
        print(f"[pointcloud] GT mesh={gt_mesh_path} | gt_pc n={len(gt_pc)}")

        # Get DA3-aligned evaluation params per dataset: down_sample (voxel meters), threshold (F-score meters), crop_margin
        pc_params = get_pointcloud_eval_params(scene.get("source_dataset"))

        if len(gt_pc) > 0 and "pred_depth" in predictions:
            # AMB3R's raw depth is the one geometrically paired with its predicted
            # camera pose. Keep pred_depth for depth metrics, but use raw depth for
            # point-cloud fusion.
            if adapter.name().lower() == "amb3r" and "pred_depth_raw" in predictions:
                pred_depth = predictions["pred_depth_raw"]
            else:
                pred_depth = predictions["pred_depth"]

            # Scale alignment: derive scale via Umeyama pose alignment (consistent with the DA3 protocol)
            if "pred_pose" in predictions:
                _, sim3 = procrustes_alignment(predictions["pred_pose"], gt_poses)
                scale = sim3["s"]
            else:
                # Fall back to median scale when there is no pred_pose
                valid_flat = valid_mask & (pred_depth > 1e-6) & np.isfinite(pred_depth)
                scale = float(np.median(gt_depth[valid_flat]) /
                              (np.median(pred_depth[valid_flat]) + 1e-8)) if valid_flat.any() else 1.0
            pred_depth_scaled = pred_depth * scale

            # (1) pointcloud_gt_pose: pred_depth + gt_pose (isolates depth quality)
            _mesh_p = _pcd_p = None
            pc_gt_pose = fuse_depth_to_pointcloud(
                pred_depth_scaled, gt_poses, gt_intrinsic, images_raw,
                source_dataset=scene.get("source_dataset"),
                save_mesh_path=_mesh_p, save_pcd_path=_pcd_p)
            if len(pc_gt_pose) > 0:
                result["pointcloud_gt_pose"] = compute_pointcloud_metrics(
                    pc_gt_pose, gt_pc, **pc_params)

            # (2) pointcloud_pred_pose: pred_depth + pred_pose (joint depth+pose quality)
            # Use the Sim3(s,R,t) obtained from pose procrustes to first map pred cameras into GT-world:
            #   R_new = R_sim @ R_pred,  c_new = s_sim * R_sim @ c_pred + t_sim
            # Scale depth by the same s_sim (i.e. pred_depth_scaled), then unproject.
            # The resulting point cloud is already in GT-world, much more stable
            # than NN-based align_pointcloud_procrustes, and the voxel downsampling
            # inside fuse stays in meter scale.
            if "pred_pose" in predictions:
                pred_intrinsic = predictions.get("pred_intrinsic", gt_intrinsic)
                s_sim = float(sim3["s"])
                R_sim = sim3["R"].astype(np.float32)
                t_sim = sim3["t"].astype(np.float32)

                pred_poses = np.asarray(predictions["pred_pose"], dtype=np.float32)
                aligned_poses = np.zeros_like(pred_poses)
                aligned_poses[:, :3, :3] = np.einsum(
                    "ij,njk->nik", R_sim, pred_poses[:, :3, :3]
                )
                aligned_poses[:, :3, 3] = (
                    s_sim * (pred_poses[:, :3, 3] @ R_sim.T) + t_sim
                )

                _mesh_p = _pcd_p = None
                pc_pred_pose = fuse_depth_to_pointcloud(
                    pred_depth_scaled, aligned_poses, pred_intrinsic, images_raw,
                    source_dataset=scene.get("source_dataset"),
                    save_mesh_path=_mesh_p, save_pcd_path=_pcd_p)
                if len(pc_pred_pose) > 0:
                    result["pointcloud_pred_pose"] = compute_pointcloud_metrics(
                        pc_pred_pose, gt_pc, **pc_params)

        # (3) Directly-output pred_pointcloud (e.g. VGGT world_points) - disabled:
        #     NN-based Procrustes alignment is unstable; uniformly use only the gt_pose / pred_pose paths.
        # if len(gt_pc) > 0 and "pred_pointcloud" in predictions:
        #     pred_pc = predictions["pred_pointcloud"]
        #     if pred_pc.shape[0] == valid_mask.size:
        #         pred_pc = pred_pc[valid_mask.reshape(-1)]
        #     finite_mask = np.all(np.isfinite(pred_pc), axis=1) & (np.linalg.norm(pred_pc, axis=1) > 1e-6)
        #     pred_pc = pred_pc[finite_mask]
        #     # Deterministic subsample to cap KDTree cost (GT is ~1M; matching
        #     # orders makes chamfer O(1M·logM) instead of O(20M·logM)).
        #     _MAX_PRED_PTS = 1_000_000
        #     if len(pred_pc) > _MAX_PRED_PTS:
        #         _rng = np.random.RandomState(0)
        #         _idx = _rng.choice(len(pred_pc), _MAX_PRED_PTS, replace=False)
        #         pred_pc = pred_pc[_idx]
        #     if len(pred_pc) > 0:
        #         pred_pc = align_pointcloud_procrustes(pred_pc, gt_pc)
        #         result["pointcloud"] = compute_pointcloud_metrics(
        #             pred_pc, gt_pc, **pc_params)

    eval_timing["pointcloud_eval"] = round(time.time() - t_pc, 4)
    eval_timing["eval_total"] = round(
        eval_timing["depth_eval"] + eval_timing["pose_eval"] + eval_timing["pointcloud_eval"], 4)

    return result, eval_timing


def _resume_signature(model_name, scene_index, tags, max_scenes,
                      ordered_scene_ids):
    """Identify a run for resume purposes; mismatch → discard partial."""
    return {
        "model": model_name,
        "scene_index": scene_index,
        "tags": tags,
        "max_scenes": max_scenes,
        "ordered_scene_ids": list(ordered_scene_ids),
    }


def _load_partial(partial_path, signature):
    """Load `dense.json.partial`-style file. Return (results, timing_accum) or None."""
    if not os.path.isfile(partial_path):
        return None
    try:
        with open(partial_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[resume] failed to read {partial_path}: {e}; ignoring partial.")
        return None
    if data.get("signature") != signature:
        print(f"[resume] signature mismatch in {partial_path}; ignoring partial "
              f"(re-running from scratch).")
        return None
    results = data.get("all_results") or []
    timing_accum = data.get("timing_accum") or {}
    return results, timing_accum


def _save_partial(partial_path, signature, all_results, timing_accum):
    """Atomic write of partial progress. Failures are logged and swallowed."""
    tmp_path = partial_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(partial_path) or '.', exist_ok=True)
        with open(tmp_path, 'w') as f:
            json.dump({
                "signature": signature,
                "all_results": all_results,
                "timing_accum": timing_accum,
            }, f, default=str)
        os.replace(tmp_path, partial_path)
    except Exception as e:
        print(f"[resume] failed to write {partial_path}: {e}")
        try:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="SpatialBenchBenchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file (benchmark/configs/*.yaml). '
                             'CLI args override config values.')

    # Model parameters
    parser.add_argument('--model', type=str, default=None,
                        help='Model adapter name (da3, vggt, mapanything, ...)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Inference device')
    parser.add_argument('--checkpoints-dir', type=str, default=DEFAULT_CHECKPOINTS_DIR,
                        help='Where to put downloaded HuggingFace weights (e.g. DA3/VGGT/..)')

    # Data parameters
    parser.add_argument('--scene-index', type=str, default=None,
                        help='Path to scene_index.json')

    # Filter parameters
    parser.add_argument('--tags', type=str, default=None,
                        help='Tag filter (e.g., "sparse+indoor", "wrist|ego")')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Max number of scenes (for quick testing)')
    parser.add_argument('--scene-id', type=str, default=None,
                        help='Run only this single scene_id (overrides tags/max-scenes filtering '
                             'after they are applied; YAML key: scene_id).')

    # Evaluation parameters
    parser.add_argument('--eval-metrics', nargs='+',
                        choices=ALL_EVAL_METRICS, default=None,
                        help='Metric categories to evaluate (default: all). '
                             'Choices: depth, pose, trajectory, pointcloud')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Ropedia confidence threshold')
    parser.add_argument('--depth-alignment', choices=['median', 'lstsq'],
                        default='median', help='Depth alignment method')

    # Shuffle mode: only applies to end2end models
    parser.add_argument('--shuffle-seed', type=int, default=None,
                        help='Base seed for shuffling frame order within each scene. None=no shuffle (default). '
                             'Only applies to end2end models (benchmark/configs/end2end/). '
                             'When set, each scene derives a deterministic permutation from (seed + hash(scene_id)); '
                             'GT and images stay aligned.')
    parser.add_argument('--priority-datasets', nargs='+', default=None,
                        help='Move scenes of the specified source_dataset to the front of the queue. '
                             'Used in dense sweep so that high-GPU-memory scenes like ropedia run first, '
                             'so the scheduler can immediately skip to the next model on OOM.')

    # Output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output JSON path (auto-named if not set)')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Save GT and predicted point clouds as GLB files')
    parser.add_argument('--vis-conf-percent', type=float, default=20.0,
                        help='Filter out lowest N%% confidence points in visualization '
                             '(0=no filter, 50=keep top 50%%)')

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
            f"shuffling is only meaningful for end2end models "
            f"(benchmark/configs/end2end/)."
        )

    # Print the effective configuration
    print(f"\n[Config] model={args.model}, checkpoint={args.checkpoint}, "
          f"device={args.device}")
    print(f"[Config] checkpoints_dir={args.checkpoints_dir}")
    print(f"[Config] scene_index={args.scene_index}, tags={args.tags}, "
          f"max_scenes={args.max_scenes}, scene_id={args.scene_id}")
    print(f"[Config] eval_metrics={args.eval_metrics or 'all'}, "
          f"depth_alignment={args.depth_alignment}")
    if args.shuffle_seed is not None:
        print(f"[Config] shuffle_seed={args.shuffle_seed} (frame order randomized per scene)")
    if args.priority_datasets:
        print(f"[Config] priority_datasets={args.priority_datasets} (these run first)")
    if args.visualize:
        print(f"[Config] visualize=True, vis_conf_percent={args.vis_conf_percent}")

    # ---- 1. Load model ----
    print(f"\n[1/3] Loading model '{args.model}'...")
    t_model_load = time.time()
    adapter = get_adapter(args.model)
    # Inject model-specific inference params from YAML (chunk_size, overlap, etc.)
    extra = getattr(args, 'model_extra_params', {})
    if extra:
        adapter.configure(**extra)
        print(f"[Config] model_extra_params={extra}")
    adapter.load_model(checkpoint=args.checkpoint, device=args.device)
    model_load_time = time.time() - t_model_load
    print(f"  Model loaded in {model_load_time:.2f}s")

    # Count model parameters
    model_params = adapter.get_model_params()
    if model_params:
        print(f"  Parameters: {model_params['total_params_M']:.2f}M total "
              f"({model_params['trainable_params']/1e6:.2f}M trainable, "
              f"{model_params['frozen_params']/1e6:.2f}M frozen)")

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

    # Single-frame mode: selecting the single tag automatically restricts eval to depth only
    if args.tags and "single" in args.tags:
        args.eval_metrics = ["depth"]
        print(f"[Config] Single-frame mode detected, eval_metrics forced to ['depth']")

    # Resolution override (read from the config file, e.g. {height: 512, align: 16})
    resolution_override = cfg.get('resolution_override', None)

    dataset = BenchmarkDataset(
        scene_index_path=args.scene_index,
        tags=tags,
        tag_operator=tag_operator,
        max_scenes=args.max_scenes,
        conf_threshold=args.conf_threshold,
        resolution_override=resolution_override,
        shuffle_seed=args.shuffle_seed,
        priority_datasets=args.priority_datasets,
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
    # Layout: results/{run_name}/{run_name}.json + results/{run_name}/visualization/gt/ + pred/
    if args.output:
        output_path = args.output
        run_dir = os.path.dirname(output_path) or args.output_dir
    else:
        auto_name = generate_output_name(
            adapter.name(), args.scene_index, args.tags
        )
        if args.shuffle_seed is not None:
            auto_name = f"{auto_name}_shuffle{args.shuffle_seed}"
        run_dir = os.path.join(args.output_dir, auto_name)
        output_path = os.path.join(run_dir, f"{auto_name}.json")

    os.makedirs(run_dir, exist_ok=True)
    output_dir = run_dir

    # Only save input images in single-scene debug mode (--scene-id); otherwise a
    # dense sweep would write a massive number of PNGs (110 scenes x 1000+ frames x N models)
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
        print(f"  Visualization output: {os.path.join(run_dir, 'visualization')}")

    # ---- 3. Evaluation loop ----
    print(f"\n[3/3] Running evaluation on {len(dataset)} scenes...")
    all_results = []

    # Accumulate timing statistics for each stage
    timing_accum = {
        "data_loading": [], "inference": [], "eval_total": [],
        "depth_eval": [], "pose_eval": [], "pointcloud_eval": [],
        "visualization": [],
    }

    # ---- Resume: write a partial after each finished scene so a hard crash does not lose prior progress ----
    partial_path = output_path + ".partial"
    ordered_scene_ids = [s["scene_id"] for s in dataset.scenes]
    resume_signature = _resume_signature(
        adapter.name(), args.scene_index, args.tags, args.max_scenes,
        ordered_scene_ids,
    )
    done_scene_ids = set()
    loaded_partial = _load_partial(partial_path, resume_signature)
    if loaded_partial is not None:
        prev_results, prev_timing = loaded_partial
        # Only mark "successful" scenes as done; failed entries with errors are not counted and will be re-run next time
        for r in prev_results:
            if "error" not in r:
                done_scene_ids.add(r["scene_id"])
                all_results.append(r)
        for k, v in prev_timing.items():
            if k in timing_accum and isinstance(v, list):
                timing_accum[k] = list(v)
        print(f"[resume] loaded {len(done_scene_ids)} completed scenes from "
              f"{partial_path}; will skip them.")

    benchmark_start = time.time()

    for idx in range(len(dataset)):
        # Peek at scene_id (cheap) to decide whether to load the data
        cheap_scene_id = dataset.scenes[idx]["scene_id"]
        if cheap_scene_id in done_scene_ids:
            print(f"\n  [{idx+1}/{len(dataset)}] {cheap_scene_id} "
                  f"[resume] skipping (already evaluated)")
            continue

        # ---- Data loading timing ----
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
            # ---- Model inference timing ----
            t0 = time.time()
            predictions = adapter.predict(scene)
            inference_time = time.time() - t0

            # ---- Evaluation timing (internally split into depth/pose/pointcloud) ----
            t_eval = time.time()
            scene_result, eval_timing = evaluate_scene(
                scene, predictions, adapter,
                depth_alignment=args.depth_alignment,
                eval_metrics=args.eval_metrics,
                gt_mesh_path=gt_mesh_path,
            )
            total_eval_time = time.time() - t_eval

            # ---- Visualization timing ----
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

            # ---- Print per-scene timing breakdown ----
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
            if "tgm" in scene_result:
                parts.append(f"tgm={scene_result['tgm'].get('tgm', float('nan')):.4f}")
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
            # Flush the partial after every scene: a hard crash loses only the current scene
            _save_partial(partial_path, resume_signature,
                          all_results, timing_accum)
            scene = None
            predictions = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    benchmark_total = time.time() - benchmark_start

    # ---- Print global timing summary ----
    print(f"\n{'='*70}")
    print(f"  TIMING SUMMARY  ({len(all_results)} scenes)")
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

    if args.shuffle_seed is not None and isinstance(report.get("meta"), dict):
        report["meta"]["shuffle_seed"] = args.shuffle_seed

    write_json_report(report, output_path)
    write_overall(report, output_dir)
    print_summary(report)

    # Clean up the partial after the entire evaluation has been successfully persisted
    try:
        if os.path.isfile(partial_path):
            os.unlink(partial_path)
    except Exception as e:
        print(f"[resume] failed to remove {partial_path}: {e}")


if __name__ == '__main__':
    main()
