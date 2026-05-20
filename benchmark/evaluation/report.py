"""
Evaluation report generation and aggregation: aggregates per-scene results into a multi-axis statistical report.
"""
import json
import os
import numpy as np
from datetime import datetime
from collections import defaultdict


def aggregate_metric_dict(metric_dicts):
    """Aggregate a list of metric dicts, computing mean/median/std.

    Non-finite values (NaN / +/-Inf) are treated as bad cases and excluded. Typical source:
    when the pointcloud prediction is empty, ``compute_pointcloud_metrics`` returns
    ``float('inf')`` as a sentinel; without filtering, the whole column's mean would
    be polluted to inf and break downstream table computations.
    """
    if not metric_dicts:
        return {}

    keys = metric_dicts[0].keys()
    result = {}
    for key in keys:
        values = [d[key] for d in metric_dicts
                  if np.isfinite(d.get(key, float('nan')))]
        if values:
            result[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
            }
        else:
            result[key] = {"mean": float('nan'), "median": float('nan'), "std": float('nan')}
    return result


def generate_output_name(model_name, scene_index_path, tags=None, resolution=None):
    """Automatically generate an output filename from model, benchmark, and tags.

    Format: {model}_{benchmark}_{tags}_{WxH}_{timestamp}
    Example: vggt_droid_scenes_sparse+indoor_518x378_20260320_045411

    Args:
        model_name: model name
        scene_index_path: scene-index file path
        tags: tag filter string
        resolution: (W, H) resolution

    Returns:
        str: filename without directory or extension
    """
    parts = [model_name.lower()]

    # The benchmark name is extracted from the scene_index filename
    if scene_index_path:
        bench_name = os.path.splitext(os.path.basename(scene_index_path))[0]
        parts.append(bench_name)

    # Tags
    if tags:
        tag_str = tags.replace("|", "_or_").replace("+", "_")
        parts.append(tag_str)
    else:
        parts.append("all")

    # Resolution
    if resolution:
        parts.append(f"{resolution[0]}x{resolution[1]}")

    # Timestamp
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))

    return "_".join(parts)


def build_report(model_name, query_tags, per_scene_results, checkpoint=None, resolution=None, model_params=None):
    """Build the full evaluation report."""
    report = {
        "meta": {
            "model": model_name,
            "checkpoint": checkpoint or "",
            "query": query_tags or "all",
            "resolution": list(resolution) if resolution else [],
            "num_scenes": len(per_scene_results),
            "timestamp": datetime.now().isoformat(),
            "model_params": model_params or {},
        },
        "per_scene": per_scene_results,
    }

    # Overall aggregation
    ALL_METRIC_TYPES = ["depth_metric", "depth", "tgm", "camera", "trajectory", "pointcloud", "pointcloud_gt_pose", "pointcloud_pred_pose"]
    report["aggregate"] = {}
    for metric_type in ALL_METRIC_TYPES:
        dicts = [r[metric_type] for r in per_scene_results if metric_type in r]
        if dicts:
            report["aggregate"][metric_type] = aggregate_metric_dict(dicts)

    # Group by source_dataset
    by_dataset = defaultdict(list)
    for r in per_scene_results:
        by_dataset[r.get("source_dataset", "unknown")].append(r)

    report["per_dataset_breakdown"] = {}
    for ds, results in by_dataset.items():
        breakdown = {"num_scenes": len(results)}
        for metric_type in ALL_METRIC_TYPES:
            dicts = [r[metric_type] for r in results if metric_type in r]
            if dicts:
                breakdown[metric_type] = aggregate_metric_dict(dicts)
        report["per_dataset_breakdown"][ds] = breakdown

    # Group by each tag value
    by_tag = defaultdict(list)
    for r in per_scene_results:
        for axis, value in r.get("tags", {}).items():
            by_tag[value].append(r)

    report["per_tag_breakdown"] = {}
    for tag_value, results in by_tag.items():
        breakdown = {"num_scenes": len(results)}
        for metric_type in ALL_METRIC_TYPES:
            dicts = [r[metric_type] for r in results if metric_type in r]
            if dicts:
                breakdown[metric_type] = aggregate_metric_dict(dicts)
        report["per_tag_breakdown"][tag_value] = breakdown

    # Efficiency statistics
    timings = [r.get("timing", {}).get("inference_seconds", 0) for r in per_scene_results]
    if timings:
        report["efficiency"] = {
            "mean_inference_seconds": float(np.mean(timings)),
            "total_inference_seconds": float(np.sum(timings)),
        }

    return report


_DEPTH_KEYS = ["abs_rel", "sq_rel", "rmse", "delta_1.25", "delta_1.25^2", "delta_1.25^3",
               "inlier_1.03", "inlier_1.05", "inlier_1.10"]
_TGM_KEYS = ["tgm"]
_CAMERA_KEYS = ["racc_3", "racc_5", "tacc_3", "tacc_5", "auc_5", "auc_15", "auc_30"]
_TRAJECTORY_KEYS = ["ate", "rpe_t", "rpe_r"]
_POINTCLOUD_KEYS = ["f_score", "overall"]
_METRIC_GROUP_SPEC = [
    ("depth_metric", _DEPTH_KEYS),
    ("depth", _DEPTH_KEYS),
    ("tgm", _TGM_KEYS),
    ("camera", _CAMERA_KEYS),
    ("trajectory", _TRAJECTORY_KEYS),
    ("pointcloud", _POINTCLOUD_KEYS),
    ("pointcloud_gt_pose", _POINTCLOUD_KEYS),
    ("pointcloud_pred_pose", _POINTCLOUD_KEYS),
]


def _summary_from_agg(agg):
    """Extract mean values from an aggregate dict in a fixed group/key order and return a nested dict.

    Used for both the overall top-level metrics and the per_dataset per-dataset summary,
    ensuring the same granularity at both places.
    """
    summary = {}
    for group, keys in _METRIC_GROUP_SPEC:
        g = agg.get(group)
        if not g:
            continue
        picked = {k: round(g[k]["mean"], 4) for k in keys if k in g}
        if picked:
            summary[group] = picked
    return summary


def build_overall(report):
    """Extract a condensed overall metric summary from the full report.

    Keeps only the mean values of the most important aggregated metrics for fast model comparison.
    per_dataset and the top level share the same metric-extraction logic to guarantee
    complete metrics for every dataset.

    Returns:
        dict: condensed report
    """
    meta = report["meta"]
    agg = report.get("aggregate", {})

    overall = {
        "model": meta["model"],
        "checkpoint": meta["checkpoint"],
        "query": meta["query"],
        "num_scenes": meta["num_scenes"],
        "resolution": meta["resolution"],
        "timestamp": meta["timestamp"],
        "model_params": meta.get("model_params", {}),
    }

    # Top-level metrics: full set of means for depth/tgm/camera/trajectory/pointcloud_*
    overall.update(_summary_from_agg(agg))

    # Efficiency
    eff = report.get("efficiency", {})
    if eff:
        overall["mean_inference_seconds"] = round(eff.get("mean_inference_seconds", 0), 3)

    # Per-dataset breakdown: same structure as top level, each dataset has the full group/key means
    per_ds = report.get("per_dataset_breakdown", {})
    if per_ds:
        overall["per_dataset"] = {}
        for ds, bd in per_ds.items():
            ds_summary = {"num_scenes": bd.get("num_scenes", 0)}
            ds_summary.update(_summary_from_agg(bd))
            overall["per_dataset"][ds] = ds_summary

    return overall


def write_json_report(report, output_path):
    """Write the report to a JSON file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {output_path}")


def write_overall(report, output_dir):
    """Append the condensed overall report to output_dir/overall.json.

    overall.json is a list; each evaluation appends one record, making it easy
    to compare multiple models side by side.
    """
    overall = build_overall(report)
    overall_path = os.path.join(output_dir, "overall.json")

    # Read existing records
    existing = []
    if os.path.isfile(overall_path):
        try:
            with open(overall_path, 'r') as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
        except (json.JSONDecodeError, Exception):
            existing = []

    existing.append(overall)

    os.makedirs(output_dir, exist_ok=True)
    with open(overall_path, 'w') as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"Overall summary appended to {overall_path}")


def print_summary(report):
    """Print a concise summary of the evaluation results."""
    meta = report["meta"]
    print(f"\n{'='*60}")
    print(f"Model: {meta['model']}  |  Query: {meta['query']}  |  Scenes: {meta['num_scenes']}")
    params = meta.get("model_params", {})
    if params:
        print(f"Params: {params.get('total_params_M', 0):.2f}M total "
              f"({params.get('trainable_params', 0)/1e6:.2f}M trainable, "
              f"{params.get('frozen_params', 0)/1e6:.2f}M frozen)")
    print(f"{'='*60}")

    agg = report.get("aggregate", {})

    depth_display_keys = ['abs_rel', 'rmse', 'delta_1.25', 'inlier_1.03', 'inlier_1.05', 'inlier_1.10']
    for depth_type, label in [("depth_metric", "Depth Metrics (Metric, no alignment)"),
                               ("depth", "Depth Metrics (Aligned)")]:
        if depth_type in agg:
            d = agg[depth_type]
            print(f"\n[{label}]")
            for key in depth_display_keys:
                if key in d:
                    print(f"  {key:>15s}: mean={d[key]['mean']:.4f}  median={d[key]['median']:.4f}")

    if "tgm" in agg and "tgm" in agg["tgm"]:
        t = agg["tgm"]["tgm"]
        print(f"\n[Temporal Geometric Motion]")
        print(f"  {'tgm':>15s}: mean={t['mean']:.4f}  median={t['median']:.4f}")

    if "camera" in agg:
        c = agg["camera"]
        print(f"\n[Camera Pose Metrics]")
        for key in ['racc_3', 'racc_5', 'tacc_3', 'tacc_5', 'auc_3', 'auc_5', 'auc_15', 'auc_30']:
            if key in c:
                print(f"  {key:>15s}: mean={c[key]['mean']:.4f}  median={c[key]['median']:.4f}")

    if "trajectory" in agg:
        t = agg["trajectory"]
        print(f"\n[Trajectory Metrics (Sim(3) aligned)]")
        for key in ['ate', 'rpe_t', 'rpe_r']:
            if key in t:
                print(f"  {key:>15s}: mean={t[key]['mean']:.4f}  median={t[key]['median']:.4f}")

    pc_display_keys = ['f_score', 'overall']
    for pc_type, label in [("pointcloud_gt_pose", "Point Cloud (GT Pose / recon_posed)"),
                           ("pointcloud_pred_pose", "Point Cloud (Pred Pose / recon_unposed)"),
                           ("pointcloud", "Point Cloud (Direct)")]:
        if pc_type in agg:
            p = agg[pc_type]
            print(f"\n[{label}]")
            for key in pc_display_keys:
                if key in p:
                    print(f"  {key:>18s}: mean={p[key]['mean']:.4f}  median={p[key]['median']:.4f}")

    # Per-dataset breakdown
    breakdown = report.get("per_dataset_breakdown", {})
    if breakdown:
        print(f"\n[Per-Dataset Breakdown]")
        for ds, bd in breakdown.items():
            parts = [f"{ds} ({bd['num_scenes']} scenes)"]
            if "depth_metric" in bd and "abs_rel" in bd["depth_metric"]:
                parts.append(f"abs_rel_metric={bd['depth_metric']['abs_rel']['mean']:.4f}")
            if "depth" in bd and "abs_rel" in bd["depth"]:
                parts.append(f"abs_rel={bd['depth']['abs_rel']['mean']:.4f}")
            if "tgm" in bd and "tgm" in bd["tgm"]:
                parts.append(f"tgm={bd['tgm']['tgm']['mean']:.4f}")
            if "camera" in bd and "racc_5" in bd["camera"]:
                parts.append(f"racc_5={bd['camera']['racc_5']['mean']:.4f}")
            if "camera" in bd and "tacc_5" in bd["camera"]:
                parts.append(f"tacc_5={bd['camera']['tacc_5']['mean']:.4f}")
            if "trajectory" in bd and "ate" in bd["trajectory"]:
                parts.append(f"ate={bd['trajectory']['ate']['mean']:.4f}")
            if "pointcloud_gt_pose" in bd:
                pg = bd["pointcloud_gt_pose"]
                if "f_score" in pg:
                    parts.append(f"f_gt={pg['f_score']['mean']:.4f}")
                if "overall" in pg:
                    parts.append(f"ov_gt={pg['overall']['mean']:.4f}")
            if "pointcloud_pred_pose" in bd:
                pp = bd["pointcloud_pred_pose"]
                if "f_score" in pp:
                    parts.append(f"f_pred={pp['f_score']['mean']:.4f}")
                if "overall" in pp:
                    parts.append(f"ov_pred={pp['overall']['mean']:.4f}")
            if "pointcloud" in bd:
                pd = bd["pointcloud"]
                if "f_score" in pd:
                    parts.append(f"f_pc={pd['f_score']['mean']:.4f}")
                if "overall" in pd:
                    parts.append(f"ov_pc={pd['overall']['mean']:.4f}")
            print(f"  {'  |  '.join(parts)}")

    print(f"\n{'='*60}\n")
