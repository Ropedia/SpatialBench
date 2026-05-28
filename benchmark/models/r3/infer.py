"""Run R3 inference and save outputs for the viewer."""

import os

# Reduce CUDA memory fragmentation with expandable segments
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import torch
import numpy as np
import cv2
import imageio.v2 as iio
from omegaconf import OmegaConf

from R3.models.r3 import R3 as DA3Wrapper
from R3.utils.pose_enc import pose_encoding_to_extri_intri
from depth_anything_3.utils.geometry import affine_inverse
from R3.utils.config_resolve import resolve_model_config
from R3.utils.input_io import parse_seq_path, prepare_image_views


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference and save outputs for the viewer.")
    parser.add_argument("--seq_path", type=str, default="examples/indoor", help="Path to image dir or video")
    parser.add_argument("--output_dir", type=str, default="scratch/infer/default", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--size", type=int, default=504, help="Image resize target")
    parser.add_argument("--config_name", type=str, default="r3-large")
    parser.add_argument("--ckpt", type=str, default="ckpt/r3.safetensors", help="Explicit checkpoint path. If empty, auto-resolved from --config_name.")
    parser.add_argument("--max_frames", type=int, default=0, help="Limit number of frames (0=all)")
    parser.add_argument("--frame_stride", type=int, default=1, help="Use every N-th frame from input")
    parser.add_argument(
        "--wrapper_mode",
        type=str,
        default="online",
        choices=["online", "offline"],
        help="Use online streaming reconstruction or offline batch pose reconstruction",
    )
    parser.add_argument(
        "--rel_pose_reconstruction_method",
        type=str,
        default="greedy",
        choices=["greedy", "pgo"],
        help="Method used to reconstruct absolute poses from relative pose predictions",
    )
    parser.add_argument(
        "--rel_pose_topn_conf",
        type=int,
        default=999,
        help="Number of highest-confidence incoming edges to keep during reconstruction",
    )
    parser.add_argument(
        "--rel_pose_score_mode",
        type=str,
        default="auto",
        choices=["auto", "shared", "mean", "min", "translation", "rotation", "separate"],
        help="How to score relative-pose edges when split translation/rotation confidence is available",
    )
    parser.add_argument(
        "--pgo_num_iters",
        type=int,
        default=100,
        help="Requested number of PGO refinement iterations",
    )
    parser.add_argument("--pgo_lr", type=float, default=0.05, help="PGO optimizer learning rate hint")
    parser.add_argument(
        "--pgo_weight_T",
        type=float,
        default=1.0,
        help="Translation weight inside the PGO objective",
    )
    parser.add_argument(
        "--pgo_weight_R",
        type=float,
        default=0.5,
        help="Rotation weight inside the PGO objective",
    )
    parser.add_argument(
        "--pgo_weight_fl",
        type=float,
        default=0.1,
        help="FoV weight inside the PGO objective",
    )
    parser.add_argument(
        "--pgo_init_prior_weight",
        type=float,
        default=1e-4,
        help="Global prior weight that keeps refined poses near the greedy initialization",
    )
    parser.add_argument(
        "--pgo_keyframe_stride",
        type=int,
        default=0,
        help="If > 0, keep every Nth greedy pose fixed during PGO as a keyframe scaffold",
    )
    parser.add_argument(
        "--edge_percentile_cutoff",
        type=float,
        default=0.0,
        help="Drop the lowest-confidence fraction of PGO edges before optimization",
    )
    parser.add_argument("--pgo_geman_mcclure_c", type=float, default=0.0, help="Geman-McClure c (0=disabled)")
    parser.add_argument("--pgo_dcs_phi", type=float, default=0.0, help="DCS phi (0=disabled)")
    parser.add_argument(
        "--pgo_max_translation_per_frame", type=float, default=0.0, help="Physical plausibility filter (0=disabled)"
    )
    parser.add_argument(
        "--attention_mode", type=str, default="", help="Override attention mode (causal/window/window_wo_sink)"
    )
    parser.add_argument("--attention_window_size", type=int, default=0, help="Window size for window attention modes")
    parser.add_argument(
        "--online_kv_cache_mode",
        type=str,
        default="all",
        choices=["all", "dynamic"],
        help="Online KV-cache retention mode: all keeps every frame, dynamic keeps keyframes plus a recent window",
    )
    parser.add_argument(
        "--online_kv_backend",
        type=str,
        default="dense",
        choices=["dense", "paged"],
        help="KV cache storage backend. 'paged' uses flashinfer paged attention (mode='causal' only).",
    )
    parser.add_argument(
        "--flashinfer_page_size",
        type=int,
        default=0,
        help="Page size for paged KV backend (0 = backend default).",
    )
    parser.add_argument(
        "--online_recent_frames",
        type=int,
        default=0,
        help="Number of recent frames to retain when online_kv_cache_mode=dynamic",
    )
    parser.add_argument(
        "--bootstrap_full_attention_frames",
        type=int,
        default=0,
        help="Run a single full-attention pass on the first N frames to seed online all-KV state (0 disables).",
    )
    parser.add_argument(
        "--online_verbose",
        action="store_true",
        help="Enable online diagnostic logging during sequential inference",
    )
    parser.add_argument(
        "--bank_initial_frames",
        type=int,
        default=1,
        help="Number of initial real frames that stay resident in dynamic mode",
    )
    parser.add_argument(
        "--keyframe_mode",
        type=str,
        default="interval",
        choices=["interval", "novelty"],
        help="Keyframe selection mode used by dynamic retention and fallback registry",
    )
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=10,
        help="Store every Nth frame when keyframe_mode=interval",
    )
    parser.add_argument(
        "--keyframe_novelty_threshold",
        type=float,
        default=0.985,
        help="Add a keyframe when max similarity to the bank stays below this threshold",
    )
    parser.add_argument(
        "--keyframe_max_interval",
        type=int,
        default=30,
        help="Force a keyframe after this many frames without one in novelty mode",
    )
    parser.add_argument(
        "--keyframe_max_keyframes",
        type=int,
        default=100,
        help="Maximum number of keyframes tracked by the registry",
    )
    parser.add_argument(
        "--keyframe_pose_confidence_ratio",
        type=float,
        default=0.0,
        help="Gate keyframe admission at reliability >= ratio * warmup_mean (0 disables).",
    )
    parser.add_argument(
        "--pose_max_recent",
        type=int,
        default=0,
        help="Local chain: only use N most recent frames for pose averaging (0=all)",
    )
    # Confidence-based fallback.
    parser.add_argument(
        "--online_fallback_enabled",
        action="store_true",
        help="Enable confidence-based fallback with KV flush + bridge re-run + scale alignment",
    )
    parser.add_argument(
        "--fallback_drought_length", type=int, default=3, help="Consecutive low-confidence frames to trigger fallback"
    )
    parser.add_argument(
        "--fallback_drought_threshold",
        type=float,
        default=1.0,
        help="Confidence threshold for fallback drought detection (softplus space)",
    )
    parser.add_argument(
        "--fallback_drought_threshold_pct",
        type=float,
        default=50.0,
        help="Fallback drought threshold as a percentage of the warmup mean confidence from the current segment (0 to disable)",
    )
    parser.add_argument(
        "--fallback_drought_warmup_frames",
        type=int,
        default=5,
        help="Number of early segment frames used to build the percentage-threshold warmup mean; initial frame-0 reference is skipped",
    )
    parser.add_argument(
        "--fallback_num_bridge_frames", type=int, default=5, help="Number of bridge frames to re-run after fallback"
    )
    parser.add_argument(
        "--fallback_min_bridge_baseline_ratio",
        type=float,
        default=0.0,
        help="Parallax gate ratio. When >0, walk back through good frames and only accept those whose translation delta exceeds ratio * 75th-percentile baseline.",
    )
    parser.add_argument(
        "--fallback_max_bridge_lookback",
        type=int,
        default=0,
        help="Max frames to look back when parallax-gated bridge selection is enabled (0 = num_bridge_frames).",
    )
    parser.add_argument(
        "--evict_low_conf_threshold",
        type=float,
        default=0.0,
        help="Evict frames from KV cache when post_score is below this threshold (0 to disable)",
    )
    parser.add_argument(
        "--evict_low_conf_threshold_pct",
        type=float,
        default=0.0,
        help="Evict frames below this percentage of the warmup confidence baseline; warmup skips the segment anchor/frame 0 (0 to disable)",
    )
    parser.add_argument(
        "--evict_low_conf_warmup_frames",
        type=int,
        default=3,
        help="Number of initial segment frames used for relative low-confidence eviction baseline; the segment anchor/frame 0 is skipped",
    )
    parser.add_argument(
        "--fallback_ref_mode",
        type=str,
        default="bridge",
        choices=["bridge", "keyframe"],
        help="How to select ref frame during fallback",
    )
    parser.add_argument(
        "--min_segment_frames",
        type=int,
        default=0,
        help="Minimum frames in a segment before drought can trigger fallback",
    )
    parser.add_argument(
        "--max_segment_frames",
        type=int,
        default=0,
        help="Force fallback when segment exceeds this length to bound KV cache (0=disabled)",
    )
    parser.add_argument(
        "--fallback_replay_attention",
        type=str,
        default="full",
        choices=["full", "causal"],
        help="Attention mode for bridge replay (full=all frames see each other, causal=sequential)",
    )
    parser.add_argument(
        "--fallback_skip_confidence_check",
        action="store_true",
        help="Always accept the fallback replay even if bridge confidence does not improve",
    )
    parser.add_argument(
        "--disable_segment_pgo",
        action="store_true",
        help="Disable segment PGO after each accepted fallback",
    )
    parser.add_argument(
        "--depth_scale_mode",
        type=str,
        default="ransac",
        choices=["ransac", "huber", "huber_conf", "weighted_median"],
        help="Depth scale estimation mode: ransac, huber (no conf weights), or huber_conf (with conf weights)",
    )
    parser.add_argument(
        "--fallback_debug_dir",
        type=str,
        default="",
        help="If set, save fallback depth comparison figures to this directory",
    )
    parser.add_argument(
        "--metric_scale_enabled",
        action="store_true",
        help="Load DA3-metric alongside the main model and anchor scale at frame 0 + after each fallback",
    )
    parser.add_argument(
        "--metric_model_name",
        type=str,
        default="depth-anything/DA3METRIC-LARGE",
        help="HuggingFace repo id for the DA3 metric model",
    )
    parser.add_argument(
        "--metric_min_conf",
        type=float,
        default=1.02,
        help="Min main-model depth confidence (softplus space) for a pixel to count in metric-scale median",
    )
    parser.add_argument(
        "--metric_bootstrap_frames",
        type=int,
        default=1,
        help="Pool the metric-scale anchor across the first N frames (>=2 batches DA3-metric on those frames)",
    )
    parser.add_argument(
        "--runtime_stats_every",
        type=int,
        default=0,
        help="Write FPS/VRAM stats every N online frames to runtime_stats.jsonl (0 disables).",
    )
    parser.add_argument(
        "--compute_sky_mask",
        action="store_true",
        help="Export sky/non-sky masks when the selected model emits a sky tensor.",
    )
    parser.add_argument("--sky_mask_threshold", type=float, default=0.3, help="Sky mask threshold.")
    a = parser.parse_args()

    seq_path = a.seq_path
    output_dir = a.output_dir
    device = a.device
    size = a.size
    config_name = a.config_name

    # Resolve config and checkpoint
    repo_root = os.path.dirname(__file__)
    model_config = resolve_model_config(config_name=config_name, checkpoint_dir=a.ckpt, repo_root=repo_root)
    print(f"Using configuration: {model_config['config']}")

    cfg = OmegaConf.load(model_config["config"])
    if "model" in cfg and "net" in cfg.model and "da3_cfg" in cfg.model.net:
        print("Extracting da3_cfg from experiment config...")
        da3_cfg = cfg.model.net.da3_cfg
    else:
        da3_cfg = cfg

    # Override attention mode in da3_cfg if requested
    if a.attention_mode:
        da3_cfg.net.attention_mode = a.attention_mode
        print(f"Overriding attention_mode to: {a.attention_mode}")
    if a.attention_window_size > 0:
        da3_cfg.net.attention_window_size = a.attention_window_size
        print(f"Overriding attention_window_size to: {a.attention_window_size}")

    # Create model in online mode
    print("Loading model...")
    model = DA3Wrapper(
        da3_cfg=da3_cfg,
        teacher_embed_dim=2048,
        student_embed_dim=2048,
        freeze="none",
        online_mode=(a.wrapper_mode == "online"),
        online_kv_cache_mode=a.online_kv_cache_mode,
        online_kv_backend=a.online_kv_backend,
        flashinfer_page_size=a.flashinfer_page_size,
        online_recent_frames=a.online_recent_frames,
        online_verbose=(True if a.online_verbose else None),
        bank_initial_frames=a.bank_initial_frames,
        keyframe_mode=a.keyframe_mode,
        keyframe_interval=a.keyframe_interval,
        keyframe_novelty_threshold=a.keyframe_novelty_threshold,
        keyframe_max_interval=a.keyframe_max_interval,
        keyframe_max_keyframes=a.keyframe_max_keyframes,
        keyframe_pose_confidence_ratio=a.keyframe_pose_confidence_ratio,
        online_fallback_enabled=a.online_fallback_enabled,
        drought_length=a.fallback_drought_length,
        drought_threshold=a.fallback_drought_threshold,
        drought_threshold_pct=a.fallback_drought_threshold_pct,
        drought_threshold_warmup_frames=a.fallback_drought_warmup_frames,
        num_bridge_frames=a.fallback_num_bridge_frames,
        min_bridge_baseline_ratio=a.fallback_min_bridge_baseline_ratio,
        max_bridge_lookback=a.fallback_max_bridge_lookback,
        evict_low_conf_threshold=a.evict_low_conf_threshold,
        evict_low_conf_threshold_pct=a.evict_low_conf_threshold_pct,
        evict_low_conf_warmup_frames=a.evict_low_conf_warmup_frames,
        fallback_ref_mode=a.fallback_ref_mode,
        min_segment_frames=a.min_segment_frames,
        max_segment_frames=a.max_segment_frames,
        fallback_replay_attention=a.fallback_replay_attention,
        fallback_skip_confidence_check=a.fallback_skip_confidence_check,
        depth_scale_mode=a.depth_scale_mode,
        disable_segment_pgo=a.disable_segment_pgo,
        metric_scale_enabled=a.metric_scale_enabled,
        metric_model_name=a.metric_model_name,
        metric_min_conf=a.metric_min_conf,
        metric_bootstrap_frames=a.metric_bootstrap_frames,
        compute_sky_mask=a.compute_sky_mask,
        sky_mask_threshold=a.sky_mask_threshold,
    ).to(device)
    if a.fallback_debug_dir:
        model.fallback_debug_dir = a.fallback_debug_dir

    ckpt_path = model_config["ckpt_path"]
    print(f"Loading checkpoint from {ckpt_path}...")
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("net."):
            key = key[len("net.") :]
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = "da3." + key[len("model.") :]
        if not key.startswith("da3.") and not key.startswith("projections."):
            if ("da3." + key) in model.state_dict():
                key = "da3." + key
        new_state_dict[key] = value

    filtered = {k: v for k, v in new_state_dict.items() if k in model.state_dict()}
    print(f"Matched {len(filtered)} keys from checkpoint out of {len(model.state_dict())} model keys.")
    model.load_state_dict(filtered, strict=False)
    model.eval()

    # Prepare input
    img_paths, tmpdirname = parse_seq_path(seq_path)
    # Apply frame_stride and max_frames
    if a.frame_stride > 1:
        img_paths = img_paths[:: a.frame_stride]
    if a.max_frames > 0:
        img_paths = img_paths[: a.max_frames]
    print(f"Using {len(img_paths)} images (stride={a.frame_stride}, max={a.max_frames or 'all'})")
    views = prepare_image_views(img_paths, size, revisit=0, update=True)
    if tmpdirname is not None:
        import shutil

        shutil.rmtree(tmpdirname)

    # Save original images before model preprocessing (model applies ImageNet normalization)
    # Input views have images in [-1, 1] range from load_images (Normalize(0.5, 0.5, 0.5))
    original_imgs = [0.5 * (view["img"].squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) for view in views]

    # Build batch (list of view dicts).
    batch = views
    for view in batch:
        for k in view:
            if isinstance(view[k], torch.Tensor):
                view[k] = view[k].to(device)

    # Run inference — reset peak-memory counter so we capture this run's peak only.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        baseline_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        baseline_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Running {a.wrapper_mode} inference on {len(batch)} frames...")
    runtime_stats_path = ""
    if a.runtime_stats_every > 0:
        os.makedirs(output_dir, exist_ok=True)
        runtime_stats_path = os.path.join(output_dir, "runtime_stats.jsonl")
        open(runtime_stats_path, "w", encoding="utf-8").close()
    start = time.time()
    rel_pose_reconstruction_kwargs = {
        "topn_conf": a.rel_pose_topn_conf,
        "score_mode": a.rel_pose_score_mode,
        "pgo_num_iters": a.pgo_num_iters,
        "pgo_lr": a.pgo_lr,
        "pgo_weight_T": a.pgo_weight_T,
        "pgo_weight_R": a.pgo_weight_R,
        "pgo_weight_fl": a.pgo_weight_fl,
        "pgo_init_prior_weight": a.pgo_init_prior_weight,
        "pgo_keyframe_stride": a.pgo_keyframe_stride,
        "edge_percentile_cutoff": a.edge_percentile_cutoff,
        "geman_mcclure_c": a.pgo_geman_mcclure_c,
        "dcs_phi": a.pgo_dcs_phi,
        "max_translation_per_frame": a.pgo_max_translation_per_frame,
    }
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.clear_online_state()
            if a.wrapper_mode == "offline":
                predictions = model(
                    batch,
                    mode="causal",
                    use_ray_pose=False,
                    rel_pose_reconstruction_method=a.rel_pose_reconstruction_method,
                    rel_pose_reconstruction_kwargs=rel_pose_reconstruction_kwargs,
                )
            else:
                predictions = model(
                    batch,
                    use_ray_pose=False,
                    pose_max_recent=a.pose_max_recent,
                    bootstrap_full_attention_frames=a.bootstrap_full_attention_frames,
                    runtime_stats_every=a.runtime_stats_every,
                    runtime_stats_path=runtime_stats_path,
                    rel_pose_reconstruction_method=a.rel_pose_reconstruction_method,
                    rel_pose_reconstruction_kwargs=rel_pose_reconstruction_kwargs,
                )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    elapsed = time.time() - start
    print(f"Inference done in {elapsed:.1f}s ({len(batch) / elapsed:.1f} fps)")
    if torch.cuda.is_available():
        print(
            f"GPU mem | baseline: alloc={baseline_alloc:.2f}GB reserved={baseline_reserved:.2f}GB"
            f" | peak: alloc={peak_alloc:.2f}GB reserved={peak_reserved:.2f}GB"
            f" | delta_alloc={peak_alloc - baseline_alloc:.2f}GB"
        )

    # Extract pose and depth
    H, W = predictions["images"].shape[-2:]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))
    c2w = affine_inverse(extrinsic)  # w2c -> c2w

    S = predictions["depth"].shape[1]
    output_frame_ids = list(predictions.get("output_frame_ids", range(S)))
    if len(output_frame_ids) != S:
        raise ValueError(
            f"output_frame_ids length {len(output_frame_ids)} does not match prediction length {S}"
        )

    # Extract per-frame pose confidence from frame_post_scores.
    # Frame 0 has no incoming relative-pose edge, so its confidence is undefined.
    # The online runtime may keep an internal high sentinel for the reference frame;
    # do not export that sentinel as a measured confidence value.
    frame_post_scores = predictions.get("frame_post_scores", {})
    pose_conf_per_frame = np.full(S, np.nan, dtype=np.float32)
    for out_idx, fid in enumerate(output_frame_ids):
        if int(fid) == 0:
            continue
        score = frame_post_scores.get(fid, 0.0)
        pose_conf_per_frame[out_idx] = score if np.isfinite(score) else 0.0

    # Save outputs
    sky_mask = predictions.get("sky_mask")
    non_sky_mask = predictions.get("non_sky_mask")
    raw_sky = predictions.get("sky")
    has_sky_output = any(isinstance(x, torch.Tensor) for x in (sky_mask, non_sky_mask, raw_sky))

    output_subdirs = ["depth", "depth_vis", "conf", "color", "camera"]
    if has_sky_output:
        output_subdirs.append("sky_mask")
        if isinstance(non_sky_mask, torch.Tensor):
            output_subdirs.append("non_sky_mask")
        if isinstance(raw_sky, torch.Tensor):
            output_subdirs.append("sky")

    for sub in output_subdirs:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # Dump all run parameters for reproducibility
    import json as _json

    run_params = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    run_params["rel_pose_reconstruction_kwargs"] = rel_pose_reconstruction_kwargs
    run_params["num_frames"] = S
    run_params["output_frame_ids"] = [int(fid) for fid in output_frame_ids]
    run_params["inference_time_s"] = round(time.time() - start, 1)
    with open(os.path.join(output_dir, "run_params.json"), "w") as _f:
        _json.dump(run_params, _f, indent=2, default=str)

    for i in range(S):
        src_i = int(output_frame_ids[i])
        if src_i < 0 or src_i >= len(original_imgs):
            raise ValueError(
                f"output_frame_ids[{i}]={src_i} is outside loaded image range {len(original_imgs)}"
            )
        depth = predictions["depth"][0, i, ..., 0].cpu().numpy()
        conf = predictions["depth_conf"][0, i].cpu().numpy()
        img = np.clip(original_imgs[src_i], 0, 1)
        pose = c2w[0, i].cpu().numpy()
        intrins = intrinsic[0, i].cpu().numpy()

        np.save(os.path.join(output_dir, "depth", f"{i:06d}.npy"), depth)
        np.save(os.path.join(output_dir, "conf", f"{i:06d}.npy"), conf)
        iio.imwrite(os.path.join(output_dir, "color", f"{i:06d}.png"), (img * 255).astype(np.uint8))
        np.savez(os.path.join(output_dir, "camera", f"{i:06d}.npz"), pose=pose, intrinsics=intrins)

        if isinstance(sky_mask, torch.Tensor):
            mask = sky_mask[0, i].cpu().numpy().astype(np.uint8)
            np.save(os.path.join(output_dir, "sky_mask", f"{i:06d}.npy"), mask.astype(bool))
            iio.imwrite(os.path.join(output_dir, "sky_mask", f"{i:06d}.png"), mask * 255)
        elif isinstance(raw_sky, torch.Tensor):
            mask = (raw_sky[0, i].cpu().numpy() >= a.sky_mask_threshold).astype(np.uint8)
            np.save(os.path.join(output_dir, "sky_mask", f"{i:06d}.npy"), mask.astype(bool))
            iio.imwrite(os.path.join(output_dir, "sky_mask", f"{i:06d}.png"), mask * 255)

        if isinstance(non_sky_mask, torch.Tensor):
            mask = non_sky_mask[0, i].cpu().numpy().astype(np.uint8)
            np.save(os.path.join(output_dir, "non_sky_mask", f"{i:06d}.npy"), mask.astype(bool))
            iio.imwrite(os.path.join(output_dir, "non_sky_mask", f"{i:06d}.png"), mask * 255)

        if isinstance(raw_sky, torch.Tensor):
            sky = raw_sky[0, i].cpu().numpy()
            np.save(os.path.join(output_dir, "sky", f"{i:06d}.npy"), sky)

        # Save depth visualization as colored PNG
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            d_norm = depth.copy()
            vmin, vmax = np.percentile(depth[valid], (2, 98))
            d_norm = np.clip((d_norm - vmin) / (vmax - vmin + 1e-8), 0, 1)
            d_colored = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.imwrite(os.path.join(output_dir, "depth_vis", f"{i:06d}.png"), d_colored)

        if (i + 1) % 50 == 0:
            print(f"  Saved {i + 1}/{S} frames")

    # Save pose confidence log
    np.save(os.path.join(output_dir, "pose_conf.npy"), pose_conf_per_frame)

    # Save pose edge log if available (for bridge/anchor visualization)
    if hasattr(model, "_pose_edge_log"):
        edges = list(model._pose_edge_log.get_all_edges())
        # Append every prior accepted fallback's bridge edges so visualization can circle
        # all bridges across the run, not just the most recent one.
        historical = list(getattr(model, "_historical_bridge_edges", []))
        if historical:
            edges = historical + edges
        frame_id_to_output_idx = {
            int(frame_id): out_idx for out_idx, frame_id in enumerate(output_frame_ids)
        }
        if edges:
            edge_records = [
                {
                    "frame_i": frame_id_to_output_idx[int(e.frame_i)],
                    "frame_j": frame_id_to_output_idx[int(e.frame_j)],
                    "confidence": e.confidence,
                    "edge_type": e.edge_type,
                }
                for e in edges
                if int(e.frame_i) in frame_id_to_output_idx
                and int(e.frame_j) in frame_id_to_output_idx
            ]
            import json

            if edge_records:
                with open(os.path.join(output_dir, "pose_edge_log.json"), "w") as f:
                    json.dump(edge_records, f)
                type_counts = {}
                for e in edge_records:
                    type_counts[e["edge_type"]] = type_counts.get(e["edge_type"], 0) + 1
                print(f"  Pose edge log saved: {len(edge_records)} edges {type_counts}")

    print(f"\nAll {S} frames saved to {output_dir}")
    if has_sky_output:
        print("Sky mask outputs saved.")
    elif a.compute_sky_mask:
        print("Sky mask requested, but this model did not emit a sky tensor.")
    measured_pose_conf = pose_conf_per_frame[np.isfinite(pose_conf_per_frame)]
    print("\nPer-frame pose confidence (avg incoming rel_pose_conf; reference frame excluded):")
    if measured_pose_conf.size:
        print(
            f"  mean={measured_pose_conf.mean():.4f}  min={measured_pose_conf.min():.4f}  max={measured_pose_conf.max():.4f}"
        )
    else:
        print("  no measured pose confidence values")


if __name__ == "__main__":
    main()
