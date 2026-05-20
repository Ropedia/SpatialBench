"""
LingBot-Map Streaming model adapter.

Corresponds to lingbot-map/demo.py --mode streaming streaming inference path:
  - use lingbot_map.models.gct_stream.GCTStream
  - model.inference_streaming(images, num_scale_frames, keyframe_interval, output_device)
  - Phase 1: initial num_scale_frames frames perform bidirectional attention (scale token estimates scale)
  - Phase 2: feed the remaining frames one by one + KV cache sliding window, supports keyframe mechanism

Differences from lingbot_map_adapter (windowed by default):
  - mode fixed to streaming, hard-codes gct_stream.GCTStream
  - automatic keyframe_interval: N > 320 uses ceil(N / 320), avoid very long sequence KV cache OOM
    (matches demo.py default behavior)
  - supports offload_to_cpu: offload each frame output to CPU as it is computed, greatly reducing peak VRAM

LingBot-Map outputs:
  - pose_enc:          (B, S, 9)   [T(3), quat(4), fov_h, fov_w]
  - depth:             (B, S, H, W, 1)
  - depth_conf:        (B, S, H, W)
  - world_points:      (B, S, H, W, 3)
  - world_points_conf: (B, S, H, W)

Pose convention (following lingbot_map_adapter measured result):
  - pose_encoding_to_extri_intri returnsis actually c2w (not w2c)
  - w2c_extrinsics is obtained by closed-form inversion of c2w
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter
from benchmark.utils.paths import DEFAULT_CHECKPOINTS_DIR


@register_adapter("lingbot_map_stream")
class LingBotMapStreamAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"
        # ---- inference parameters (YAML can be overridden) ----
        self.num_scale_frames = 8          # initial scale frame count (Phase 1)
        self.keyframe_interval = None      # None=automatic: N<=320 uses 1, otherwise ceil(N/320)
        self.use_sdpa = False              # Default FlashInfer (aligned with demo.py)
        self.enable_3d_rope = True
        self.max_frame_num = 1024
        self.kv_cache_sliding_window = 64
        self.kv_cache_scale_frames = 8
        self.camera_num_iterations = 4     # camera head iteration count (4=default accuracy, 1=fastest)
        self.image_size = 518
        self.patch_size = 14
        # official checkpoint (robbyant/lingbot-map) does not contain point_head weights
        self.enable_point = False
        # per-frame outputs offloaded to CPU (required for very long sequences to avoid OOM)
        self.offload_to_cpu = False

    def name(self):
        return "LingBot-Map-Stream"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device

        # Streaming-specific GCTStream (gct_stream.py), distinct from windowed gct_stream_window.
        from lingbot_map.models.gct_stream import GCTStream

        self.model = GCTStream(
            img_size=self.image_size,
            patch_size=self.patch_size,
            enable_3d_rope=self.enable_3d_rope,
            max_frame_num=self.max_frame_num,
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.kv_cache_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=self.use_sdpa,
            enable_point=self.enable_point,
            camera_num_iterations=self.camera_num_iterations,
        )

        # checkpoint supports (local file / HF repo_id / default)
        ckpt_path = None
        if checkpoint and os.path.isfile(checkpoint):
            ckpt_path = checkpoint
        else:
            from benchmark.utils.hf_weights import ensure_hf_snapshot
            repo_id = checkpoint or "robbyant/lingbot-map"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snap = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            for fname in os.listdir(snap):
                if fname.endswith((".pt", ".bin", ".pth", ".safetensors")):
                    ckpt_path = os.path.join(snap, fname)
                    break
            assert ckpt_path is not None, f"No weights found in {snap}"

        print(f"[LingBotMapStreamAdapter] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        res = self.model.load_state_dict(state_dict, strict=False)
        print(f"[LingBotMapStreamAdapter] Loaded: missing={len(res.missing_keys)}, "
              f"unexpected={len(res.unexpected_keys)}")

        self.model = self.model.to(device).eval()

        # Cast aggregator (DINOv2-style trunk) to the inference dtype, saving 2-3 GB VRAM
        # (demo.py same approach; heads remain fp32 and are handled inside autocast)
        if torch.cuda.is_available():
            dev_cap = torch.cuda.get_device_capability()[0]
            amp_dtype = torch.bfloat16 if dev_cap >= 8 else torch.float16
            if amp_dtype != torch.float32 and getattr(self.model, "aggregator", None) is not None:
                print(f"[LingBotMapStreamAdapter] Casting aggregator to {amp_dtype}")
                self.model.aggregator = self.model.aggregator.to(dtype=amp_dtype)

        print(f"[LingBotMapStreamAdapter] GCTStream (streaming) on {device}")

    def _resolve_keyframe_interval(self, num_frames):
        """Automatic keyframe_interval selection consistent with demo.py.

        - User explicitly specified (>0): use directly
        - otherwise: N <= 320 -> 1; N > 320 -> ceil(N / 320)
        """
        if self.keyframe_interval is not None and self.keyframe_interval > 0:
            return int(self.keyframe_interval)
        if num_frames > 320:
            ki = (num_frames + 319) // 320
            print(f"[LingBotMapStreamAdapter] Auto keyframe_interval={ki} "
                  f"(num_frames={num_frames} > 320)")
            return ki
        return 1

    def predict(self, scene):
        """Run LingBot-Map streaming inference."""
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
        from lingbot_map.utils.geometry import closed_form_inverse_se3_general

        images_raw = scene['images_raw']   # (N, 3, H, W) in [0, 1]
        N, _, H, W = images_raw.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"[LingBotMapStreamAdapter] input resolution {H}x{W} must be a multiple of "
            f"patch_size={self.patch_size} (set width=518, align=14 in YAML resolution_override)"
        )

        images = images_raw.to(self.device)

        # autocast dtype (consistent with demo.py)
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        num_scale = min(self.num_scale_frames, N)
        keyframe_interval = self._resolve_keyframe_interval(N)
        output_device = torch.device("cpu") if self.offload_to_cpu else None

        if keyframe_interval > 1:
            print(f"[LingBotMapStreamAdapter] Streaming: N={N}, scale={num_scale}, "
                  f"keyframe_interval={keyframe_interval}")

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
            predictions = self.model.inference_streaming(
                images,
                num_scale_frames=num_scale,
                keyframe_interval=keyframe_interval,
                output_device=output_device,
            )

        # pose_enc (B, S, 9) → c2w (B, S, 3, 4) + intrinsic (B, S, 3, 3)
        # Note: the function name contains "extri", but LingBot-Map ckpt output is actually c2w (Measured)
        pose_enc = predictions["pose_enc"]
        # pose_enc must be moved back to the original device for matrix operations if offloaded to CPU
        pose_enc_dev = pose_enc.to(self.device) if pose_enc.device.type == "cpu" else pose_enc
        extrinsic_c2w, intrinsic = pose_encoding_to_extri_intri(
            pose_enc_dev, image_size_hw=(H, W)
        )

        # c2w → w2c via closed-form inverse
        c2w_44 = torch.zeros(
            (*extrinsic_c2w.shape[:-2], 4, 4),
            device=extrinsic_c2w.device, dtype=extrinsic_c2w.dtype,
        )
        c2w_44[..., :3, :4] = extrinsic_c2w
        c2w_44[..., 3, 3] = 1.0
        w2c_44 = closed_form_inverse_se3_general(c2w_44)
        w2c = w2c_44[..., :3, :4]

        # Remove the batch dimension
        if extrinsic_c2w.ndim == 4:
            extrinsic_c2w = extrinsic_c2w[0]
        if w2c.ndim == 4:
            w2c = w2c[0]
        if intrinsic.ndim == 4:
            intrinsic = intrinsic[0]

        result = {}
        result['pred_pose'] = extrinsic_c2w.detach().cpu().float().numpy().astype(np.float32)
        result['w2c_extrinsics'] = w2c.detach().cpu().float().numpy().astype(np.float32)
        result['pred_intrinsic'] = intrinsic.detach().cpu().float().numpy().astype(np.float32)

        # depth: depth (B, S, H, W, 1) -> (S, H, W)
        if "depth" in predictions:
            depth = predictions["depth"]
            if depth.ndim == 5 and depth.shape[-1] == 1:
                depth = depth.squeeze(-1)
            if depth.ndim == 4 and depth.shape[0] == 1:
                depth = depth[0]
            result['pred_depth'] = depth.detach().cpu().float().numpy().astype(np.float32)

        # Confidence: prefer world_points_conf (more discriminative for 3D); fall back to depth_conf.
        conf_key = "world_points_conf" if "world_points_conf" in predictions else\
                   ("depth_conf" if "depth_conf" in predictions else None)
        if conf_key is not None:
            conf = predictions[conf_key]
            if conf.ndim == 4 and conf.shape[0] == 1:
                conf = conf[0]
            result['pred_confidence'] = conf.detach().cpu().float().numpy().astype(np.float32)

        # point cloud: enable_point=False has no world_points outputs, the framework falls back to back-projection
        if "world_points" in predictions:
            wp = predictions["world_points"]
            if wp.ndim == 5 and wp.shape[0] == 1:
                wp = wp[0]
            wp_np = wp.detach().cpu().float().numpy()

            valid = np.isfinite(wp_np).all(axis=-1)
            if 'pred_confidence' in result:
                conf_np = result['pred_confidence']
                thr = np.percentile(conf_np[valid], 50) if valid.any() else 0.0
                valid = valid & (conf_np >= thr)

            if valid.any():
                pts = wp_np[valid]
                img_np = images_raw.permute(0, 2, 3, 1).numpy()
                colors = img_np[valid]
                result['pred_pointcloud'] = pts.astype(np.float32)
                result['pred_pointcloud_colors'] = colors.astype(np.float32)

        # Clear VRAM (especiallylong-sequence + KV cache residue)
        del predictions
        if hasattr(self.model, "clean_kv_cache"):
            try:
                self.model.clean_kv_cache()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def supports_metric_depth(self):
        return False   # Outputs relative scale

    def requires_intrinsics(self):
        return False

    def normalize_gt_poses(self, scene):
        """Align to the first camera and scale by average point distance, matching the VGGT family."""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )
