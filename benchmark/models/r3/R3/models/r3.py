import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from omegaconf import DictConfig

from depth_anything_3.model.da3 import DepthAnything3Net
import torchvision.transforms as T
from R3.models.online.revisit import resolve_online_verbose
from R3.models.r3_wrapper import (
    R3OnlineInferenceMixin,
    R3OutputMixin,
    R3SetupMixin,
)

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class R3(R3OnlineInferenceMixin, R3OutputMixin, R3SetupMixin, nn.Module):
    def __init__(
        self,
        da3_cfg: Union[DictConfig, dict, str],
        teacher_embed_dim: Optional[int] = None,
        student_embed_dim: Optional[int] = None,
        freeze: str = "none",
        export_feat_layers: List[int] = None,
        disable_depth_head: bool = False,
        online_mode: bool = True,
        online_kv_cache_mode: str = "all",
        online_kv_backend: str = "dense",
        flashinfer_page_size: int = 0,
        online_recent_frames: int = 5,
        bank_initial_frames: int = 1,
        keyframe_mode: str = "interval",
        keyframe_interval: int = 10,
        keyframe_novelty_threshold: float = 0.985,
        keyframe_max_interval: int = 20,
        keyframe_max_keyframes: int = 100,
        keyframe_pose_confidence_ratio: float = 0.0,
        online_verbose: Optional[bool] = True,
        online_memory_verbose: Optional[bool] = None,
        online_finalize_pose_reconstruction: bool = False,
        rel_pose_reconstruction_method: str = "greedy",
        rel_pose_reconstruction_kwargs: Optional[Dict] = None,
        global_lr_scale: float = 1.0,
        online_fallback_enabled: bool = False,
        drought_length: int = 3,
        drought_threshold: float = 1.0,
        drought_threshold_pct: float = 0.0,
        drought_threshold_warmup_frames: int = 5,
        num_bridge_frames: int = 10,
        min_bridge_baseline_ratio: float = 0.0,
        max_bridge_lookback: int = 0,
        fallback_scale_epsilon: float = 1e-4,
        evict_low_conf_threshold: float = 0.0,
        evict_low_conf_threshold_pct: float = 0.0,
        evict_low_conf_warmup_frames: int = 3,
        fallback_ref_mode: str = "bridge",
        min_segment_frames: int = 0,
        max_segment_frames: int = 250,
        fallback_replay_attention: str = "full",
        fallback_skip_confidence_check: bool = False,
        depth_scale_mode: str = "ransac",
        disable_segment_pgo: bool = False,
        metric_scale_enabled: bool = False,
        metric_model_name: str = "depth-anything/DA3METRIC-LARGE",
        metric_min_conf: float = 1.02,
        metric_bootstrap_frames: int = 1,
        compute_sky_mask: bool = False,
        sky_mask_threshold: float = 0.3,
    ):
        super().__init__()

        # Store lr scale for global backbone blocks (used by configure_optimizers)
        self.global_lr_scale = float(global_lr_scale)

        # Initialize DepthAnything3Net
        if isinstance(da3_cfg, str):
            from omegaconf import OmegaConf

            da3_cfg = OmegaConf.load(da3_cfg)

        if isinstance(da3_cfg, (dict, DictConfig)):
            # Remove __object__ if present, as it's not an argument for DepthAnything3Net
            if "__object__" in da3_cfg:
                # Make a copy to avoid modifying the original config if it's reused
                if isinstance(da3_cfg, DictConfig):
                    from omegaconf import OmegaConf

                    da3_cfg = OmegaConf.to_container(da3_cfg, resolve=True)
                else:
                    da3_cfg = da3_cfg.copy()
                da3_cfg.pop("__object__")

            self.da3 = DepthAnything3Net(**da3_cfg)
        else:
            self.da3 = da3_cfg

        self.disable_depth_head = disable_depth_head
        if self.disable_depth_head:
            self._disable_model_depth_head(self.da3)
        self.teacher_embed_dim = teacher_embed_dim
        self.student_embed_dim = student_embed_dim
        self.export_feat_layers = (
            export_feat_layers if export_feat_layers is not None else []
        )
        self.online_mode = online_mode
        self.online_kv_cache_mode = str(online_kv_cache_mode)
        if self.online_kv_cache_mode not in {"all", "dynamic"}:
            raise ValueError(
                f"Unsupported online_kv_cache_mode: {self.online_kv_cache_mode}"
            )
        self.online_kv_backend = str(online_kv_backend)
        if self.online_kv_backend not in {"dense", "paged"}:
            raise ValueError(f"Unsupported online_kv_backend: {self.online_kv_backend}")
        self.flashinfer_page_size = int(flashinfer_page_size)
        self.online_recent_frames = max(int(online_recent_frames), 0)
        self.bank_initial_frames = max(int(bank_initial_frames), 0)
        self.keyframe_mode = str(keyframe_mode)
        if self.keyframe_mode not in {"interval", "novelty"}:
            raise ValueError(f"Unsupported keyframe_mode: {self.keyframe_mode}")
        self.keyframe_interval = max(int(keyframe_interval), 1)
        self.keyframe_novelty_threshold = float(keyframe_novelty_threshold)
        self.keyframe_max_interval = max(int(keyframe_max_interval), 1)
        self.keyframe_max_keyframes = max(int(keyframe_max_keyframes), 0)
        self.keyframe_pose_confidence_ratio = max(float(keyframe_pose_confidence_ratio), 0.0)
        self.online_verbose = resolve_online_verbose(
            online_verbose=online_verbose,
            online_memory_verbose=online_memory_verbose,
            online_revisit_verbose=None,
        )
        self.online_finalize_pose_reconstruction = bool(
            online_finalize_pose_reconstruction
        )
        self.rel_pose_reconstruction_method = str(rel_pose_reconstruction_method)
        self.rel_pose_reconstruction_kwargs = (
            dict(rel_pose_reconstruction_kwargs)
            if rel_pose_reconstruction_kwargs is not None
            else {}
        )
        self.rel_pose_reconstruction_kwargs.setdefault("topn_conf", 1000)
        self.online_state = None

        # Projection layers for feature distillation if dimensions differ
        self.projections = nn.ModuleDict()
        if (
            teacher_embed_dim is not None
            and student_embed_dim is not None
            and teacher_embed_dim != student_embed_dim
        ):
            pass

        # Fallback mechanism for confidence-based reanchoring
        self.online_fallback_enabled = online_fallback_enabled
        self.drought_length = drought_length
        self.drought_threshold = drought_threshold
        self.drought_threshold_pct = drought_threshold_pct
        self.drought_threshold_warmup_frames = drought_threshold_warmup_frames
        self.num_bridge_frames = num_bridge_frames
        self.fallback_scale_epsilon = fallback_scale_epsilon
        self.evict_low_conf_threshold = evict_low_conf_threshold
        self.evict_low_conf_threshold_pct = max(
            float(evict_low_conf_threshold_pct), 0.0
        )
        self.evict_low_conf_warmup_frames = max(
            int(evict_low_conf_warmup_frames), 0
        )
        self.fallback_ref_mode = str(fallback_ref_mode)
        if self.fallback_ref_mode not in {"bridge", "keyframe"}:
            raise ValueError(f"Unsupported fallback_ref_mode: {self.fallback_ref_mode}")
        self.min_segment_frames = max(int(min_segment_frames), 0)
        self.max_segment_frames = max(int(max_segment_frames), 0)
        self.fallback_replay_attention = str(fallback_replay_attention)
        if self.fallback_replay_attention not in {"full", "causal"}:
            raise ValueError(
                f"Unsupported fallback_replay_attention: {self.fallback_replay_attention}"
            )
        self.fallback_skip_confidence_check = bool(fallback_skip_confidence_check)
        self.depth_scale_mode = str(depth_scale_mode)
        if self.depth_scale_mode not in {
            "ransac",
            "huber",
            "huber_conf",
            "weighted_median",
        }:
            raise ValueError(f"Unsupported depth_scale_mode: {self.depth_scale_mode}")
        self.disable_segment_pgo = bool(disable_segment_pgo)
        self.compute_sky_mask_enabled = bool(compute_sky_mask)
        self.sky_mask_threshold = float(sky_mask_threshold)
        self._in_fallback = False
        self.fallback_debug_dir = None
        self._fallback_debug_count = 0

        from R3.models.online.fallback import (
            FallbackManager,
            ImageRingBuffer,
            PoseEdgeLog,
            ScaleEstimator,
        )

        self._fallback_manager = FallbackManager(
            enabled=online_fallback_enabled,
            drought_length=drought_length,
            drought_threshold=drought_threshold,
            drought_threshold_pct=drought_threshold_pct,
            drought_threshold_warmup_frames=drought_threshold_warmup_frames,
            num_bridge_frames=num_bridge_frames,
            min_bridge_baseline_ratio=min_bridge_baseline_ratio,
            max_bridge_lookback=max_bridge_lookback,
            fallback_scale_epsilon=fallback_scale_epsilon,
        )
        self._scale_estimator = ScaleEstimator(epsilon=fallback_scale_epsilon)
        self._pose_edge_log = PoseEdgeLog()
        # Snapshot of bridge edges from every accepted fallback. Every replay overwrites
        # self._pose_edge_log with the trial log (which only contains the freshly written
        # bridge edges + this segment's normal edges), so without this snapshot the exported
        # log would only ever show the most recent bridge.
        self._historical_bridge_edges: list = []
        buffer_capacity = num_bridge_frames * 3 + drought_length
        self._image_buffer = ImageRingBuffer(capacity=buffer_capacity)
        self._depth_buffer = ImageRingBuffer(
            capacity=buffer_capacity
        )  # reuse same class for depth
        self._depth_conf_buffer = ImageRingBuffer(capacity=buffer_capacity)
        self._keyframe_registry = self._make_empty_keyframe_registry()
        self._previous_bridge_frame_ids: set[int] = set()
        # Refs that already failed a fallback acceptance; consulted by
        # _resolve_fallback_ref_id so the next attempt does not re-pick the same bad anchor.
        # Cleared once a fallback is accepted (registry is replaced anyway).
        self._rejected_ref_ids: set[int] = set()
        # Per-frame post-score record that survives fallback runtime-state flush. Each normal
        # inference step mirrors state.frame_post_scores into this dict so the end-of-run
        # visualization / export has a non-empty value for every processed frame.
        self._persistent_post_scores: dict[int, float] = {}
        # Frames explicitly rejected by low-confidence KV eviction. These are also removed
        # from the returned online predictions so downstream exports naturally skip them.
        self._evicted_output_frame_ids: set[int] = set()

        # Metric-scale anchor: when enabled, a frozen DA3 metric model provides
        # an absolute-scale depth reference used to set state.scale_factor at
        # sequence start and after each fallback reset.
        self.metric_scale_enabled = bool(metric_scale_enabled)
        self.metric_model_name = str(metric_model_name)
        self.metric_min_conf = float(metric_min_conf)
        self.metric_bootstrap_frames = max(int(metric_bootstrap_frames), 1)
        self.da3_metric = None
        if self.metric_scale_enabled:
            from depth_anything_3.api import DepthAnything3

            metric_wrapper = DepthAnything3.from_pretrained(self.metric_model_name)
            self.da3_metric = metric_wrapper.model
            self.da3_metric.eval()
            for p in self.da3_metric.parameters():
                p.requires_grad = False

        self._metric_bootstrap_images: dict[int, torch.Tensor] = {}
        self._metric_bootstrap_depths: dict[int, torch.Tensor] = {}
        self._metric_bootstrap_confs: dict[int, torch.Tensor] = {}
        self._metric_bootstrap_done = False

        self.set_freeze(freeze)

    def forward(
        self,
        images: Union[torch.Tensor, List[Dict[str, torch.Tensor]]],
        mode: str = "causal",
        **kwargs,
    ):
        """
        Forward pass wrapper to match STream3R output format.

        Args:
            images: [B, S, 3, H, W] or List of views
        """
        if isinstance(images, list):
            # List of views from dataloader → stack [B, 3, H, W] → [B, S, 3, H, W]
            images = torch.stack([view["img"] for view in images], dim=1)

        # Auto-detect input range and apply ImageNet normalization if needed.
        # ImageNet-normalized inputs have ~min ≈ -2.12, ~max ≈ 2.64; use a guard band so
        # the dataloader's [-1, 1] range never gets passed through unnormalized.
        img_min = images.amin().item()
        img_max = images.amax().item()
        if img_max <= 1.5 and img_min >= -1.05:
            if img_min < -0.05:
                # Range is [-1, 1] → rescale to [0, 1] first.
                # In-place to avoid a 2× peak-VRAM spike on long input
                # sequences: the out-of-place form allocates a same-shape
                # copy before freeing the operand. Long sequences (1000+
                # frames) on 8 GB cards OOM here otherwise.
                images = images.add_(1.0).div_(2.0)
            # Chunk NORMALIZE along the sequence (S) axis for the same
            # reason — torchvision's Normalize allocates a same-shape
            # output. Per-chunk peak is bounded by _S_CHUNK frames.
            _S_CHUNK = 32
            for _s in range(0, images.shape[1], _S_CHUNK):
                _e = min(_s + _S_CHUNK, images.shape[1])
                images[:, _s:_e] = NORMALIZE(images[:, _s:_e])
        # else: assume input is already ImageNet-normalized; pass through.

        if self._should_use_online_mode(mode):
            return self._forward_online(images, mode=mode, **kwargs)

        B, S, C, H, W = images.shape
        # Run DA3
        # We need to ensure we request aux features if we are in training mode (for distillation)
        da3_kwargs = dict(kwargs)
        export_feat_layers = da3_kwargs.pop(
            "export_feat_layers", self.export_feat_layers
        )
        rel_pose_reconstruction_method = da3_kwargs.pop(
            "rel_pose_reconstruction_method", self.rel_pose_reconstruction_method
        )
        rel_pose_reconstruction_kwargs = da3_kwargs.pop(
            "rel_pose_reconstruction_kwargs", self.rel_pose_reconstruction_kwargs
        )
        # pose_max_recent is only used by the online path; remove it before passing to DA3
        da3_kwargs.pop("pose_max_recent", None)

        output = self.da3(
            images,
            export_feat_layers=export_feat_layers,
            **da3_kwargs,
        )

        predictions = self._format_predictions(
            output,
            images,
            H,
            W,
            rel_pose_reconstruction_method,
            rel_pose_reconstruction_kwargs,
            online_mode=False,
        )
        return predictions

    def forward_online_step(self, frame_images, mode: str = "causal", **kwargs):
        if frame_images.dim() == 4:
            frame_images = frame_images.unsqueeze(1)
        if frame_images.dim() != 5 or frame_images.shape[1] != 1:
            raise ValueError(
                "forward_online_step expects frame_images with shape [B, 1, 3, H, W]"
            )
        state = self._ensure_online_state(frame_images.shape[0])
        online_options, da3_kwargs = self._extract_online_step_options(kwargs)
        return self._run_online_step(
            frame_images,
            state,
            online_options=online_options,
            da3_kwargs=da3_kwargs,
        )
