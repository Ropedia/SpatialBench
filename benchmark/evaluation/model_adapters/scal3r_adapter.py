"""
Scal3R model adapter.
Directly call Scal3R internal API for inference (no subprocess).

Scal3R (Scalable Test-Time Training for 3D Reconstruction) outputs:
  - c2w: (N, 4, 4) cam2world pose
  - ixt: (N, 3, 3) intrinsics
  - dpt_map: (N, H*W, 2) depth map
"""
import sys
import os
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter
from benchmark.utils.hf_weights import ensure_hf_snapshot
from benchmark.utils.paths import DEFAULT_CHECKPOINTS_DIR


@register_adapter("scal3r")
class Scal3RAdapter(ModelAdapter):

    def __init__(self):
        self.model = None
        self.device = "cuda"
        self._checkpoint_path = None
        self._loop_ckpt_path = ''      # SALAD ckpt resolved absolute path (load_model Fill)
        self._config_path = None
        self._dataset_cfg = None
        self._scal3r_dir = None
        self.block_size = None         # pipeline block size (None -> Scal3R default 60)
        self.overlap_size = None       # block overlap size (None -> Scal3R default 30)
        self.loop_size = 20            # loop size
        self.use_xyz_align = 0         # whether to use xyz alignment
        self.test_use_amp = False      # whether to use AMP during testing
        self.use_loop = 1              # whether to enable loop closure (Scal3R upstream default 1)
        self.loop_ckpt = None          # SALAD ckpt path (None -> dino_salad.ckpt next to scal3r.pt)

    def name(self):
        return "Scal3R"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        """Load the Scal3R model into memory."""
        self.device = device

        # Scal3R has two supported local layouts:
        #   1) bundled: benchmark/models/scal3r is the release root and
        #      benchmark/models is the Python import root.
        #   2) external clone: Scal3R is both the release root and import root.
        # The release root must contain configs/ because Scal3R config files
        # include paths such as "configs/base.yaml".
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        models_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
        bundled_release_root = os.path.join(models_root, 'scal3r')
        external_release_root = os.path.join(project_root, 'Scal3R')

        layout_candidates = [
            (bundled_release_root, models_root),
            (external_release_root, external_release_root),
            (models_root, models_root),  # legacy fallback
        ]
        scal3r_release_root = None
        scal3r_import_root = None
        for release_root, import_root in layout_candidates:
            cfg = os.path.join(release_root, 'configs', 'models', 'scal3r.yaml')
            pkg = os.path.join(import_root, 'scal3r', '__init__.py')
            if os.path.isfile(cfg) and os.path.isfile(pkg):
                scal3r_release_root = release_root
                scal3r_import_root = import_root
                self._config_path = cfg
                break

        if scal3r_release_root is None:
            raise FileNotFoundError(
                "Could not find Scal3R configs. Expected "
                "benchmark/models/scal3r/configs/models/scal3r.yaml. "
                "If you have an external Scal3R clone, copy Scal3R/configs "
                "into benchmark/models/scal3r/configs."
            )

        self._scal3r_dir = scal3r_release_root
        # Add the Python import root to sys.path (so import scal3r works).
        if scal3r_import_root not in sys.path:
            sys.path.insert(0, scal3r_import_root)

        def _resolve_local_path(path):
            """Resolve user paths relative to cwd/project root before Scal3R sees them.

            Scal3R resolves relative checkpoint paths against its release root. Benchmark
            configs, however, use paths relative to the SpatialBench project root.
            """
            if not path:
                return None
            path = os.path.expanduser(str(path))
            candidates = [path] if os.path.isabs(path) else [
                os.path.abspath(path),
                os.path.abspath(os.path.join(project_root, path)),
                os.path.abspath(os.path.join(scal3r_release_root, path)),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
            return None

        # Find the checkpoint
        checkpoint_path = _resolve_local_path(checkpoint)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            self._checkpoint_path = os.path.abspath(checkpoint_path)
        elif checkpoint_path and os.path.isdir(checkpoint_path):
            for f in os.listdir(checkpoint_path):
                if f.endswith('.pt') or f.endswith('.pth'):
                    self._checkpoint_path = os.path.abspath(
                        os.path.join(checkpoint_path, f))
                    break
        else:
            if checkpoint and str(checkpoint).endswith(('.pt', '.pth')):
                raise FileNotFoundError(
                    f"Scal3R checkpoint path does not exist: {checkpoint}. "
                    "Expected a local .pt/.pth file or a Hugging Face repo id."
                )
            repo_id = checkpoint or "xbillowy/Scal3R"
            weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
            snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
            for f in os.listdir(snapshot_dir):
                if f.endswith('.pt') or f.endswith('.pth'):
                    self._checkpoint_path = os.path.abspath(
                        os.path.join(snapshot_dir, f))
                    break

        # Resolve the SALAD loop-detector ckpt:
        #   1) the user explicitly gave a path in YAML -> use it
        #   2) otherwise use scal3r.pt same directory dino_salad.ckpt
        #   3) if neither is found -> leave an empty string (Scal3R will fall back to backup loop detection without SALAD)
        loop_ckpt_path = _resolve_local_path(self.loop_ckpt)
        if loop_ckpt_path and os.path.isfile(loop_ckpt_path):
            self._loop_ckpt_path = os.path.abspath(loop_ckpt_path)
        elif self._checkpoint_path:
            cand = os.path.join(os.path.dirname(self._checkpoint_path), 'dino_salad.ckpt')
            self._loop_ckpt_path = os.path.abspath(cand) if os.path.isfile(cand) else ''
        else:
            self._loop_ckpt_path = ''
        if self.use_loop and not self._loop_ckpt_path:
            print('[Scal3RAdapter] WARNING: use_loop=1 but dino_salad.ckpt not found; '
                  'Scal3R will fall back to non-SALAD loop detection.')

        # Point the Scal3R release root to the correct directory (contains configs/ and data/)
        import scal3r.engine.path as _scal3r_path
        import scal3r.engine.config as _scal3r_config
        _release_root = lambda: os.path.abspath(self._scal3r_dir)
        _scal3r_path.get_release_root = _release_root
        _scal3r_config.get_release_root = _release_root

        from scal3r.models import build_sampler_from_config
        from scal3r.engine.path import resolve_release_path

        config_path = self._config_path or os.path.join(
            self._scal3r_dir, 'configs', 'models', 'scal3r.yaml')
        config_path = str(resolve_release_path(config_path))
        self.model, self._dataset_cfg = build_sampler_from_config(
            config_path, torch.device(device), self._checkpoint_path)
        self.model.eval()
        print(f"[Scal3RAdapter] Model loaded on {device}")

    def prepare(self, scene):
        """Data preparation stage(not counted in inference timing): 
        Write images as PNGs to a temporary directory, build the args dotdict, and call scal3r load_data
        Build batches/indices.store the results on self for predict().
        """
        import shutil
        # Defensive: if the previous predict did not clean up normally, clean it up first
        old_dir = getattr(self, '_tmp_dir', None)
        if old_dir and os.path.isdir(old_dir):
            shutil.rmtree(old_dir, ignore_errors=True)

        images_raw = scene['images_raw']  # (N, 3, H, W) tensor [0, 1]
        N = images_raw.shape[0]

        # Save images to a temporary directory for the Scal3R dataloader to read
        self._tmp_dir = tempfile.mkdtemp(prefix='scal3r_')
        img_dir = os.path.join(self._tmp_dir, 'images')
        result_dir = os.path.join(self._tmp_dir, 'results')
        runtime_dir = os.path.join(result_dir, 'runtime')
        os.makedirs(img_dir)
        os.makedirs(result_dir)
        os.makedirs(runtime_dir)

        import cv2
        for i in range(N):
            img_np = (images_raw[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(img_dir, f'{i:06d}.png'), img_bgr)

        # Build args (simulate backend.parse_args output)
        from scal3r.utils.base_utils import DotDict as dotdict
        self._args = dotdict(
            config=self._config_path,
            checkpoint=self._checkpoint_path or '',
            input_dir=img_dir,
            result_dir=result_dir,
            runtime_dir=runtime_dir,
            image_patterns='*.png',
            max_images=-1,
            preprocess_workers=1,
            block_size=self.block_size if self.block_size is not None else 60,
            overlap_size=self.overlap_size if self.overlap_size is not None else 30,
            loop_size=self.loop_size,
            loop_ckpt=self._loop_ckpt_path,
            use_xyz_align=self.use_xyz_align,
            max_align_points_per_frame=None,
            test_use_amp=self.test_use_amp,
            save_dpt=1,
            save_xyz=0,
            downsample_xyz_ratio=0.15,
            confidence_xyz_threshold=0.75,
            use_loop=int(bool(self.use_loop)),
            streaming_state=0,
            offload_batches=0,
            offload_outputs=0,
            cleanup_offload=1,
            offload_dir='',
            probe_dir='',
            stop_after_stage='',
            pgo_workers=32,
            device=self.device,
        )

        from scal3r.pipelines.backend import load_data
        self._batches, self._indices = load_data(self._dataset_cfg, self._args)

    def predict(self, scene):
        """Run Scal3R inference (Directly callinternal API, no subprocess).
        Only includes forward + post_process + result extraction, not writing PNGs / load_data
        (these are in prepare()).
        """
        # Compatibility fallback: when predict() is called directly without prepare() first(such as old run_benchmark), 
        # automatically run prepare once.
        if not getattr(self, '_args', None):
            self.prepare(scene)

        try:
            images_raw = scene['images_raw']
            N, _, H, W = images_raw.shape

            from scal3r.pipelines.backend import forward, post_process

            # Forward inference
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
                output = forward(self.model, self._batches, self._args)

            # Post-processing: aligns with blocks, get c2w / ixt / dpt_map
            # must pass load_data written by args.n_blocks_loop through, otherwise when use_loop=1
            # loop closure blocks will be treated as normal base blocks for adjacent PGO, breaking multi-block inference
            processed, output, batches, indices, visualize = post_process(
                output, self._batches, self._indices, self._args,
                n_blocks_loop=self._args.get('n_blocks_loop', 0),
                alignment='sim3_wet',
                use_xyz_align=self._args.use_xyz_align,
            )

            # Extract results
            c2w = processed.output.c2w[:N]  # (N, 4, 4)
            ixt = processed.output.ixt[:N]  # (N, 3, 3)

            result = {}

            # pred_pose: c2w (N, 3, 4)
            result['pred_pose'] = c2w[:, :3, :4].astype(np.float32)

            # w2c_extrinsics: (N, 3, 4), align to the first frame so w2c[0] = [I | 0]
            n_poses = len(c2w)
            c2w_44 = np.zeros((n_poses, 4, 4), dtype=np.float32)
            c2w_44[:, :3, :4] = c2w[:, :3, :4]
            c2w_44[:, 3, 3] = 1.0
            R0 = c2w_44[0, :3, :3]
            t0 = c2w_44[0, :3, 3]
            inv_c2w0 = np.eye(4, dtype=np.float32)
            inv_c2w0[:3, :3] = R0.T
            inv_c2w0[:3, 3] = -R0.T @ t0
            c2w_aligned = np.matmul(inv_c2w0, c2w_44)  # c2w_aligned[0] = I
            # c2w -> w2c (closed-form SE3 inverse)
            w2c_aligned = np.zeros((n_poses, 3, 4), dtype=np.float32)
            for i in range(n_poses):
                R = c2w_aligned[i, :3, :3]
                t = c2w_aligned[i, :3, 3]
                w2c_aligned[i, :3, :3] = R.T
                w2c_aligned[i, :3, 3] = -R.T @ t
            result['w2c_extrinsics'] = w2c_aligned

            # pred_intrinsic
            result['pred_intrinsic'] = ixt.astype(np.float32)

            # pred_depth: (N, H, W) - output resolution must equal input H, W
            if hasattr(processed.output, 'dpt_map') and processed.output.dpt_map is not None:
                dpt_map = processed.output.dpt_map[:N]  # (N, H*W, 2) or (N, H*W, 1)
                if dpt_map.ndim == 3 and dpt_map.shape[-1] >= 1:
                    depth = dpt_map[..., 0]  # (N, H*W)
                else:
                    depth = dpt_map
                n_pixels = depth.shape[1]
                assert n_pixels == H * W, (
                    f"[Scal3RAdapter] depth resolution mismatch: "
                    f"n_pixels={n_pixels}, expected H*W={H * W} (H={H}, W={W})"
                )
                depth = depth.reshape(N, H, W)
                assert depth.shape == (N, H, W), (
                    f"[Scal3RAdapter] depth shape mismatch: "
                    f"got {depth.shape}, expected ({N}, {H}, {W})"
                )
                result['pred_depth'] = depth.astype(np.float32)

            torch.cuda.empty_cache()
            return result
        finally:
            # Clean up prepare() created tmp dir and state fields
            tmp = getattr(self, '_tmp_dir', None)
            if tmp and os.path.isdir(tmp):
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
            self._tmp_dir = None
            self._args = None
            self._batches = None
            self._indices = None

    def supports_metric_depth(self):
        return False

    def normalize_gt_poses(self, scene):
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )

    def get_model_params(self):
        if self.model is not None:
            return super().get_model_params()
        return {}
