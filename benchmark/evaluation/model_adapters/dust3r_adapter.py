"""
DUSt3R 模型适配器。

该适配器使用 DUSt3R 的 pairwise inference + global alignment 流程，
并把输出转换为 SpatialBench 统一评测契约：
  - pred_depth: (N, H, W)
  - pred_pose: (N, 3, 4) cam2world
  - w2c_extrinsics: (N, 3, 4) world-to-camera
  - pred_intrinsic: (N, 3, 3)
  - pred_pointcloud: (M, 3)
  - pred_confidence: (N, H, W)
"""
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch

from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter


_DUST3R_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "dust3r_root")
)
_DUST3R_PREFIXES = ("dust3r", "croco", "models")


@contextmanager
def _dust3r_context():
    """临时切换 DUSt3R 依赖路径，并在退出时恢复同名模块。

    DUSt3R/CroCo 使用顶层包名 `dust3r` 和 `models`。仓库里还有其他模型
    也带有同名依赖快照，所以这里按 adapter root 做隔离，避免交叉污染。
    """
    saved_modules = {
        key: value
        for key, value in sys.modules.items()
        if any(key == prefix or key.startswith(prefix + ".") for prefix in _DUST3R_PREFIXES)
    }
    for key in saved_modules:
        sys.modules.pop(key, None)

    saved_path = sys.path[:]
    root_abs = os.path.abspath(_DUST3R_ROOT)
    clean_path = []
    for entry in sys.path:
        entry_abs = os.path.abspath(entry or os.getcwd())
        if entry_abs == root_abs:
            continue
        clean_path.append(entry)
    sys.path[:] = [root_abs] + clean_path

    try:
        yield
    finally:
        loaded = [
            key
            for key in sys.modules
            if any(key == prefix or key.startswith(prefix + ".") for prefix in _DUST3R_PREFIXES)
        ]
        for key in loaded:
            sys.modules.pop(key, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def _align_c2w_to_first(poses_c2w):
    """将预测相机位姿对齐到第一帧坐标系。"""
    r0 = poses_c2w[0, :3, :3]
    t0 = poses_c2w[0, :3, 3]
    first_inv = np.eye(4, dtype=poses_c2w.dtype)
    first_inv[:3, :3] = r0.T
    first_inv[:3, 3] = -r0.T @ t0
    return np.asarray([first_inv @ poses_c2w[i] for i in range(len(poses_c2w))])


def _c2w_to_w2c(c2w_44):
    """批量把 cam2world 4x4 转为 world-to-camera 3x4。"""
    w2c_list = []
    for i in range(len(c2w_44)):
        r = c2w_44[i, :3, :3]
        t = c2w_44[i, :3, 3]
        w2c = np.zeros((3, 4), dtype=np.float32)
        w2c[:3, :3] = r.T
        w2c[:3, 3] = -r.T @ t
        w2c_list.append(w2c)
    return np.stack(w2c_list).astype(np.float32)


def _build_dust3r_views(images_raw):
    """把 SpatialBench 的 [0, 1] 图像张量转换成 DUSt3R view list。"""
    n, _, h, w = images_raw.shape
    images_normed = (images_raw - 0.5) / 0.5
    views = []
    for i in range(n):
        views.append({
            "img": images_normed[i:i + 1],
            "true_shape": np.int32([[h, w]]),
            "idx": i,
            "instance": str(i),
        })
    return views


def _scene_graph_for_num_frames(num_frames):
    """短序列使用 complete graph，长序列改用滑窗以降低显存峰值。"""
    if num_frames <= 30:
        return "complete"
    winsize = min(10, num_frames - 1)
    return f"swin-{winsize}-noncyclic"


def _extract_aligned_result(scene_opt, images_raw, num_frames, height, width):
    """从 DUSt3R global aligner 中抽取 SpatialBench 结果。"""
    result = {}

    poses_c2w = scene_opt.get_im_poses().detach().cpu().numpy()
    poses_c2w = _align_c2w_to_first(poses_c2w)
    result["pred_pose"] = poses_c2w[:, :3, :4].astype(np.float32)
    result["w2c_extrinsics"] = _c2w_to_w2c(poses_c2w)

    depthmaps = scene_opt.get_depthmaps()
    depth_list = []
    for i in range(num_frames):
        depth = depthmaps[i].detach().cpu().numpy()
        assert depth.shape == (height, width), (
            f"DUSt3R depth resolution mismatch: pred {depth.shape}, "
            f"expected ({height}, {width})"
        )
        depth_list.append(depth)
    result["pred_depth"] = np.stack(depth_list).astype(np.float32)

    try:
        intrinsic = scene_opt.get_intrinsics().detach().cpu().numpy()
        result["pred_intrinsic"] = intrinsic.astype(np.float32)
    except Exception:
        focals = scene_opt.get_focals().detach().cpu().numpy()
        intrinsic = np.zeros((num_frames, 3, 3), dtype=np.float32)
        for i in range(num_frames):
            focal = float(focals[i])
            intrinsic[i] = np.array(
                [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                dtype=np.float32,
            )
        result["pred_intrinsic"] = intrinsic

    pts3d_list = scene_opt.get_pts3d()
    masks = scene_opt.get_masks()

    all_points = []
    all_colors = []
    conf_list = []
    for i in range(num_frames):
        pts = pts3d_list[i].detach().cpu().numpy()
        mask = masks[i].detach().cpu().numpy()

        conf = mask.astype(np.float32)
        if conf.shape != (height, width):
            import cv2
            conf = cv2.resize(conf, (width, height), interpolation=cv2.INTER_NEAREST)
        conf_list.append(conf)

        valid = mask & np.isfinite(pts).all(axis=-1)
        pts_valid = pts[valid]
        if len(pts_valid) > 0:
            all_points.append(pts_valid)
            image = images_raw[i].permute(1, 2, 0).numpy()
            if pts.shape[:2] != (height, width):
                import cv2
                image = cv2.resize(image, (pts.shape[1], pts.shape[0]))
            all_colors.append(image[valid])

    if all_points:
        result["pred_pointcloud"] = np.concatenate(all_points, axis=0).astype(np.float32)
        result["pred_pointcloud_colors"] = np.concatenate(all_colors, axis=0).astype(np.float32)

    result["pred_confidence"] = np.stack(conf_list).astype(np.float32)
    return result


@register_adapter("dust3r")
class Dust3RAdapter(ModelAdapter):
    def __init__(self):
        self.model = None
        self.device = "cuda"
        self.image_size = 512
        self.niter = 300
        self.schedule = "cosine"
        self.lr = 0.01

    def name(self):
        return "DUSt3R"

    def load_model(self, checkpoint=None, device="cuda", weights_dir=None):
        self.device = device
        with _dust3r_context():
            from dust3r.model import AsymmetricCroCo3DStereo

            if checkpoint and (os.path.isfile(checkpoint) or os.path.isdir(checkpoint)):
                self.model = AsymmetricCroCo3DStereo.from_pretrained(checkpoint)
                print(f"[Dust3RAdapter] Model loaded from {checkpoint}")
            else:
                repo_id = checkpoint or "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
                if not os.path.exists(repo_id):
                    from benchmark.utils.hf_weights import ensure_hf_snapshot
                    from benchmark.utils.paths import DEFAULT_CHECKPOINTS_DIR

                    weights_dir = weights_dir or DEFAULT_CHECKPOINTS_DIR
                    try:
                        snapshot_dir = ensure_hf_snapshot(repo_id, local_root=weights_dir)
                        self.model = AsymmetricCroCo3DStereo.from_pretrained(snapshot_dir)
                    except Exception:
                        self.model = AsymmetricCroCo3DStereo.from_pretrained(repo_id)
                else:
                    self.model = AsymmetricCroCo3DStereo.from_pretrained(repo_id)
                print(f"[Dust3RAdapter] Model loaded from {repo_id}")

            self.model = self.model.to(device)
            self.model.eval()
        print(f"[Dust3RAdapter] DUSt3R on {device}")

    def predict(self, scene):
        """运行 DUSt3R 推理并返回 SpatialBench 格式结果。"""
        with _dust3r_context():
            from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
            from dust3r.image_pairs import make_pairs
            from dust3r.inference import inference

            images_raw = scene["images_raw"]
            single_frame = images_raw.shape[0] == 1
            if single_frame:
                images_raw = images_raw.repeat(2, 1, 1, 1)
            num_frames, _, height, width = images_raw.shape

            views = _build_dust3r_views(images_raw)
            pairs = make_pairs(
                views,
                scene_graph=_scene_graph_for_num_frames(num_frames),
                prefilter=None,
                symmetrize=True,
            )

            with torch.no_grad():
                output = inference(pairs, self.model, self.device, batch_size=8, verbose=True)

            torch.cuda.empty_cache()

            mode = (
                GlobalAlignerMode.PairViewer
                if num_frames <= 2
                else GlobalAlignerMode.PointCloudOptimizer
            )
            scene_opt = global_aligner(output, device=self.device, mode=mode)
            if mode != GlobalAlignerMode.PairViewer:
                try:
                    loss = scene_opt.compute_global_alignment(
                        init="mst",
                        niter=self.niter,
                        schedule=self.schedule,
                        lr=self.lr,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    fallback_niter = max(50, self.niter // 3)
                    print(
                        "    [DUSt3R] OOM during global_alignment; "
                        f"retry niter={fallback_niter}"
                    )
                    loss = scene_opt.compute_global_alignment(
                        init="mst",
                        niter=fallback_niter,
                        schedule=self.schedule,
                        lr=self.lr,
                    )
                print(f"    [DUSt3R] Global alignment loss: {loss:.4f}")

            result = _extract_aligned_result(scene_opt, images_raw, num_frames, height, width)

            if single_frame:
                result["pred_depth"] = result["pred_depth"][:1]
                result.pop("pred_pose", None)
                result.pop("w2c_extrinsics", None)
                result["pred_confidence"] = result["pred_confidence"][:1]
                if "pred_intrinsic" in result:
                    result["pred_intrinsic"] = result["pred_intrinsic"][:1]

            torch.cuda.empty_cache()
            return result

    def supports_metric_depth(self):
        return False

    def requires_intrinsics(self):
        return False

    def normalize_gt_poses(self, scene):
        """DUSt3R 评测时按首帧和平均点距归一化 GT 位姿。"""
        return self.normalize_camera_extrinsics_and_points(
            scene["extrinsic"], scene["world_points"], scene["valid_mask"]
        )
