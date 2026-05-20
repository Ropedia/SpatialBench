"""
Benchmark scene-level dataset: deterministic loading, no random sampling, no data augmentation.
Each __getitem__ returns all fixed frames of one complete scene.
"""
import json
import os

import numpy as np
import torch
import torchvision.transforms as T

from benchmark.datasets.data_readers import AdtReader, \
DroidReader, DtuReader, Eth3dReader, HiroomReader, KittiOdometryReader, LingbotReader, NrgbdReader, \
OmniworldReader, RLBenchReader, RoboLabReader, RoboTwinReader, RopediaReader, ScannetppReader, \
SevenScenesReader, TanksAndTemplesReader, TumReader, VkittiReader, WaymoReader


class TagRegistry:
    """Scene tag query system: supports filtering scenes by tag."""

    def __init__(self, scene_index_path):
        with open(scene_index_path, 'r') as f:
            self.scenes = json.load(f)

        # Build inverted index: tag_value -> set of scene indices
        self._tag_to_idxs = {}
        for i, scene in enumerate(self.scenes):
            for axis, value in scene.get("tags", {}).items():
                if value not in self._tag_to_idxs:
                    self._tag_to_idxs[value] = set()
                self._tag_to_idxs[value].add(i)
            # Also index by source_dataset
            ds = scene.get("source_dataset", "")
            if ds not in self._tag_to_idxs:
                self._tag_to_idxs[ds] = set()
            self._tag_to_idxs[ds].add(i)

    def query(self, tags, operator="AND"):
        """Query scenes by tags.

        Args:
            tags: list[str], list of tag values (e.g. ["sparse", "indoor"])
            operator: "AND" or "OR"

        Returns:
            list[dict]: matched scene entries
        """
        if not tags:
            return self.scenes

        sets = []
        for tag in tags:
            if tag in self._tag_to_idxs:
                sets.append(self._tag_to_idxs[tag])
            else:
                if operator == "AND":
                    return []  # Under AND mode, if any tag has no match the result is empty
                sets.append(set())

        if operator == "AND":
            result_idxs = set.intersection(*sets)
        else:
            result_idxs = set.union(*sets)

        return [self.scenes[i] for i in sorted(result_idxs)]

    def query_string(self, query_str):
        """Parse a query string.
        "sparse+indoor" -> AND
        "sparse|dense" -> OR
        """
        if '+' in query_str:
            tags = query_str.split('+')
            return self.query(tags, operator="AND")
        elif '|' in query_str:
            tags = query_str.split('|')
            return self.query(tags, operator="OR")
        else:
            return self.query([query_str])

    def list_tags(self):
        """List all tag axes and their possible values."""
        axes = {}
        for scene in self.scenes:
            for axis, value in scene.get("tags", {}).items():
                if axis not in axes:
                    axes[axis] = set()
                axes[axis].add(value)
        return {k: sorted(v) for k, v in axes.items()}

    def stats(self, tags=None):
        """Statistics."""
        scenes = self.query(tags) if tags else self.scenes
        per_dataset = {}
        for s in scenes:
            ds = s.get("source_dataset", "unknown")
            per_dataset[ds] = per_dataset.get(ds, 0) + 1
        return {
            "total": len(scenes),
            "per_dataset": per_dataset,
        }


class BenchmarkDataset(torch.utils.data.Dataset):
    """Deterministic scene-level benchmark dataset.

    Each __getitem__ returns all fixed frames of one scene, used for model inference and evaluation.
    No randomness, no data augmentation.

    Data layout (SpatialBenchmark, pre-split by view_density):
        <benchmark_root>/<density>/<dataset>/<scene_path>/{images,depths,poses,intrinsics,...,meta.json}
    Where density in {single, sparse, medium, dense}; each scene folder's images/ is already filtered
    per frame, so use list(range(n_frames)) as positional indices to load all frames of the scene.

    Args:
        scene_index_path: path to scene_index.json (e.g. all_scenes.json)
        benchmark_root: SpatialBenchmark root directory
        tags: tag filter (e.g. ["sparse", "indoor"])
        tag_operator: "AND" or "OR"
        max_scenes: maximum number of scenes (used for quick testing)
        conf_threshold: Ropedia confidence threshold
    """

    # ImageNet normalization
    IMG_NORM = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # SpatialBenchmark root pre-split by density (per-frame data is read from here):
    #   <root>/<density>/<dataset>/<scene_path>/{images,depths,...,meta.json}
    DEFAULT_BENCHMARK_ROOT = "SpatialBenchmark"

    def __init__(self,
                 scene_index_path,
                 benchmark_root=None,
                 tags=None,
                 tag_operator="AND",
                 max_scenes=None,
                 resolution_override=None,
                 conf_threshold=0.3,
                 shuffle_seed=None,
                 priority_datasets=None):

        self.registry = TagRegistry(scene_index_path)
        self.shuffle_seed = shuffle_seed

        # Filter scenes
        if tags:
            self.scenes = self.registry.query(tags, operator=tag_operator)
        else:
            self.scenes = self.registry.scenes

        if max_scenes:
            self.scenes = self.scenes[:max_scenes]

        # Move high-VRAM / OOM-prone datasets to the front of the queue, so OOM is triggered on the
        # first scene; combined with run_dense_benchmark.py's OOM detection this kills the process
        # immediately and skips to the next model
        if priority_datasets:
            priority_set = set(priority_datasets)
            head = [s for s in self.scenes
                    if s.get("source_dataset") in priority_set]
            tail = [s for s in self.scenes
                    if s.get("source_dataset") not in priority_set]
            self.scenes = head + tail

        # SpatialBenchmark root pre-split by density; per-frame loading reads from here
        self.benchmark_root = (
            benchmark_root if benchmark_root is not None else self.DEFAULT_BENCHMARK_ROOT
        )
        # Placeholder: external code (run_benchmark.py, metrics.get_gt_mesh_path) uses
        # dataset.data_roots.get(ds) to look up GT mesh resources; we no longer hold the original
        # SpatialBench root, so return None and let the upper layer gracefully skip.
        self.data_roots = {}

        # Reader instances
        self._readers = {
            "droid": DroidReader(),
            "ropedia": RopediaReader(conf_threshold=conf_threshold),
            "tum": TumReader(),
            "nrgbd": NrgbdReader(),
            "7scenes": SevenScenesReader(),
            "adt": AdtReader(),
            "robotwin": RoboTwinReader(),
            "rlbench": RLBenchReader(),
            "tanks_and_temples": TanksAndTemplesReader(),
            "dtu": DtuReader(),
            "eth3d": Eth3dReader(),
            "omniworld": OmniworldReader(),
            "lingbot": LingbotReader(),
            "vkitti": VkittiReader(),
            "waymo": WaymoReader(),
            "kitti_odometry": KittiOdometryReader(),
            "robolab": RoboLabReader(),
            "hiroom": HiroomReader(),
            "scannetpp": ScannetppReader(),
        }

        # Resolution override (used for models needing special resolutions, e.g. Fast3R)
        if resolution_override is not None:
            for rdr in self._readers.values():
                rdr._resolution_override = resolution_override

        print(f"[BenchmarkDataset] Loaded {len(self.scenes)} scenes "
              f"(tags={tags}, operator={tag_operator})")
        print(f"[BenchmarkDataset] conf_threshold: {conf_threshold}")
        if resolution_override:
            print(f"[BenchmarkDataset] resolution_override: {resolution_override}")
        if priority_datasets:
            n_pri = sum(1 for s in self.scenes
                        if s.get("source_dataset") in set(priority_datasets))
            print(f"[BenchmarkDataset] priority_datasets={list(priority_datasets)} "
                  f"({n_pri} scenes moved to front)")
        if shuffle_seed is not None:
            print(f"[BenchmarkDataset] frame shuffle ENABLED, base seed={shuffle_seed} "
                  f"(per-scene seed = base + hash(scene_id))")
        print(f"[BenchmarkDataset] benchmark_root: {self.benchmark_root}")
        print(f"[BenchmarkDataset] per-dataset default resolution/z_far:")
        for ds, rdr in self._readers.items():
            print(f"    {ds}: resolution={rdr.DEFAULT_RESOLUTION}, z_far={rdr.DEFAULT_Z_FAR}")

    def __len__(self):
        return len(self.scenes)

    @staticmethod
    def _permute_scene(scene, perm):
        """Permute all per-frame fields in scene according to perm (np.ndarray of length N).

        Non per-frame fields (scene_id, source_dataset, tags) are kept unchanged.
        After permutation, scene[key][i] corresponds to the original scene[key][perm[i]]; images /
        depth / extrinsic / intrinsic / valid_mask / sky_mask / world_points / frame_indices are
        all synchronized so that the pred[i] <-> gt[i] relationship is preserved.
        """
        perm_list = perm.tolist()
        scene['images']        = scene['images'][perm]
        scene['images_raw']    = scene['images_raw'][perm]
        scene['depth']         = scene['depth'][perm]
        scene['extrinsic']     = scene['extrinsic'][perm]
        scene['intrinsic']     = scene['intrinsic'][perm]
        scene['valid_mask']    = scene['valid_mask'][perm]
        scene['world_points']  = scene['world_points'][perm]
        if scene.get('sky_mask') is not None:
            scene['sky_mask']  = scene['sky_mask'][perm]
        scene['frame_indices'] = [scene['frame_indices'][i] for i in perm_list]
        scene['frame_permutation'] = perm_list
        return scene

    @staticmethod
    def _depth_to_world_points(depth, extrinsic, intrinsic):
        """Unproject depth maps into a world-coordinate point cloud.

        Args:
            depth: (N, H, W) float32 depth map
            extrinsic: (N, 3, 4) float32 cam2world
            intrinsic: (N, 3, 3) float32 intrinsics

        Returns:
            world_points: (N, H, W, 3) float32 3D points in the world coordinate system
        """
        N, H, W = depth.shape
        world_points = np.zeros((N, H, W, 3), dtype=np.float32)

        u, v = np.meshgrid(np.arange(W), np.arange(H))  # (H, W)

        for i in range(N):
            K = intrinsic[i]
            fu, fv = K[0, 0], K[1, 1]
            cu, cv = K[0, 2], K[1, 2]

            z = depth[i]  # (H, W)
            x_cam = (u - cu) * z / fu
            y_cam = (v - cv) * z / fv
            cam_coords = np.stack((x_cam, y_cam, z), axis=-1).astype(np.float32)  # (H, W, 3)

            # extrinsic is c2w (3, 4); use it directly for the transform
            R = extrinsic[i, :3, :3]  # (3, 3)
            t = extrinsic[i, :3, 3]   # (3,)
            world_points[i] = cam_coords @ R.T + t

        return world_points

    def __getitem__(self, idx):
        """Load all fixed frames of one scene.

        Returns:
            dict:
                scene_id: str
                source_dataset: str
                tags: dict
                images: Tensor (N, 3, H, W)  - ImageNet normalized
                images_raw: Tensor (N, 3, H, W)  - unnormalized (for visualization)
                depth: np.ndarray (N, H, W) float32
                extrinsic: np.ndarray (N, 3, 4) float32  - cam2world
                intrinsic: np.ndarray (N, 3, 3) float32
                valid_mask: np.ndarray (N, H, W) bool
                frame_indices: list[int]
        """
        scene = self.scenes[idx]
        source = scene["source_dataset"]
        scene_path = scene["scene_path"]
        frame_indices = scene["frame_indices"]
        density = scene.get("tags", {}).get("view_density")
        if density is None:
            raise ValueError(
                f"scene {scene.get('scene_id')!r} missing tags.view_density; "
                f"required to resolve SpatialBenchmark per-density path"
            )

        # Each SpatialBenchmark scene directory is already filtered by frame_indices, so read by
        # positions 0..N-1 directly.
        frame_data_root = os.path.join(self.benchmark_root, density, source)
        N = len(frame_indices)
        positions = list(range(N))

        reader = self._readers.get(source)
        if reader is None:
            raise ValueError(f"No reader for dataset '{source}'")

        # Read raw data (resolution and z_far are controlled internally by the reader)
        raw = reader.read_scene(frame_data_root, scene_path, positions)

        # Images: normalized + raw
        images_norm = []
        images_raw = []
        for img in raw['images']:
            images_norm.append(self.IMG_NORM(img))
            images_raw.append(T.ToTensor()(img))

        images = torch.stack(images_norm)       # (N, 3, H, W)
        images_raw = torch.stack(images_raw)    # (N, 3, H, W)

        # Depth
        depth = np.stack(raw['depths'])         # (N, H, W)

        # Extrinsics & Intrinsics
        extrinsic = np.stack(raw['extrinsics']).astype(np.float32)  # (N, 3, 4)
        intrinsic = np.stack(raw['intrinsics']).astype(np.float32)  # (N, 3, 3)

        # Valid mask: depth > 0 and finite (z_far is already filtered in the reader)
        valid_mask = (depth > 0) & np.isfinite(depth)

        # Sky mask (optional, only some readers provide it; None means the dataset has no sky-region
        # annotations)
        sky_mask_list = raw.get('sky_masks')
        if sky_mask_list is not None and all(m is not None for m in sky_mask_list):
            sky_mask = np.stack(sky_mask_list)  # (N, H, W) bool
        else:
            sky_mask = None

        # Compute world points: depth + intrinsic -> cam coords -> c2w -> world coords
        world_points = self._depth_to_world_points(depth, extrinsic, intrinsic)

        out = {
            'scene_id': scene['scene_id'],
            'source_dataset': source,
            'tags': scene.get('tags', {}),
            'images': images,
            'images_raw': images_raw,
            'depth': depth,
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'valid_mask': valid_mask,
            'sky_mask': sky_mask,
            'world_points': world_points,
            'frame_indices': list(frame_indices),
        }

        if self.shuffle_seed is not None and N > 1:
            seed = (self.shuffle_seed + hash(scene['scene_id'])) & 0x7FFFFFFF
            perm = np.random.RandomState(seed).permutation(N)
            out = self._permute_scene(out, perm)

        return out
