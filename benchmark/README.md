# SpatialBench User Guide

> The unified evaluation harness for SpatialBench. The current scene index ships **546 scenes across 19 datasets**, four density levels (`sparse` / `medium` / `dense` / `single`), and six tag axes. Adapters are added incrementally — see [Currently Supported Models](#currently-supported-models) for what is ready today and what is still pending.

## Quick Start

### Run an Evaluation in 30 Seconds

```bash
# 1. Pick a model config file and run directly
python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/vggt_eval.yaml

# 2. Or override config parameters via CLI
python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/vggt_eval.yaml \
    --tags "droid+sparse" --max-scenes 5 --visualize
```


### Config File Layout

Configs are grouped by **inference paradigm** so the harness can apply the right defaults (e.g. `--shuffle-seed` is rejected for streaming models):

| Directory | Paradigm | Status |
|-----------|----------|--------|
| `benchmark/configs/end2end/` | Feed-forward, all frames at once | **populated** (`vggt`, `vggt_omega`, `da3_{small,base,large,giant}`, `da3nested`, `fastvggt`, `mapanything`, `omnivggt`, `pi3`, `pi3x`, `worldmirror`, `amb3r`) |
| `benchmark/configs/online/` | Per-frame / chunked streaming | **populated** (`infinitevggt`, `lingbot_map_{window,stream}`, `stream3r_{stream,window}`, `streamvggt`, `page4d`) |
| `benchmark/configs/chunk/` | Sliding-window chunk reconstruction | **populated** (`vggt_long`, `pi_long`, `da3_streaming`) |
| `benchmark/configs/ttt/` | Test-time training | **populated** (`scal3r`, `loger`, `loger_star`, `zipmap`, `vgg_ttt`) |
| `benchmark/configs/prior/` | GT prior injection (intrinsics / depth) | **populated** (`da3_giant_prior`, `mapanything_prior`, `omnivggt_prior`, `pi3x_prior`, `worldmirror_prior`) |
| `benchmark/configs/optimization/` | Iterative global alignment (DUSt3R-style) | **populated** (`dust3r`, `mast3r`) |
| `benchmark/configs/slam/` | SLAM-based backends | reserved, empty today |


A complete `end2end` config looks like this (`benchmark/configs/end2end/vggt_eval.yaml`):

```yaml
# ---- Model ----
model: vggt                     # must match an adapter registered under model_adapters/
checkpoint: null                # checkpoint path or HF name; null = auto-download
device: cuda

# ---- Data ----
scene_index: benchmark/scene_indices/all_scenes.json
scene_id: null                  # null = no specific scene restriction

# ---- Scene filtering ----
tags: sparse+scannetpp          # tag expression — see syntax below
max_scenes: null                # limit number of scenes; null = all

# ---- Evaluation parameters ----
eval_metrics:                   # metric categories to compute (null = all)
  - depth                       #   depth: abs_rel, rmse, delta, ...
  - pose                        #   pose: racc, tacc, auc (pairwise)
  - trajectory                  #   trajectory: ATE, RPE (Sim(3) aligned)
  # - pointcloud                #   pointcloud: chamfer, f-score
resolution: null                # [W, H] override; null = use reader default
depth_alignment: median         # median | lstsq (only used by relative-depth models)

# ---- <ModelName> inference parameters ----
# Model-specific knobs are auto-forwarded to the adapter. Examples:
#   ref_view_strategy: first   # DA3: first | saddle_balanced
#   chunk_size: 60             # streaming/chunked models
#   overlap: 30

# ---- Output ----
output_dir: results/end2end/vggt
output: null                    # explicit output path (overrides auto-name)

# ---- Visualization ----
visualize: true                 # save GT / predicted GLB point clouds
vis_conf_percent: 10.0          # filter the lowest N% confidence points (0 = no filter)
```

### Tag Filter Syntax

The `tags` field selects which scenes from the scene index to evaluate:

| Syntax | Meaning | Example |
|--------|---------|---------|
| `dataset` | All scenes from a single dataset | `droid`, `dtu`, `tanks_and_temples` |
| `tag1+tag2` | AND: matches both | `dtu+dense`, `droid+sparse+indoor` |
| `tag1\|tag2` | OR: matches either | `sparse\|dense` |
| `null` | No filter, all scenes | |

### Single-Scene Selection (`scene_id`)

Set `scene_id` to run exactly one scene by its unique ID. When `scene_id` is
set, it overrides the `tags` and `max_scenes` selection, so the runner evaluates
that scene even if it does not match the configured tag expression.

Scene IDs are listed in the scene-index JSON files under
[`benchmark/scene_indices/`](scene_indices/), especially
[`benchmark/scene_indices/all_scenes.json`](scene_indices/all_scenes.json).

YAML example:

```yaml
scene_index: benchmark/scene_indices/all_scenes.json
scene_id: ropedia_slab_01_sparse
tags: scannetpp+dense           # ignored because scene_id is set
max_scenes: 3                   # ignored because scene_id is set
```

CLI example:

```bash
python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/vggt_eval.yaml \
    --scene-id ropedia_slab_01_sparse
```

Available tag values (from `benchmark/scene_indices/all_scenes.json`):

| Tag axis | Possible values |
|----------|-----------------|
| `source_dataset` | `7scenes`, `adt`, `droid`, `dtu`, `eth3d`, `hiroom`, `kitti_odometry`, `lingbot`, `nrgbd`, `omniworld`, `rlbench`, `robolab`, `robotwin`, `ropedia`, `scannetpp`, `tanks_and_temples`, `tum`, `vkitti`, `waymo` |
| `view_density` | `sparse` (5 frames, FPS), `medium` (10 frames, uniform), `dense` (~13 frames, optimal segment), `single` (1 frame) |
| `environment` | `indoor`, `outdoor` |
| `dynamics` | `static`, `dynamic` |
| `view_type` | `wrist`, `egoview`, `normal` |
| `data_type` | `real`, `simulation` |

> **Note on `single`**: `single`-density records contain a single frame, so pose / trajectory / point-cloud metrics are undefined. The harness auto-restricts `eval_metrics` to `["depth"]` whenever the tag expression includes `single` (see [`run_benchmark.py:510`](evaluation/run_benchmark.py#L510)).

### Currently Supported Models

The adapter registry currently exposes the following models (others on the [parent README](../README.md#-models) roadmap are not yet wired up):

| Model | Config | Extra | Metric Depth | Inference Params | Notes |
|-------|--------|-------|:---:|------|-------|
| **Optimization-based** | | | | | |
| DUSt3R | `optimization/dust3r_eval.yaml` | `[optimization]` | — | `niter`, `schedule`, `lr` | Pairwise inference + global alignment |
| MASt3R | `optimization/mast3r_eval.yaml` | `[optimization]` | — | `niter`, `schedule`, `lr` | MASt3R backbone with DUSt3R-style global alignment |
| **End-to-End** | | | | | |
| VGGT | `end2end/vggt_eval.yaml` | `[vggt]` | — | — | Depth + pose + point cloud |
| VGGT-Omega | `end2end/vggt_omega_eval.yaml` | `[vggt]` | — | `resolution_override` | VGGT-Omega 1B camera + depth, patch-size 16 |
| DA3-Small / Base / Large / Giant | `end2end/da3_{small,base,large,giant}_eval.yaml` | `[da3]` | — | `ref_view_strategy` | Depth Anything 3 (ViT-S / B / L / G) |
| DA3-Nested | `end2end/da3nested_eval.yaml` | `[da3]` | ✓ | `ref_view_strategy` | DA3 GIANT + LARGE dual-branch metric-scale |
| FastVGGT | `end2end/fastvggt_eval.yaml` | `[vggt]` | — | `merging`, `merge_ratio`, `enable_point` | VGGT + token merging |
| MAPAnything | `end2end/mapanything_eval.yaml` | `[mapanything]` | ✓ | `memory_efficient_inference`, `use_amp`, … | Supports GT prior (pose/depth/intrinsic) |
| OmniVGGT | `end2end/omnivggt_eval.yaml` | `[vggt]` | — | — | VGGT + omnidirectional, supports GT prior |
| π³ | `end2end/pi3_eval.yaml` | `[vggt]` | — | — | Pi3 multi-view reconstruction |
| π³-X | `end2end/pi3x_eval.yaml` | `[vggt]` | ✓ | — | Pi3X metric-scale variant |
| WorldMirror | `end2end/worldmirror_eval.yaml` | `[vggt]` | — | `cond_flags` | HunyuanWorld-Mirror, supports GT prior |
| AMB3R | `end2end/amb3r_eval.yaml` | `[amb3r]` | ✓ | `resolution`, `data_type` | VGGT enc/dec + PTV3 backend, vendored deps |
| **Online / Streaming** | | | | | |
| InfiniteVGGT | `online/infinitevggt_eval.yaml` | `[streaming]` | — | `total_budget` | KV-cache token budget for long sequences |
| LingbotMap (Window) | `online/lingbot_map_window_eval.yaml` | `[lingbot-map]` | — | `window_size`, `overlap_size`, … | Windowed mode (GCTStreamWindow) |
| LingbotMap (Stream) | `online/lingbot_map_stream_eval.yaml` | `[lingbot-map]` | — | `num_scale_frames`, `keyframe_interval` | Causal streaming mode (GCTStream) |
| Stream3R (Stream) | `online/stream3r_stream_eval.yaml` | `[vggt]` | — | `mode=causal` | STream3R causal aggregator |
| Stream3R (Window) | `online/stream3r_window_eval.yaml` | `[vggt]` | — | `mode=window` | STream3R sliding-window aggregator |
| StreamVGGT | `online/streamvggt_eval.yaml` | `[streaming]` | — | — | Real-time streaming VGGT |
| PAGE4D | `online/page4d_eval.yaml` | `[vggt]` | — | — | VGGT-based 4D reconstruction |
| **Chunk-wise** | | | | | |
| VGGT-Long | `chunk/vggt_long_eval.yaml` | `[vggt]` | — | `chunk_size`, `overlap` | Sim(3)-aligned chunk reconstruction |
| π³-Long | `chunk/pi_long_eval.yaml` | `[vggt]` | — | `chunk_size`, `overlap` | Sim(3)-aligned chunk reconstruction |
| DA3-Streaming | `chunk/da3_streaming_eval.yaml` | `[da3]` | ✓ | `chunk_size`, `overlap`, `ref_view_strategy` | DA3 streaming variant |
| **Test-Time Training** | | | | | |
| Scal3R | `ttt/scal3r_eval.yaml` | `[scal3r]` | — | `block_size`, `overlap_size`, `loop_size`, … | Scale-aware TTT |
| LoGeR | `ttt/loger_eval.yaml` | `[vggt]` | — | `window_size`, `overlap_size`, `variant=LoGeR` | LoGeR base variant |
| LoGeR* | `ttt/loger_star_eval.yaml` | `[vggt]` | — | `window_size`, `overlap_size`, `se3=true`, `variant=LoGeR_star` | LoGeR* SE(3)-aligned variant |
| ZipMap | `ttt/zipmap_eval.yaml` | `[zipmap]` | — | `variant`, `affine_invariant`, `align_first_view`, `window_size` | Vendored ZipMap TTT adapter; default checkpoint `checkpoints/zipmap/checkpoint_aff_inv.pt` |
| VGG-TTT | `ttt/vgg_ttt_eval.yaml` | `[vgg_ttt]` | — | `num_ttt_steps`, `memory_efficient_inference`, `use_global_pred` | Adapter name `vgg_ttt`; vendored source under `benchmark/models/vgg_ttt/`; auto-downloads `nvidia/vgg-ttt` |

Install the per-model extra(s) listed above before running the config (see [parent README "Per-model extras"](../README.md#per-model-extras)).

> **Adding a model**: drop an adapter file under [`benchmark/evaluation/model_adapters/`](evaluation/model_adapters/) with `@register_adapter("name")`, import it in [`run_benchmark.py`](evaluation/run_benchmark.py), and put a YAML config under the appropriate `benchmark/configs/<paradigm>/` directory. See [Integrating a New Model](#integrating-a-new-model).

### Common Evaluation Commands

```bash
# Evaluate VGGT on the full scene index
python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/vggt_eval.yaml

# Evaluate DA3-Giant on dense ScanNet++ scenes
python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/da3_giant_eval.yaml \
    --tags "scannetpp+dense"

# Quick smoke test: 3 scenes, with visualization
python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/vggt_eval.yaml \
    --tags "droid+sparse" --max-scenes 3 --visualize

# Inspect aggregated results
cat results/overall.json | python3 -m json.tool
```

### Currently Supported Datasets (19)

Scene counts below come directly from [`benchmark/scene_indices/all_scenes.json`](scene_indices/all_scenes.json) (546 total scenes; each physical scene typically appears under multiple density levels).

| Dataset | # Scenes | Densities | Environment | Type | Notes |
|---------|:---:|---|---|---|---|
| 7-Scenes | 28 | sparse / medium / dense / single | indoor | real / static | Indoor localization |
| ADT | 16 | sparse / medium / dense / single | indoor | real / dynamic | Aria Digital Twin |
| DROID | 64 | sparse / medium / dense / single | indoor | real / dynamic | Robot manipulation (wrist view) |
| DTU | 39 | sparse / medium / single | indoor | real / static | Multi-view stereo |
| ETH3D | 24 | sparse / medium / single | indoor / outdoor | real / static | High-precision MVS (COLMAP) |
| HiRoom | 18 | sparse / medium / single | indoor | simulation / static | Synthetic indoor (aliasing_mask filtered) |
| KITTI-Odometry | 11 | dense | outdoor | real / dynamic | KITTI odometry split (LiDAR depth) |
| Lingbot | 50 | single | indoor / outdoor | real / dynamic | Lingbot robot single-frame scenes |
| NRGBD | 32 | sparse / medium / dense / single | indoor | real / static | Neural RGB-D |
| OmniWorld | 19 | sparse / medium / dense / single | outdoor | simulation / dynamic | Game-engine virtual outdoor scenes |
| RLBench | 38 | sparse / medium / dense / single | indoor | simulation | Robot simulation tasks |
| RoboLab | 32 | sparse / medium / dense / single | indoor | simulation / dynamic | Isaac Sim synthetic (wrist view) |
| RoboTwin | 31 | sparse / medium / dense / single | indoor | simulation | Bimanual robot simulation |
| Ropedia | 8 | sparse / medium / dense / single | indoor | real / dynamic | Robot egocentric view |
| ScanNet++ | 44 | sparse / medium / dense / single | indoor | real / static | iPhone subset (COLMAP + rendered depth) |
| Tanks & Temples | 16 | sparse / medium / dense / single | outdoor | real / static | Outdoor large scenes (RobustMVD) |
| TUM | 24 | sparse / medium / dense / single | indoor | real / dynamic | RGB-D SLAM |
| VKITTI | 20 | sparse / medium / dense / single | outdoor | simulation / dynamic | Virtual KITTI 2 |
| Waymo | 32 | sparse / medium / dense / single | outdoor | real / dynamic | Waymo Open (LiDAR depth) |

Aggregate breakdown across the index:
- **By density**: `sparse` 129 · `medium` 129 · `dense` 109 · `single` 179
- **By environment**: `indoor` 426 · `outdoor` 120
- **By dynamics**: `static` 251 · `dynamic` 295
- **By view type**: `wrist` 179 · `normal` 343 · `egoview` 24
- **By data type**: `real` 362 · `simulation` 184

---

## Overview

SpatialBench is a deterministic, density-aware framework for evaluating 3D reconstruction models across paradigms. Key features:

- **Deterministic frame selection**: Test frames for each scene are precomputed (FPS / uniform / optimal-contiguous-segment / single) and pinned, so all users evaluate on exactly the same frames.
- **Scene-level loading**: Each `__getitem__` returns every selected frame of a complete scene, suited for multi-view models.
- **Multi-axis tags**: Compose queries across `view_density` / `environment` / `dynamics` / `view_type` / `source_dataset` / `data_type`.
- **YAML config files**: Each model has its own default config grouped by paradigm; edit it directly or override via CLI.
- **Model adapters**: Unified interface — adding a new model only requires implementing `predict()`.

---

## Directory Layout

```
benchmark/
├── configs/                       # Evaluation config files (grouped by paradigm)
│   ├── end2end/                   #   feed-forward, all frames at once  [populated]
│   ├── online/                    #   per-frame / streaming             [populated]
│   ├── optimization/              #   iterative global alignment        [populated]
│   ├── prior/                     #   GT prior injection                [populated]
│   ├── slam/                      #   SLAM backends                     [reserved]
│   └── ttt/                       #   test-time training                [populated]
├── datasets/
│   ├── data_readers.py            # per-dataset readers (DROID / Ropedia / DTU / …)
│   └── benchmark_dataset.py       # BenchmarkDataset + TagRegistry
├── evaluation/
│   ├── alignment.py               # prediction ↔ GT alignment (median / lstsq / procrustes)
│   ├── metrics.py                 # depth / pose / point-cloud metric computation
│   ├── report.py                  # JSON report generation and aggregation
│   ├── run_benchmark.py           # main evaluation entry point
│   └── model_adapters/
│       ├── base_adapter.py        # ModelAdapter abstract base class
│       ├── vggt_adapter.py        # @register_adapter("vggt")
│       └── da3_adapter.py         # @register_adapter("da3") — auto-detects S/B/L/G
├── models/                        # bundled model source
│   ├── depth_anything_3/
│   ├── dust3r_root/
│   ├── mast3r_root/
│   ├── vgg_ttt/
│   ├── zipmap/
│   └── vggt/
├── utils/                         # helpers (cropping, image_ranking, visualization)
├── scene_indices/                 # precomputed per-dataset + merged scene indices
│   ├── all_scenes.json            # ← merged index used by every config
│   ├── droid_scenes.json
│   ├── ropedia_scenes.json
│   └── … (one per dataset)
└── README.md                      # this file
```

---

## Step 1: Prepare the Datasets

The benchmark datasets are released on Hugging Face as a single bundle [`HarrisonPENG/SpatialBenchmark`](https://huggingface.co/datasets/HarrisonPENG/SpatialBenchmark). Download with the CLI:

```bash
# Full benchmark (all four density splits)
huggingface-cli download HarrisonPENG/SpatialBenchmark --repo-type dataset --local-dir SpatialBenchmark

# Or restrict to specific density regime(s) — single / sparse / medium / dense
huggingface-cli download HarrisonPENG/SpatialBenchmark --repo-type dataset \
    --local-dir SpatialBenchmark --include "sparse/*"
```

After downloading, the tree is organized **by density first, then dataset** (each density split contains the datasets it covers — see the [Currently Supported Datasets](#currently-supported-datasets-19) table for which densities each dataset ships):

```
SpatialBenchmark/
├── single/         # 1 anchor frame per scene
│   ├── 7scenes/  adt/  droid/  dtu/  eth3d/  hiroom/  lingbot/  nrgbd/
│   ├── omniworld/  rlbench/  robolab/  robotwin/  ropedia/  scannetpp/
│   └── tanks_and_temples/  tum/  vkitti/  waymo/
├── sparse/         # 5 frames, farthest-point sampled in pose space
│   ├── 7scenes/  adt/  droid/  dtu/  eth3d/  hiroom/  nrgbd/  omniworld/
│   ├── rlbench/  robolab/  robotwin/  ropedia/  scannetpp/  tanks_and_temples/
│   └── tum/  vkitti/  waymo/
├── medium/         # 10 frames, uniform stride
│   └── (same dataset list as sparse)
├── dense/          # ~13 frames, optimal contiguous segment
│   └── 7scenes/  adt/  droid/  kitti_odometry/  nrgbd/  omniworld/  rlbench/
│       robolab/  robotwin/  ropedia/  scannetpp/  tanks_and_temples/
│       tum/  vkitti/  waymo/
└── _split_log.jsonl
```

### Per-scene Layout

Each scene under `<density>/<dataset>/<scene_id>/` is normalized to a uniform on-disk format so all readers in [`benchmark/datasets/data_readers.py`](datasets/data_readers.py) consume the same files. A DROID scene, for example:

```
SpatialBenchmark/sparse/droid/Fri_Jul_14_15:19:25_2023/18026681/
├── images/            # RGB (*.png), one per selected frame
├── depths/            # uint16 PNG, /1000 → metric meters (per meta.depth_format)
├── depth_masks/       # uint8, 0 = invalid / flying-point pixels
├── intrinsics/        # shared intrinsic.npy (3, 3); per-frame <stem>.npy for some datasets
├── poses/             # per-frame cam2world (3, 4); loader also accepts (4, 4)
└── meta.json          # scene metadata: tags, frame_indices, depth_format, intrinsic_mode, ...
```

Datasets with extra modalities (Ropedia confidence masks, ScanNet++ rendered depth, etc.) add a sibling subdirectory inside the scene folder; see the docstrings in [`data_readers.py`](datasets/data_readers.py) for the exact spec of every dataset.

> **No preprocessing required**: scene indices are pre-built and shipped with the repo under [`benchmark/scene_indices/`](scene_indices/). Every evaluation reads from the merged [`all_scenes.json`](scene_indices/all_scenes.json) (546 scenes, 19 datasets, 4 densities) — just point each config's `scene_index:` at this file.

---

## Step 2: Prepare a Model

### Bundled Model Source

| Model family | Source path | Default checkpoint |
|---|---|---|
| VGGT | [`benchmark/models/vggt/`](models/vggt/) | auto-download `facebook/VGGT-1B` |
| VGGT-Omega | [`benchmark/models/vggt_omega/`](models/vggt_omega/) | `checkpoints/VGGT-Omega` or auto-download `facebook/VGGT-Omega` |
| Depth Anything 3 | [`benchmark/models/depth_anything_3/`](models/depth_anything_3/) | `depth-anything/DA3-GIANT-1.1` (DA3 adapter auto-detects variant from checkpoint name: `DA3-SMALL` / `DA3-BASE` / `DA3-LARGE-1.1` / `DA3-GIANT-1.1`) |
| DUSt3R | [`benchmark/models/dust3r_root/`](models/dust3r_root/) | `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` |
| MASt3R | [`benchmark/models/mast3r_root/`](models/mast3r_root/) | `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` |
| ZipMap | [`benchmark/models/zipmap/`](models/zipmap/) | `checkpoints/zipmap/checkpoint_aff_inv.pt`; download with `hf download coast01/ZipMap checkpoint_aff_inv.pt --local-dir checkpoints/zipmap` |
| VGG-TTT | [`benchmark/models/vgg_ttt/`](models/vgg_ttt/) | auto-download `nvidia/vgg-ttt`; or set `checkpoint` to a local Hugging Face snapshot directory |

DUSt3R / MASt3R vendor their own CroCo copies. CroCo's CUDA RoPE extension is
optional; when it is not compiled, the models use the slower PyTorch fallback.
ZipMap vendors the inference source under `benchmark/models/zipmap/`; install
its benchmark extra with `pip install -e ".[zipmap]"` before running
`benchmark/configs/ttt/zipmap_eval.yaml`.
VGG-TTT vendors the NVIDIA source under `benchmark/models/vgg_ttt/`; install
its benchmark extra with `pip install -e ".[vgg_ttt]"` before running
`benchmark/configs/ttt/vgg_ttt_eval.yaml`.

### Integrating a New Model

**1. Create the adapter file** `benchmark/evaluation/model_adapters/your_model_adapter.py`:

```python
import numpy as np
import torch
from benchmark.evaluation.model_adapters import register_adapter
from benchmark.evaluation.model_adapters.base_adapter import ModelAdapter

@register_adapter("your_model")
class YourModelAdapter(ModelAdapter):

    def name(self):
        return "YourModel"

    def load_model(self, checkpoint=None, device="cuda"):
        self.device = device
        from your_model import YourModel
        self.model = YourModel.load(checkpoint).to(device).eval()

    def predict(self, scene):
        """
        `scene` provides:
          - images:     Tensor (N, 3, H, W)  ImageNet-normalized
          - images_raw: Tensor (N, 3, H, W)  [0, 1] unnormalized
          - intrinsic:  ndarray (N, 3, 3)    camera intrinsics
          - depth:      ndarray (N, H, W)    GT depth (DO NOT use during inference!)
          - extrinsic:  ndarray (N, 3, 4)    GT cam2world pose (DO NOT use during inference!)

        Return a dict with any subset of the following keys
        (missing keys cause the corresponding metric to be skipped):
          - pred_depth:      ndarray (N, H, W)  predicted depth
          - pred_pose:       ndarray (N, 3, 4)  predicted cam2world poses
          - w2c_extrinsics:  ndarray (N, 3, 4)  predicted world-to-camera poses
          - pred_pointcloud: ndarray (M, 3)     predicted point cloud
          - pred_confidence: ndarray (N, H, W)  per-pixel confidence
        """
        images = scene['images_raw'].to(self.device)
        with torch.no_grad():
            output = self.model(images)
        pred_c2w = output['poses'].cpu().numpy()  # benchmark standard: cam2world
        pred_w2c = self._invert_se3(pred_c2w)     # required for pairwise camera metrics
        return {
            'pred_depth': output['depth'].cpu().numpy(),
            'pred_pose':  pred_c2w,
            'w2c_extrinsics': pred_w2c,
        }

    def supports_metric_depth(self):
        return True  # True = compare directly; False = align via median / lstsq scaling
```

**2. Register the import in `run_benchmark.py`** (add a line near the other adapter imports):

```python
import benchmark.evaluation.model_adapters.your_model_adapter
```

**3. Create the config file** `benchmark/configs/<paradigm>/your_model_eval.yaml`. Pick the directory that matches the inference paradigm (`end2end` / `online` / `optimization` / `prior` / `slam` / `ttt`).

**4. Run the evaluation** (see Step 3).

### Key Caveats

- `pred_pose` must be **cam2world** (3×4). Do not put world-to-camera poses in this key.
- If your adapter returns `pred_pose`, it must also return `w2c_extrinsics` as **world-to-camera** (3×4). Pairwise camera pose metrics (`racc` / `tacc` / `auc`) read `w2c_extrinsics`, while trajectory and point-cloud paths read `pred_pose`.
- Convert both directions inside the adapter: if your model emits world-to-camera, store it in `w2c_extrinsics` and invert it for `pred_pose`; if it emits cam2world, store it in `pred_pose` and invert it for `w2c_extrinsics`.
- `pred_depth` should match the input resolution `(H, W)`. Resize inside the adapter if needed.
- `supports_metric_depth()` controls how depth is evaluated:
  - `True`: reports two sets of depth metrics — `depth_metric` (no alignment) **and** `depth` (after alignment).
  - `False`: reports only `depth` (after `median` / `lstsq` alignment).

---

## Step 3: Run the Evaluation

### Recommended: Use a Config File

Every supported model ships a YAML config with sensible defaults. Edit the config directly, or override fields via CLI.

```bash
# Use the VGGT default config
python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/vggt_eval.yaml

# Config + CLI override
python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/vggt_eval.yaml \
    --tags "sparse+indoor" --max-scenes 5 --visualize

# Use the DA3-Giant config
python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/da3_giant_eval.yaml
```

**Precedence**: explicit CLI arguments > config file values > argparse defaults

### Pure CLI (no config file)

```bash
python benchmark/evaluation/run_benchmark.py \
    --model vggt \
    --scene-index benchmark/scene_indices/all_scenes.json \
    --tags "sparse" \
    --output-dir results
```

### Full CLI Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | YAML config file path | None |
| `--model` | Registered adapter name (`vggt`, `da3`, …) | (required) |
| `--checkpoint` | Checkpoint path or HuggingFace name | None (auto-download) |
| `--device` | Inference device | `cuda` |
| `--checkpoints-dir` | Cache directory for auto-downloaded HF weights | `DEFAULT_CHECKPOINTS_DIR` |
| `--scene-index` | Scene index JSON | (required) |
| `--scene-id` | Run one scene by unique `scene_id` from `benchmark/scene_indices/`; overrides `tags` and `max_scenes` | None |
| `--tags` | Tag expression (`+` = AND, `\|` = OR) | None (all) |
| `--max-scenes` | Cap the number of scenes (smoke test) | None |
| `--eval-metrics` | Subset of `{depth, pose, trajectory, pointcloud}` | None (all) |
| `--conf-threshold` | Ropedia confidence threshold | `0.5` |
| `--depth-alignment` | `median` \| `lstsq` (relative-depth models only) | `median` |
| `--shuffle-seed` | Base seed for per-scene frame-order shuffling. **end2end / prior / optimization only** — rejected on `/online/` configs. Per-scene permutation = `seed + hash(scene_id)`; GT/images stay aligned. | None |
| `--priority-datasets` | Move scenes of the named `source_dataset`(s) to the front of the queue. | None |
| `--output-dir` | Output directory | `results` |
| `--output` | Explicit output JSON path (overrides auto-name) | None |
| `--visualize` | Save GT / predicted GLB point clouds | False |
| `--vis-conf-percent` | Visualization confidence percentile filter (0 = keep all) | `20.0` |

### Tag Query Examples

- `"sparse"` — all sparse scenes
- `"sparse+indoor"` — sparse AND indoor
- `"sparse|dense"` — sparse OR dense
- `"scannetpp+dense"` — dense ScanNet++ scenes
- `"single"` — single-frame scenes only (eval auto-restricted to depth)

See the [Tag Filter Syntax](#tag-filter-syntax) table above for the full list of axes and values.

### Frame-Order Shuffle (`shuffle_seed`)

Off by default. Setting `shuffle_seed: <int>` in YAML (or `--shuffle-seed <int>` on the CLI) deterministically permutes the **frame order within each sequence** before it is fed to the model. Use it to stress-test models whose output silently depends on input frame ordering (e.g. those that cache the first frame as an anchor).

The shuffle is applied in `BenchmarkDataset.__getitem__` ([`benchmark/datasets/benchmark_dataset.py:378`](datasets/benchmark_dataset.py#L378)) — `images`, `depth`, `extrinsic`, `intrinsic`, `valid_mask`, `sky_mask`, `world_points`, `frame_indices` are all permuted **together** with the same `perm`, so the `pred[i] ↔ gt[i]` relationship is preserved and metrics stay comparable across runs with different seeds.

```yaml
# benchmark/configs/end2end/vggt_eval.yaml
shuffle_seed: 42          # any int → deterministic per-scene frame shuffle
# shuffle_seed: null      # default → frames kept in the index-defined order
```

- Per-scene seed = `shuffle_seed + hash(scene_id)`, so the same `shuffle_seed` reproduces the same permutation every run.
- Output filename gets a `_shuffle{seed}` suffix and the seed is recorded under `meta.shuffle_seed` in the report JSON.
- **Restriction**: rejected for `/online/` configs (streaming models are designed for sequential temporal input). Allowed for `end2end` / `optimization` / `prior` / `slam` / `ttt`.

---

## Step 4: Result Analysis

### Output Files

Each evaluation produces two files:

1. **Detailed report** `results/<output_dir>/{model}_{benchmark}_{tags}_{WxH}_{timestamp}.json` — per-scene results and multi-axis aggregates.
2. **Overview file** `results/overall.json` — one record appended per evaluation; only core metrics are kept, ideal for side-by-side comparison.

> **Scene-level resumability**: After each scene completes, `run_benchmark.py` atomically writes a progress snapshot (with a signature covering `model` / `scene_index` / `tags` / `max_scenes` / ordered scene IDs) to `<output>.partial`. After a crash or OOM, the next run skips already-finished scenes. The partial is discarded on signature mismatch and removed on successful completion.

### `overall.json` Format

```json
[
  {
    "model": "VGGT",
    "checkpoint": "",
    "query": "all",
    "num_scenes": 45,
    "depth": { "abs_rel": 0.0812, "rmse": 0.134, "delta_1.25": 0.952 },
    "camera": { "racc_5": 0.85, "auc_5": 55.3, "auc_15": 78.2 },
    "mean_inference_seconds": 2.1,
    "per_dataset": {
      "droid":   { "num_scenes": 25, "abs_rel": 0.092, "racc_5": 0.79 },
      "ropedia": { "num_scenes": 20, "abs_rel": 0.068, "racc_5": 0.92 }
    }
  },
  { "model": "DA3", "query": "all", "...": "..." }
]
```

### Detailed Report Structure

```
{
  "meta":                  { ... },   # evaluation metadata (model, config, timestamp)
  "per_scene":             [ ... ],   # per-scene detailed results
  "aggregate":             { ... },   # global aggregates (mean / median / std)
  "per_dataset_breakdown": { ... },   # grouped by source_dataset
  "per_tag_breakdown":     { ... },   # grouped by tag
  "efficiency":            { ... }    # inference time, peak memory, etc.
}
```

### Metric Definitions

#### Depth Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| `abs_rel` | Mean absolute relative error `|pred - gt| / gt` | lower is better |
| `sq_rel` | Mean squared relative error `(pred - gt)² / gt` | lower is better |
| `rmse` | Root mean squared error | lower is better |
| `log_rmse` | RMSE in log space | lower is better |
| `delta_1.03` / `delta_1.05` / `delta_1.10` / `delta_1.25` | Fraction of pixels with `max(pred/gt, gt/pred)` below the threshold | higher is better |

For metric-depth models the report contains both `depth_metric` (no alignment) and `depth` (after median / lstsq alignment).

#### Camera Pose Metrics

All pose metrics are based on the **pairwise relative pose error between frame pairs**. Predicted poses are aligned to GT via **Procrustes alignment** before computation.

| Metric | Description | Range |
|--------|-------------|-------|
| `racc_3` / `racc_5` | Fraction of frame pairs with rotation error < 3° / 5° | [0, 1] |
| `tacc_3` / `tacc_5` | Fraction of frame pairs with translation-direction error < 3° / 5° | [0, 1] |
| `auc_3` / `auc_5` / `auc_15` / `auc_30` | AUC of `max(rot_err, trans_err)` | [0, 100] |

#### Trajectory Metrics

| Metric | Description |
|--------|-------------|
| `ATE` | Absolute trajectory error after Sim(3) alignment (evo) |
| `RPE` | Relative pose error after Sim(3) alignment (evo) |

#### Point Cloud Metrics

| Metric | Description |
|--------|-------------|
| `chamfer_distance` | Mean bidirectional nearest-neighbor distance |
| `f_score` | F1 at distance threshold τ = 0.05 |

---

## Appendix: Full Workflow Cheat Sheet

```bash
# 1. Prepare data (download to ./SpatialBenchmark/)
#    Scene indices are pre-built and ship with the repo at
#    benchmark/scene_indices/all_scenes.json — no preprocessing step needed.

# 2. Run evaluation (recommended: use a config file)
python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/vggt_eval.yaml

python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/da3_giant_eval.yaml

# 2b. With visualization
python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/vggt_eval.yaml \
    --visualize --vis-conf-percent 50

# 3. Inspect results
cat results/overall.json | python3 -m json.tool
```
