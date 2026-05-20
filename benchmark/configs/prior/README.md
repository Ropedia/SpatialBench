# GT Prior Evaluation Configs

This directory contains configs for running SpatialBench with ground-truth
prior injection. Use these configs with:

```bash
python benchmark/evaluation/run_benchmark_with_prior.py \
  --config benchmark/configs/prior/omnivggt_prior_eval.yaml
```

The prior runner uses the same scene index, tag filtering, metrics, output, and
visualization conventions as the standard benchmark runner, but additionally
passes selected ground-truth camera, depth, and intrinsic inputs into adapters
that support them.

## Prior switches

The common YAML keys are:

```yaml
use_gt_camera: false            # GT pose + GT intrinsics
use_gt_depth: false             # GT depth maps
# use_gt_intrinsic_only: false  # GT intrinsics only, no pose
gt_camera_ratio: 1.0            # fraction of frames receiving camera prior
gt_depth_ratio: 1.0             # fraction of frames receiving depth prior
gt_seed: 42                     # deterministic frame selection seed
```

Equivalent CLI flags can be used to enable priors and override ratios:

```bash
python benchmark/evaluation/run_benchmark_with_prior.py \
  --config benchmark/configs/prior/mapanything_prior_eval.yaml \
  --use-gt-camera \
  --use-gt-depth \
  --gt-camera-ratio 0.5 \
  --gt-depth-ratio 0.25 \
  --gt-seed 42
```

Notes:

- `use_gt_camera` means pose and intrinsics together. It sets
  `use_pose=True` and `use_intrinsic=True` inside the runner.
- `use_gt_intrinsic_only` injects intrinsics without pose. Leave
  `use_gt_camera: false`; if both are set, `use_gt_camera` takes precedence.
- `gt_camera_ratio` controls the frames used for camera prior. `gt_depth_ratio`
  controls the frames used for depth prior. The two frame sets are sampled
  independently.
- For models with `partial=True`, ratios in `(0, 1)` select a deterministic
  per-scene subset of frames. For models with `partial=False`, any positive
  ratio is treated as all frames.
- CLI boolean flags only turn options on. To turn off a prior that is enabled
  in a YAML file, edit the YAML file.

## Method support

| Config | Model key | Partial frames | Camera prior setting | Depth prior setting | Notes |
| --- | --- | --- | --- | --- | --- |
| `da3_giant_prior_eval.yaml` | `da3` | No | `use_gt_camera: true`, `gt_camera_ratio: 1.0` | Not supported. Keep `use_gt_depth: false`. | DA3 accepts GT camera inputs all-or-nothing. |
| `danext_prior_eval.yaml` | `danext` | No | `use_gt_camera: true`, `gt_camera_ratio: 1.0` | Not supported. Keep `use_gt_depth: false`. | Placeholder config. The `danext` adapter is not wired up yet. |
| `mapanything_prior_eval.yaml` | `mapanything` | Yes | `use_gt_camera: true`, set `gt_camera_ratio` to `0.0-1.0` | `use_gt_depth: true`, set `gt_depth_ratio` to `0.0-1.0` | Supports camera, depth, and intrinsic-only priors. Frame 0 is added to camera prior when needed. |
| `omnivggt_prior_eval.yaml` | `omnivggt` | Yes | `use_gt_camera: true`, set `gt_camera_ratio` to `0.0-1.0` | `use_gt_depth: true`, set `gt_depth_ratio` to `0.0-1.0` | `camera_gt_index` controls pose and intrinsics together. Frame 0 is added to camera prior when needed. |
| `pi3x_prior_eval.yaml` | `pi3x` | Yes | `use_gt_camera: true`, set `gt_camera_ratio` to `0.0-1.0` | `use_gt_depth: true`, set `gt_depth_ratio` to `0.0-1.0` | Uses separate masks for depth, rays/intrinsics, and pose. |
| `worldmirror_prior_eval.yaml` | `worldmirror` | No | `use_gt_camera: true`, `gt_camera_ratio: 1.0` | `use_gt_depth: true`, `gt_depth_ratio: 1.0` | All-or-nothing. Internally maps to `cond_flags=[pose, depth, intrinsic]`. |

## Common recipes

Camera prior only:

```yaml
use_gt_camera: true
use_gt_depth: false
gt_camera_ratio: 1.0
gt_depth_ratio: 0.0
```

Depth prior only, for models that support depth prior:

```yaml
use_gt_camera: false
use_gt_depth: true
gt_camera_ratio: 0.0
gt_depth_ratio: 1.0
```

Partial camera and depth prior, for `mapanything`, `omnivggt`, or `pi3x`:

```yaml
use_gt_camera: true
use_gt_depth: true
gt_camera_ratio: 0.5
gt_depth_ratio: 0.5
gt_seed: 42
```

Intrinsic-only prior, for models that support standalone intrinsics:

```yaml
use_gt_camera: false
use_gt_depth: false
use_gt_intrinsic_only: true
gt_camera_ratio: 1.0
```

Single-scene debugging:

```bash
python benchmark/evaluation/run_benchmark_with_prior.py \
  --config benchmark/configs/prior/pi3x_prior_eval.yaml \
  --scene-id <scene_id> \
  --visualize
```

When `--scene-id` is used, the runner also saves the input images under the run
directory, which is useful for checking the exact frames that were evaluated.

## Outputs

If `output` is not set, the runner writes results under `output_dir` using an
auto-generated run name that includes the enabled prior types and ratios, for
example:

```text
results/prior/omnivggt_prior/omnivggt_all_scenes_sparse_nrgbd_YYYYMMDD_HHMMSS_prior_pose50_depth25_intr50/
```

The JSON report records the effective prior configuration in `meta.gt_prior`.
Each scene result also contains a `gt_prior` block with:

- `camera_gt_indices`
- `depth_gt_indices`
- `n_camera_gt`
- `n_depth_gt`
- `n_total_frames`
- `gt_camera_ratio`
- `gt_depth_ratio`

Use these fields to verify which frames actually received GT inputs.
