<div align="center">
<h1 style="border-bottom: none; margin-bottom: 0px ">R³: 3D Reconstruction via Relative Regression</h1>

<a href='https://kevinxu02.github.io/r3-site/'><img src='https://img.shields.io/badge/Project_Page-R3-green' alt='Project Page'></a>
<a href='https://huggingface.co/KevinXu02/R3'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-blue' alt='Checkpoints'></a>
<a href='https://arxiv.org/abs/2605.26519'><img src='https://img.shields.io/badge/arXiv-2605.26519-b31b1b' alt='arXiv'></a>

</div>

This work presents **R³**, a feed-forward model that reconstructs camera poses and dense geometry from arbitrarily long video streams via *relative-pose regression*. Instead of regressing every camera in one global frame, R³ predicts confidence-weighted pairwise relative poses on top of a Depth Anything 3 backbone, then assembles a consistent global trajectory downstream.

Two ideas keep the modeling minimal:

- A **lightweight pairwise pose MLP** sits on a DA3 backbone — no recurrent state, no TTT modules, no extra transformer.
- A **single learned confidence per edge** (decoupled into rotation and translation) drives loss weighting, pose aggregation, and keyframe-bank management.

With **372M parameters** (≈⅓ of recent 1B-class models), R³ matches or surpasses state-of-the-art streaming methods on pose estimation and dense reconstruction, runs at **20+ FPS**, and scales to **thousands of frames** under a bounded memory budget.

## 📰 News

- **2026-05-26:** Inference-only public release with `r3` and `r3_long` checkpoints.

## 🗂️ Release TODO

- [ ] Evaluation code.
- [ ] Training code.

## 🚀 Quick Start

### 📦 Installation

```bash
conda env create -f environment.yml
conda activate r3
pip install -e .
```

If you already have a CUDA-enabled PyTorch environment, install dependencies directly:

```bash
pip install -r requirements.txt
pip install -e .
```

### 🧱 Checkpoints

Place weights under:

```text
ckpt/r3.safetensors
ckpt/r3_long.safetensors
```

Both are available on [Hugging Face](https://huggingface.co/KevinXu02/R3):

| Name      | File                                                                                       | Train views | Best for                       | Notes                                                                                                       |
|-----------|--------------------------------------------------------------------------------------------|-------------|--------------------------------|-------------------------------------------------------------------------------------------------------------|
| `r3`      | [`r3.safetensors`](https://huggingface.co/KevinXu02/R3/blob/main/r3.safetensors)           | 4–32        | Indoor / small-coverage scenes | Default checkpoint, reported in the paper. Stronger local consistency on short clips.                       |
| `r3_long` | [`r3_long.safetensors`](https://huggingface.co/KevinXu02/R3/blob/main/r3_long.safetensors) | 32–100      | Outdoor / long trajectories    | Used by `--mode long` and `--mode strided` unless `--ckpt` is passed explicitly.                             |

### 💻 Run the Demo

```bash
python demo.py --seq_path examples/indoor --no_viewer
```

`demo.py` runs inference with `infer.py`, writes depth / color / confidence / camera files to `--output_dir`, and then opens the saved run in a [Viser](https://viser.studio/) viewer.
By default, the demo uses `--mode test`, which keeps all KV entries and skips fallback / metric scale for a quick smoke run. Use `--mode local`, `--mode long`, or `--mode strided` for the release presets.
Sky-mask export is available through `--compute_sky_mask` when the selected model emits a `sky` tensor; the default R3 checkpoint does not emit one.

Presets cover the common regimes:

```bash
python demo.py --mode test     # quick test run, all KV cache
python demo.py --mode local    # indoor scenes, small coverage
python demo.py --mode long     # long trajectories, large outdoor scenes
python demo.py --mode strided  # temporally strided video
```

(`--mode short`, `--mode sampled`, and `--mode sparse` are kept as legacy aliases.)

To reopen a saved run without re-running inference:

```bash
python view.py --data_dir scratch/demo/<run_name>
```

## 🙏 Acknowledgement

Our code is built upon the following repositories:

- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [STream3R](https://github.com/NIRVANALAN/STream3R)

We thank the authors for their excellent work.

## 📝 Citation

If R³ is useful in your research or projects, please cite:

```bibtex
@article{r3_2026,
  title  = {R^3: 3D Reconstruction via Relative Regression},
  author = {Anonymous},
  year   = {2026},
  note   = {Paper coming soon}
}
```

Please also cite the works above if you use this codebase.
