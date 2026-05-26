<div align="center">
<h1>VGG-T³: Offline Feed-Forward 3D Reconstruction at Scale</h1>

<a href="https://arxiv.org/abs/2602.23361"><img src="https://img.shields.io/badge/arXiv-2602.23361-b31b1b.svg" alt="arXiv"></a>
<a href="https://research.nvidia.com/labs/dvl/projects/vgg-ttt/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/nvidia/vgg-ttt"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" alt="Hugging Face Model"></a>

**[NVIDIA](https://www.nvidia.com/)** &nbsp;&nbsp;&nbsp; **[University of Toronto](https://www.utoronto.ca/)** &nbsp;&nbsp;&nbsp; **[Vector Institute](https://vectorinstitute.ai/)**

[Sven Elflein](https://selflein.github.io/), [Ruilong Li](https://www.liruilong.cn/), [Sérgio Agostinho](https://sergioagostinho.com/), [Zan Gojcic](https://zgojcic.github.io/), [Laura Leal-Taixé](https://dvl.in.tum.de/team/lealtaixe/), [Qunjie Zhou](https://research.nvidia.com/labs/dvl/author/qunjie-zhou/), [Aljosa Osep](https://aljosaosep.github.io/)
</div>

## Overview

VGG-T³ processes large image collections significantly faster than other feed-forward methods (_1k images in <1 minute vs. 10 minutes for VGGT_) by replacing the quadratic-scaling softmax attention in the global attention layers with a linear alternative based on test-time training.

## Quick Start

Clone this repo and then install (preferably in a conda environment):
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install .
```

VGG-T³ is compatible with the VGGT API and can be used in a similar way:

```python
from vggttt.nets.vggt.models.vggt import VGGT
from vggttt.nets.vggt.img import load_and_preprocess_images

vggttt = VGGT.from_pretrained("nvidia/vgg-ttt").eval().cuda()

image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]
images = load_and_preprocess_images(image_names).to("cuda")

preds = vggttt.infer(images)
# Dict containing the predicted outputs with the following keys:
#  - 'pose':        [#images, 4, 4]  Camera-to-world transformation
#  - 'intrinsics':  [#images, 3, 3]  Pinhole camera matrix
#  - 'pts3d':       [#images, height, width, 3]  Per-pixel points in world coordinates
#  - 'conf':        [#images, height, width]  Per-pixel confidence in range ]1, inf[
#  - 'depth':       [#images, height, width, 1]  Per-pixel depth
```


## Demo

We provide an interactive web interface to perform 3D reconstruction of images and videos and visualize the result.

```bash
python vggttt/demo.py
```
Note: When running on a remote server you need to forward both the viser **and** Gradio port. See the CLI output for details.

## Evaluation

Find details on how to reproduce the results in the paper [here](./vggttt/evaluation/README.md).

## Training
We release the training harness, however, dataset implementations and preprocessing code is missing. We are currently in the process of checking feasibility for releasing the relevant code.

## Acknowledgmens

We are also grateful to several other open-source repositories that we drew inspiration from or built upon during the development of our pipeline:
- [VGGT](https://github.com/facebookresearch/vggt)
- [Pi3](https://github.com/yyfz/Pi3)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [MapAnything](https://github.com/facebookresearch/map-anything)
- [LaCT](https://github.com/a1600012888/LaCT)


## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{elflein2026vggttt,
  title     = {VGG-T\textsuperscript{3}: Offline Feed-Forward 3D Reconstruction at Scale},
  author    = {Elflein, Sven and Li, Ruilong and Agostinho, S{\'e}rgio and Gojcic, Zan and Leal-Taix{\'e}, Laura and Zhou, Qunjie and Osep, Aljosa},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```


## License

The code and model are released under the [NVIDIA OneWay Noncommercial License](./LICENSE), with the following exceptions:

- `vggttt/nets/vggt/` — released under the [VGGT license](./vggttt/nets/vggt/LICENSE.txt).
- `vggttt/nets/ttt.py` — adapted from [LaCT](https://github.com/a1600012888/LaCT) and released under the [MIT License](https://github.com/a1600012888/LaCT/blob/main/LICENSE).
- `vggttt/evaluation/pointmaps/utils.py` — adapted from [CUT3R](https://github.com/CUT3R/CUT3R) and released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

See [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md) for the full license texts of all third-party components.
