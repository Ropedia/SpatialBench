# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# CroCo 子包导入路径（适配 SpatialBench 的 mast3r_root vendored 布局）
# --------------------------------------------------------

import sys
import os.path as path

HERE_PATH = path.normpath(path.dirname(__file__))
# benchmark/models/mast3r_root/dust3r/utils/ -> benchmark/models/mast3r_root/croco
CROCO_REPO_PATH = path.normpath(path.join(HERE_PATH, '..', '..', 'croco'))
CROCO_MODELS_PATH = path.join(CROCO_REPO_PATH, 'models')

if path.isdir(CROCO_MODELS_PATH):
    sys.path.insert(0, CROCO_REPO_PATH)
    # 处理 'models' 命名空间冲突：benchmark/models/src/ 下可能有其他项目的 models/ 包。
    # 如果 sys.modules 中已有 models 且不是 croco 的，临时替换。
    _prev_models = sys.modules.pop('models', None)
    import models as _croco_models  # noqa: force import from croco
    if _prev_models is not None and 'croco' not in str(getattr(_prev_models, '__file__', '')):
        # 保存 croco 的 models，稍后两者都可用
        sys.modules['croco_models'] = _croco_models
        sys.modules['models'] = _croco_models  # dust3r 代码用 'from models.xxx'
else:
    raise ImportError(f"croco is not initialized, could not find: {CROCO_MODELS_PATH}.\n "
                      "Please ensure croco models are copied to benchmark/models/mast3r_root/croco/")
