# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R 子包导入路径（适配 SpatialBench 的 mast3r_root vendored 布局）
# --------------------------------------------------------
# 原始 MASt3R 从 ../../dust3r/ git submodule 中导入 dust3r，
# 这里改为从 benchmark/models/mast3r_root/dust3r 包导入。

import sys
import os.path as path

HERE_PATH = path.normpath(path.dirname(__file__))
# benchmark/models/mast3r_root/mast3r/utils/ -> benchmark/models/mast3r_root/
SRC_ROOT = path.normpath(path.join(HERE_PATH, '..', '..'))
DUSt3R_LIB_PATH = path.join(SRC_ROOT, 'dust3r')

if path.isdir(DUSt3R_LIB_PATH):
    # 将 benchmark/models/src/ 加入 sys.path，使 import dust3r 可用
    if SRC_ROOT not in sys.path:
        sys.path.insert(0, SRC_ROOT)
else:
    raise ImportError(
        f"dust3r is not initialized, could not find: {DUSt3R_LIB_PATH}.\n"
        "Please ensure dust3r is available at benchmark/models/mast3r_root/dust3r/"
    )
