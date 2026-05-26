# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

try:
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
except Exception:
    pass


import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HYDRA_FULL_ERROR"] = "1"
