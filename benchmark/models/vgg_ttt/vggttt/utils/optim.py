# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import contextlib
import logging
import re
from functools import partial
from typing import Any, TypedDict

import torch

_logger = logging.getLogger(__name__)


def freeze_weights(module: torch.nn.Module, patterns: list[str]) -> None:
    for param_name, param in module.named_parameters():
        if any(re.search(regex, param_name) for regex in patterns):
            param.requires_grad = False


class OptimGroup(TypedDict):
    patterns: list[str]
    optim_options: dict[str, Any]


def make_parameter_groups(
    module: torch.nn.Module,
    groups: list[OptimGroup] = (),
) -> list[dict[str, Any]]:
    """Create parameter groups for an optimizer based on regex patterns on the parameter names.

    Example for groups:
    ```
    [
        {
            "patterns": ["Conv2d"],
            "group_options": {"lr": 0.01},
        }
    ]
    ```
    """
    parameter_groups = []
    model_parameters = {k: v for k, v in module.named_parameters() if v.requires_grad}
    for group in groups:
        group_params = []
        group_param_names = []
        for param_name, param in model_parameters.items():
            if any(re.search(regex, param_name) for regex in group["patterns"]):
                group_params.append(param)
                group_param_names.append(param_name)

        # Remove parameters that were assigned to parameter group from global set of parameters
        for param_name in group_param_names:
            model_parameters.pop(param_name)

        if group_params:
            _logger.info(
                "Assigned options %s to %s. Total parameters in group: %s",
                group["optim_options"],
                group["patterns"],
                f"{sum(p.numel() for p in group_params):,}",
            )

            # Ignore frozen params
            group_params = [p for p in group_params if p.requires_grad]
            if group_params:
                parameter_groups.append({"params": group_params, **group["optim_options"]})
        else:
            invalid_patterns = group["patterns"]
            raise ValueError(f"No parameters were found for {invalid_patterns}.")

    # Assign all remaining parameters that did not match any regex to default group
    if model_parameters:
        parameter_groups.append({"params": [p for p in model_parameters.values() if p.requires_grad]})

    # Remove parameters that have zero learning rate
    optimize_params = [p for p in parameter_groups if "lr" not in p or p["lr"] != 0.0]
    return optimize_params


def checkpoint(func, *args, enabled: bool, offload: bool = False, use_reentrant: bool = False, **kwargs):
    # Offload actually does something: ~20% memory savings. Iteration speed goes from 0.07s/iter -> 0.06s/iter.
    if not enabled:
        return func(*args)

    save_on_cpu_context = (
        partial(torch.autograd.graph.save_on_cpu, pin_memory=True) if offload else contextlib.nullcontext
    )

    with save_on_cpu_context():
        return torch.utils.checkpoint.checkpoint(
            func,
            *args,
            use_reentrant=use_reentrant,
            **kwargs,
        )


def load_weights(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    """Loads weights into a model from a state.

    Additionally considers the case where there is a shape mismatch which usually leads to an error.
    """
    current_model_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        if k not in current_model_dict:
            new_state_dict[k] = v  # This will be handled later by `load_state_dict`
            continue

        current_model_shape = current_model_dict[k].shape
        if v.shape == current_model_shape:
            new_state_dict[k] = v
        else:
            _logger.warning(f"Skipping key {k} ({v.shape}) due to mismatch with model's shape {current_model_shape}")

    return model.load_state_dict(new_state_dict, strict=False)
