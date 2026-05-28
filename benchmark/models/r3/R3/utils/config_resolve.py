"""Auto-resolve a config_name to (config_path, ckpt_path).

Search order:
  config: configs/{name}.yaml, depth_anything_3/configs/{name}.yaml
  ckpt:   ckpt/<config_name>/last.ckpt, ckpt/**/*.{safetensors,ckpt}, logs/**/checkpoints/*

Token-overlap + mtime is used to disambiguate when multiple checkpoints match.
"""

import glob
import os
from importlib import resources

from omegaconf import OmegaConf


def normalize_config_name(config_name: str) -> str:
    return os.path.splitext(os.path.basename(config_name))[0]


def infer_ckpt_path_from_config_name(config_name: str, repo_root: str = None):
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    log_root = os.path.join(repo_root, "logs")
    ckpt_root = os.path.join(repo_root, "ckpt")

    patterns = [
        os.path.join(ckpt_root, config_name, "last.ckpt"),
        os.path.join(ckpt_root, config_name, "*.safetensors"),
        os.path.join(ckpt_root, config_name, "*.ckpt"),
        os.path.join(ckpt_root, "**", "*.safetensors"),
        os.path.join(ckpt_root, "**", "*.ckpt"),
        os.path.join(log_root, "**", f"*{config_name}*", "checkpoints", "*.ckpt"),
        os.path.join(log_root, "**", f"*{config_name}*", "checkpoints", "*.safetensors"),
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))
    if not candidates:
        return None

    tokens = [t for t in config_name.replace("-", "_").split("_") if t]

    def _score(path):
        path_lower = path.lower()
        return sum(1 for token in tokens if token.lower() in path_lower)

    candidates = sorted(set(candidates), key=lambda p: (_score(p), os.path.getmtime(p)), reverse=True)
    return candidates[0]


def resolve_model_config(config_name: str, checkpoint_dir: str, repo_root: str = None):
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if not config_name:
        raise ValueError("--config_name is required when using auto resolve")

    if os.path.exists(config_name):
        config_path = config_name
        normalized_name = normalize_config_name(config_name)
    else:
        normalized_name = normalize_config_name(config_name)
        config_candidates = [
            os.path.join(repo_root, "configs", f"{normalized_name}.yaml"),
            os.path.join(repo_root, "R3", "configs", f"{normalized_name}.yaml"),
            os.path.join(repo_root, "depth_anything_3", "configs", f"{normalized_name}.yaml"),
        ]
        config_path = next((p for p in config_candidates if os.path.exists(p)), None)
        if config_path is None:
            for package in ("R3", "depth_anything_3"):
                try:
                    candidate = resources.files(package).joinpath(
                        "configs", f"{normalized_name}.yaml"
                    )
                except (ModuleNotFoundError, AttributeError):
                    continue
                if candidate.is_file():
                    config_path = str(candidate)
                    break
        if config_path is None:
            raise FileNotFoundError(
                f"Cannot infer config path for '{normalized_name}'. Looked in: {config_candidates}"
            )

    ckpt_path = checkpoint_dir or None
    if not ckpt_path:
        try:
            cfg = OmegaConf.load(config_path)
            task_name = cfg.get("task_name", None)
            if task_name:
                checkpoints_dir = os.path.join(repo_root, "ckpt", task_name)
                if os.path.isdir(checkpoints_dir):
                    last = os.path.join(checkpoints_dir, "last.ckpt")
                    if os.path.exists(last):
                        ckpt_path = last
        except Exception:
            ckpt_path = None

    if not ckpt_path:
        ckpt_path = infer_ckpt_path_from_config_name(normalized_name, repo_root=repo_root)

    if ckpt_path is None:
        raise FileNotFoundError("Cannot infer checkpoint path. Pass --checkpoint_dir explicitly.")

    return {"config": config_path, "ckpt_path": ckpt_path}
