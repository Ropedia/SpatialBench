import glob
import os
from typing import Iterable, Optional

from huggingface_hub import snapshot_download


def _safe_local_dir_name(repo_id: str) -> str:
    # repo_id usually contains "/", which cannot be directly concatenated as a single path to avoid generating overly deep directories.
    return repo_id.replace("/", "__")


def _has_any_weight_file(local_dir: str, exts: Iterable[str]) -> bool:
    for ext in exts:
        for p in glob.glob(os.path.join(local_dir, f"*{ext}")):
            if os.path.isfile(p):
                return True
    return False


def ensure_hf_snapshot(
    repo_id: str,
    *,
    local_root: str,
    revision: Optional[str] = None,
) -> str:
    """
    Ensure the HuggingFace repo is available locally, and place the download under your specified checkpoints directory whenever possible.

    Returns: local snapshot directory path (containing config.json + weight files).
    """
    os.makedirs(local_root, exist_ok=True)
    target_dir = os.path.join(local_root, _safe_local_dir_name(repo_id))

    config_path = os.path.join(target_dir, "config.json")
    # PyTorchModelHubMixin typically requires config.json + weights (which may be safetensors or bin).
    has_config = os.path.isfile(config_path)
    has_weights = _has_any_weight_file(target_dir, exts=[".safetensors", ".bin", ".pt", ".ckpt"])

    if has_config and has_weights:
        return target_dir

    # local_dir specifies the output directory; if the directory exists it will reuse/resume the download as much as possible.
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target_dir,
    )
    return target_dir

