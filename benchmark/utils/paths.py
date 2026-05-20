import os


def _repo_root() -> str:
    # benchmark/utils/paths.py -> .../code/EgoExpert_benchmark
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Per your machine environment: DROID raw data root directory
DEFAULT_DROID_ROOT = "/mnt/local/lihao/phs_datasets/droid"

# Per your requirements: place HuggingFace weights uniformly under checkpoints
DEFAULT_CHECKPOINTS_DIR = os.path.join(_repo_root(), "checkpoints")

