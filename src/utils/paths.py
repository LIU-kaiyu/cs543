from pathlib import Path
from typing import Optional
import yaml


_PROJECT_ROOT: Optional[Path] = None
_CONFIG: Optional[dict] = None


def project_root() -> Path:
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        # Walk up from this file until we find configs/dataset_paths.yaml
        candidate = Path(__file__).resolve()
        for parent in candidate.parents:
            if (parent / "configs" / "dataset_paths.yaml").exists():
                _PROJECT_ROOT = parent
                return _PROJECT_ROOT
        raise RuntimeError("Could not locate project root (no configs/dataset_paths.yaml found)")
    return _PROJECT_ROOT


def _load_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        cfg_path = project_root() / "configs" / "dataset_paths.yaml"
        # with open(cfg_path) as f:
        with open(cfg_path, encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_dataset_path(dataset: str, key: str) -> Path:
    cfg = _load_config()
    raw = cfg[dataset][key]
    p = Path(raw)
    if not p.is_absolute():
        p = project_root() / p
    return p


def get_output_path(name: str) -> Path:
    cfg = _load_config()
    raw = cfg["outputs"][name]
    p = Path(raw)
    if not p.is_absolute():
        p = project_root() / p
    p.mkdir(parents=True, exist_ok=True)
    return p
