from pathlib import Path
from typing import Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - only used in lean local envs
    yaml = None


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
        with open(cfg_path, encoding="utf-8") as f:
            if yaml is not None:
                _CONFIG = yaml.safe_load(f)
            else:
                _CONFIG = _load_simple_yaml(f.read())
    return _CONFIG


def _load_simple_yaml(text: str) -> dict:
    """
    Parse the small two-level dataset_paths.yaml file without PyYAML.

    This keeps local utility scripts usable in partially configured
    environments; full YAML support is still provided by PyYAML when present.
    """
    cfg: dict[str, dict[str, str]] = {}
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            cfg[current_section] = {}
            continue

        if current_section is None or ":" not in line:
            continue

        key, value = line.strip().split(":", 1)
        cfg[current_section][key.strip()] = value.strip()

    return cfg


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
