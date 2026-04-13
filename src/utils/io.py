from pathlib import Path
import numpy as np
import pandas as pd


def save_npy(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(str(path))


def save_metrics_csv(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)


def load_metrics_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(str(path))


def save_metrics_parquet(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)
