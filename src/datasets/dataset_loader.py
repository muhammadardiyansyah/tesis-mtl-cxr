"""Dataset loader for partially labelled multi-task chest X-rays."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from src.utils.config import PROJECT_ROOT, get_config_path, load_config


PathInput = Union[str, Path]


def resolve_csv_path(csv_file: PathInput) -> Path:
    """Resolve a CSV consistently from the project root or current directory."""
    normalized = str(csv_file).replace("\\", "/")
    supplied_path = Path(normalized)

    if supplied_path.is_absolute():
        resolved = supplied_path
    else:
        candidates = [Path.cwd() / supplied_path, PROJECT_ROOT / supplied_path]
        data_position = normalized.lower().find("data/")
        if data_position >= 0:
            candidates.append(PROJECT_ROOT / normalized[data_position:])
        resolved = next(
            (candidate.resolve() for candidate in candidates if candidate.exists()),
            candidates[0].resolve(),
        )

    if not resolved.exists():
        raise FileNotFoundError(f"File CSV tidak ditemukan: {resolved}")
    return resolved


def resolve_image_path(stored_path: str, raw_data_dir: PathInput) -> Path:
    """Map a portable path stored in metadata to the configured raw-data root."""
    if pd.isna(stored_path):
        raise ValueError("Kolom image_path berisi nilai kosong.")

    normalized = str(stored_path).strip().replace("\\", "/")
    direct_path = Path(normalized)
    if direct_path.is_absolute() and direct_path.exists():
        return direct_path

    marker = "data/raw/"
    marker_position = normalized.lower().find(marker)
    if marker_position >= 0:
        relative_path = normalized[marker_position + len(marker) :]
    else:
        relative_path = normalized.lstrip("./")
    return Path(raw_data_dir) / Path(relative_path)


class ChestXrayDataset(Dataset):
    """CXR dataset with cardiomegaly and tuberculosis label masks."""

    REQUIRED_COLUMNS = {"image_path", "cardiomegaly", "tb", "source"}

    def __init__(
        self,
        csv_file: PathInput,
        transform=None,
        raw_data_dir: Optional[PathInput] = None,
    ) -> None:
        self.csv_path = resolve_csv_path(csv_file)
        self.dataframe = pd.read_csv(self.csv_path)
        missing_columns = self.REQUIRED_COLUMNS - set(self.dataframe.columns)
        if missing_columns:
            raise ValueError(f"Kolom CSV tidak lengkap: {sorted(missing_columns)}")

        if raw_data_dir is None:
            config = load_config()
            raw_data_dir = get_config_path(config, "raw_data")
        self.raw_data_dir = Path(raw_data_dir)
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"Folder dataset mentah tidak ditemukan: {self.raw_data_dir}"
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def get_image_path(self, idx: int) -> Path:
        stored_path = self.dataframe.iloc[idx]["image_path"]
        return resolve_image_path(stored_path, self.raw_data_dir)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        image_path = self.get_image_path(idx)
        context = (
            f"Indeks CSV : {idx}\n"
            f"Path       : {image_path}\n"
            f"Sumber     : {row['source']}"
        )

        if not image_path.is_file():
            raise FileNotFoundError(f"Citra tidak ditemukan.\n{context}")
        if image_path.stat().st_size == 0:
            raise RuntimeError(f"Citra berukuran 0 byte.\n{context}")

        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
                image.load()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise RuntimeError(
                f"Citra rusak atau tidak dapat didekode.\n{context}\nError: {error}"
            ) from error

        raw_labels = [row["cardiomegaly"], row["tb"]]
        labels = []
        for value in raw_labels:
            try:
                labels.append(float(value))
            except (TypeError, ValueError):
                labels.append(np.nan)

        labels_array = np.asarray(labels, dtype=np.float32)
        mask_array = np.isin(labels_array, [0.0, 1.0])
        labels_array = np.where(mask_array, labels_array, 0.0).astype(np.float32)

        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            torch.tensor(labels_array, dtype=torch.float32),
            torch.tensor(mask_array, dtype=torch.float32),
        )
