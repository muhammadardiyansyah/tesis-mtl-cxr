from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset

from src.utils.config import (
    PROJECT_ROOT,
    get_config_path,
    load_config,
)


PathInput = Union[str, Path]


def resolve_csv_path(
    csv_file: PathInput
) -> Path:
    """
    Menentukan lokasi CSV secara konsisten,
    baik dijalankan dari root proyek maupun notebook.
    """

    normalized = str(csv_file).replace("\\", "/")
    supplied_path = Path(normalized)

    if supplied_path.is_absolute():
        resolved = supplied_path
    else:
        candidates = [
            Path.cwd() / supplied_path,
            PROJECT_ROOT / supplied_path,
        ]

        lower_path = normalized.lower()
        data_position = lower_path.find("data/")

        if data_position >= 0:
            relative_from_data = normalized[data_position:]

            candidates.append(
                PROJECT_ROOT / relative_from_data
            )

        resolved = next(
            (
                candidate.resolve()
                for candidate in candidates
                if candidate.exists()
            ),
            candidates[0].resolve(),
        )

    if not resolved.exists():
        raise FileNotFoundError(
            f"File CSV tidak ditemukan: {resolved}"
        )

    return resolved


def resolve_image_path(
    stored_path: str,
    raw_data_dir: PathInput
) -> Path:
    """
    Mengubah path lama di CSV menjadi path
    yang mengarah ke folder raw pada HDD.
    """

    if pd.isna(stored_path):
        raise ValueError(
            "Kolom image_path berisi nilai kosong."
        )

    normalized = str(stored_path).strip()
    normalized = normalized.replace("\\", "/")

    direct_path = Path(normalized)

    if direct_path.is_absolute() and direct_path.exists():
        return direct_path

    marker = "data/raw/"
    marker_position = normalized.lower().find(marker)

    if marker_position >= 0:
        relative_path = normalized[
            marker_position + len(marker):
        ]
    else:
        relative_path = normalized.lstrip("./")

    return (
        Path(raw_data_dir)
        / Path(relative_path)
    )


class ChestXrayDataset(Dataset):
    """
    Dataset multi-task untuk:
    1. Kardiomegali
    2. Tuberkulosis

    Label 0 dan 1 dianggap valid.
    NaN dan -1 dianggap tidak tersedia
    dan ditutup menggunakan mask.
    """

    REQUIRED_COLUMNS = {
        "image_path",
        "cardiomegaly",
        "tb",
        "source",
    }

    def __init__(
        self,
        csv_file: PathInput,
        transform=None,
        raw_data_dir: Optional[PathInput] = None,
    ):
        self.csv_path = resolve_csv_path(csv_file)

        self.dataframe = pd.read_csv(
            self.csv_path
        )

        missing_columns = (
            self.REQUIRED_COLUMNS
            - set(self.dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "Kolom CSV tidak lengkap. "
                f"Kolom yang hilang: {missing_columns}"
            )

        if raw_data_dir is None:
            config = load_config()

            raw_data_dir = get_config_path(
                config,
                "raw_data"
            )

        self.raw_data_dir = Path(raw_data_dir)

        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                "Folder dataset mentah tidak ditemukan: "
                f"{self.raw_data_dir}"
            )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def get_image_path(
        self,
        idx: int
    ) -> Path:
        stored_path = self.dataframe.iloc[
            idx
        ]["image_path"]

        return resolve_image_path(
            stored_path=stored_path,
            raw_data_dir=self.raw_data_dir,
        )

    def __getitem__(
        self,
        idx: int
    ):
        row = self.dataframe.iloc[idx]

        image_path = self.get_image_path(idx)

        if not image_path.exists():
            raise FileNotFoundError(
                "Citra tidak ditemukan.\n"
                f"Indeks CSV : {idx}\n"
                f"Path       : {image_path}\n"
                f"Sumber     : {row['source']}"
            )

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        raw_labels = [
            row["cardiomegaly"],
            row["tb"],
        ]

        labels = []

        for value in raw_labels:
            try:
                labels.append(float(value))
            except (TypeError, ValueError):
                labels.append(np.nan)

        labels = np.asarray(
            labels,
            dtype=np.float32,
        )

        mask = np.isin(
            labels,
            [0.0, 1.0]
        )

        labels = np.where(
            mask,
            labels,
            0.0
        ).astype(np.float32)

        labels = torch.tensor(
            labels,
            dtype=torch.float32,
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32,
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, labels, mask