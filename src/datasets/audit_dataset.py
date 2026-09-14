import json
from collections import Counter
from pathlib import Path

import pandas as pd

from PIL import Image
from tqdm import tqdm

from src.datasets.dataset_loader import (
    resolve_image_path,
)
from src.utils.config import (
    PROJECT_ROOT,
    get_config_path,
    load_config,
)


RANDOM_SEED = 42
READABILITY_SAMPLE_PER_SOURCE = 50


def calculate_label_statistics(
    series: pd.Series
) -> dict:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return {
        "negative_0": int((numeric == 0).sum()),
        "positive_1": int((numeric == 1).sum()),
        "uncertain_-1": int((numeric == -1).sum()),
        "missing_nan": int(numeric.isna().sum()),
        "other_values": int(
            (
                numeric.notna()
                & ~numeric.isin([-1, 0, 1])
            ).sum()
        ),
    }


def main():
    config = load_config()

    raw_data_dir = get_config_path(
        config,
        "raw_data",
    )

    metadata_path = (
        PROJECT_ROOT
        / "data"
        / "metadata"
        / "harmonized_clean.csv"
    )

    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "data_audit"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata tidak ditemukan: {metadata_path}"
        )

    print("=" * 60)
    print("AUDIT DATASET TESIS MTL CXR")
    print("=" * 60)
    print(f"Metadata : {metadata_path}")
    print(f"Raw data : {raw_data_dir}")
    print()

    dataframe = pd.read_csv(metadata_path)

    required_columns = {
        "image_path",
        "cardiomegaly",
        "tb",
        "source",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Kolom metadata hilang: {missing_columns}"
        )

    cardiomegaly = pd.to_numeric(
        dataframe["cardiomegaly"],
        errors="coerce",
    )

    tuberculosis = pd.to_numeric(
        dataframe["tb"],
        errors="coerce",
    )

    cardiomegaly_valid = (
        cardiomegaly.isin([0, 1])
    )

    tuberculosis_valid = (
        tuberculosis.isin([0, 1])
    )

    no_valid_label = (
        ~cardiomegaly_valid
        & ~tuberculosis_valid
    )

    source_counts = {
        str(key): int(value)
        for key, value
        in dataframe["source"].value_counts().items()
    }

    missing_records = []
    resolved_paths = set()
    duplicate_path_count = 0
    existing_count = 0

    print(
        f"Memeriksa {len(dataframe):,} path citra..."
    )
    print(
        "Proses ini dapat memerlukan beberapa menit "
        "karena data berada di HDD eksternal."
    )
    print()

    for index, row in tqdm(
        dataframe.iterrows(),
        total=len(dataframe),
        desc="Memeriksa path",
    ):
        resolved_path = resolve_image_path(
            stored_path=row["image_path"],
            raw_data_dir=raw_data_dir,
        )

        normalized_path = str(
            resolved_path
        ).lower()

        if normalized_path in resolved_paths:
            duplicate_path_count += 1
        else:
            resolved_paths.add(normalized_path)

        if resolved_path.is_file():
            existing_count += 1
        else:
            missing_records.append({
                "row_index": int(index),
                "source": row["source"],
                "stored_path": row["image_path"],
                "resolved_path": str(resolved_path),
            })

    missing_dataframe = pd.DataFrame(
        missing_records
    )

    missing_output = (
        output_dir / "missing_images.csv"
    )

    missing_dataframe.to_csv(
        missing_output,
        index=False,
    )

    # Pemeriksaan sampel citra rusak/tidak terbaca
    unreadable_records = []

    for source, group in dataframe.groupby(
        "source"
    ):
        sample_size = min(
            READABILITY_SAMPLE_PER_SOURCE,
            len(group),
        )

        sample = group.sample(
            n=sample_size,
            random_state=RANDOM_SEED,
        )

        print(
            f"Memeriksa keterbacaan "
            f"{sample_size} citra dari {source}..."
        )

        for index, row in sample.iterrows():
            image_path = resolve_image_path(
                stored_path=row["image_path"],
                raw_data_dir=raw_data_dir,
            )

            if not image_path.is_file():
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()

            except Exception as error:
                unreadable_records.append({
                    "row_index": int(index),
                    "source": source,
                    "image_path": str(image_path),
                    "error": str(error),
                })

    unreadable_dataframe = pd.DataFrame(
        unreadable_records
    )

    unreadable_output = (
        output_dir / "unreadable_sample.csv"
    )

    unreadable_dataframe.to_csv(
        unreadable_output,
        index=False,
    )

    summary = {
        "metadata_file": str(metadata_path),
        "raw_data_directory": str(raw_data_dir),
        "total_rows": int(len(dataframe)),
        "source_counts": source_counts,
        "cardiomegaly_labels": (
            calculate_label_statistics(
                dataframe["cardiomegaly"]
            )
        ),
        "tb_labels": (
            calculate_label_statistics(
                dataframe["tb"]
            )
        ),
        "rows_without_valid_label": int(
            no_valid_label.sum()
        ),
        "existing_images": int(existing_count),
        "missing_images": int(
            len(missing_records)
        ),
        "duplicate_paths": int(
            duplicate_path_count
        ),
        "unreadable_images_in_sample": int(
            len(unreadable_records)
        ),
        "readability_sample_per_source": (
            READABILITY_SAMPLE_PER_SOURCE
        ),
        "random_seed": RANDOM_SEED,
    }

    summary_output = (
        output_dir / "audit_summary.json"
    )

    with summary_output.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("HASIL AUDIT")
    print("=" * 60)
    print(f"Total metadata       : {len(dataframe):,}")
    print(f"Citra ditemukan      : {existing_count:,}")
    print(f"Citra tidak ditemukan: {len(missing_records):,}")
    print(f"Path duplikat        : {duplicate_path_count:,}")
    print(
        "Tanpa label valid    : "
        f"{int(no_valid_label.sum()):,}"
    )
    print(
        "Sampel tidak terbaca : "
        f"{len(unreadable_records):,}"
    )
    print()
    print("Jumlah berdasarkan sumber:")

    for source, count in source_counts.items():
        print(f"  {source}: {count:,}")

    print()
    print(f"Ringkasan: {summary_output}")
    print(f"Path hilang: {missing_output}")
    print(f"Sampel rusak: {unreadable_output}")


if __name__ == "__main__":
    main()