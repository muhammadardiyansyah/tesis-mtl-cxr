"""Dataset path, label, and image-integrity audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.datasets.dataset_loader import resolve_image_path
from src.utils.config import PROJECT_ROOT, get_config_path, load_config


RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit integritas dataset CXR.")
    parser.add_argument(
        "--readability",
        choices=("none", "sample", "full"),
        default="sample",
        help="full mendekode seluruh citra; sample hanya mengambil sampel per sumber.",
    )
    parser.add_argument("--sample-per-source", type=int, default=50)
    return parser.parse_args()


def calculate_label_statistics(series: pd.Series) -> dict[str, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    return {
        "negative_0": int((numeric == 0).sum()),
        "positive_1": int((numeric == 1).sum()),
        "uncertain_-1": int((numeric == -1).sum()),
        "missing_nan": int(numeric.isna().sum()),
        "other_values": int(
            (numeric.notna() & ~numeric.isin([-1, 0, 1])).sum()
        ),
    }


def decode_error(image_path: Path) -> str | None:
    """Return an error message, or None when the complete image decodes."""
    try:
        with Image.open(image_path) as image:
            image.load()
            image.convert("RGB")
        return None
    except Exception as error:  # Audit must record every decoder failure.
        return f"{type(error).__name__}: {error}"


def record_from_row(index, row, resolved_path: Path, reason: str, error="") -> dict:
    return {
        "row_index": int(index),
        "source": str(row["source"]),
        "stored_path": str(row["image_path"]).replace("\\", "/"),
        "resolved_path": str(resolved_path),
        "reason": reason,
        "error": error,
    }


def main() -> None:
    args = parse_args()
    if args.sample_per_source < 1:
        raise ValueError("--sample-per-source minimal 1.")

    config = load_config()
    raw_data_dir = get_config_path(config, "raw_data")
    metadata_path = PROJECT_ROOT / "data" / "metadata" / "harmonized_clean.csv"
    output_dir = PROJECT_ROOT / "reports" / "data_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {metadata_path}")

    dataframe = pd.read_csv(metadata_path)
    required = {"image_path", "cardiomegaly", "tb", "source"}
    missing_columns = required - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Kolom metadata hilang: {sorted(missing_columns)}")

    cardio = pd.to_numeric(dataframe["cardiomegaly"], errors="coerce")
    tb = pd.to_numeric(dataframe["tb"], errors="coerce")
    no_valid_label = ~cardio.isin([0, 1]) & ~tb.isin([0, 1])
    source_counts = {
        str(key): int(value) for key, value in dataframe["source"].value_counts().items()
    }

    print("=" * 68)
    print("AUDIT INTEGRITAS DATASET TESIS MTL CXR")
    print("=" * 68)
    print(f"Metadata          : {metadata_path}")
    print(f"Raw data          : {raw_data_dir}")
    print(f"Mode keterbacaan : {args.readability}")
    if args.readability == "full":
        print("Seluruh citra akan didekode; proses dapat berlangsung beberapa jam.")

    resolved_paths: set[str] = set()
    duplicate_path_count = 0
    existing_count = 0
    invalid_records: list[dict] = []
    checked_decode_count = 0

    sample_indices: set[int] = set()
    if args.readability == "sample":
        for _, group in dataframe.groupby("source"):
            sample_size = min(args.sample_per_source, len(group))
            sample_indices.update(
                int(index)
                for index in group.sample(
                    n=sample_size,
                    random_state=RANDOM_SEED,
                ).index
            )

    for index, row in tqdm(
        dataframe.iterrows(),
        total=len(dataframe),
        desc="Audit citra",
    ):
        image_path = resolve_image_path(row["image_path"], raw_data_dir)
        normalized_path = str(image_path).casefold()
        if normalized_path in resolved_paths:
            duplicate_path_count += 1
        else:
            resolved_paths.add(normalized_path)

        if not image_path.is_file():
            invalid_records.append(
                record_from_row(index, row, image_path, "missing_file")
            )
            continue

        existing_count += 1
        try:
            file_size = image_path.stat().st_size
        except OSError as error:
            invalid_records.append(
                record_from_row(index, row, image_path, "stat_error", str(error))
            )
            continue
        if file_size == 0:
            invalid_records.append(
                record_from_row(index, row, image_path, "zero_byte")
            )
            continue

        should_decode = args.readability == "full" or int(index) in sample_indices
        if should_decode:
            checked_decode_count += 1
            error = decode_error(image_path)
            if error is not None:
                invalid_records.append(
                    record_from_row(index, row, image_path, "decode_error", error)
                )

    invalid_columns = [
        "row_index",
        "source",
        "stored_path",
        "resolved_path",
        "reason",
        "error",
    ]
    invalid_dataframe = pd.DataFrame(invalid_records, columns=invalid_columns)
    invalid_output = output_dir / "invalid_images.csv"
    invalid_dataframe.to_csv(invalid_output, index=False)

    reason_counts = Counter(record["reason"] for record in invalid_records)
    summary = {
        "metadata_file": str(metadata_path),
        "raw_data_directory": str(raw_data_dir),
        "readability_mode": args.readability,
        "total_rows": int(len(dataframe)),
        "source_counts": source_counts,
        "cardiomegaly_labels": calculate_label_statistics(dataframe["cardiomegaly"]),
        "tb_labels": calculate_label_statistics(dataframe["tb"]),
        "rows_without_valid_label": int(no_valid_label.sum()),
        "existing_images": int(existing_count),
        "invalid_images": int(len(invalid_records)),
        "invalid_reason_counts": dict(reason_counts),
        "decoded_images_checked": int(checked_decode_count),
        "duplicate_paths": int(duplicate_path_count),
        "random_seed": RANDOM_SEED,
    }
    summary_output = output_dir / "audit_summary.json"
    summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("HASIL AUDIT")
    print("=" * 68)
    print(f"Total metadata       : {len(dataframe):,}")
    print(f"Citra ditemukan      : {existing_count:,}")
    print(f"Citra invalid        : {len(invalid_records):,}")
    print(f"Citra didekode       : {checked_decode_count:,}")
    print(f"Path duplikat        : {duplicate_path_count:,}")
    print(f"Tanpa label valid    : {int(no_valid_label.sum()):,}")
    for reason, count in sorted(reason_counts.items()):
        print(f"  {reason}: {count:,}")
    print(f"\nRingkasan            : {summary_output}")
    print(f"Manifest citra invalid: {invalid_output}")


if __name__ == "__main__":
    main()
