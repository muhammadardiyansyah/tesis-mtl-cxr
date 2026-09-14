"""Build leakage-safe train/validation/test splits for the MTL CXR thesis.

Run from the project root with:
    python build_patient_level_splits.py

The script is intentionally non-destructive. It reads harmonized_clean.csv and
writes a new versioned split directory. Existing train/val/test CSV files are
not overwritten.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "data" / "metadata" / "harmonized_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits" / "patient_level_v1"
READY_METADATA_CSV = (
    PROJECT_ROOT / "data" / "metadata" / "harmonized_model_ready_v1.csv"
)
REPORT_JSON = OUTPUT_DIR / "split_report.json"

RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

REQUIRED_COLUMNS = {"image_path", "cardiomegaly", "tb", "source"}
VALID_SOURCES = {"nih", "chexpert", "tbx11k"}


def normalize_relative_path(value: object) -> str:
    """Return a portable path relative to the configured raw-data folder."""
    if pd.isna(value):
        raise ValueError("image_path contains a missing value")

    text = str(value).strip().replace("\\", "/")
    marker = "data/raw/"
    position = text.lower().find(marker)
    if position >= 0:
        text = text[position + len(marker) :]

    while text.startswith("./") or text.startswith("../"):
        if text.startswith("./"):
            text = text[2:]
        elif text.startswith("../"):
            text = text[3:]

    return text


def extract_patient_id(source: str, image_path: str) -> str:
    """Extract a source-prefixed patient/group identifier."""
    normalized = image_path.replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]
    stem = Path(filename).stem

    if source == "chexpert":
        match = re.search(r"(?:^|/)patient(\d+)(?:/|$)", normalized.lower())
        if not match:
            raise ValueError(f"CheXpert patient ID not found in: {image_path}")
        return f"chexpert_patient{match.group(1)}"

    if source == "nih":
        patient_token = stem.split("_")[0]
        if not patient_token.isdigit():
            raise ValueError(f"NIH patient ID not found in: {image_path}")
        return f"nih_patient{patient_token}"

    if source == "tbx11k":
        # TBX11K does not provide a patient identifier in the current metadata.
        # Each image is therefore treated as one independent grouping unit.
        return f"tbx11k_image_{stem.lower()}"

    raise ValueError(f"Unsupported source: {source}")


def apply_label_policy(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented binary label policy for each source.

    CheXpert cardiomegaly policy:
      blank/NaN -> 0 (unmentioned-as-weak-negative)
      0         -> 0
      1         -> 1
      -1        -> missing/masked (U-Ignore)

    A value of -1 in the output means that the task is unavailable for that
    image and must be ignored by MaskedBCELoss.
    """
    result = dataframe.copy()
    result["source"] = result["source"].astype(str).str.strip().str.lower()

    unknown_sources = set(result["source"].unique()) - VALID_SOURCES
    if unknown_sources:
        raise ValueError(f"Unknown dataset sources: {sorted(unknown_sources)}")

    raw_cardio = pd.to_numeric(result["cardiomegaly"], errors="coerce")
    raw_tb = pd.to_numeric(result["tb"], errors="coerce")

    cardio = pd.Series(-1, index=result.index, dtype="int8")
    tb = pd.Series(-1, index=result.index, dtype="int8")

    nih_mask = result["source"].eq("nih")
    nih_valid = nih_mask & raw_cardio.isin([0, 1])
    cardio.loc[nih_valid] = raw_cardio.loc[nih_valid].astype("int8")

    chexpert_mask = result["source"].eq("chexpert")
    chexpert_known = chexpert_mask & raw_cardio.isin([0, 1])
    cardio.loc[chexpert_known] = raw_cardio.loc[chexpert_known].astype("int8")
    cardio.loc[chexpert_mask & raw_cardio.isna()] = 0
    # CheXpert -1 remains -1 (U-Ignore).

    tbx_mask = result["source"].eq("tbx11k")
    tbx_valid = tbx_mask & raw_tb.isin([0, 1])
    tb.loc[tbx_valid] = raw_tb.loc[tbx_valid].astype("int8")

    result["cardiomegaly"] = cardio
    result["tb"] = tb

    usable = result["cardiomegaly"].isin([0, 1]) | result["tb"].isin([0, 1])
    return result.loc[usable].copy()


def split_source_groups(source_df: pd.DataFrame) -> pd.DataFrame:
    """Split one dataset source by patient/group with stratification."""
    source = str(source_df["source"].iloc[0])
    target_column = "tb" if source == "tbx11k" else "cardiomegaly"

    group_table = (
        source_df.groupby("patient_id", as_index=False)[target_column]
        .max()
        .rename(columns={target_column: "group_label"})
    )

    invalid_group_labels = ~group_table["group_label"].isin([0, 1])
    if invalid_group_labels.any():
        raise ValueError(f"Invalid group labels found for source {source}")

    train_groups, temporary_groups = train_test_split(
        group_table,
        test_size=VAL_FRACTION + TEST_FRACTION,
        stratify=group_table["group_label"],
        random_state=RANDOM_SEED,
    )

    relative_test_fraction = TEST_FRACTION / (VAL_FRACTION + TEST_FRACTION)
    val_groups, test_groups = train_test_split(
        temporary_groups,
        test_size=relative_test_fraction,
        stratify=temporary_groups["group_label"],
        random_state=RANDOM_SEED,
    )

    split_by_group = {}
    split_by_group.update(dict.fromkeys(train_groups["patient_id"], "train"))
    split_by_group.update(dict.fromkeys(val_groups["patient_id"], "val"))
    split_by_group.update(dict.fromkeys(test_groups["patient_id"], "test"))

    result = source_df.copy()
    result["split"] = result["patient_id"].map(split_by_group)

    if result["split"].isna().any():
        raise RuntimeError(f"Some {source} rows did not receive a split")

    return result


def count_labels(dataframe: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        "negative_0": int(dataframe[column].eq(0).sum()),
        "positive_1": int(dataframe[column].eq(1).sum()),
        "masked_-1": int(dataframe[column].eq(-1).sum()),
    }


def build_report(dataframe: pd.DataFrame, removed_rows: int) -> dict:
    report: dict = {
        "version": "patient_level_v1",
        "random_seed": RANDOM_SEED,
        "label_policy": {
            "chexpert_blank_cardiomegaly": "0_weak_negative",
            "chexpert_uncertain_minus_1": "masked_u_ignore",
            "unavailable_task": "masked_minus_1",
        },
        "input_rows": int(len(dataframe) + removed_rows),
        "usable_rows": int(len(dataframe)),
        "removed_without_active_label": int(removed_rows),
        "splits": {},
        "patient_overlap": {},
    }

    for split_name in ("train", "val", "test"):
        split_df = dataframe.loc[dataframe["split"].eq(split_name)]
        report["splits"][split_name] = {
            "rows": int(len(split_df)),
            "patients_or_groups": int(split_df["patient_id"].nunique()),
            "sources": {
                str(key): int(value)
                for key, value in split_df["source"].value_counts().items()
            },
            "cardiomegaly": count_labels(split_df, "cardiomegaly"),
            "tb": count_labels(split_df, "tb"),
        }

    patient_sets = {
        name: set(dataframe.loc[dataframe["split"].eq(name), "patient_id"])
        for name in ("train", "val", "test")
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patient_sets[first] & patient_sets[second]
        report["patient_overlap"][f"{first}_vs_{second}"] = int(len(overlap))

    return report


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input metadata not found: {INPUT_CSV}")

    dataframe = pd.read_csv(INPUT_CSV)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    input_rows = len(dataframe)
    dataframe["image_path"] = dataframe["image_path"].map(normalize_relative_path)
    dataframe = apply_label_policy(dataframe)
    removed_rows = input_rows - len(dataframe)

    dataframe["patient_id"] = [
        extract_patient_id(source, image_path)
        for source, image_path in zip(
            dataframe["source"], dataframe["image_path"], strict=True
        )
    ]

    split_parts = []
    for source in sorted(VALID_SOURCES):
        source_df = dataframe.loc[dataframe["source"].eq(source)].copy()
        if source_df.empty:
            raise ValueError(f"No usable rows found for source: {source}")
        split_parts.append(split_source_groups(source_df))

    final_df = pd.concat(split_parts, ignore_index=True)
    final_df = final_df[
        [
            "image_path",
            "cardiomegaly",
            "tb",
            "source",
            "patient_id",
            "split",
        ]
    ]

    if final_df["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image paths remain in model-ready metadata")

    report = build_report(final_df, removed_rows)
    if any(report["patient_overlap"].values()):
        raise RuntimeError(f"Patient leakage detected: {report['patient_overlap']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    READY_METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(READY_METADATA_CSV, index=False)

    for split_name in ("train", "val", "test"):
        split_df = final_df.loc[final_df["split"].eq(split_name)].copy()
        split_df = split_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        split_df.to_csv(OUTPUT_DIR / f"{split_name}.csv", index=False)

    with REPORT_JSON.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("=" * 68)
    print("PATIENT-LEVEL SPLIT COMPLETED")
    print("=" * 68)
    print(f"Input rows                    : {input_rows:,}")
    print(f"Usable rows                   : {len(final_df):,}")
    print(f"Removed (no active label)     : {removed_rows:,}")
    for split_name in ("train", "val", "test"):
        details = report["splits"][split_name]
        print(
            f"{split_name.capitalize():<13} rows/patients    : "
            f"{details['rows']:,} / {details['patients_or_groups']:,}"
        )
    print(f"Patient overlap train-val     : {report['patient_overlap']['train_vs_val']}")
    print(f"Patient overlap train-test    : {report['patient_overlap']['train_vs_test']}")
    print(f"Patient overlap val-test      : {report['patient_overlap']['val_vs_test']}")
    print(f"Output directory              : {OUTPUT_DIR}")
    print(f"Report                        : {REPORT_JSON}")


if __name__ == "__main__":
    main()
