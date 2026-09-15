"""Safely extract source archives and validate Kaggle training paths."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm


EXPECTED_SOURCE_DIRS = ("chestxray14", "chexpert", "tbx11k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="/kaggle/input/mtl-cxr-private-data/archives",
        help="Folder Kaggle Dataset yang berisi tiga arsip sumber.",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/mtl_cxr_data/raw",
        help="Folder raw data yang dapat ditulis oleh notebook.",
    )
    parser.add_argument(
        "--val-csv",
        default=(
            "/kaggle/input/mtl-cxr-private-data/splits/"
            "patient_level_v1/val.csv"
        ),
        help="Validation CSV untuk memeriksa kecocokan path.",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    return parser.parse_args()


def _safe_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Path arsip tidak aman: {member_name}") from error
    return destination


def extract_zip(path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            _safe_destination(output_dir, member.filename)
        for member in tqdm(members, desc=f"Ekstrak {path.name}"):
            archive.extract(member, output_dir)


def extract_tar(path: Path, output_dir: Path) -> None:
    with tarfile.open(path, mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            _safe_destination(output_dir, member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Tipe anggota arsip tidak diizinkan: {member.name}")
        for member in tqdm(members, desc=f"Ekstrak {path.name}"):
            archive.extract(member, output_dir)


def find_archives(input_dir: Path) -> list[Path]:
    supported_suffixes = (".zip", ".tar", ".tar.gz", ".tgz")
    archives = [
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and any(path.name.lower().endswith(suffix) for suffix in supported_suffixes)
    ]
    return sorted(archives)


def validate_layout(output_dir: Path, val_csv: Path, sample_size: int) -> None:
    missing_dirs = [
        name for name in EXPECTED_SOURCE_DIRS if not (output_dir / name).is_dir()
    ]
    if missing_dirs:
        raise FileNotFoundError(
            "Folder sumber tidak ditemukan setelah ekstraksi: "
            f"{missing_dirs}. Arsip harus dibuat dari root data/raw."
        )
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV tidak ditemukan: {val_csv}")

    dataframe = pd.read_csv(val_csv, usecols=["image_path"], nrows=sample_size)
    missing_paths = [
        value
        for value in dataframe["image_path"].astype(str)
        if not (output_dir / value).is_file()
    ]
    if missing_paths:
        preview = "\n".join(missing_paths[:5])
        raise FileNotFoundError(
            f"{len(missing_paths)} dari {len(dataframe)} sampel tidak ditemukan:\n{preview}"
        )

    total, used, free = shutil.disk_usage(output_dir)
    del total, used
    print(f"Validasi path berhasil: {len(dataframe)} sampel ditemukan.")
    print(f"Sisa ruang /kaggle/working: {free / (1024**3):.1f} GiB")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    val_csv = Path(args.val_csv).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = output_dir / ".prepared"
    layout_ready = all((output_dir / name).is_dir() for name in EXPECTED_SOURCE_DIRS)
    if marker.exists() and layout_ready:
        print("Data sudah diekstrak pada sesi ini; ekstraksi dilewati.")
    else:
        archives = find_archives(input_dir)
        if not archives:
            raise FileNotFoundError(f"Tidak ada arsip didukung di: {input_dir}")
        print(f"Ditemukan {len(archives)} arsip.")
        for archive_path in archives:
            lower_name = archive_path.name.lower()
            if lower_name.endswith(".zip"):
                extract_zip(archive_path, output_dir)
            else:
                extract_tar(archive_path, output_dir)
        marker.touch()

    validate_layout(output_dir, val_csv, args.sample_size)


if __name__ == "__main__":
    main()
