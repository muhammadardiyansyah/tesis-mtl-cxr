"""Safely restore invalid CXR files from their original ZIP archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pulihkan citra invalid secara selektif dari arsip ZIP asli."
    )
    parser.add_argument(
        "--manifest",
        default="reports/data_audit/invalid_images.csv",
        help="Manifest hasil audit_dataset.py.",
    )
    parser.add_argument("--nih-archive", required=True)
    parser.add_argument("--chexpert-archive", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Lakukan pemulihan. Tanpa opsi ini program hanya melakukan dry-run.",
    )
    parser.add_argument(
        "--report",
        default="reports/data_audit/recovery_report.json",
    )
    return parser.parse_args()


def normalize_archive_path(value: str) -> str:
    return str(PurePosixPath(str(value).replace("\\", "/"))).lstrip("./").casefold()


def desired_suffix(record: dict[str, str]) -> str:
    stored = normalize_archive_path(record["stored_path"])
    source = record["source"].strip().casefold()
    marker = f"{source}/"
    if source == "nih":
        return f"images/{PurePosixPath(stored).name}"
    if marker in stored:
        return stored.split(marker, maxsplit=1)[1]
    return stored


def find_member(
    archive_members: list[zipfile.ZipInfo],
    record: dict[str, str],
) -> tuple[zipfile.ZipInfo | None, str]:
    suffix = desired_suffix(record)
    exact_matches = []
    for member in archive_members:
        normalized = normalize_archive_path(member.filename)
        if normalized == suffix or normalized.endswith(f"/{suffix}"):
            exact_matches.append(member)

    if len(exact_matches) == 1:
        return exact_matches[0], "found"
    if len(exact_matches) > 1:
        return None, f"ambiguous:{len(exact_matches)}"

    if record["source"].strip().casefold() == "nih":
        basename = PurePosixPath(suffix).name
        basename_matches = [
            member
            for member in archive_members
            if PurePosixPath(normalize_archive_path(member.filename)).name == basename
        ]
        if len(basename_matches) == 1:
            return basename_matches[0], "found_by_basename"
        if len(basename_matches) > 1:
            return None, f"ambiguous_basename:{len(basename_matches)}"

    return None, "not_found"


def validate_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.load()
            image.convert("RGB")
        return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_one(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    destination: Path,
) -> tuple[str, str, int, str]:
    if destination.exists() and destination.stat().st_size > 0:
        return "skipped_nonzero_destination", "", destination.stat().st_size, ""
    if not destination.parent.is_dir():
        return "missing_destination_directory", "", 0, ""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=destination.parent,
            prefix=f"{destination.name}.restore_",
            suffix=".tmp",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with archive.open(member, "r") as source_file:
                shutil.copyfileobj(source_file, temporary_file)

        if temporary_path.stat().st_size == 0:
            return "archive_member_zero_byte", "", 0, ""
        valid, error = validate_image(temporary_path)
        if not valid:
            return "archive_member_invalid", error, temporary_path.stat().st_size, ""

        digest = sha256_file(temporary_path)
        restored_size = temporary_path.stat().st_size
        os.replace(temporary_path, destination)
        temporary_path = None
        return "restored", "", restored_size, digest
    except Exception as error:
        return "archive_read_error", f"{type(error).__name__}: {error}", 0, ""
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    archives = {
        "nih": Path(args.nih_archive).resolve(),
        "chexpert": Path(args.chexpert_archive).resolve(),
    }
    for label, archive_path in archives.items():
        if not archive_path.is_file():
            raise FileNotFoundError(f"Arsip {label} tidak ditemukan: {archive_path}")
        if not zipfile.is_zipfile(archive_path):
            raise ValueError(f"Arsip {label} bukan ZIP yang valid: {archive_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        records = list(csv.DictReader(file))
    if not records:
        raise ValueError("Manifest tidak berisi citra invalid.")

    required_columns = {"source", "stored_path", "resolved_path", "reason"}
    missing_columns = required_columns - set(records[0])
    if missing_columns:
        raise ValueError(f"Kolom manifest hilang: {sorted(missing_columns)}")

    grouped_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        source = record["source"].strip().casefold()
        if source not in archives:
            raise ValueError(f"Belum ada arsip untuk source: {source}")
        grouped_records[source].append(record)

    mode = "apply" if args.apply else "dry_run"
    print("=" * 68)
    print("PEMULIHAN CITRA INVALID")
    print("=" * 68)
    print(f"Mode     : {mode}")
    print(f"Manifest : {manifest_path}")
    print(f"Jumlah   : {len(records)}")

    results: list[dict] = []
    for source, source_records in grouped_records.items():
        archive_path = archives[source]
        print(f"\nMembaca indeks arsip {source}: {archive_path.name}")
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            print(f"Anggota arsip: {len(members):,}")
            for record in source_records:
                member, lookup_status = find_member(members, record)
                result = {
                    "source": source,
                    "stored_path": record["stored_path"],
                    "destination": record["resolved_path"],
                    "archive_member": member.filename if member else None,
                    "lookup_status": lookup_status,
                    "status": (
                        "archive_member_zero_byte"
                        if member is not None and member.file_size == 0
                        else "ready" if member else lookup_status
                    ),
                    "error": "",
                    "restored_size": 0,
                    "sha256": "",
                }
                if args.apply and member is not None and member.file_size > 0:
                    status, error, size, digest = restore_one(
                        archive,
                        member,
                        Path(record["resolved_path"]),
                    )
                    result.update(
                        status=status,
                        error=error,
                        restored_size=size,
                        sha256=digest,
                    )
                results.append(result)

    status_counts = Counter(result["status"] for result in results)
    report = {
        "mode": mode,
        "manifest": str(manifest_path),
        "archives": {key: str(value) for key, value in archives.items()},
        "total_records": len(records),
        "status_counts": dict(status_counts),
        "results": results,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nHasil:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"Report: {report_path}")

    problem_statuses = {
        "not_found",
        "missing_destination_directory",
        "archive_member_zero_byte",
        "archive_member_invalid",
        "archive_read_error",
    }
    has_problem = any(
        status in problem_statuses or status.startswith("ambiguous")
        for status in status_counts
    )
    if has_problem:
        raise SystemExit(1)
    if not args.apply:
        print("\nDry-run selesai. Tidak ada file yang diubah.")


if __name__ == "__main__":
    main()
