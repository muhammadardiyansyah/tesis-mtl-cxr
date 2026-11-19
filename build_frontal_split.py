"""Membangun split frontal-only dari patient_level_v1.

Aturan: buang baris CheXpert yang image_path-nya memuat 'lateral'. NIH dan
TBX11K tidak diubah, dan batas pasien tetap sama (penyaringan dilakukan di
dalam setiap split, bukan pembagian ulang). Hasil yang diharapkan (verifikasi
18 Sep 2026): train 211.893, val 47.247, test 45.596; overlap pasien 0;
pos_weight kardiomegali train 10.1854.
"""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data/splits/patient_level_v1"
DST = ROOT / "data/splits/patient_level_frontal_v1"
EXPECTED = {"train": 211893, "val": 47247, "test": 45596}


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    report, ok = {}, True
    for split, expected in EXPECTED.items():
        df = pd.read_csv(SRC / f"{split}.csv")
        lateral = (df["source"] == "chexpert") & df["image_path"].str.lower().str.contains("lateral")
        frontal = df[~lateral].reset_index(drop=True)
        frontal.to_csv(DST / f"{split}.csv", index=False)
        active = frontal.loc[frontal["cardiomegaly"].isin([0, 1]), "cardiomegaly"]
        report[split] = {
            "rows": int(len(frontal)),
            "lateral_removed": int(lateral.sum()),
            "patients": int(frontal["patient_id"].nunique()),
            "cardiomegaly_positive": int((active == 1).sum()),
            "cardiomegaly_negative": int((active == 0).sum()),
        }
        ok &= len(frontal) == expected
        print(f"{split:5s}: {len(frontal):,} (target {expected:,}) | lateral dibuang {int(lateral.sum()):,}")

    pids = {s: set(pd.read_csv(DST / f"{s}.csv", usecols=["patient_id"])["patient_id"]) for s in EXPECTED}
    overlap = {
        "train_val": len(pids["train"] & pids["val"]),
        "train_test": len(pids["train"] & pids["test"]),
        "val_test": len(pids["val"] & pids["test"]),
    }
    pos_weight = report["train"]["cardiomegaly_negative"] / report["train"]["cardiomegaly_positive"]
    ok &= sum(overlap.values()) == 0 and abs(pos_weight - 10.1854) < 1e-3

    json.dump({
        "split_version": "patient_level_frontal_v1",
        "source_split": "patient_level_v1",
        "rule": "drop CheXpert rows whose image_path contains 'lateral'",
        "splits": report,
        "patient_overlap": overlap,
        "pos_weight_cardiomegaly": round(pos_weight, 4),
    }, open(DST / "split_report.json", "w"), indent=2)

    print("Overlap pasien:", overlap, f"| pos_weight {pos_weight:.4f}")
    print("STATUS:", "SESUAI" if ok else "TIDAK SESUAI")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
