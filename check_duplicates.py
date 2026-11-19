"""Cek duplikat dan near-duplicate TBX11K lintas split (sebelum test dibuka).

Latar: TBX11K tidak menyediakan ID pasien, sehingga setiap citra dianggap satu
pasien. Citra yang identik atau hampir identik di train dan test akan membuat
estimasi performa TBC terlalu optimistis.

Metode (ditetapkan a priori, tidak disetel setelah melihat hasil):
  - Duplikat persis : MD5 isi berkas.
  - Near-duplicate  : perceptual hash (pHash 64-bit: grayscale 32x32 -> DCT ->
                      blok 8x8 frekuensi rendah tanpa DC -> ambang median);
                      pasangan dengan jarak Hamming <= 6 bit ditandai.

Keluaran (folder reports/duplicates/):
  - tbx11k_duplicate_pairs.csv          : semua pasangan lintas split yang ditandai
  - test_rows_excluded_tbx11k_dup.csv    : row index test.csv untuk analisis
                                           sensitivitas (evaluate_run metrics --exclude)
  - duplicate_report.json                : ringkasan
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data/splits/patient_level_frontal_v1"
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/kaggle/working/mtl_cxr_data/raw")
OUT = ROOT / "reports/duplicates"
HAMMING_THRESHOLD = 6
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    freq = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].flatten()[1:]
    bits = np.concatenate([[False], freq > np.median(freq)])  # 64 bit, bit DC = 0
    return np.packbits(bits)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    rows = []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(SPLIT_DIR / f"{split}.csv")
        frame["row_index"] = np.arange(len(frame))
        frame["split"] = split
        rows.append(frame[frame["source"] == "tbx11k"])
    tbx = pd.concat(rows, ignore_index=True)
    print(f"Citra TBX11K: {len(tbx):,}")

    hashes, digests = [], []
    for i, rel in enumerate(tbx["image_path"]):
        path = RAW / rel
        hashes.append(phash(path))
        digests.append(md5(path))
        if (i + 1) % 1000 == 0:
            print(f"  hash {i + 1:,}/{len(tbx):,}", flush=True)
    codes = np.stack(hashes)
    tbx["md5"] = digests

    pairs = []
    split_arr = tbx["split"].to_numpy()
    for i in range(len(tbx) - 1):
        dist = POPCOUNT[np.bitwise_xor(codes[i + 1:], codes[i])].sum(axis=1)
        for j in np.flatnonzero(dist <= HAMMING_THRESHOLD) + i + 1:
            if split_arr[i] == split_arr[j]:
                continue
            a, b = tbx.iloc[i], tbx.iloc[j]
            pairs.append({
                "split_a": a["split"], "row_a": int(a["row_index"]), "image_a": a["image_path"], "tb_a": int(a["tb"]),
                "split_b": b["split"], "row_b": int(b["row_index"]), "image_b": b["image_path"], "tb_b": int(b["tb"]),
                "hamming": int(dist[j - i - 1]), "exact_md5": bool(a["md5"] == b["md5"]),
            })
    pair_df = pd.DataFrame(pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(OUT / "tbx11k_duplicate_pairs.csv", index=False)

    test_rows = set()
    for _, p in pair_df.iterrows() if len(pair_df) else []:
        for side in ("a", "b"):
            if p[f"split_{side}"] == "test":
                test_rows.add(int(p[f"row_{side}"]))
    pd.DataFrame({"row_index": sorted(test_rows)}).to_csv(
        OUT / "test_rows_excluded_tbx11k_dup.csv", index=False)

    def count(s1, s2):
        if not len(pair_df):
            return 0
        m = ((pair_df.split_a == s1) & (pair_df.split_b == s2)) | ((pair_df.split_a == s2) & (pair_df.split_b == s1))
        return int(m.sum())

    test_tbx = tbx[tbx["split"] == "test"]
    excluded = test_tbx[test_tbx["row_index"].isin(test_rows)]
    report = {
        "method": "md5 + pHash 64-bit (DCT 32x32 -> 8x8)", "hamming_threshold": HAMMING_THRESHOLD,
        "tbx11k_images": int(len(tbx)),
        "cross_split_pairs": {"train-val": count("train", "val"), "train-test": count("train", "test"),
                              "val-test": count("val", "test")},
        "exact_md5_cross_split_pairs": int(pair_df["exact_md5"].sum()) if len(pair_df) else 0,
        "label_conflicts": int((pair_df.tb_a != pair_df.tb_b).sum()) if len(pair_df) else 0,
        "test_tbx11k_images": int(len(test_tbx)),
        "test_rows_flagged": int(len(excluded)),
        "test_flagged_tb_positive": int((excluded["tb"] == 1).sum()),
    }
    (OUT / "duplicate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
