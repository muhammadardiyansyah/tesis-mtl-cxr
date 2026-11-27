# ===================================================================================
# Evaluasi eksperimen tambahan v3 (ANALYSIS_PLAN.md bagian 12)
# Tempel sel ini MENGGANTIKAN bagian pelatihan (blok LANES sampai akhir) pada salinan
# notebook tesis-training-runs-b3. Bagian awal (clone repo, symlink data, build split)
# tetap dipakai apa adanya.
#
# Urutan dipaksa: (1) prediksi validation + threshold Youden untuk KEENAM run dulu,
# (2) baru prediksi test, masing-masing satu kali, (3) statistik, (4) uji eksternal TBC.
# evaluate_run.py sendiri menolak menimpa prediksi atau threshold.
# ===================================================================================
import json, shutil, subprocess, sys
from pathlib import Path

REPO = Path("/kaggle/working/tesis-mtl-cxr")
EVAL = Path("/kaggle/working/eval_v3")
EXT_OUT = Path("/kaggle/working/external_v3")
INPUT = Path("/kaggle/input")
SEEDS = (42, 43, 44)
MTL_V3 = [f"resnet18_mtl_frontal_v3_s{s}" for s in SEEDS]
STL_V2 = [f"resnet18_stl_cardiomegaly_frontal_v2_s{s}" for s in SEEDS]
MTL_V2 = [f"resnet18_mtl_frontal_v2_s{s}" for s in SEEDS]           # pembanding P12-A
STL_TB = [f"resnet18_stl_tuberculosis_frontal_v1_s{s}" for s in SEEDS]  # pembanding P12-C

def sh(*args):
    print("\n>>>", " ".join(str(a) for a in args[:4]), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=REPO, check=True)

def ev(*args):
    sh("evaluate_run.py", "--out-dir", EVAL, *args)

# Indeks sekali jalan: hanya checkpoint *_best.pt dan threshold beku (dataset citra dilewati)
import os
SKIP = {"images", "imgs", "train", "valid", "test", "cxr_png", "mask", "leftmask", "rightmask", "manualmask"}
INDEX = []
for root, dirs, files in os.walk(INPUT, followlinks=True):
    dirs[:] = [d for d in dirs if d.lower() not in SKIP and not d.lower().startswith("images_")]
    INDEX += [Path(root) / f for f in files if f.endswith(("_best.pt", "_frozen_thresholds.json"))]

def find_one(name):
    hits = sorted(p for p in INDEX if p.name == name)
    if len(hits) != 1:
        raise SystemExit(f"{name}: ditemukan {len(hits)} berkas, harus tepat 1: {hits}")
    return hits[0]

print("Commit:", subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip())
EVAL.mkdir(exist_ok=True)

# 0. Cari checkpoint v3 (dari dua dataset output sesi 1 dan sesi 2)
CKPT = {run: find_one(f"{run}_best.pt") for run in MTL_V3 + STL_V2}
for run, p in CKPT.items():
    print(f"{run:45s} {p}")

# 0b. Salin artefak evaluasi v2 yang sudah ada (hanya dibaca, tidak dihitung ulang)
for run in MTL_V2 + STL_TB:
    src = find_one(f"{run}_frozen_thresholds.json").parent
    dst = EVAL / run
    if not dst.exists():
        shutil.copytree(src, dst)
    print("salin", run, "<-", src)

# 1. Validation -> threshold beku (keenam run v3)
for run in MTL_V3 + STL_V2:
    ev("predict", "--run", run, "--checkpoint", CKPT[run], "--split", "val")
    ev("threshold", "--run", run)

# 2. Test internal: satu kali per run
for run in MTL_V3 + STL_V2:
    ev("predict", "--run", run, "--checkpoint", CKPT[run], "--split", "test")
    ev("metrics", "--run", run)

# 3a. P12-A: MTL v3 vs MTL v2 (seed sama). Nama berkas compare diganti agar tidak tertimpa.
for s in SEEDS:
    a = f"resnet18_mtl_frontal_v3_s{s}"
    ev("compare", "--a", a, "--b-cardiomegaly", f"resnet18_mtl_frontal_v2_s{s}",
       "--b-tuberculosis", f"resnet18_mtl_frontal_v2_s{s}")
    (EVAL / f"compare_{a}_vs_stl.json").rename(EVAL / f"P12A_{a}_vs_mtl_v2.json")

# 3b. P12-B: MTL v3 vs STL kardio v2 (TBC vs STL TBC v1, deskriptif / P12-C)
for s in SEEDS:
    a = f"resnet18_mtl_frontal_v3_s{s}"
    ev("compare", "--a", a, "--b-cardiomegaly", f"resnet18_stl_cardiomegaly_frontal_v2_s{s}",
       "--b-tuberculosis", f"resnet18_stl_tuberculosis_frontal_v1_s{s}")
    (EVAL / f"compare_{a}_vs_stl.json").rename(EVAL / f"P12B_{a}_vs_stl.json")

# 3c. Ringkasan mean ± SD
ev("summarize", "--runs", *MTL_V3, "--label", "MTL_v3")
ev("summarize", "--runs", *STL_V2, "--label", "STL_kardio_v2")

# 3d. P12-C probe lintas sumber: proporsi citra NIH/CheXpert test yang dinyatakan TBC
import numpy as np, pandas as pd
test = pd.read_csv(REPO / "data/splits/patient_level_frontal_v1/test.csv")
rows = []
for run in MTL_V3 + MTL_V2:
    d = np.load(EVAL / run / f"{run}_test_predictions.npz")
    t = json.loads((EVAL / run / f"{run}_frozen_thresholds.json").read_text())["thresholds"]["tuberculosis"]
    src = test["source"].to_numpy()[d["row_index"]]
    p_tb = d["probabilities"][:, 1]
    for name in ("nih", "chexpert"):
        m = src == name
        rows.append({"run": run, "source": name, "n": int(m.sum()), "threshold_tb": t,
                     "prop_positive_tb": float((p_tb[m] >= t).mean())})
probe = pd.DataFrame(rows)
probe.to_csv(EVAL / "P12C_probe_lintas_sumber.csv", index=False)
print(probe.to_string(index=False))

# 4. P12-C uji eksternal TBC (Shenzhen + Montgomery), prosedur bagian 8, threshold beku
tbx_root = next(p for p in Path("/kaggle/working/mtl_cxr_data/raw").iterdir() if "tbx" in p.name.lower())
sh("evaluate_external.py", "--search-roots", INPUT, "--tbx-root", tbx_root,
   "--eval-dir", EVAL, "--checkpoint-roots", *sorted({CKPT[r].parent for r in MTL_V3}), "--out-dir", EXT_OUT, "--runs", *MTL_V3)

print("\nSELESAI. Unduh folder eval_v3 dan external_v3.")
