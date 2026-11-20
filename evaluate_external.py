"""Uji eksternal head TBC pada Shenzhen dan Montgomery (NLM; Jaeger et al., 2014).

Rencana dikunci di experiments/ANALYSIS_PLAN.md bagian 8 sebelum skrip ini dijalankan.

- Label dari nama berkas: CHNCXR_####_X.png / MCUCXR_####_X.png (X: 0 normal, 1 TB).
- Model dan threshold BEKU dari evaluasi internal (tidak ada penyetelan pada data eksternal).
- PNG 12/16-bit dikonversi ke 8-bit dengan penskalaan min-max per citra sebelum
  transform evaluasi yang sama dengan internal (pad-to-square, resize 224, normalisasi ImageNet).
- Cek citra identik terhadap 8.400 citra TBX11K yang dipakai penelitian: thumbnail 32x32,
  korelasi Pearson > 0,995 ditandai dan dikeluarkan dari analisis utama.
- Satu citra = satu pasien (kedua set berisi satu radiografi per kasus).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluate_run import (REPORTED, all_metrics, cluster_weights, delong_paired,  # noqa: E402
                          sha256, BOOTSTRAP_SEED)

PATTERNS = {"shenzhen": re.compile(r"^CHNCXR_(\d{4})_([01])\.png$", re.I),
            "montgomery": re.compile(r"^MCUCXR_(\d{4})_([01])\.png$", re.I)}
DUP_CORR = 0.995


# ----------------------------------------------------------------------------
def _priority(path: Path) -> int:
    """Utamakan rilis NLM asli (folder CXR_png) di atas salinan dalam TBX11K/imgs/extra."""
    text = str(path).lower()
    return 0 if "cxr_png" in text else (2 if "tbx11k" in text else 1)


def discover(search_roots: list[Path]) -> pd.DataFrame:
    """Cari citra Shenzhen/Montgomery; satu entri per kasus.

    Folder masker (leftMask/rightMask/ManualMask) memakai nama berkas yang sama dengan
    citra rontgen, sehingga semua path yang memuat 'mask' dikecualikan.
    """
    best: dict[str, tuple[int, Path, str, int, str]] = {}
    for root in search_roots:
        for path in root.rglob("*.png"):
            if any("mask" in part.lower() for part in path.parts):
                continue
            for name, pattern in PATTERNS.items():
                m = pattern.match(path.name)
                if not m:
                    continue
                key = path.name.upper()
                cand = (_priority(path), path, name, int(m.group(2)), f"{name}_{m.group(1)}")
                if key not in best or cand[0] < best[key][0]:
                    best[key] = cand
    rows = [{"dataset": n, "case_id": c, "label": y, "path": str(p)} for _, p, n, y, c in best.values()]
    if not rows:
        return pd.DataFrame(columns=["dataset", "case_id", "label", "path"])
    return pd.DataFrame(rows).sort_values(["dataset", "case_id"]).reset_index(drop=True)


def load_8bit(path: str) -> Image.Image:
    """Baca PNG 8/12/16-bit menjadi RGB 8-bit tanpa clipping."""
    with Image.open(path) as image:
        image.load()
        if image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.asarray(image, dtype=np.float64)
            lo, hi = arr.min(), arr.max()
            arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
            return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
        return image.convert("RGB")


def thumb(image: Image.Image) -> np.ndarray:
    v = np.asarray(image.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float64).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ----------------------------------------------------------------------------
def predict(frame: pd.DataFrame, checkpoint: Path, batch_size: int = 32, use_clahe: bool = False) -> np.ndarray:
    import torch
    from src.datasets.transforms import get_eval_transforms
    from src.models.multitask_model import MultiTaskModel

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in state.get("model_state_dict", state).items()}
    tasks = sorted({k.split(".")[1] for k in sd if k.startswith("heads.")},
                   key=("cardiomegaly", "tuberculosis").index)
    model = MultiTaskModel(backbone_name="resnet18", pretrained=False, dropout=0.20, task_names=tuple(tasks))
    model.load_state_dict(sd, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tf = get_eval_transforms(224, use_clahe=use_clahe)
    j = tasks.index("tuberculosis")
    out = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            batch = torch.stack([tf(load_8bit(p)) for p in frame["path"].iloc[start:start + batch_size]])
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(batch.to(device))
            out.append(torch.sigmoid(logits.float())[:, j].cpu().numpy())
    return np.concatenate(out)


def bootstrap_ci(y, p, t, groups, iterations):
    point = all_metrics(y, p, t)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = {k: [] for k in REPORTED}
    for w in cluster_weights(groups, rng, iterations):
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        m = all_metrics(y, p, t, w)
        for k in REPORTED:
            boot[k].append(m[k])
    return {k: {"estimate": point[k], "ci95": [float(np.nanpercentile(boot[k], 2.5)),
                                              float(np.nanpercentile(boot[k], 97.5))]} for k in REPORTED} | \
           {"confusion": {k: int(point[k]) for k in ("tp", "fp", "fn", "tn")}}


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-roots", nargs="+", required=True, help="folder yang memuat PNG Shenzhen/Montgomery")
    ap.add_argument("--tbx-root", required=True, help="folder TBX11K (berisi imgs/health, sick, tb)")
    ap.add_argument("--split-dir", default=str(ROOT / "data/splits/patient_level_frontal_v1"))
    ap.add_argument("--eval-dir", required=True, help="folder eval internal (threshold beku)")
    ap.add_argument("--checkpoint-roots", nargs="+", required=True)
    ap.add_argument("--out-dir", default="/kaggle/working/external")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--runs", nargs="+", help="default: MTL dan STL TBC seed 42-44. "
                    "Run bernama *clahe* otomatis memakai transform CLAHE.")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1. Data eksternal
    frame = discover([Path(r) for r in args.search_roots])
    if frame.empty:
        raise SystemExit("Berkas CHNCXR_*/MCUCXR_* tidak ditemukan.")
    print(frame.groupby(["dataset", "label"]).size().rename("n").to_string(), flush=True)

    # 2. Cek citra identik vs TBX11K yang dipakai penelitian
    used = pd.concat([pd.read_csv(Path(args.split_dir) / f"{s}.csv") for s in ("train", "val", "test")])
    used = used[used["source"] == "tbx11k"]["image_path"].str.replace("tbx11k/", "", regex=False)
    tbx = np.stack([thumb(load_8bit(str(Path(args.tbx_root) / rel))) for rel in used])
    ext = np.stack([thumb(load_8bit(p)) for p in frame["path"]])
    corr = ext @ tbx.T
    frame["max_corr_tbx11k"] = corr.max(axis=1)
    frame["nearest_tbx11k"] = used.to_numpy()[corr.argmax(axis=1)]
    frame["duplicate_of_training_data"] = frame["max_corr_tbx11k"] > DUP_CORR
    print(f"Citra eksternal identik dengan TBX11K penelitian (r > {DUP_CORR}): "
          f"{int(frame['duplicate_of_training_data'].sum())}", flush=True)
    frame.to_csv(out / "external_manifest.csv", index=False)
    analysis = frame[~frame["duplicate_of_training_data"]].reset_index(drop=True)

    # 3. Prediksi semua run dengan threshold beku internal
    runs = args.runs or ([f"resnet18_mtl_frontal_v2_s{s}" for s in (42, 43, 44)] +
                         [f"resnet18_stl_tuberculosis_frontal_v1_s{s}" for s in (42, 43, 44)])
    preds, report = {}, {"plan": "experiments/ANALYSIS_PLAN.md bagian 8", "dup_corr_threshold": DUP_CORR,
                         "n_excluded_duplicates": int(frame["duplicate_of_training_data"].sum()), "runs": {}}
    for run in runs:
        ckpt = next((h for r in args.checkpoint_roots for h in Path(r).rglob(f"{run}_best.pt")), None)
        if ckpt is None:
            raise FileNotFoundError(run)
        t = json.loads((Path(args.eval_dir) / run / f"{run}_frozen_thresholds.json").read_text())["thresholds"]["tuberculosis"]
        p = predict(analysis, ckpt, use_clahe="clahe" in run)
        preds[run] = p
        analysis[f"p_{run}"] = p
        report["runs"][run] = {"checkpoint_sha256": sha256(ckpt), "threshold": t, "subsets": {}}
        for name, sub in [("gabungan", analysis)] + list(analysis.groupby("dataset")):
            m = analysis.index.isin(sub.index)
            y = analysis.loc[m, "label"].to_numpy()
            res = bootstrap_ci(y, p[m], t, analysis.loc[m, "case_id"].to_numpy(), args.iterations)
            report["runs"][run]["subsets"][name] = {"n": int(m.sum()), "positive": int(y.sum()), **res}
            e = res
            print(f"{run:42s} {name:10s} n={m.sum():4d} AUC {e['roc_auc']['estimate']:.4f} "
                  f"[{e['roc_auc']['ci95'][0]:.4f}, {e['roc_auc']['ci95'][1]:.4f}] "
                  f"sens {e['sensitivity']['estimate']:.3f} spes {e['specificity']['estimate']:.3f}", flush=True)
    analysis.to_csv(out / "external_predictions.csv", index=False)

    # 4. MTL vs STL TBC per seed (gabungan): DeLong + bootstrap berpasangan selisih AUC
    report["compare"] = {}
    y = analysis["label"].to_numpy()
    g = analysis["case_id"].to_numpy()
    for s in (42, 43, 44):
        ka, kb = f"resnet18_mtl_frontal_v2_s{s}", f"resnet18_stl_tuberculosis_frontal_v1_s{s}"
        if ka not in preds or kb not in preds:
            continue
        a, b = preds[ka], preds[kb]
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        d = [roc_auc_score(y, a, sample_weight=w) - roc_auc_score(y, b, sample_weight=w)
             for w in cluster_weights(g, rng, args.iterations)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        report["compare"][f"s{s}"] = {"delong": delong_paired(y, a, b), "auc_diff_ci95": [float(lo), float(hi)]}
        dl = report["compare"][f"s{s}"]["delong"]
        print(f"MTL vs STL TBC s{s}: selisih AUC {dl['difference']:+.4f} [{lo:+.4f}, {hi:+.4f}] DeLong p={dl['p_value']:.3g}")
    (out / "external_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nTersimpan di {out}")


if __name__ == "__main__":
    main()
