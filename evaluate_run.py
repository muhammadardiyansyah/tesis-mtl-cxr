"""Evaluasi terpadu untuk run MTL dan STL (menggantikan sel notebook 37-56).

Alur protokol (urutan dipaksa oleh skrip):
  1. predict   --split val   -> probabilitas validation
  2. threshold               -> Youden J dari validation, lalu DIBEKUKAN
  3. predict   --split test  -> hanya boleh jika threshold sudah beku,
                                 dan hanya sekali (tidak menimpa)
  4. metrics                 -> metrik test + CI 95% bootstrap per pasien
  5. compare                 -> bootstrap berpasangan + uji DeLong (MTL vs STL)
  6. summarize               -> mean ± SD lintas seed

Format prediksi seragam untuk MTL dan STL: array [N, 2] dengan urutan tugas
(cardiomegaly, tuberculosis). Tugas yang tidak dimiliki model diisi NaN.
Satu format ini mencegah KeyError seperti pada sel 52 (17 Sep 2026).

Contoh:
  python evaluate_run.py predict --run resnet18_mtl_frontal_v2_s42 \
      --checkpoint /kaggle/input/.../resnet18_mtl_frontal_v2_s42_best.pt --split val
  python evaluate_run.py threshold --run resnet18_mtl_frontal_v2_s42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
TASKS = ("cardiomegaly", "tuberculosis")
LABEL_COLUMNS = {"cardiomegaly": "cardiomegaly", "tuberculosis": "tb"}
DEFAULT_SPLIT_DIR = ROOT / "data/splits/patient_level_frontal_v1"
DEFAULT_RAW = Path("/kaggle/working/mtl_cxr_data/raw")
DEFAULT_OUT = Path("/kaggle/working/eval")
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 42


# ----------------------------------------------------------------------------
# Utilitas berkas
# ----------------------------------------------------------------------------
def paths_for(out_dir: Path, run: str) -> dict[str, Path]:
    base = out_dir / run
    return {
        "dir": base,
        "val": base / f"{run}_val_predictions.npz",
        "thresholds": base / f"{run}_frozen_thresholds.json",
        "test": base / f"{run}_test_predictions.npz",
        "metrics": base / f"{run}_test_metrics.json",
        "meta": base / f"{run}_prediction_meta.json",
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    data = {key: archive[key] for key in archive.files}
    if tuple(str(t) for t in data["task_names"]) != TASKS:
        raise ValueError(f"Urutan tugas tidak sesuai di {path}")
    return data


def groups_for(split_csv: Path, row_index: np.ndarray) -> np.ndarray:
    """patient_id sudah unik per sumber (mis. nih_patient..., tbx11k_image_...)."""
    frame = pd.read_csv(split_csv, usecols=["patient_id"])
    return frame["patient_id"].astype(str).to_numpy()[row_index]


# ----------------------------------------------------------------------------
# Metrik
# ----------------------------------------------------------------------------
def threshold_metrics(y, p, threshold, weight=None):
    """Metrik berbasis threshold; weight = jumlah kemunculan pada bootstrap."""
    w = np.ones_like(p, dtype=float) if weight is None else weight
    pred = p >= threshold
    tp = float(w[(pred) & (y == 1)].sum())
    fp = float(w[(pred) & (y == 0)].sum())
    fn = float(w[(~pred) & (y == 1)].sum())
    tn = float(w[(~pred) & (y == 0)].sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    total = tp + fp + fn + tn
    return {
        "f1": f1, "precision": prec, "sensitivity": sens, "specificity": spec,
        "npv": npv, "accuracy": (tp + tn) / total,
        "balanced_accuracy": (sens + spec) / 2,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def calibration(y, p, weight=None, bins=10):
    """Brier score dan expected calibration error (bin lebar sama)."""
    w = np.ones_like(p, dtype=float) if weight is None else weight
    brier = float(np.sum(w * (p - y) ** 2) / w.sum())
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = which == b
        if w[sel].sum() > 0:
            gap = abs(np.average(p[sel], weights=w[sel]) - np.average(y[sel], weights=w[sel]))
            ece += w[sel].sum() / w.sum() * gap
    return {"brier": brier, "ece": float(ece)}


def all_metrics(y, p, threshold, weight=None):
    out = {
        "roc_auc": float(roc_auc_score(y, p, sample_weight=weight)),
        "average_precision": float(average_precision_score(y, p, sample_weight=weight)),
    }
    out.update(threshold_metrics(y, p, threshold, weight))
    out.update(calibration(y, p, weight))
    return out


REPORTED = ["roc_auc", "average_precision", "f1", "precision", "sensitivity",
            "specificity", "npv", "accuracy", "balanced_accuracy", "brier", "ece"]


def cluster_weights(groups: np.ndarray, rng, iterations: int):
    """Bootstrap per pasien: sampel pasien dengan pengembalian -> bobot per citra."""
    unique, inverse = np.unique(groups, return_inverse=True)
    for _ in range(iterations):
        draws = rng.integers(0, len(unique), size=len(unique))
        counts = np.bincount(draws, minlength=len(unique))
        yield counts[inverse].astype(float)


# ----------------------------------------------------------------------------
# DeLong berpasangan (Sun & Xu, 2014) untuk selisih AUC dua model
# ----------------------------------------------------------------------------
def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_paired(y, p1, p2):
    from scipy.stats import norm

    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v01, v10 = [], [], []
    for p in (p1, p2):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(p)
        aucs.append((tz[pos].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[pos] - tx) / n)
        v10.append(1.0 - (tz[neg] - ty) / m)
    s = np.cov(np.vstack(v01)) / m + np.cov(np.vstack(v10)) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else np.inf
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "difference": float(diff),
            "z": float(z), "p_value": float(2 * norm.sf(abs(z)))}


# ----------------------------------------------------------------------------
# Perintah: predict
# ----------------------------------------------------------------------------
def cmd_predict(args):
    import torch
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, str(ROOT))
    from src.datasets.dataset_loader import ChestXrayDataset
    from src.datasets.transforms import get_eval_transforms
    from src.models.multitask_model import MultiTaskModel

    files = paths_for(Path(args.out_dir), args.run)
    target = files[args.split]
    if target.exists():
        raise SystemExit(f"DITOLAK: {target.name} sudah ada. Prediksi tidak ditimpa.")
    if args.split == "test" and not files["thresholds"].exists():
        raise SystemExit("DITOLAK: threshold belum dibekukan. Jalankan 'threshold' dulu.")

    checkpoint_path = Path(args.checkpoint)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state.get("model_state_dict", state)
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model_tasks = sorted({k.split(".")[1] for k in state_dict if k.startswith("heads.")},
                         key=TASKS.index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(backbone_name="resnet18", pretrained=False, dropout=0.20,
                           task_names=tuple(model_tasks))
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    split_csv = Path(args.split_dir) / f"{args.split}.csv"
    dataset = ChestXrayDataset(split_csv, transform=get_eval_transforms(224, use_clahe=args.clahe),
                               raw_data_dir=args.raw_data)
    frame = dataset.dataframe
    active = np.zeros(len(frame), dtype=bool)
    for task in model_tasks:
        active |= frame[LABEL_COLUMNS[task]].isin([0, 1]).to_numpy()
    row_index = np.flatnonzero(active)

    loader = DataLoader(Subset(dataset, row_index.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    probs, labels, masks = [], [], []
    with torch.no_grad():
        for step, (images, y, m) in enumerate(loader, 1):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
            p = torch.sigmoid(logits.float()).cpu().numpy()
            full = np.full((len(p), len(TASKS)), np.nan, dtype=np.float32)
            for j, task in enumerate(model_tasks):
                full[:, TASKS.index(task)] = p[:, j]
            probs.append(full)
            labels.append(y.numpy())
            masks.append(m.numpy().astype(bool))
            if step % 50 == 0 or step == len(loader):
                print(f"  batch {step}/{len(loader)}", flush=True)

    probabilities = np.concatenate(probs)
    mask_arr = np.concatenate(masks)
    for task in TASKS:  # tugas yang tidak dimiliki model -> mask False
        if task not in model_tasks:
            mask_arr[:, TASKS.index(task)] = False

    files["dir"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, row_index=row_index, labels=np.concatenate(labels),
                        masks=mask_arr, probabilities=probabilities,
                        task_names=np.array(TASKS))
    meta = json.loads(files["meta"].read_text()) if files["meta"].exists() else {}
    meta[args.split] = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": state.get("epoch") if isinstance(state, dict) else None,
        "model_tasks": model_tasks, "clahe": bool(args.clahe), "split_csv": str(split_csv), "rows": int(len(row_index)),
        "code_commit": git_commit(), "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(files["meta"], meta)
    print(f"Tersimpan: {target} ({len(row_index):,} citra, tugas {model_tasks})")


# ----------------------------------------------------------------------------
# Perintah: threshold (hanya dari validation)
# ----------------------------------------------------------------------------
def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    finite = np.isfinite(thr)
    best = np.argmax(tpr[finite] - fpr[finite])
    return float(thr[finite][best])


def cmd_threshold(args):
    files = paths_for(Path(args.out_dir), args.run)
    if files["thresholds"].exists():
        raise SystemExit("DITOLAK: threshold sudah dibekukan dan tidak boleh diubah.")
    if files["test"].exists():
        raise SystemExit("DITOLAK: prediksi test sudah ada; threshold tidak boleh dipilih ulang.")
    data = load_predictions(files["val"])
    thresholds, summary = {}, {}
    for j, task in enumerate(TASKS):
        sel = data["masks"][:, j]
        if not sel.any():
            continue
        y = data["labels"][sel, j].astype(int)
        p = data["probabilities"][sel, j].astype(float)
        t = youden_threshold(y, p)
        thresholds[task] = t
        summary[task] = {"n": int(sel.sum()), "positive": int(y.sum()),
                         **{k: v for k, v in all_metrics(y, p, t).items()
                            if k in REPORTED}}
    write_json(files["thresholds"], {
        "run": args.run, "selection_dataset": "validation", "selection_method": "youden_j",
        "test_set_used_for_selection": False, "thresholds": thresholds,
        "validation_metrics_at_threshold": summary,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
    })
    for task, t in thresholds.items():
        s = summary[task]
        print(f"{task:13s} threshold {t:.6f} | val AUC {s['roc_auc']:.4f} "
              f"sens {s['sensitivity']:.4f} spes {s['specificity']:.4f}")
    print(f"Threshold DIBEKUKAN: {files['thresholds']}")


# ----------------------------------------------------------------------------
# Perintah: metrics (test + CI bootstrap per pasien)
# ----------------------------------------------------------------------------
def cmd_metrics(args):
    files = paths_for(Path(args.out_dir), args.run)
    data = load_predictions(files["test"])
    thresholds = json.loads(files["thresholds"].read_text())["thresholds"]
    target = files["metrics"]
    excluded = 0
    if args.exclude:
        # Analisis sensitivitas yang sudah direncanakan (mis. tanpa near-duplicate)
        drop = pd.read_csv(args.exclude)["row_index"].to_numpy()
        keep = ~np.isin(data["row_index"], drop)
        excluded = int((~keep).sum())
        data = {k: (v[keep] if k != "task_names" else v) for k, v in data.items()}
        target = files["dir"] / f"{args.run}_test_metrics_{Path(args.exclude).stem}.json"
        print(f"Analisis sensitivitas: {excluded} citra test dikeluarkan ({args.exclude})")
    groups_all = groups_for(Path(args.split_dir) / "test.csv", data["row_index"])
    report = {"run": args.run, "split": "test", "bootstrap_unit": "patient",
              "iterations": args.iterations, "seed": BOOTSTRAP_SEED,
              "excluded_rows_file": args.exclude, "excluded_rows": excluded, "tasks": {}}
    for j, task in enumerate(TASKS):
        sel = data["masks"][:, j]
        if task not in thresholds or not sel.any():
            continue
        y = data["labels"][sel, j].astype(int)
        p = data["probabilities"][sel, j].astype(float)
        g = groups_all[sel]
        t = thresholds[task]
        point = all_metrics(y, p, t)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boot = {k: [] for k in REPORTED}
        for w in cluster_weights(g, rng, args.iterations):
            if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
                continue
            m = all_metrics(y, p, t, w)
            for k in REPORTED:
                boot[k].append(m[k])
        report["tasks"][task] = {
            "n_images": int(len(y)), "n_patients": int(len(np.unique(g))),
            "positive": int(y.sum()), "threshold": t,
            "confusion": {k: int(point[k]) for k in ("tp", "fp", "fn", "tn")},
            "metrics": {k: {"estimate": point[k],
                            "ci95": [float(np.nanpercentile(boot[k], 2.5)),
                                     float(np.nanpercentile(boot[k], 97.5))]}
                        for k in REPORTED},
        }
        print(f"\n{task.upper()} (n={len(y):,}, pasien={len(np.unique(g)):,}, threshold={t:.4f})")
        for k in REPORTED:
            e, (lo, hi) = point[k], report["tasks"][task]["metrics"][k]["ci95"]
            print(f"  {k:18s} {e:.4f}  [{lo:.4f}, {hi:.4f}]")
    write_json(target, report)
    print(f"\nTersimpan: {target}")


# ----------------------------------------------------------------------------
# Perintah: compare (paired bootstrap + DeLong)
# ----------------------------------------------------------------------------
def task_vectors(out_dir, run, task, split_dir):
    files = paths_for(out_dir, run)
    data = load_predictions(files["test"])
    j = TASKS.index(task)
    sel = data["masks"][:, j]
    rows = data["row_index"][sel]
    t = json.loads(files["thresholds"].read_text())["thresholds"][task]
    return rows, data["labels"][sel, j].astype(int), data["probabilities"][sel, j].astype(float), t


def cmd_compare(args):
    out_dir = Path(args.out_dir)
    test_csv = Path(args.split_dir) / "test.csv"
    b_runs = {"cardiomegaly": args.b_cardiomegaly, "tuberculosis": args.b_tuberculosis}
    report = {"comparison": f"{args.a} minus STL", "a": args.a, "b": b_runs,
              "iterations": args.iterations, "seed": BOOTSTRAP_SEED,
              "bootstrap_unit": "patient", "tasks": {}}
    for task, b_run in b_runs.items():
        if not b_run:
            continue
        ra, ya, pa, ta = task_vectors(out_dir, args.a, task, args.split_dir)
        rb, yb, pb, tb = task_vectors(out_dir, b_run, task, args.split_dir)
        common = np.intersect1d(ra, rb)
        ia, ib = np.searchsorted(ra, common), np.searchsorted(rb, common)
        y, pa, pb = ya[ia], pa[ia], pb[ib]
        if not np.array_equal(y, yb[ib]):
            raise ValueError(f"Label {task} berbeda antar run; urutan data tidak cocok.")
        g = groups_for(test_csv, common)
        ma, mb = all_metrics(y, pa, ta), all_metrics(y, pb, tb)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        diffs = {k: [] for k in REPORTED}
        for w in cluster_weights(g, rng, args.iterations):
            if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
                continue
            xa, xb = all_metrics(y, pa, ta, w), all_metrics(y, pb, tb, w)
            for k in REPORTED:
                diffs[k].append(xa[k] - xb[k])
        task_report = {"n_images": int(len(y)), "n_patients": int(len(np.unique(g))),
                       "delong_auc": delong_paired(y, pa, pb), "metrics": {}}
        print(f"\n{task.upper()}: {args.a} vs {b_run} (n={len(y):,})")
        for k in REPORTED:
            d = np.asarray(diffs[k])
            d = d[np.isfinite(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p_boot = min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (len(d) + 1))
            task_report["metrics"][k] = {"a": ma[k], "b": mb[k], "difference": ma[k] - mb[k],
                                         "ci95": [float(lo), float(hi)],
                                         "bootstrap_p": float(p_boot),
                                         "ci_excludes_zero": bool(lo > 0 or hi < 0)}
            print(f"  {k:18s} {ma[k]:.4f} vs {mb[k]:.4f} | selisih {ma[k]-mb[k]:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]")
        dl = task_report["delong_auc"]
        print(f"  DeLong AUC: selisih {dl['difference']:+.4f}, z={dl['z']:.2f}, p={dl['p_value']:.4g}")
        report["tasks"][task] = task_report
    target = out_dir / f"compare_{args.a}_vs_stl.json"
    write_json(target, report)
    print(f"\nTersimpan: {target}")


# ----------------------------------------------------------------------------
# Perintah: summarize (mean ± SD lintas seed)
# ----------------------------------------------------------------------------
def cmd_summarize(args):
    out_dir = Path(args.out_dir)
    rows = []
    for run in args.runs:
        report = json.loads(paths_for(out_dir, run)["metrics"].read_text())
        for task, body in report["tasks"].items():
            rows.append({"run": run, "task": task,
                         **{k: v["estimate"] for k, v in body["metrics"].items()}})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / f"summary_{args.label}_per_run.csv", index=False)
    agg = frame.groupby("task")[REPORTED].agg(["mean", "std", "count"])
    agg.to_csv(out_dir / f"summary_{args.label}_mean_sd.csv")
    for task, row in agg.iterrows():
        print(f"\n{args.label} | {task} (n run = {int(row[('roc_auc', 'count')])})")
        for k in REPORTED:
            print(f"  {k:18s} {row[(k, 'mean')]:.4f} ± {row[(k, 'std')]:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict")
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--raw-data", default=str(DEFAULT_RAW))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--clahe", action="store_true",
                   help="Wajib untuk run yang dilatih dengan preprocessing.clahe: true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("threshold")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_threshold)

    p = sub.add_parser("metrics")
    p.add_argument("--run", required=True)
    p.add_argument("--iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    p.add_argument("--exclude", help="CSV berkolom row_index (analisis sensitivitas)")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("compare")
    p.add_argument("--a", required=True, help="run MTL")
    p.add_argument("--b-cardiomegaly", help="run STL kardiomegali")
    p.add_argument("--b-tuberculosis", help="run STL TBC")
    p.add_argument("--iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("summarize")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--label", required=True)
    p.set_defaults(func=cmd_summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
