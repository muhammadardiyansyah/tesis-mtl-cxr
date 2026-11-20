"""Grad-CAM dan Grad-CAM++ untuk model MTL utama (RQ3), dengan evaluasi kuantitatif.

Rencana: experiments/ANALYSIS_PLAN.md bagian 11 (dikunci sebelum dijalankan).

Isi:
  1. Peta saliency per tugas (logit head kardiomegali / TBC) dari lapisan konvolusi terakhir
     ResNet18 (backbone.model.layer4); implementasi manual tanpa pustaka tambahan.
  2. Kasus kualitatif dipilih sistematis dari prediksi yang sudah ada (threshold beku):
     TP/TN/FP/FN paling yakin per tugas dan per sumber, ditambah kasus eksternal
     (Shenzhen/Montgomery).
  3. Evaluasi kuantitatif pada Montgomery (138 citra, masker paru manual NLM):
       - lung_energy  : proporsi energi peta di dalam masker paru
       - pointing_hit : apakah titik maksimum peta jatuh di dalam paru
       - border_energy: proporsi energi pada bingkai luar 8% citra (indikasi shortcut tepi/penanda)
       - cardiac_proxy_energy (head kardiomegali): energi pada area antara kedua paru di
         separuh bawah kotak paru (proksi mediastinum/jantung; bukan masker jantung).
     Pembanding acuan: proporsi luas area yang sama (nilai harapan untuk peta acak seragam).
  4. Sanity check randomisasi (bobot layer4 + head diacak): korelasi Spearman antara peta asli
     dan peta model teracak. Korelasi tinggi berarti peta tidak bergantung pada bobot terlatih.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluate_external import discover, load_8bit  # noqa: E402
from src.datasets.transforms import PadToSquare, get_eval_transforms  # noqa: E402
from src.models.multitask_model import MultiTaskModel  # noqa: E402

TASKS = ("cardiomegaly", "tuberculosis")
SIZE = 224
BORDER = 0.08
TOP_K = 3


# ----------------------------------------------------------------------------
def load_model(checkpoint: Path, device) -> MultiTaskModel:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in state.get("model_state_dict", state).items()}
    tasks = tuple(sorted({k.split(".")[1] for k in sd if k.startswith("heads.")}, key=TASKS.index))
    model = MultiTaskModel(backbone_name="resnet18", pretrained=False, dropout=0.20, task_names=tasks)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


class CAM:
    """Grad-CAM dan Grad-CAM++ pada backbone.model.layer4 untuk satu head tugas."""

    def __init__(self, model: MultiTaskModel):
        self.model = model
        self.layer = model.backbone.model.layer4
        self.acts = None
        self.layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out):
        self.acts = out
        if out.requires_grad:  # saat torch.no_grad() (prediksi biasa) hook gradien dilewati
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x: torch.Tensor, task: str) -> dict[str, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        j = self.model.task_names.index(task)
        logits[:, j].sum().backward()
        a, g = self.acts.detach(), self.grads.detach()
        out = {"prob": torch.sigmoid(logits[:, j]).detach().cpu().numpy()}
        w_cam = g.mean(dim=(2, 3), keepdim=True)  # Grad-CAM: gradien rata-rata per kanal
        g2, g3 = g ** 2, g ** 3                    # Grad-CAM++ (Chattopadhay et al., 2018)
        denom = 2 * g2 + (a * g3).sum(dim=(2, 3), keepdim=True)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        w_pp = (alpha * torch.relu(g)).sum(dim=(2, 3), keepdim=True)
        for name, w in (("gradcam", w_cam), ("gradcampp", w_pp)):
            cam = torch.relu((w * a).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[:, 0]
            mx = cam.flatten(1).max(dim=1).values.clamp_min(1e-12)[:, None, None]
            out[name] = (cam / mx).cpu().numpy()
        return out


def geometric(img: Image.Image) -> Image.Image:
    """Transform geometri yang sama dengan evaluasi (pad-to-square, resize 224)."""
    return transforms.Resize((SIZE, SIZE))(PadToSquare(fill=0)(img))


# ----------------------------------------------------------------------------
def mask_metrics(cam: np.ndarray, lung: np.ndarray) -> dict[str, float]:
    total = cam.sum() + 1e-12
    border = np.zeros_like(lung)
    b = int(round(BORDER * SIZE))
    border[:b, :] = border[-b:, :] = border[:, :b] = border[:, -b:] = 1
    ys, xs = np.nonzero(lung)
    proxy = np.zeros_like(lung)
    if len(ys):
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        mid = (y0 + y1) // 2
        proxy[mid:y1 + 1, x0:x1 + 1] = 1
        proxy = proxy * (1 - lung)
    iy, ix = np.unravel_index(np.argmax(cam), cam.shape)
    return {
        "lung_energy": float((cam * lung).sum() / total), "lung_area": float(lung.mean()),
        "pointing_hit": int(lung[iy, ix] > 0),
        "border_energy": float((cam * border).sum() / total), "border_area": float(border.mean()),
        "cardiac_proxy_energy": float((cam * proxy).sum() / total), "cardiac_proxy_area": float(proxy.mean()),
    }


def mask_index(roots: list[Path]) -> dict[str, dict[str, Path]]:
    """Indeks masker paru Montgomery NLM: {nama berkas: {'leftMask': path, 'rightMask': path}}."""
    index: dict[str, dict[str, Path]] = {}
    for root in roots:
        for side in ("leftMask", "rightMask"):
            for path in root.rglob(f"{side}/MCUCXR_*.png"):
                index.setdefault(path.name.upper(), {})[side] = path
    return index


def montgomery_mask(cxr_path: Path, index: dict) -> np.ndarray | None:
    """Gabungan masker kiri+kanan dalam geometri 224 yang sama dengan citra.

    Dipakai hanya bila kedua masker ada DAN rasio aspek citra sama dengan masker
    (salinan yang sudah diubah ukurannya secara tidak proporsional akan tidak sejajar).
    """
    entry = index.get(cxr_path.name.upper(), {})
    if set(entry) != {"leftMask", "rightMask"}:
        return None
    with Image.open(cxr_path) as im:
        w, h = im.size
    parts = []
    for side in ("leftMask", "rightMask"):
        m = Image.open(entry[side]).convert("L")
        if abs(m.size[0] / m.size[1] - w / h) > 0.01:
            return None
        parts.append(np.asarray(geometric(m), dtype=np.float64) > 127)
    return (parts[0] | parts[1]).astype(np.float64)


# ----------------------------------------------------------------------------
def overlay_figure(rows: list[dict], title: str, path: Path) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.6 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        base = np.asarray(row["image"].convert("L"))
        for c, (key, label) in enumerate((("orig", "Citra"), ("gradcam", "Grad-CAM"), ("gradcampp", "Grad-CAM++"))):
            ax = axes[r, c]
            ax.imshow(base, cmap="gray")
            if key != "orig":
                ax.imshow(row[key], cmap="jet", alpha=0.40, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=9)
            if c == 0:
                ax.set_ylabel(row["caption"], fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_cases(cam: CAM, frame: pd.DataFrame, task: str, title: str, path: Path, device) -> list[dict]:
    tf = get_eval_transforms(SIZE)
    rows, records = [], []
    for _, case in frame.iterrows():
        img = load_8bit(case["path"])
        res = cam(tf(img)[None].to(device), task)
        rows.append({"image": geometric(img), "gradcam": res["gradcam"][0], "gradcampp": res["gradcampp"][0],
                     "caption": f"{case['group']}\ny={case['label']} p={res['prob'][0]:.3f}"})
        records.append({"task": task, "group": case["group"], "path": case["path"], "label": int(case["label"]),
                        "prob": float(res["prob"][0])})
    overlay_figure(rows, title, path)
    return records


def select_internal(npz: Path, csv: Path, raw: Path, thresholds: dict) -> dict[tuple, pd.DataFrame]:
    d = np.load(npz)
    test = pd.read_csv(csv)
    out = {}
    for j, task in enumerate(TASKS):
        m = d["masks"][:, j]
        df = pd.DataFrame({"row": d["row_index"][m], "label": d["labels"][m, j].astype(int),
                           "prob": d["probabilities"][m, j]})
        df["source"] = test["source"].to_numpy()[df["row"]]
        df["path"] = [str(raw / p) for p in test["image_path"].to_numpy()[df["row"]]]
        pred = df["prob"] >= thresholds[task]
        df["cat"] = np.select([pred & (df.label == 1), ~pred & (df.label == 0), pred & (df.label == 0)],
                              ["TP", "TN", "FP"], "FN")
        for source, g in df.groupby("source"):
            picks = []
            for cat in ("TP", "TN", "FP", "FN"):
                gc = g[g.cat == cat].sort_values("prob", ascending=cat in ("TN", "FN"))
                picks.append(gc.head(TOP_K).assign(group=f"{source} {cat}"))
            out[(task, source)] = pd.concat(picks)
    return out


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="resnet18_mtl_frontal_v2_s42_best.pt")
    ap.add_argument("--eval-dir", required=True, help="folder eval berisi prediksi test & threshold beku")
    ap.add_argument("--run", default="resnet18_mtl_frontal_v2_s42")
    ap.add_argument("--raw-data", default="/kaggle/working/mtl_cxr_data/raw")
    ap.add_argument("--split-dir", default=str(ROOT / "data/splits/patient_level_frontal_v1"))
    ap.add_argument("--external-roots", nargs="+", required=True)
    ap.add_argument("--out-dir", default="/kaggle/working/xai")
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)
    cam = CAM(model)
    thr = json.loads((Path(args.eval_dir) / args.run / f"{args.run}_frozen_thresholds.json").read_text())["thresholds"]
    records = []

    # 1. Kasus internal (test) per tugas dan sumber
    cases = select_internal(Path(args.eval_dir) / args.run / f"{args.run}_test_predictions.npz",
                            Path(args.split_dir) / "test.csv", Path(args.raw_data), thr)
    for (task, source), frame in cases.items():
        records += run_cases(cam, frame, task, f"{args.run} | head {task} | test {source}",
                             out / f"cases_{task}_{source}.png", device)

    # 2. Kasus eksternal (head TBC): positif palsu paling yakin dan TP per dataset
    ext = discover([Path(r) for r in args.external_roots])
    tf = get_eval_transforms(SIZE)
    with torch.no_grad():
        probs = []
        for i in range(0, len(ext), 32):
            x = torch.stack([tf(load_8bit(p)) for p in ext["path"].iloc[i:i + 32]]).to(device)
            probs.append(torch.sigmoid(model(x)[:, model.task_names.index("tuberculosis")]).cpu().numpy())
    ext["prob"] = np.concatenate(probs)
    pred = ext["prob"] >= thr["tuberculosis"]
    ext["cat"] = np.select([pred & (ext.label == 1), ~pred & (ext.label == 0), pred & (ext.label == 0)],
                           ["TP", "TN", "FP"], "FN")
    for ds, g in ext.groupby("dataset"):
        picks = [g[g.cat == c].sort_values("prob", ascending=c in ("TN", "FN")).head(TOP_K).assign(group=f"{ds} {c}")
                 for c in ("TP", "TN", "FP", "FN")]
        records += run_cases(cam, pd.concat(picks), "tuberculosis", f"{args.run} | head TBC | eksternal {ds}",
                             out / f"cases_tuberculosis_external_{ds}.png", device)
    pd.DataFrame(records).to_csv(out / "xai_selected_cases.csv", index=False)

    # 3. Kuantitatif: Montgomery dengan masker paru manual
    quant = []
    masks = mask_index([Path(r) for r in args.external_roots])
    mc = ext[ext.dataset == "montgomery"]
    for _, case in mc.iterrows():
        lung = montgomery_mask(Path(case["path"]), masks)
        if lung is None:
            continue
        x = tf(load_8bit(case["path"]))[None].to(device)
        for task in TASKS:
            res = cam(x, task)
            for method in ("gradcam", "gradcampp"):
                quant.append({"case_id": case["case_id"], "label_tb": int(case["label"]), "task": task,
                              "method": method, "prob": float(res["prob"][0]), "tb_pred_positive":
                              int(case["prob"] >= thr["tuberculosis"]), **mask_metrics(res[method][0], lung)})
    q = pd.DataFrame(quant)
    q.to_csv(out / "xai_montgomery_quantitative.csv", index=False)
    summary = {"n_montgomery_cases": int(len(mc)), "n_montgomery_with_aligned_mask": int(q["case_id"].nunique()) if len(q) else 0}
    if len(q):
        cols = ["lung_energy", "lung_area", "pointing_hit", "border_energy", "border_area",
                "cardiac_proxy_energy", "cardiac_proxy_area"]
        agg = q.groupby(["task", "method"])[cols].mean().round(4).reset_index()
        summary["montgomery_mean"] = agg.to_dict(orient="records")
        by = q[q.task == "tuberculosis"].groupby(["method", "label_tb"])[
            ["lung_energy", "pointing_hit", "border_energy"]].mean().round(4).reset_index()
        summary["montgomery_tb_head_by_label"] = by.to_dict(orient="records")

    # 4. Sanity check randomisasi bobot (layer4 + head)
    rand = load_model(Path(args.checkpoint), device)  # salinan bersih (tanpa hook bawaan)
    torch.manual_seed(0)
    for module in list(rand.backbone.model.layer4.modules()) + list(rand.heads.modules()):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    rand_cam = CAM(rand.eval())
    sanity = []
    sample = pd.concat([f.head(5) for f in cases.values()]).head(20)
    for _, case in sample.iterrows():
        x = tf(load_8bit(case["path"]))[None].to(device)
        task = "tuberculosis" if "tbx11k" in case["group"] else "cardiomegaly"
        a, b = cam(x, task), rand_cam(x, task)
        for method in ("gradcam", "gradcampp"):
            rho = spearmanr(a[method][0][::4, ::4].ravel(), b[method][0][::4, ::4].ravel()).correlation
            sanity.append({"group": case["group"], "task": task, "method": method, "spearman": float(rho)})
    s = pd.DataFrame(sanity)
    s.to_csv(out / "xai_sanity_randomization.csv", index=False)
    summary["sanity_spearman_mean"] = s.groupby("method")["spearman"].mean().round(4).to_dict()
    (out / "xai_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nGambar dan tabel tersimpan di {out}")


if __name__ == "__main__":
    main()
