# Menjalankan eksperimen tambahan v3 di Kaggle (ANALYSIS_PLAN bagian 12)

Prasyarat: notebook Kaggle dengan akselerator **GPU T4 x2**, persiapan data sama seperti
notebook `tesis-training-runs-b2` (repo di `/kaggle/working/tesis-mtl-cxr`, data di
`/kaggle/working/mtl_cxr_data/raw`). Repo harus di-*clone* pada commit yang memuat
amandemen bagian 12. Catat hash commit di sel pertama.

Enam run, dua berjalan paralel (satu per GPU). Perkiraan kasar dari waktu per epoch v2:
MTL ±30 menit/epoch, STL kardiomegali ±20 menit/epoch; dengan early stopping ±10–14 epoch,
satu run MTL ±5–7 jam dan satu run STL ±3,5–5 jam. Batas sesi Kaggle 12 jam: jika sesi
habis, run yang belum selesai dilanjutkan dari checkpoint `*_latest.pt` dengan `--resume` (lihat catatan di bawah Sel 2).

Pembagian yang disarankan (GPU 0 = MTL, GPU 1 = STL):

| Sesi | GPU 0 | GPU 1 |
|---|---|---|
| 1 | MTL v3 s42 | STL kardio v2 s42, lalu s43 |
| 2 | MTL v3 s43 | STL kardio v2 s44 |
| 3 | MTL v3 s44 | – |

## Sel 1: cek commit dan smoke test (CPU/GPU, beberapa menit)

```python
import subprocess
repo = "/kaggle/working/tesis-mtl-cxr"
print(subprocess.run(["git", "-C", repo, "log", "--oneline", "-3"], capture_output=True, text=True).stdout)
# smoke test: 2 batch, memastikan config dan argumen baru terbaca
subprocess.run(["python", "run_training.py", "--config", "configs/config.kaggle_frontal_v3_s42.yaml",
                "--no-local-config", "--mode", "smoke"], cwd=repo, check=True)
subprocess.run(["python", "run_single_task_training_frontal.py", "--task", "cardiomegaly",
                "--mode", "smoke", "--seed", "42", "--steps-per-epoch", "1472", "--version", "v2"],
               cwd=repo, check=True)
```

Pastikan log smoke test MTL menampilkan nama `resnet18_mtl_frontal_v3_s42` dan STL menampilkan
`resnet18_stl_cardiomegaly_frontal_v2_s42` serta `Langkah per epoch : 2` (mode smoke).

## Sel 2: jalankan dua run paralel (ubah daftar JOBS per sesi)

```python
import os, subprocess, json, time
from pathlib import Path

repo = "/kaggle/working/tesis-mtl-cxr"
logs = Path("/kaggle/working/run_logs"); logs.mkdir(exist_ok=True)

def mtl(seed, resume=False):
    c = ["python", "run_training.py", "--config", f"configs/config.kaggle_frontal_v3_s{seed}.yaml",
         "--no-local-config", "--mode", "full"]
    return c + (["--resume"] if resume else [])

def stl(seed, resume=False):
    c = ["python", "run_single_task_training_frontal.py", "--task", "cardiomegaly", "--mode", "full",
         "--seed", str(seed), "--steps-per-epoch", "1472", "--version", "v2"]
    return c + (["--resume"] if resume else [])

# Sesi 1: GPU0 -> MTL s42; GPU1 -> STL s42 lalu s43 (berurutan)
JOBS = {
    "0": [("MTL_v3_s42", mtl(42))],
    "1": [("STL_kardio_v2_s42", stl(42)), ("STL_kardio_v2_s43", stl(43))],
}

def run_queue(gpu, queue):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    cmd = " && ".join(" ".join(c) + f" > {logs}/{name}.log 2>&1" for name, c in queue)
    return subprocess.Popen(cmd, shell=True, cwd=repo, env=env)

start = time.time()
procs = {gpu: run_queue(gpu, q) for gpu, q in JOBS.items()}
for gpu, p in procs.items():
    p.wait()
    print("GPU", gpu, "selesai, return code", p.returncode)
print("Durasi total (jam):", round((time.time() - start) / 3600, 2))
```

Catatan melanjutkan run yang terputus: `--resume` gagal bila checkpoint belum ada, jadi pakai
`resume=True` hanya untuk run yang sudah sempat berjalan. Karena `/kaggle/working` kosong di sesi
baru, tambahkan output sesi sebelumnya sebagai *Input* notebook, lalu salin
`*_latest.pt` ke `/kaggle/working/checkpoints_frontal_v3` (MTL) atau
`/kaggle/working/checkpoints_single_task_frontal` (STL) sebelum menjalankan `mtl(seed, resume=True)`
atau `stl(seed, resume=True)`. Jangan mengubah argumen lain.

## Sel 3: simpan keluaran

Unduh folder berikut setelah setiap sesi (checkpoint `*_best.pt` diperlukan untuk evaluasi):
`checkpoints_frontal_v3`, `outputs_frontal_v3`, `checkpoints_single_task_frontal`,
`outputs_single_task_frontal`, `run_logs`.

Evaluasi (prediksi validation → threshold beku → prediksi test → perbandingan) dijalankan
setelah keenam run selesai, dengan `evaluate_run.py` seperti pada run v2.
