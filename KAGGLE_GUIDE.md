# Menjalankan MTL CXR pada Kaggle GPU

## Struktur dataset privat

Buat Kaggle Dataset privat dengan slug `mtl-cxr-private-data` dan struktur:

```text
archives/
  chestxray14.tar
  chexpert.tar
  tbx11k.tar
metadata/
  harmonized_model_ready_v1.csv
splits/patient_level_v1/
  train.csv
  val.csv
  test.csv
  split_report.json
```

Ketiga arsip harus dibuat dari folder `data/raw` agar masing-masing memiliki
folder teratas `chestxray14`, `chexpert`, dan `tbx11k`. Dataset harus tetap
private dan tidak boleh dipublikasikan.

## Urutan aman

1. Tambahkan dataset privat pada Kaggle Notebook.
2. Aktifkan `GPU T4 x2`; pipeline tahap awal menggunakan satu T4.
3. Buka `notebooks/kaggle_gpu_training.ipynb` dan jalankan cell berurutan.
4. Jalankan smoke test terlebih dahulu.
5. Jalankan pilot satu epoch hanya setelah smoke test berhasil.
6. Unduh checkpoint dan history sebelum menutup sesi.

## Perintah utama

```bash
python run_training.py --config configs/config.kaggle.yaml --no-local-config --mode smoke
python run_training.py --config configs/config.kaggle.yaml --no-local-config --mode pilot
python run_training.py --config configs/config.kaggle.yaml --no-local-config --mode full
python run_training.py --config configs/config.kaggle.yaml --no-local-config --mode full --resume
```

Checkpoint `best` menyimpan model dengan validation loss terendah. Checkpoint
`latest` diperbarui setiap epoch dan dipakai untuk melanjutkan sesi terputus.
