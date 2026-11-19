# Rencana Analisis (dikunci sebelum test set dibuka)

Dokumen ini di-commit **sebelum** prediksi test untuk run final dijalankan
(19 November 2026). Hash commit dokumen ini menjadi bukti bahwa model, threshold,
endpoint, dan uji statistik ditetapkan sebelum hasil test dilihat. Perubahan
setelah test dibuka hanya boleh berupa penambahan analisis yang ditandai
"pasca-hoc" di tesis, bukan pengubahan rencana ini.

## 1. Model yang dievaluasi

Pipeline utama: split `patient_level_frontal_v1`, ResNet18 pralatih ImageNet,
tanpa CLAHE, commit kode `65a34fb`.

| Peran | Run (seed 42, 43, 44) |
|---|---|
| Model utama (MTL) | `resnet18_mtl_frontal_v2_s{seed}` |
| Pembanding STL kardiomegali | `resnet18_stl_cardiomegaly_frontal_v1_s{seed}` |
| Pembanding STL TBC | `resnet18_stl_tuberculosis_frontal_v1_s{seed}` |
| Ablation CLAHE (sekunder) | `resnet18_mtl_frontal_clahe_v1_s{seed}` |

Pipeline utama **tidak diganti** berdasarkan hasil ablation CLAHE, apa pun hasilnya.
Checkpoint yang dievaluasi adalah checkpoint dengan validation loss terendah
(aturan early stopping yang sama untuk MTL dan STL); tidak ada pemilihan epoch lain.

## 2. Threshold

Youden J pada validation set, per run dan per tugas, lalu dibekukan
(`evaluate_run.py threshold`). Tidak ada threshold yang dipilih dari test set.
Test set dievaluasi satu kali per run.

## 3. Endpoint primer (RQ2)

Selisih AUC-ROC (MTL − STL) pada test set, per tugas:

- per seed: bootstrap berpasangan per pasien (1.000 iterasi, seed 42) dan uji DeLong berpasangan;
- lintas seed: mean ± SD dari 3 seed.

Aturan interpretasi (ditetapkan sekarang):
- "MTL lebih baik/lebih buruk" hanya jika arah selisih sama pada ketiga seed **dan**
  CI 95% bootstrap tidak memuat 0 pada ketiga seed;
- selain itu disimpulkan "tidak ada perbedaan yang konsisten";
- sebagai acuan klinis, selisih |ΔAUC| < 0,02 dianggap tidak bermakna secara praktis
  meskipun signifikan secara statistik.

## 4. Endpoint sekunder

Pada threshold beku: sensitivitas, spesifisitas, presisi, NPV, F1, akurasi,
balanced accuracy, confusion matrix; average precision (AUPRC); kalibrasi
(Brier score, ECE 10 bin). Semua dengan CI 95% bootstrap per pasien.
Efisiensi: jumlah parameter dan ukuran berkas satu model MTL vs dua model STL.

## 5. Analisis tambahan yang direncanakan

1. **Per sumber data:** metrik kardiomegali terpisah untuk NIH dan CheXpert (MTL vs STL).
2. **Uji shortcut sumber dataset:** distribusi probabilitas head TBC model MTL pada citra
   non-TBX11K (NIH/CheXpert, tanpa label TBC) dan proporsi di atas threshold beku;
   sebaliknya head kardiomegali pada citra TBX11K. Dilaporkan deskriptif pada validation
   dan test. Probabilitas yang hampir selalu ~0 pada citra non-TBX11K dibaca sebagai
   indikasi model mengenali sumber data, bukan patologi.
3. **Sensitivitas duplikat TBX11K:** `check_duplicates.py` (MD5 + pHash, Hamming ≤ 6)
   dijalankan sebelum test dibuka. Metrik TBC dihitung ulang tanpa citra test yang
   memiliki pasangan duplikat/near-duplicate di train atau val (`metrics --exclude`).
4. **Ablation CLAHE:** MTL+CLAHE vs MTL utama, protokol sama (threshold dari validation,
   test sekali), dilaporkan sebagai analisis sekunder.
5. **XAI (RQ3):** Grad-CAM dan Grad-CAM++ per tugas pada model MTL seed 42 (dipilih a priori),
   dengan kasus TP/TN/FP/FN yang dipilih secara sistematis (probabilitas paling yakin) per
   tugas dan per sumber. Interpretasi kualitatif disertai catatan keterbatasan saliency map.

## 6. Uji eksternal (data RS Universitas Hasanuddin)

Model yang dibawa ke uji eksternal: `resnet18_mtl_frontal_v2_s42` beserta threshold
validation-nya, tanpa penyetelan ulang. Data eksternal tidak dipakai untuk training,
validasi, maupun pemilihan threshold.

## 7. Transparansi

- Test set (publik) pernah dibuka untuk run pendahuluan 17 Sep 2026 (protokol frontal+lateral)
  dan 18 Sep 2026 (`resnet18_mtl_frontal_v1`, checkpoint hilang). Perubahan ke frontal-only
  didorong kriteria inklusi proposal (PA/AP), bukan hasil test. Run final dilatih ulang dengan
  konfigurasi yang sama dan multi-seed.
- Penyimpangan dari proposal yang dilaporkan di Bab III: AdamW (bukan Adam), pos_weight
  kelas, loss total = rata-rata berbobot tugas aktif, augmentasi konservatif tanpa horizontal
  flip, source-balanced sampler, CLAHE dipindah menjadi ablation, NIH versi 224×224,
  implementasi Grad-CAM sendiri, eksklusi artefak berat tidak dilakukan manual.
