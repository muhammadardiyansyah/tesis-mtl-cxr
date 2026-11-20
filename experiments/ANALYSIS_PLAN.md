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

## 8. Amandemen 20 Nov 2026: uji eksternal TBC (dikunci sebelum dijalankan)

Alasan: analisis 5.2 (uji shortcut) menunjukkan head TBC model MTL memprediksi positif
pada 34-43% citra NIH dan 66-81% citra CheXpert (prevalensi TBC sebenarnya jauh < 1%),
sehingga AUC TBC internal hanya berlaku di dalam TBX11K. Uji eksternal ini adalah opsi
cadangan proposal 2.3.5 (dataset publik lain yang tidak dipakai training).

- Data: Shenzhen (662 citra) dan Montgomery (138 citra), NLM (Jaeger et al., 2014).
  Label dari nama berkas (`CHNCXR/MCUCXR_####_X`, X = 0 normal, 1 TB). Folder `imgs/extra`
  TBX11K tidak pernah dipakai di training/validasi/test internal.
- Cek kebocoran: citra eksternal dengan korelasi thumbnail 32x32 > 0,995 terhadap salah
  satu dari 8.400 citra TBX11K penelitian dikeluarkan dari analisis utama dan dilaporkan;
  pasangan yang ditandai diperiksa visual.
- Model: `resnet18_mtl_frontal_v2_s42` (utama, sesuai bagian 6) dan seluruh seed MTL serta
  STL TBC (pembanding). Threshold = threshold beku validation internal. Tidak ada
  penyetelan threshold, kalibrasi ulang, atau fine-tuning pada data eksternal.
- Endpoint: AUC-ROC, sensitivitas, spesifisitas, AUPRC, kalibrasi per dataset dan gabungan,
  CI 95% bootstrap (1.000); MTL vs STL TBC per seed dengan DeLong + bootstrap berpasangan.
- Interpretasi yang ditetapkan sekarang: penurunan AUC eksternal > 0,05 dibanding internal
  atau spesifisitas eksternal < 0,80 pada threshold beku dinyatakan sebagai kegagalan
  generalisasi head TBC. Hasil dilaporkan apa adanya.
- Catatan: pHash (bagian 5.3) terbukti tidak spesifik untuk CXR (1.082 pasangan "duplikat"
  berlabel berbeda); analisis duplikat utama memakai MD5 (pasca-hoc, ditandai).

## 9. Rencana uji eksternal RS Universitas Hasanuddin (melengkapi bagian 6; dikunci 20 Nov 2026, sebelum data diterima)

Dasar etik: Rekomendasi Persetujuan Etik No. 1276/UN4.6.4.5.31/PP36/2026 (protokol UH26101021),
berlaku 18 Nov 2026 - 18 Nov 2027. Data tidak diunggah ke Kaggle; inferensi dijalankan secara
lokal di lingkungan Universitas Hasanuddin dengan checkpoint dan threshold beku.

**Besar sampel.** Metode Buderer (1996) untuk estimasi sensitivitas/spesifisitas:
n = z^2 * p(1-p) / d^2 per kelompok. Dengan Se = Sp = 0,80, CI 95% (z = 1,96):
presisi +-0,10 -> 62 kasus per kelompok (minimum); presisi +-0,075 -> 110.
Target: >=100 kasus TBC positif, >=100 kasus kardiomegali positif, >=100 citra normal
(total +-300-400 citra), sesuai rekomendasi minimum 100 kejadian untuk uji eksternal
(Collins, Ogundimu, & Altman, 2016). Bila jumlah < 62 per kelompok, hasil dilaporkan sebagai
studi pendahuluan dengan CI lebar.

**Desain pengambilan.** Retrospektif, sampel diperkaya (kasus dan kontrol diambil terpisah).
Konsekuensi: PPV/NPV tidak diestimasi; yang dilaporkan sensitivitas, spesifisitas, AUC, AUPRC,
kalibrasi. Kriteria inklusi: CXR frontal PA/AP dewasa, kualitas diagnostik. Eksklusi: citra
lateral, artefak berat yang menutupi toraks, label tidak dapat dipastikan.

**Standar acuan.** TBC: konfirmasi bakteriologis (GeneXpert/BTA) atau diagnosis klinis dokter
(jenis konfirmasi dicatat per kasus dan dianalisis terpisah). Kardiomegali: CTR > 0,5 menurut
bacaan radiolog. Setiap citra berlabel untuk kedua kondisi; citra "normal" negatif untuk keduanya.

**Model dan analisis.** `resnet18_mtl_frontal_v2_s42` + threshold beku validation internal
(utama); seed lain dan STL sebagai pembanding. Tidak ada fine-tuning, kalibrasi ulang, atau
pemilihan threshold dari data RS. Metrik per tugas dengan CI 95% bootstrap per pasien.

**Hipotesis yang ditetapkan sebelum melihat data** (berdasarkan bagian 5.2 dan 8): head TBC
diperkirakan menghasilkan spesifisitas rendah (banyak positif palsu) pada data RS karena
belajar ciri sumber TBX11K; head kardiomegali diperkirakan menurun sedang dibanding internal.
Hasil dilaporkan apa adanya, termasuk bila hipotesis ini tidak terbukti.

## 10. Amandemen 20 Nov 2026: evaluasi ablation CLAHE (dikunci sebelum dievaluasi)

Run: `resnet18_mtl_frontal_clahe_v1_s{42,43,44}` (sudah dilatih, commit 38e63ca; belum ada
prediksi validation/test yang dilihat). Protokol sama dengan run utama: threshold Youden dari
validation (dengan transform CLAHE), dibekukan, test dibuka sekali.

- Pembanding: MTL utama tanpa CLAHE pada seed yang sama (`resnet18_mtl_frontal_v2_s{seed}`),
  bootstrap berpasangan per pasien + DeLong, per tugas; ringkasan mean ± SD.
- Uji eksternal TBC (Shenzhen + Montgomery, 800 citra; prosedur bagian 8) juga dijalankan
  untuk run CLAHE. Hipotesis a priori: CLAHE mengurangi perbedaan kontras antar-sumber
  sehingga AUC eksternal TBC dapat naik; dinyatakan bermakna hanya bila arah sama pada
  ketiga seed dan CI bootstrap selisih AUC (CLAHE − utama) tidak memuat 0 pada ketiga seed.
- Hasil ablation tidak mengubah pipeline utama (bagian 1).

## 11. Amandemen 20 Nov 2026: analisis XAI (RQ3), dikunci sebelum peta saliency dilihat

Model: `resnet18_mtl_frontal_v2_s42` (bagian 5.5). Lapisan target: `backbone.model.layer4`.
Metode: Grad-CAM dan Grad-CAM++ per head tugas (logit), implementasi manual (`run_xai.py`).

- Kualitatif: kasus TP/TN/FP/FN paling yakin (3 per kategori) per tugas dan per sumber dari
  prediksi test yang sudah ada (threshold beku), ditambah kasus eksternal Shenzhen/Montgomery
  untuk head TBC.
- Kuantitatif (Montgomery, masker paru manual NLM; hanya citra yang geometrinya sejajar dengan
  maskernya): proporsi energi peta di dalam paru (`lung_energy`) dibanding proporsi luas paru;
  pointing game (titik maksimum di dalam paru); proporsi energi pada bingkai luar 8%
  (`border_energy`); untuk head kardiomegali, energi pada proksi mediastinum/jantung (area di
  antara kedua paru pada separuh bawah kotak paru; bukan masker jantung).
- Tafsiran a priori: head TBC dianggap berfokus anatomis bila `lung_energy` rata-rata jelas di atas
  proporsi luas paru dan pointing game > 0,5; dianggap mendukung hipotesis shortcut bila
  `border_energy` melebihi proporsi luas bingkai atau `lung_energy` tidak lebih tinggi dari luas paru.
- Sanity check (randomisasi bobot layer4 + head): korelasi Spearman peta asli vs model teracak;
  rata-rata > 0,5 berarti peta tidak cukup bergantung pada bobot terlatih, sehingga interpretasi
  visual harus dibatasi.
- Keterbatasan yang dinyatakan: saliency map bukan bukti kausal; proksi jantung bukan anotasi jantung.
