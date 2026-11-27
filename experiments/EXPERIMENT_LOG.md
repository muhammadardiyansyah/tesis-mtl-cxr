# Log Eksperimen

| Tanggal | Run | Split | Seed | Status |
|---|---|---|---|---|
| 16 Sep 2026 | resnet18_mtl_baseline_v1 | patient_level_v1 (frontal+lateral) | 42 | Selesai; test dibuka 17 Sep |
| 17 Sep 2026 | resnet18_stl_cardiomegaly_v1, resnet18_stl_tuberculosis_v1 | patient_level_v1 | 42 | Selesai; test dibuka 17 Sep |
| 18 Sep 2026 | resnet18_mtl_frontal_v1 | patient_level_frontal_v1 | 42 | Selesai; test dibuka 18 Sep. Checkpoint dan prediksi hilang (sesi Kaggle berakhir tanpa disimpan); angka tersimpan di Laporan 18 Sep |
| 19 Nov 2026 | resnet18_mtl_frontal_v2_s{42,43,44} | patient_level_frontal_v1 | 42/43/44 | Selesai; s42 batch 1 (commit 65a34fb), s43/s44 batch 2 (commit 5a5f5a6, kode pelatihan identik); best epoch 7/6/9 |
| 19 Nov 2026 | resnet18_stl_{cardiomegaly,tuberculosis}_frontal_v1_s{42,43,44} | patient_level_frontal_v1 | 42/43/44 | Selesai; STL kardio s42 batch 1 (65a34fb), sisanya batch 2 (5a5f5a6); best epoch kardio 7/4/11, TBC 4/8/7 |
| 19 Nov 2026 | resnet18_mtl_frontal_clahe_v1_s{42,43,44} | patient_level_frontal_v1 | 42/43/44 | Ablation CLAHE (commit 38e63ca); dievaluasi 20 Nov (ANALYSIS_PLAN bagian 10) |
| 26–27 Nov 2026 | resnet18_mtl_frontal_v3_s{42,43,44}, resnet18_stl_cardiomegaly_frontal_v2_s{42,43,44} | patient_level_frontal_v1 | 42/43/44 | Eksperimen tambahan (ANALYSIS_PLAN bagian 12), commit 65669ea, steps_per_epoch 1.472 (terverifikasi di log dan history). Sesi 1: MTL s42, STL s42/s43; sesi 2: MTL s43/s44, STL s44. Best epoch (val loss) MTL 7/14/2, STL kardio 7/6/15 |

Protokol: threshold Youden J dipilih di validation lalu dibekukan; test dievaluasi
satu kali per run setelah seluruh run selesai; tidak ada pemilihan model dari test.
Catatan transparansi: test set pernah dibuka untuk run 17 dan 18 Sep; perubahan ke
frontal-only didorong kesesuaian dengan kriteria inklusi proposal (PA/AP), bukan hasil test.

Rencana analisis dikunci sebelum test dibuka: experiments/ANALYSIS_PLAN.md.

Evaluasi: test internal run v2 dan STL v1 dibuka 20 Nov 2026 (evaluate_run.py, commit 5a5f5a6);
uji eksternal TBC final dan XAI 20 Nov 2026 (ANALYSIS_PLAN bagian 8 dan 11).
Artefak: results tesis-training-runs (batch 1), results tesis-training-runs-b2 (batch 2),
results tesis-eval-final, results 3 tesis-external-tb, results tesis-eval-clahe, results tesis-xai,
results tesis-training-runs-b3 (v3 sesi 1), results tesis-training-runs-b3 sesi 2 (v3 sesi 2).
Evaluasi v3: notebook tesis-eval-v3 (experiments/sel_evaluasi_v3.py), dijalankan 27 Nov 2026;
checkpoint dari dataset Kaggle tesis-v3-checkpoints (enam *_best.pt, disalin dari kedua folder sesi).
