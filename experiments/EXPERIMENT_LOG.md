# Log Eksperimen

| Tanggal | Run | Split | Seed | Status |
|---|---|---|---|---|
| 16 Sep 2026 | resnet18_mtl_baseline_v1 | patient_level_v1 (frontal+lateral) | 42 | Selesai; test dibuka 17 Sep |
| 17 Sep 2026 | resnet18_stl_cardiomegaly_v1, resnet18_stl_tuberculosis_v1 | patient_level_v1 | 42 | Selesai; test dibuka 17 Sep |
| 18 Sep 2026 | resnet18_mtl_frontal_v1 | patient_level_frontal_v1 | 42 | Selesai; test dibuka 18 Sep. Checkpoint dan prediksi hilang (sesi Kaggle berakhir tanpa disimpan); angka tersimpan di Laporan 18 Sep |
| 19 Nov 2026 | resnet18_mtl_frontal_v2_s{42,43,44} | patient_level_frontal_v1 | 42/43/44 | Selesai (commit 65a34fb); best epoch s42 = 7 |
| 19 Nov 2026 | resnet18_stl_{cardiomegaly,tuberculosis}_frontal_v1_s{42,43,44} | patient_level_frontal_v1 | 42/43/44 | Selesai (commit 65a34fb) |
| 19 Nov 2026 | resnet18_mtl_frontal_clahe_v1_s{42,43,44} | patient_level_frontal_v1 | 42/43/44 | Ablation CLAHE, direncanakan (lihat ANALYSIS_PLAN.md) |

Protokol: threshold Youden J dipilih di validation lalu dibekukan; test dievaluasi
satu kali per run setelah seluruh run selesai; tidak ada pemilihan model dari test.
Catatan transparansi: test set pernah dibuka untuk run 17 dan 18 Sep; perubahan ke
frontal-only didorong kesesuaian dengan kriteria inklusi proposal (PA/AP), bukan hasil test.

Rencana analisis dikunci sebelum test dibuka: experiments/ANALYSIS_PLAN.md.
