# Dokumentasi Perubahan: Perbaikan Preprocessing Notebook dan Sinkronisasi FastAPI

Tanggal: 20 Oktober 2026

## 1. Ringkasan masalah

Model dilatih dari data yang sudah melewati preprocessing manual di notebook,
tetapi artefak yang disimpan tidak membawa parameter preprocessing. API lalu
menerima nilai mentah dan langsung mengirimkannya ke model, padahal model
dilatih memakai nilai yang sudah di-scale, di-transform, dan di-cap. Akibatnya
prediksi API tidak konsisten dengan hasil training.

Bug yang ditemukan di notebook lama:

1. Data uji memakai `power.fit_transform(...)`, bukan `power.transform(...)`,
   sehingga transformer di-fit ulang pada data uji (kebocoran data).
2. Capping memakai `Q3 = quantile(0.57)` padahal rumus IQR standar memakai
   `quantile(0.75)`.
3. Capping diterapkan juga ke fitur ordinal (`stress_group`, `support_group`),
   sehingga `IQR = 0` dan kedua fitur menjadi konstan. Model lama tidak memakai
   informasi stres dan dukungan sosial.
4. API mengubah `age_group` menjadi angka bebas. Nilai default `"23"` menjadi
   `23.0`, padahal training memakai kode `0/1/2`.
5. Parameter preprocessing tidak disimpan, sehingga API tidak mungkin
   mereproduksi hasil training.

## 2. Solusi

Semua preprocessing dan training sekarang berada di dalam
`prediksi_branour.ipynb`. Tidak ada file Python tambahan di folder model.

Notebook menyimpan dua artefak:

- `preprocessing.joblib`: dictionary berisi `model_features`,
  `ordinal_features`, `ordinal_labels`, `normal_features`, `skewed_features`,
  objek `StandardScaler`, objek `PowerTransformer`, `cap_bounds`,
  `target_map`, `target_labels`, dan `default_input`.
- `model classification.joblib` / `.pkl`: classifier
  `ExtraTreesClassifier(n_estimators=500, random_state=42)`.

API memuat kedua artefak tersebut. Scaling, power transform, capping, dan
mapping ordinal dijalankan memakai parameter dari `preprocessing.joblib`,
sehingga input API diproses sama persis seperti saat training.

### Urutan preprocessing

1. Mapping ordinal:
   - `stress_group` dan `support_group`: `Low (0-3)=0`, `Medium (4-7)=1`,
     `High (8-10)=2`. Singkatan `Low`/`Medium`/`High` dan angka `0/1/2` juga
     diterima.
   - `age_group`: `<20=0`, `21-23=1`, `>23=2`. Angka `0/1/2` juga diterima.
   - Angka lain (misalnya `23`) ditolak dengan HTTP 422.
2. Memilih 15 fitur model. `burnout_score`, `age`, `academic_year`,
   `academic_performance`, `screen_time`, `internet_usage`, dan kolom dummy
   gender tidak dipakai, sama seperti model lama.
3. Menghitung skewness pada data training. Fitur dengan `|skew| <= 0.5`
   memakai `StandardScaler`; sisanya memakai
   `PowerTransformer(method="yeo-johnson")`.
4. Capping outlier dengan `Q1 = 0.25`, `Q3 = 0.75`, dan pengali IQR `1.5`.
   Fitur ordinal tidak di-cap.
5. Output berupa DataFrame dengan urutan kolom yang sama dengan fitur training.

SMOTE hanya dipakai saat training, tidak dipakai saat inference.

## 3. File yang berubah

### Folder `D:\project machine learning`

| File | Status | Keterangan |
| --- | --- | --- |
| `prediksi_branour.ipynb` | Diperbaiki | EDA dipertahankan; bagian preprocessing/training lama diganti dengan pipeline yang benar dan mandiri |
| `preprocessing.joblib` | Baru | Parameter preprocessing hasil fit pada data training |
| `model classification.joblib` | Diganti | Classifier hasil training notebook |
| `model classification.pkl` | Diganti | Salinan classifier |
| `model_metadata.json` | Baru | Metadata training, fitur, parameter, dan metrik |
| `legacy/` | Baru | Backup model lama sebelum ditimpa |
| `README.md` | Diperbarui | Alur training dan deploy dari notebook |
| `DOKUMENTASI_PERUBAHAN.md` | Diperbarui | Dokumen ini |

### Folder `D:\web model fastAPI`

| File | Status | Keterangan |
| --- | --- | --- |
| `API.py` | Diperbarui | Memuat classifier + `preprocessing.joblib` dan menerapkan preprocessing yang sama |
| `preprocessing.joblib` | Baru | Salinan artefak preprocessing dari notebook |
| `model classification.joblib` | Diganti | Salinan classifier dari notebook |
| `model classification.pkl` | Diganti | Salinan classifier dari notebook |
| `model_metadata.json` | Baru | Salinan metadata training |
| `nilai_default.json` | Diperbarui | `age_group` menjadi `"21-23"` |
| `DOKUMENTASI_API.md` | Diperbarui | Kontrak API dan preprocessing |

File Python sementara yang sempat dibuat (`preprocessing.py`, `train_model.py`,
`verify_sync.py`, `test_api_http.py`) sudah dihapus. Perbaikan sekarang hanya
berada di notebook dan `API.py`.

## 4. Perubahan notebook

Bagian EDA (sel 1 sampai heatmap korelasi) dipertahankan. Mulai bagian
**Preprocessing dan training final**, notebook melakukan:

1. Mendefinisikan fitur, mapping ordinal, dan nilai default API di dalam
   notebook.
2. Split stratified 70/30 dengan `random_state=42`.
3. Fit `StandardScaler` dan `PowerTransformer` pada data latih.
4. Menghitung batas capping dari data latih (`Q1=0.25`, `Q3=0.75`).
5. Transform data latih dan data uji, lalu capping.
6. SMOTE pada data latih.
7. Membandingkan `ExtraTreesClassifier` 100, 300, dan 500 pohon memakai 5-fold
   cross-validation. SMOTE berada di dalam pipeline CV agar tidak bocor.
8. Memilih model dengan akurasi CV tertinggi, yaitu 500 pohon, lalu melatihnya
   pada data latih hasil SMOTE.
9. Evaluasi akurasi, macro F1, classification report, confusion matrix (jumlah
   dan proporsi per baris), tabel feature importance, dan PCA.
10. Simpan `preprocessing.joblib`, `model classification.joblib`,
    `model classification.pkl`, dan `model_metadata.json`.
11. Salin artefak ke `D:\web model fastAPI`.
12. Verifikasi bahwa preprocessing dan prediksi dari artefak sama dengan hasil
    notebook dan model yang disalin ke FastAPI.

## 5. Hasil eksekusi bagian notebook

```
Data latih : (1400, 22)
Data uji   : (600, 22)
Sebelum SMOTE: (1400, 15)
Sesudah SMOTE: (3210, 15)
Model terpilih: ExtraTrees-500
Akurasi holdout : 0.8500
Macro F1 holdout: 0.6541
Preprocessing artifact == notebook: True
Prediksi artifact == notebook     : True
Prediksi FastAPI == notebook      : True
```

Hasil 5-fold cross-validation pada data latih:

| Model | Accuracy CV | Macro F1 CV |
| --- | --- | --- |
| ExtraTrees-500 | 0.8636 | 0.6134 |
| ExtraTrees-100 | 0.8614 | 0.5896 |
| ExtraTrees-300 | 0.8600 | 0.6102 |

Dibanding model 100 pohon, model 500 pohon menaikkan akurasi CV dari `0.8614`
ke `0.8636` dan macro F1 CV dari `0.5896` ke `0.6134`. Pada holdout, macro F1
naik dari `0.6436` ke `0.6541`; akurasi holdout `0.8500` (selisih kecil dari
`0.8517`).

Metrik per kelas (holdout):

| Kelas | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Low | 0.925 | 0.889 | 0.907 | 459 |
| Medium | 0.645 | 0.746 | 0.692 | 134 |
| High | 0.500 | 0.286 | 0.364 | 7 |

Confusion matrix: `[[408, 50, 1], [33, 100, 1], [0, 5, 2]]`.

Fitur paling berpengaruh:

| Peringkat | Fitur | Importance |
| --- | --- | --- |
| 1 | `mental_health_index` | 0.1540 |
| 2 | `stress_level` | 0.1482 |
| 3 | `dropout_risk` | 0.1298 |
| 4 | `anxiety_score` | 0.0935 |
| 5 | `depression_score` | 0.0933 |
| 6 | `sleep_hours` | 0.0849 |
| 7 | `exam_pressure` | 0.0497 |
| 8 | `support_group` | 0.0374 |
| 9 | `financial_stress` | 0.0318 |
| 10 | `social_support` | 0.0315 |

## 6. Kontrak API

- `POST /predict` menerima satu objek fitur, atau dibungkus dalam
  `data`/`features`.
- `risk_level` diabaikan jika dikirim.
- Input pengguna hanya 6 fitur: `study_hours_per_day`, `exam_pressure`,
  `anxiety_score`, `sleep_hours`, `stress_level`, dan `age`.
- Fitur model lainnya memakai nilai default dari `preprocessing.joblib`.
  Default `depression_score=2.27`, `mental_health_index=5.70`, dan
  `dropout_risk=2.61` memakai profil Medium agar kelas `High` bisa dicapai.
  Fitur di luar 6 input pengguna menghasilkan HTTP 422.
- `age` harus berupa angka 1-119 dan dikonversi menjadi `age_group`:
  `<=20`, `21-23`, dan `>=24`.
- Nilai boolean dan non-finite ditolak.
- Respons: `prediksi`, `kategori`, `kelas_model`, `probabilitas`,
  `urutan_kelas`, `urutan_kategori`.

## 7. Cara menjalankan

1. Buka `D:\project machine learning\prediksi_branour.ipynb`.
2. Jalankan notebook dari atas, termasuk bagian
   **Preprocessing dan training final**.
3. Notebook menyimpan artefak dan menyalinnya ke
   `D:\web model fastAPI`.
4. Jalankan API:

```powershell
cd "D:\web model fastAPI"
.\venv\Scripts\python.exe -m uvicorn API:app --reload
```

## 8. Catatan dan batasan

- Kelas `High` hanya berjumlah 24 dari 2.000 baris, sehingga metriknya masih
  lemah dan berfluktuasi.
- API memakai nilai default untuk fitur yang tidak dikirim. Perilaku ini
  dipertahankan dari versi sebelumnya.
- Notebook memakai split stratified agar distribusi kelas minoritas lebih
  stabil.
- Backup model lama ada di `legacy/` dan jangan dihapus jika masih ingin
  membandingkan hasil.
