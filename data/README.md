# Data

## raw/

Dua dataset eksperimen, jangan dicampur analisisnya (lihat CLAUDE.md untuk alasan):

```
raw/
├── position_front/                      ← dataset UTAMA: 10 subjek beda, posisi depan
│   ├── subject01/attys.csv, radar.csv
│   ├── subject02/attys.csv, radar.csv
│   └── ... s.d. subject10/
└── position_variation_subject01/        ← dataset SEKUNDER: 1 subjek, posisi beda-beda
    ├── front/attys.csv, radar.csv
    ├── side/attys.csv, radar.csv
    └── side_left_back/attys.csv, radar.csv
```

Aturan penamaan file: **selalu** `attys.csv` dan `radar.csv` di dalam tiap
folder subjek/posisi (metadata sudah ada di nama foldernya, jadi gak perlu
suffix nomor session lagi). Kalau nambah subjek baru, ikutin pola
`position_front/subjectNN/`. Kalau nambah posisi baru untuk subjek yang sudah
ada di `position_variation`, tinggal tambah folder baru di dalamnya.

- `attys.csv` — kolom `timestamp` (unix epoch detik) dan `value` (sinyal ECG
  mentah, ~125 Hz)
- `radar.csv` — kolom `Timestamp` (unix epoch detik) + berbagai kolom HR/fase/
  energi bawaan IWR1443BOOST (lihat CLAUDE.md untuk detail dan catatan
  reliabilitas tiap kolom)

## processed/

- `aligned_all.csv` — hasil `code/preprocessing/extract_ground_truth.py batch`,
  gabungan SEMUA subjek dan posisi dalam satu file, dengan kolom tambahan:
  - `dataset` — `position_front` atau `position_variation`
  - `subject_id` — misal `subject01`
  - `position` — misal `front`, `side`, `side_left_back`
  - `gt_heart_rate` — ground truth BPM dari deteksi R-peak Attys, sudah
    diinterpolasi ke timestamp radar

  Generate ulang file ini setiap kali ada data baru masuk ke `raw/`.

**Catatan penting**: cakupan waktu tiap sesi di `aligned_all.csv` hanya
mencakup irisan waktu yang di-cover oleh deteksi R-peak yang valid — biasanya
sedikit lebih pendek dari durasi rekaman mentah karena beberapa beat pertama/
terakhir dibuang saat perhitungan RR interval. Cek log output script untuk
detail per sesi.
