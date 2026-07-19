# Tesis: ML untuk Peningkatan Akurasi Heart Rate — TI IWR1443BOOST

Lihat `CLAUDE.md` untuk konteks riset lengkap, temuan kunci, dan konvensi kode —
file itu yang bakal dibaca otomatis oleh Claude Code tiap buka project ini.

## Struktur
```
thesis-project/
├── CLAUDE.md                          ← konteks riset lengkap (BACA INI DULU)
├── data/
│   ├── raw/
│   │   ├── position_front/            ← 10 subjek, posisi depan (dataset utama)
│   │   └── position_variation_subject01/  ← 1 subjek, posisi beda-beda
│   ├── processed/                     ← hasil alignment (ground truth + radar)
│   └── README.md
├── code/
│   ├── preprocessing/
│   │   └── extract_ground_truth.py    ← R-peak detection + alignment (mode single/batch)
│   ├── baseline/                      ← taruh code existing kamu di sini
│   ├── models/                        ← model ML akan dikembangkan di sini
│   └── evaluation/
│       └── compare_baseline.py        ← evaluasi kolom HR bawaan TI vs ground truth
├── papers/                            ← taruh PDF paper referensi di sini
└── thesis/                            ← draft bab tesis
```

## Quickstart

1. Taruh data subjek baru mengikuti konvensi folder di `data/README.md`:
   ```
   data/raw/position_front/subjectNN/attys.csv
   data/raw/position_front/subjectNN/radar.csv
   ```

2. Generate ground truth + alignment untuk SEMUA data sekaligus:
   ```bash
   cd code/preprocessing
   python extract_ground_truth.py batch ../../data/raw ../../data/processed/aligned_all.csv
   ```
   (atau mode `single` kalau cuma mau proses satu pasang file — lihat docstring
   di dalam script)

3. Cek reliabilitas output bawaan TI, otomatis breakdown per subjek/posisi:
   ```bash
   cd ../evaluation
   python compare_baseline.py ../../data/processed/aligned_all.csv
   ```

4. Taruh code model existing kamu di `code/baseline/`, lalu buka Claude Code
   di root folder ini dan minta review — dia sudah punya konteks lengkap dari
   `CLAUDE.md`.

## Hasil Validasi Awal (position_front / subject01)

Ground truth ECG: 74.3–104.2 bpm (rata-rata 89.3 bpm, fisiologis wajar).
Semua kolom HR bawaan TI Vital Sign library gagal berkorelasi dengan ground
truth (korelasi mendekati 0, lihat `CLAUDE.md` untuk tabel lengkap) — konfirmasi
awal bahwa perlu model ML berbasis fitur radar mentah. Perlu divalidasi ulang
setelah semua 10 subjek diproses.
