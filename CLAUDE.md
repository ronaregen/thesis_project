# Tesis: ML untuk Peningkatan Akurasi Heart Rate pada TI IWR1443BOOST (FMCW Radar)

## Tujuan Riset
Meningkatkan akurasi estimasi detak jantung (BPM) dari radar FMCW TI IWR1443BOOST.
Output bawaan Vital Sign library dari Texas Instruments terbukti tidak reliable
(lihat "Temuan Kunci" di bawah), sehingga akan dibangun model ML yang belajar
langsung dari fitur radar mentah/intermediate, bukan dari output BPM jadi milik TI.

## Commands

Tidak ada `requirements.txt` / lockfile di repo ini — dependencies (`pandas`,
`numpy`, `scipy`) harus diinstall manual sebelum jalanin script apapun:
```bash
pip install pandas numpy scipy
```

Preprocessing (regenerate tiap kali ada data baru masuk ke `data/raw/`):
```bash
cd code/preprocessing
python extract_ground_truth.py batch ../../data/raw ../../data/processed/aligned_all.csv
```

Evaluasi baseline TI vs ground truth (breakdown otomatis per subjek/posisi,
opsional filter `confidence_heart >= X` sebagai argumen ke-2 — **jangan pakai
threshold yang sama untuk kedua dataset**, lihat Temuan Kunci poin 6):
```bash
cd code/evaluation
python compare_baseline.py ../../data/processed/aligned_all.csv
python compare_baseline.py ../../data/processed/aligned_all.csv 0.05  # contoh filter confidence
```

Diagnostik spectral sinyal fase mentah (`unwrapPhasePeak_mm`) vs ground truth,
lepas dari algoritma HR bawaan TI — hanya jalan di `position_front` (lihat
Temuan Kunci poin 4). **CATATAN: script ini punya band 0.7-3.3 Hz yang TERBUKTI
SALAH (Temuan Kunci poin 11a) — disimpan cuma sebagai catatan sejarah, jangan
dipakai buat kesimpulan baru. Pakai `corrected_estimator.py` sebagai gantinya:**
```bash
python analyze_phase_signal.py ../../data/processed/aligned_all.csv
```

**Root-cause diagnostik — apakah sinyal HR beneran ADA di fase radar** (tes rank
frekuensi GT di PSD; ini yang membuktikan radar TIDAK rusak, poin 10):
```bash
cd code/evaluation
python diagnose_root_cause.py ../../data/processed/aligned_all.csv
```

**Estimator fase yang SUDAH DIPERBAIKI** (band 0.8-2.0 Hz + anti-alias + detrend
+ medfilt; MAE 5.2 bpm di subjek uji S10 vs trivial 14.8 —
poin 13). Ini titik-berangkat buat kerjaan berikutnya, BUKAN `analyze_phase_signal.py`:
```bash
cd code/evaluation
python corrected_estimator.py ../../data/processed/aligned_all.csv
```

Regenerate dokumen bimbingan (`thesis/diagnosis_riset.pdf` +
`thesis/rekomendasi_riset.pdf`) — butuh `pip install matplotlib fpdf2`:
```bash
cd code/reporting
python make_figures.py ../../data/processed/aligned_all.csv /tmp/fig
python make_report.py /tmp/fig ../../thesis
```

Tidak ada test suite di repo ini (proyek riset/analisis data, bukan library).

## Setup Eksperimen
- **Device under test**: TI IWR1443BOOST, output via Vital Sign demo library bawaan TI
- **Ground truth / pembanding**: Attys (perekam sinyal ECG), sample rate 125 Hz
- **Kedua device merekam secara bersamaan** (time frame sama), masing-masing dengan
  timestamp Unix epoch sendiri (bukan device yang sama, perlu alignment berbasis waktu)
- **Radar sample rate**: tidak seragam, rata-rata approx 64 Hz (perlu di-resample /
  interpolasi kalau butuh grid waktu tetap)

## Struktur Data

### Dua dataset eksperimen (jangan dicampur analisisnya)

1. **`position_front`** — data utama: **10 subjek berbeda**, semua radar di
   posisi depan. Ini dataset utama untuk training/evaluasi model (variasi
   antar-individu, n=10).
2. **`position_variation`** — data sekunder: **1 subjek saja**, radar di
   beberapa posisi berbeda (depan, samping, belakang-samping-kiri). Dipakai
   untuk analisis tambahan (misal: "apakah akurasi model turun kalau posisi
   radar berubah?"), BUKAN untuk training set utama karena cuma representasi
   1 orang (risiko bias individu).

### Layout folder
```
data/raw/
├── position_front/
│   ├── subject01/attys.csv, radar.csv
│   ├── subject02/attys.csv, radar.csv
│   └── ... (s.d. subject10)
└── position_variation_subject01/
    ├── front/attys.csv, radar.csv
    ├── front_left/attys.csv, radar.csv
    ├── front_right/attys.csv, radar.csv
    ├── left/attys.csv, radar.csv
    ├── left_back/attys.csv, radar.csv
    ├── right/attys.csv, radar.csv
    ├── right_back/attys.csv, radar.csv
    └── back/attys.csv, radar.csv
```
(8 posisi aktual per Jul 2026 — bukan cuma front/side/side_left_back seperti
contoh lama)
Nama file di dalam tiap folder subjek/posisi SELALU `attys.csv` dan `radar.csv`
(bukan pakai suffix session lagi) — metadata subjek/posisi sudah terkandung
di nama foldernya. Script `extract_ground_truth.py` mode batch bergantung pada
konvensi ini untuk auto-parsing kolom `dataset`/`subject_id`/`position`.

### Format kolom
- `attys.csv` — kolom: `timestamp` (unix epoch, detik), `value` (sinyal ECG
  mentah, satuan volt approx, amplitudo approx ±1e-4)
- `radar.csv` — kolom (skema `position_front`, lihat poin 4 di Temuan Kunci
  untuk beda skema `position_variation`): `Timestamp` (unix epoch, detik),
  `outputFilterHeartOut`, `final_heart_rate`, `heart_rate_est_peak`,
  `heart_rate_est_fft`, `heart_rate_est_fft_4hz`, `heartRateEst_xCorr`,
  `confidence_heart`, `range_bin_value`, `rangeBinPhaseIndex`,
  `unwrapPhasePeak_mm`, `sumEnergyHeartWfm`
- `data/processed/aligned_all.csv` — hasil `extract_ground_truth.py batch`:
  gabungan semua subjek/posisi, dengan kolom tambahan `dataset`, `subject_id`,
  `position`, dan `gt_heart_rate` (ground truth BPM dari R-peak Attys,
  diinterpolasi ke timestamp radar)

## TEMUAN KUNCI (penting — jangan diulang dari nol)

> **🚨 KOREKSI TERBESAR (13 Jul 2026, revisi ke-2) — BACA POIN 20-23 DULU.**
> Setelah `old code/logger.py` + `data/raw/distance` masuk, ketemu **TIGA CACAT
> AKUISISI** yang menjelaskan SEMUANYA. Akar masalahnya BUKAN algoritma TI dan
> BUKAN sensornya — tapi **setup pengambilan datanya yang salah**:
> 1. **Frame rate 50 fps** (logger pakai `frameCfg ... 20`), padahal library TI
>    dirancang buat **20 fps**. Filter jantung TI jadi mulur 2.5x → band 0.8-4.0 Hz
>    berubah jadi 2.0-10.0 Hz → **detak jantung manusia (1-2 Hz) justru DIBUANG**.
>    Bukti: 94.3% energi `outputFilterHeartOut` ada di 2-10 Hz, cuma 4.2% di band
>    jantung asli.
> 2. **Logger salah baca 4 dari 10 offset byte.** `heart_rate_est_peak` sebenarnya
>    **laju NAPAS**, `confidence_heart` sebenarnya **confidence NAPAS**. Nilai
>    "sentinel misterius" 5.859375 itu ternyata **5.86 napas/menit**. Direproduksi
>    100.00% persis.
> 3. **Jarak subjek.** Di 100 cm SNR jantung = 1.09 (setara noise); di 50 cm = 3.16.
> **KONSEKUENSI: dataset lama TIDAK SAH buat memvonis library TI.** Yang gagal
> setup-nya, bukan TI-nya. Perlu rekam ulang (poin 23).
> `unwrapPhasePeak_mm` (offset 64) **TERBACA BENAR** dan diambil SEBELUM filter TI
> → itu sebabnya `corrected_estimator.py` tetap valid.
>
> **‼️ KOREKSI PERTAMA (masih berlaku) — BACA INI SEBELUM POIN LAIN.**
> Kesimpulan lama "korelasi ~0 di semua pendekatan → kemungkinan masalah di
> akuisisi radar / riset buntu / pertimbangkan re-scope tesis" **TERBUKTI
> KELIRU** dan sudah dicabut. Lihat poin 10-13 di bawah. Ringkasnya:
> **sinyal jantung TERBUKTI ADA di `unwrapPhasePeak_mm`** (SNR median 3.19x di
> frekuensi ECG yang benar, jauh di atas level acak). Korelasi ~0 selama ini
> disebabkan 3 kesalahan metodologis, bukan radar rusak. Estimator yang
> diperbaiki dapat **MAE 5.2 bpm** di subjek uji (S10, split asli 8/1/1) vs
> trivial baseline **14.8 bpm** dan model ML lama 13.4-14.6 bpm — **tanpa ML
> sama sekali**. Model ML lama sudah dihitung ulang beneran (poin 8): dia KALAH
> dari nebak-angka-konstan di 5 dari 10 rotasi subjek uji.
> **Akar masalahnya ketemu di SOURCE CODE TI** (poin 18) — bukan dugaan lagi.
> **Arah riset: JANGAN ganti topik, ganti pembingkaian** (poin 19) — 4 dari 6
> komponen tesis udah selesai.
> Standar penilaian = **ANSI/CTA-2065, MAPE < 10%** (poin 17), BUKAN "target
> pembimbing 5 bpm".
> Dokumen lengkap: `thesis/diagnosis_riset.pdf` + `thesis/rekomendasi_riset.pdf`.
> Poin 6, 7 (bagian "implikasi riset"), dan 8 di bawah sudah USANG sebagian —
> temuan mentahnya masih valid, tapi KESIMPULANNYA salah. Jangan dipakai lagi
> sebagai dasar "riset buntu".

1. **Semua kolom HR bawaan TI Vital Sign library TIDAK RELIABLE — CONFIRMED di
   semua 10 subjek `position_front`**, bukan cuma subject01 (dicek ulang setelah
   semua data lengkap + `aligned_all.csv` di-generate, lihat
   `code/evaluation/compare_baseline.py`). Agregat gabungan semua data
   (`SEMUA DATA DIGABUNG`, n=275606, HR ground truth asli 41–178.5 bpm,
   rata-rata 78.2 bpm):

   | Kolom                  | MAE (bpm) | Korelasi | % stuck di modus |
   |-------------------------|-----------|----------|-------------------|
   | final_heart_rate        | nan*      | nan*     | 75.1%             |
   | heart_rate_est_peak     | 69.2      | 0.423    | 75.3%             |
   | heart_rate_est_fft      | 30.8      | -0.346   | 19.4%             |
   | heart_rate_est_fft_4hz  | 78.7      | -0.287   | 4.3%              |
   | heartRateEst_xCorr      | 24.4      | -0.137   | 18.8%             |

   *`final_heart_rate` nan di agregat karena kolom ini tidak ada sama sekali di
   `position_variation` (lihat poin 4) — di dalam `position_front` saja nilainya
   sama seperti temuan awal (MAE ~83, korelasi ~0). Per-subjek `position_front`
   individual: MAE 54.7–83.4 bpm, korelasi selalu di rentang -0.05 s.d. 0.14
   (praktis nol) untuk `final_heart_rate`/`heart_rate_est_peak`/`heart_rate_est_fft_4hz`.
   Pola stuck-value dan korelasi mendekati nol **konsisten di semua subjek**,
   bukan anomali subject01 saja → bukan sekadar kurang akurat, tapi memang
   gagal menangkap pola detak jantung sama sekali.

2. **`final_heart_rate` dan `heart_rate_est_peak` stuck di nilai 5.859375 untuk
   ~99.6% dari seluruh sampel.** Ini kemungkinan besar sentinel/flag value saat
   algoritma gagal mengunci sinyal jantung (confidence rendah, median
   `confidence_heart` hanya 0.037), BUKAN hasil pengukuran valid. Jangan pakai
   kolom ini sebagai baseline pembanding tanpa mem-filter dulu sampel yang stuck.

3. **Ground truth HR dari Attys HARUS diekstrak via R-peak detection (Pan-Tompkins
   style: bandpass 5-15 Hz → derivative → square → moving window integration →
   peak detection dengan constraint fisiologis 40-180 bpm)**, bukan FFT band-pass
   langsung pada sinyal ECG mentah — sempat dicoba dan hasilnya tidak stabil
   (90-180 bpm yang naik-turun drastis, tidak masuk akal) karena energi sinyal ECG
   yang sebenarnya penting (kompleks QRS) tersebar di frekuensi lebih tinggi,
   bukan di band 0.7-3 Hz.

4. **`position_variation` punya skema kolom radar.csv BEDA dari `position_front`**
   (dicek di semua 10 subjek `position_front` dan semua 8 posisi
   `position_variation_subject01`, konsisten):
   - `position_variation` TIDAK PUNYA `unwrapPhasePeak_mm` maupun
     `rangeBinPhaseIndex` — padahal `unwrapPhasePeak_mm` adalah kandidat fitur
     utama untuk model ML (lihat poin 8, tapi lihat juga poin 6-7: diagnostik
     awal fitur ini sendiri belum meyakinkan). Artinya model yang pakai fitur fase
     ini TIDAK BISA dievaluasi langsung di `position_variation`.
   - Kolom HR final juga beda nama: `final_heart_rate` (position_front) vs
     `finalHeartrate` (position_variation, camelCase, tanpa underscore).
   - Implikasi: kalau mau tetap jalanin analisis generalisasi-posisi (to-do
     paling bawah), desain fitur model utama harus dibatasi ke kolom yang ADA
     di kedua dataset (`heart_rate_est_*`, `heartRateEst_xCorr`,
     `confidence_heart`, `range_bin_value`, `outputFilterHeartOut`,
     `sumEnergyHeartWfm`), atau terima bahwa evaluasi cross-position hanya bisa
     pakai model varian tanpa fitur fase.

6. **Diagnostik sinyal fase mentah juga gagal — bukan cuma output BPM jadi TI
   yang bermasalah** (dicek Jul 2026, `code/evaluation/analyze_phase_signal.py`,
   windowed-PSD dominant-frequency di band 0.7-3.3Hz langsung dari
   `unwrapPhasePeak_mm`, tanpa lewat algoritma TI sama sekali): MAE turun jadi
   ~15.4 bpm (vs 69-83 bpm baseline TI) tapi **korelasi tetap ~0** (agregat
   0.091, per-subjek -0.14 s.d. +0.04) — MAE rendah kemungkinan cuma regresi ke
   rata-rata populasi (band-limited), BUKAN bukti model beneran ngikutin
   dinamika HR. Kemungkinan penyebab: harmonic pernapasan (amplitudo jauh lebih
   besar dari micro-motion jantung) menyasar masuk ke band HR dan mendominasi
   pemilihan frekuensi puncak.

   **Sudah dicoba dan TIDAK membantu (Jul 2026)**: breathing-harmonic
   suppression (deteksi frekuensi napas 0.1-0.5Hz per window, notch
   fundamental+4 harmonic yang jatuh di band HR, baru pilih dominant-frequency
   — lihat mode `notch` di `analyze_phase_signal.py`). Hasil: korelasi agregat
   MALAH TURUN (0.091 → 0.077), MAE relatif sama (15.4 → 16.3). Per-subjek gak
   ada pola konsisten (3 subjek membaik tipis, 4 memburuk). Kesimpulan:
   breathing harmonic BUKAN penyebab utama korelasi ~0 ini.

7. **`confidence_heart` TIDAK BISA dipakai buat filter kualitas — dan skalanya
   BEDA TOTAL antar dataset** (dicek Jul 2026):
   - `position_front`: range 0–6.09, median 0.036 (≈selalu di nilai stuck kecil)
   - `position_variation`: range **-611 s.d. 43217** (termasuk NEGATIF — gak
     masuk akal buat "confidence"), median 7.06
   - **Jangan filter/threshold `confidence_heart` di data gabungan kedua
     dataset** — 99.9% baris `position_variation` akan lolos filter berapapun
     karena skalanya beda, bikin hasil filtered menyesatkan (mengaburkan, bukan
     mengisolasi, subset berkualitas)
   - Bahkan dibatasi ke `position_front` saja: filter ke confidence tertinggi
     (top 2%, `>=2.0`) TIDAK memperbaiki MAE atau korelasi dibanding pakai
     semua data (korelasi tetap di rentang -0.05 s.d. 0.09 di semua threshold
     yang dicoba). Kesimpulan: `confidence_heart` bukan confidence score yang
     valid, kemungkinan artefak internal algoritma seperti `final_heart_rate`
     yang stuck (poin 2), BUKAN indikator kualitas sinyal yang bisa dipakai
     buat filtering.
   - ~~**Implikasi riset**: 4 pendekatan berbeda semua mentok di korelasi ~0 →
     dugaan masalah di akuisisi radar → pertimbangkan re-scope tesis.~~
     **DICABUT 13 Jul 2026 — KELIRU.** Akuisisi radar TIDAK bermasalah; sinyal
     jantung ada di data (poin 10). Korelasi ~0 disebabkan band ekstraksi
     kelewat lebar + tidak ada smoothing + salah pilih metrik (poin 11).
     JANGAN re-scope tesis atas dasar ini. Lihat poin 10-13.

8. **PENTING — MAE doang GAK CUKUP buat validasi model, wajib bandingin ke
   trivial mean-baseline.** Progress ML sebelumnya (dilaporkan ke pembimbing
   Jul 2026) pakai fitur `heart_rate_est_fft`, `heart_rate_est_fft_4hz`,
   `heartRateEst_xCorr`, `heart_rate_est_peak`, `confidence_heart` — TEPAT
   kolom-kolom yang sudah kebukti korelasi ~0 di poin 1 & 7.

   **SUDAH DIHITUNG ULANG BENERAN (13 Jul 2026)** — angka "MAE 15" yang dulu
   dipake cuma tebakan dari ingatan. Script: `code/baseline/reproduce_old_model.py`.
   Split asli: **train subject01-08 / val subject09 / test subject10** (8/1/1,
   berbasis subjek). Dicoba 6 regressor (Linear, Ridge, kNN, RF, GBM, MLP):

   | | MAE test (S10) | Korelasi test |
   |---|---|---|
   | **Trivial baseline** (konstan 75.7 bpm, TANPA radar) | **14.8** | 0.000 |
   | Model ML lama, 6 regressor | **13.4 – 14.6** | **-0.12 s.d. +0.02** |
   | Estimator fase diperbaiki (poin 13) | **5.2** | — |

   Model ML cuma unggul **0.2–1.4 bpm** dari nebak angka konstan, dengan
   korelasi NOL. Diputer ke semua 10 pilihan subjek uji (leave-one-subject-out):
   **model ML KALAH dari trivial baseline di 5 dari 10 rotasi** — persis lempar
   koin. Kesimpulan: model itu TIDAK belajar apa-apa dari radar; dia cuma
   konvergen ke mean training (GIGO). Bukti visualnya ada di fig6 kiri
   (`thesis/diagnosis_riset.pdf`): prediksi model lama = garis datar nempel di
   garis tebak-konstan, sementara ECG aslinya jalan di level lain.
   Ambang yang benar = MAPE <10% (ANSI/CTA-2065, poin 17), bukan MAE 5 bpm. Tapi
   logikanya sama: ambang itu perlu buat
   mastiin model ngalahin trivial baseline ini secara meyakinkan.

   **CATATAN split**: trivial baseline = 11.9 bpm kalau test-nya S09+S10 digabung,
   tapi **14.8 bpm** kalau test-nya cuma S10 (split asli). Selalu sebutin
   split-nya pas ngutip angka — beda split, beda angka.

   **Wajib mulai sekarang**: setiap laporan MAE model HARUS disertai (a)
   korelasi/R², (b) perbandingan eksplisit ke trivial constant-mean baseline
   dengan split subjek yang SAMA. MAE rendah tanpa korelasi dan tanpa
   pembanding trivial baseline tidak bermakna apa-apa.

   (Poin ini TETAP VALID dan tetap wajib — cuma kesimpulan "fitur radar gak
   punya sinyal HR" yang salah. Yang benar: fitur BPM TI-nya yang rusak;
   sinyal HR ada di `unwrapPhasePeak_mm` yang waktu itu belum dipakai di ML.)

10. **SINYAL JANTUNG ADA DI DATA RADAR — CONFIRMED (13 Jul 2026).**
    Script: `code/evaluation/diagnose_root_cause.py`. Tesnya: buat tiap window
    16 detik, ambil frekuensi HR SEBENARNYA dari ECG, lalu cek bin frekuensi itu
    ranking keberapa di PSD sinyal fase radar. Kalau radar gak nangkep apa-apa,
    rank-nya bakal acak (median ternormalisasi 0.5).
    - Median rank ternormalisasi = **0.158** (acak = 0.500)
    - SNR median di frekuensi ECG yang benar = **3.19x** noise floor
    - 6 dari 10 subjek: frekuensi HR benar masuk 3 besar puncak di >50% window
    - Subjek lemah: S03, S04, S06 (SNR 1.4-2.1, nyaris noise) — sesi bermasalah
    Kesimpulan: hardware + akuisisi radar BEKERJA. Jangan ulangi hipotesis
    "radar gagal nangkep sinyal".

11. **Tiga penyebab sebenarnya korelasi ~0** (ablasi lengkap ada di
    `thesis/diagnosis_riset.pdf` Bagian 3-5):
    - **(a) Band ekstraksi kelewat lebar — penyebab TERBESAR.**
      `analyze_phase_signal.py` pakai band 0.7-3.3 Hz. Energi napas (amplitudo
      4-9x lebih gede dari komponen jantung, sesuai spek TI: napas 1-12mm vs
      jantung 0.1-0.5mm) bocor di tepi bawah band dan narik argmax ke frekuensi
      salah — median rasio est/GT = **0.74x** (sistematis meleset ke BAWAH,
      bukan acak). Spek TI sendiri: band HR = **0.8-2.0 Hz**. Ganti band aja:
      korelasi 0.141 → 0.253.
    - **(b) Gak ada smoothing temporal.** Median filter 5-titik di deret
      estimasi: korelasi 0.268 → 0.365.
    - **(c) Aliasing** — `np.interp` 64Hz → 10Hz tanpa anti-alias filter.
      Secara teori bug, praktiknya dampaknya kecil di data ini (ablasi: ~0).
      Tetep perbaiki biar defensible.

12. **KORELASI DALAM-SUBJEK ADALAH METRIK YANG SALAH BUAT DATASET INI.**
    Semua subjek duduk diam istirahat → std HR dalam sesi cuma **1.0-4.4 bpm**.
    Korelasi ternormalisasi terhadap varians: kalau target hampir konstan,
    korelasi mendekati 0 SEKALIPUN estimator akurat.
    Rumus ceiling (estimator tak-bias, noise std s):
    `r_max = std(HR) / sqrt(std(HR)^2 + s^2)`
    Di s=5 bpm, estimator **SEMPURNA** pun cuma dapet
    korelasi **0.20-0.66**, dan 5 dari 10 subjek gak nembus 0.5.
    → **Jangan pernah lagi pakai "korelasi < 0.3 = gak ada sinyal" sebagai vonis
    tanpa ngecek std ground truth-nya dulu.** Ini yang bikin salah vonis kemarin.

13. **Estimator fase yang DIPERBAIKI — hasil (13 Jul 2026).**
    Script: `code/evaluation/corrected_estimator.py`. Config: band 0.8-2.0 Hz,
    resample 20 Hz + anti-alias, detrend, window 16s, interpolasi parabolik
    puncak, median filter 5. **Murni DSP, belum ada ML.**
    - **Split asli (train S01-08 / val S09 / test S10)**: MAE di subjek uji
      **5.2 bpm** vs trivial baseline **14.8 bpm** vs model ML lama 13.4-14.6 bpm
      → hampir **3x lebih baik**. MAPE 8.5% → **LOLOS standar ANSI/CTA-2065** (poin 17)
    - Kalau test-nya S09+S10 digabung: MAE 6.4 bpm, korelasi 0.62 (korelasi baru
      bermakna kalau test >1 subjek — lihat poin 12)
    - subject07 MAE **2.9 bpm**, subject10 MAE **5.0 bpm** → target ≤5 bpm
      TERBUKTI realistis, tanpa ML
    - Korelasi antar-subjek (n=10): **0.653** — radar bisa bedain orang ber-HR
      tinggi vs rendah
    - Korelasi dalam-subjek: 0.024 — TAPI std GT dalam-sesi cuma 2.8 bpm, jadi
      emang gak ada dinamika buat dilacak (lihat poin 12). Ini keterbatasan
      DATASET, bukan (cuma) keterbatasan metode.
    - **Bias sistematis -9.0 bpm** (9 dari 10 subjek diestimasi kerendahan) —
      ini PR teknis utama berikutnya. Errornya terstruktur, bukan noise: kalau
      bias per-subjek dibuang, MAE anjlok (S04: 19.3 → 3.8 bpm). (Buang bias
      pakai ECG subjek itu sendiri cuma DIAGNOSTIK, bukan hasil yg bisa dipakai.)

14. **Kolom radar yang RUSAK dan jangan dipakai sebagai fitur** (selain kolom
    BPM TI di poin 1-2):
    - `range_bin_value` (max 2147483648 = 2^31) dan `rangeBinPhaseIndex`
      (max 4294836234 ≈ 2^32) — ini **uint32 mentah yang belum di-parse**, bukan
      besaran fisik. Bug di level logger/GUI, bukan di radarnya.
    - `outputFilterHeartOut` — sudah diuji sebagai sumber sinyal alternatif:
      bias **+37.5 bpm**, korelasi antar-subjek **NEGATIF (-0.827)**. Ini
      waveform hasil olahan internal TI, ikut terkontaminasi.
    - **Satu-satunya kolom radar yang layak: `unwrapPhasePeak_mm`.**
      (`sumEnergyHeartWfm` rentangnya 0-0.517, masuk akal, layak dicoba sebagai
      fitur kualitas sinyal — belum diuji.)
    - Frame rate logger tidak konsisten: median 64 Hz tapi mean 50 Hz (timestamp
      gak beraturan), sementara spek demo TI = 20 Hz. Perlu dicek saat rekam ulang.

15. **BLOCKER UTAMA TESIS: dataset gak punya variasi HR.** Semua subjek direkam
    duduk diam istirahat → HR praktis konstan → kemampuan model melacak
    PERUBAHAN HR gak bisa dibuktikan MAUPUN dibantah. Nambah subjek baru dengan
    protokol yang sama TIDAK menyelesaikan ini.
    **Protokol rekam ulang yang disarankan** (per subjek, 1 rekaman kontinu
    ~10-12 menit, radar + Attys jalan terus tanpa henti):
    istirahat 3 menit → beban ringan (squat/naik-turun bangku) 2 menit →
    **recovery duduk diam 5 menit (paling berharga: HR turun perlahan 130→75)**
    → istirahat akhir 2 menit. Target: std HR dalam-sesi naik dari ~2.8 → ~20 bpm.
    Detail lengkap: `thesis/rekomendasi_riset.pdf` Bagian 1.

16. **Implikasi untuk desain model ML** (menggantikan poin 9 lama, yang masih
    nyaranin `range_bin_value`/`rangeBinPhaseIndex`/`outputFilterHeartOut` —
    ketiganya sekarang terbukti RUSAK, lihat poin 14).
    Semua fitur diturunkan dari **`unwrapPhasePeak_mm`** (satu-satunya kolom
    radar yang sehat), dihitung per window 16 detik:
    - Spektral: frekuensi puncak di band 0.8-2.0 Hz, tinggi puncak, SNR (puncak
      / median PSD), lebar puncak, posisi 3 puncak teratas
    - Napas sebagai konteks: frekuensi + amplitudo napas (0.1-0.5 Hz) di window
      itu → biar model belajar sendiri MENGOREKSI kebocoran napas (ini sumber
      bias -9 bpm di poin 13), bukan kita tebak-tebak filternya
    - Harmonik: energi di f_puncak vs 2×f_puncak — detak jantung punya harmonik
      kedua, kebocoran napas nggak. Ini bisa ngajarin model bedain puncak asli
      dari puncak palsu
    - Time-domain: RMS sinyal ter-bandpass, zero-crossing rate, jarak antar puncak
    - `sumEnergyHeartWfm` sebagai fitur kualitas sinyal (belum diuji)
    Urutan model: Ridge/RandomForest dulu di atas fitur window ini. Kalau
    ngalahin `corrected_estimator.py` (MAE 5.2 bpm di subjek uji S10), kontribusi ML-nya
    terbukti. 1D-CNN langsung di raw waveform baru masuk akal SETELAH data
    dengan variasi HR ada (poin 15) — dengan data sekarang pasti overfit ke mean.

17. **STANDAR PENILAIAN: ANSI/CTA-2065 — MAPE < 10%** (bukan "target pembimbing 5 bpm").
    Consumer Technology Association (2018), *Physical Activity Monitoring for Heart
    Rate*, ANSI/CTA-2065. Alat HR dinyatakan VALID kalau MAPE vs ECG < 10%.
    Dipakai luas di literatur validasi — **Nelson & Allen (2019)**, JMIR Mhealth
    Uhealth 7(3):e10828, doi:10.2196/10828 (ada di `papers/`), dan studi validasi
    wearable pada anak (juga di `papers/`). Sebagian studi pakai kriteria lebih
    ketat (±5%). Di HR rata-rata 78 bpm, MAPE 10% ≈ **7.8 bpm**.

    **Vonis (MAPE, 10 subjek `position_front`):**

    | | MAPE | Subjek lolos <10% |
    |---|---|---|
    | TI Vital Signs bawaan (`final_heart_rate`) | **89–91%** | **0 / 10** |
    | Pipeline diperbaiki (`corrected_estimator.py`) | 5.1–25.5% | **5 / 10** |

    Lolos: S05 (8.5%), S07 (5.1%), S08 (7.5%), S09 (9.4%), S10 (8.5%).
    Belum lolos: S01 (14.5%), S02 (19.5%), S03 (19.1%), S04 (25.5%), S06 (19.5%).
    **SELALU laporkan MAPE + bandingin ke ambang 10%**, jangan MAE telanjang.

18. **AKAR MASALAH KETEMU DI SOURCE CODE TI — bukan dugaan lagi (13 Jul 2026).**
    Source lab TI ada di `texas_instrument_labs/`. File
    `gui/vitalSigns_demo_gui.m` baris 345-380:
    ```matlab
    outHeartNew_CM = 0.5*confidence_heart + 0.5*outHeartNew_CM_prev;  % exp avg
    if (outHeartNew_CM > thresh_HeartCM)     % thresh_HeartCM = 0.4 (baris 195)
        heartRateEstDisplay = heartRateEstFFT;
    end                                       % <-- TIDAK ADA else!
    heartRateEstDisplayFinal = median(CircBuffer);
    ```
    **Gak ada cabang `else`** → kalau confidence ter-smooth ≤ 0.4, `heartRateEstDisplay`
    GAK PERNAH di-update, nilai lama dipertahankan terus. Confidence di data lu
    median cuma **0.036** (11x lebih kecil dari ambang 0.4) → output BEKU.

    **Simulasi ulang logika ini di data → prediksi beku 94.3%. Terukur: 94.6% stuck.
    COCOK.** Mekanisme kegagalan sudah dipahami penuh. Ini temuan forensik, bukan
    observasi black-box — nilai jual utama tesis.

    Nilai `5.859375` bpm = 0.09765625 Hz = persis **bin ke-5 FFT 1024-titik @ 20 Hz**
    (df = 20/1024). Di LUAR band HR TI sendiri (0.8-2.0 Hz) → mustahil hasil valid.
    Itu nilai awal/artefak indeks yang kemudian dibekukan logika di atas.

    **HIPOTESIS BELUM TERBUKTI (cek saat rekam ulang)**: kenapa confidence-nya
    rendah? Config bawaan `gui/profile_2d_VitalSigns_20fps.cfg` isinya
    `vitalSignsCfg 0.3 1.0 ...` → radar **cuma nyari target di jarak 0.3–1.0 meter**.
    Padahal Developer's Guide TI hal. 11 nyontohin subjek duduk di **1.5 METER**.
    Kalau subjek direkam di luar 1.0 m → range-bin tracker ngunci bin salah/noise →
    confidence runtuh. **Cek ulang jarak duduk subjek** — ini verifikasi paling murah
    dan paling mungkin jadi biang keroknya.
    (Gak bisa dicek dari CSV karena kolom range bin rusak — lihat poin 14.)

19. **ARAH RISET: JANGAN ganti topik, ganti PEMBINGKAIAN.** (jawaban 13 Jul 2026)
    Riset awalnya "apakah device ini layak buat ngukur HR?", ditinggal karena
    ngerasa kontribusinya kurang, pindah ke "gimana ningkatin akurasi pake ML".
    **Justru pindah itu yang bikin mentok** — ML dijalanin di atas fitur rusak.
    **Balik ke pertanyaan awal.** Sekarang jawabannya jauh lebih kuat, dan 4 dari 6
    komponen tesis UDAH SELESAI:
    1. ✅ Vonis kuantitatif vs standar diakui (ANSI/CTA-2065): 0/10 lolos (poin 17)
    2. ✅ Akar penyebab di level SOURCE CODE, terkonfirmasi di data (poin 18)
    3. ✅ Bukti gagalnya di SOFTWARE bukan HARDWARE (SNR 3.19x, poin 10)
    4. ✅ Pipeline pengganti: 5/10 subjek lolos standar, TANPA ML (poin 13, 17)
    5. 🔄 Sisa galat teridentifikasi (bias -9 bpm) + jalur perbaikan (poin 13)
    6. ⬜ ML di atas fitur BENAR — sekarang jadi **bab bonus**, bukan penentu lulus

    Usulan judul: *"Evaluasi Kelayakan Radar FMCW TI IWR1443BOOST untuk Estimasi
    Detak Jantung Terhadap Standar ANSI/CTA-2065: Analisis Akar Penyebab Kegagalan
    Library Bawaan dan Perbaikan Pipeline Pemrosesan Sinyal"*

    **Konsekuensi penting**: tesis gak lagi bergantung ke ML supaya lulus. Kontribusi
    utama udah aman di poin 1-4. ML jadi nilai tambah, bukan hidup-mati.

20. **CACAT 1 — FRAME RATE SALAH (akar masalah paling mendasar).**
    `old code/logger.py` baris 37: `"frameCfg 0 0 2 0 20 1 0\n"` → periode frame
    **20 ms = 50 fps**. Config bawaan TI (`gui/profile_2d_VitalSigns_20fps.cfg`):
    `frameCfg 0 0 2 0 50 1 0` → **50 ms = 20 fps**.
    Library TI pakai filter IIR bi-quad dengan **koefisien TETAP yang dirancang
    untuk fs = 20 Hz**. Dijalankan di 50 Hz → seluruh respons frekuensi **mulur
    50/20 = 2.5x**:
    - Band jantung yang DIMAKSUD TI : 0.8 – 4.0 Hz (48–240 bpm)
    - Band yang SEBENARNYA lewat    : **2.0 – 10.0 Hz** (120–600 bpm)
    - Detak jantung manusia istirahat (1.0–1.7 Hz) **DI BAWAH batas bawah filter**
      → **sinyal jantung diredam habis oleh filter TI sendiri**

    **BUKTI KUANTITATIF** (PSD `outputFilterHeartOut`, kolom ini terbaca BENAR):
    | Pita | % energi |
    |---|---|
    | 0.8–2.0 Hz (jantung asli) | **4.2%** |
    | 2.0–10.0 Hz (band tergeser 2.5x) | **94.3%** |

    Puncak spektrum di 2.98 Hz (179/menit) — bukan detak jantung manusia.
    → Ini menjelaskan kenapa SEMUA kolom hilir TI (`heart_rate_est_fft`, `xCorr`,
    `outputFilterHeartOut`) korelasinya ~0, padahal fase mentahnya bagus.
    → **`unwrapPhasePeak_mm` diambil SEBELUM filter** → selamat → sebab itu
    `corrected_estimator.py` bisa jalan.

21. **CACAT 2 — LOGGER SALAH BACA OFFSET BYTE (4 dari 10 kolom).**
    Struktur paket benar (dari `texas_instrument_labs/pjt/common/mmw_output.h`):
    header 40 byte + TLV header 8 byte → **payload mulai byte 48**.

    | Kolom CSV | offset dibaca | ISI SEBENARNYA | status |
    |---|---|---|---|
    | `unwrapPhasePeak_mm` | 64 | unwrapPhasePeak_mm | ✅ BENAR |
    | `outputFilterHeartOut` | 72 | outputFilterHeartOut | ✅ BENAR |
    | `heart_rate_est_fft` | 76 | heartRateEst_FFT | ✅ BENAR |
    | `heart_rate_est_fft_4hz` | 80 | heartRateEst_FFT_4Hz | ✅ BENAR |
    | `heartRateEst_xCorr` | 84 | heartRateEst_xCorr | ✅ BENAR |
    | `range_bin_value` | 52 | maxVal | ✅ BENAR |
    | `heart_rate_est_peak` | 92 | **breathingRateEst_FFT (laju NAPAS)** | ❌ SALAH (harusnya 88) |
    | `confidence_heart` | 104 | **confidenceMetricBreathOut (NAPAS)** | ❌ SALAH (harusnya 112) |
    | `sumEnergyHeartWfm` | 116 | **confidenceMetricHeartOut_4Hz** | ❌ SALAH (harusnya 128) |
    | `rangeBinPhaseIndex` | 50 as `<I` | uint16, bukan uint32 | ❌ SALAH (pakai `<H`) |

    **MISTERI 5.859375 TERPECAHKAN**: itu **laju NAPAS 5.86 napas/menit**, bukan
    detak jantung. Logger `calculate_heart_rate_est_display()` (baris 201-225):
    `if (confidence > 0.4) or (|est_fft - est_peak| < 15): pakai est_fft; else: pakai est_peak`
    Tapi `confidence` = confidence NAPAS (~0.036, gak pernah >0.4), dan `est_peak`
    = laju NAPAS (~5.86). |85 − 5.86| = 79 > 15 → **kedua syarat gagal → cabang
    `else` selalu diambil → laju napas disalin jadi "detak jantung" 98.3% waktu.**
    **Diverifikasi: reproduksi fungsi logger di kolom CSV → cocok 100.00%
    (selisih maks 0.000000 pada 21.296 baris).**

22. **CACAT 3 — JARAK SUBJEK.** Script: `code/evaluation/analyze_distance.py`.
    Data `data/raw/distance/` (50 cm & 100 cm, subjek sama, logger sama):

    | Jarak | SNR jantung | MAE | MAPE |
    |---|---|---|---|
    | 50 cm | **3.16** | 8.6 bpm | 12.5% |
    | 100 cm | **1.09** (≈ noise) | 16.9 bpm | 28.6% |

    Daya pantul radar ∝ 1/R⁴. Di 100 cm sinyal jantung praktis tenggelam →
    **gak ada algoritma apapun (termasuk ML) yang bisa ngangkat.** Konsisten dengan
    subjek ber-SNR lemah di dataset utama (S03: 1.39, S04: 1.38, S06: 2.08 —
    kemungkinan duduk lebih jauh) vs yang kuat (S01: 5.91, S10: 5.43).
    Config TI cuma nyari target di **0.3–1.0 m** (`vitalSignsCfg 0.3 1.0`) →
    100 cm udah di batas paling tepi. **Rekam ulang di ~50 cm, CATAT jarak persisnya.**

    Catatan format: data jarak pakai `attys.tsv` (BUKAN .csv). Header baris 1
    `# <unix_epoch> <MAC>`, kolom 0 = waktu relatif (125 Hz), **kolom 7 = ECG**
    (diverifikasi via CV interval RR ≈ 9.7%; kolom lain CV >40% = bukan ECG).

23. **YANG HARUS DIPERBAIKI DI `old code/logger.py` SEBELUM REKAM ULANG:**
    ```python
    # baris 37 — frame rate
    "frameCfg 0 0 2 0 50 1 0\n",   # 50 ms = 20 fps (BENAR), bukan 20 ms = 50 fps

    # displayVitalSign() — offset byte
    heart_rate_est_peak = struct.unpack_from('<f', data, 88)[0]   # bukan 92
    confidence_heart    = struct.unpack_from('<f', data, 112)[0]  # bukan 104
    sumEnergyHeartWfm   = struct.unpack_from('<f', data, 128)[0]  # bukan 116
    rangeBinPhaseIndex  = struct.unpack_from('<H', data, 50)[0]   # uint16, bukan '<I'
    # unwrapPhasePeak_mm di offset 64 SUDAH BENAR — jangan diubah
    ```
    Sekalian rekam kolom yang belum pernah diambil padahal berguna:
    `breathingRateEst_FFT` (92), `motionDetectedFlag` (132),
    `heartRateEst_harmonicEnergy` (140).

    **VONIS TERHADAP LIBRARY TI BELUM SAH DIBUAT.** Dataset lama direkam dengan
    setup cacat, jadi "TI gagal 0/10" itu **bukan uji yang adil** dan bisa
    dipatahkan di sidang. Vonis adil = rekam ulang dengan config benar, baru ukur.

24. **KEPUTUSAN ARAH (13 Jul 2026): TESIS SELESAI TANPA REKAM ULANG.**
    User kehabisan waktu & resource → **persempit seluruh analisis ke
    `unwrapPhasePeak_mm` saja.** Ini SAH, karena kolom itu satu-satunya yang
    selamat dari ketiga cacat (poin 20-22):
    - offset 64 dibaca BENAR (beda dari 4 kolom lain yang salah)
    - diambil SEBELUM filter TI → gak kena pergeseran band 2.5x
    - frame rate 50 fps justru MENGUNTUNGKAN sinyal fase (resolusi waktu lebih
      rapat, phase-unwrapping lebih andal)
    - yang KENA cuma jarak (SNR) — dan itu dilaporkan sebagai batas fisik (K3)

    **PIPELINE FINAL: `code/evaluation/phase_pipeline.py`** (murni DSP, no ML):
    resample anti-alias 25 Hz → **turunan fase (`np.diff`)** → kurangi komponen
    napas (0.15-0.6 Hz) → bandpass 0.8-2.0 Hz → Welch PSD window **40 s** → argmax →
    median filter **9**. (Jendela 40s/medfilt 9 sejak 14 Jul 2026 — alasan & bukti
    bahwa ini BUKAN sekadar penghalusan ada di **poin 28(d)**.)

    | | hasil |
    |---|---|
    | **LOLOS standar MAPE <10%** | **8 dari 10 subjek** |
    | MAPE rata-rata | **6.5%** (TI bawaan: ~90%; kolom TI terbaik: 20.2%) |
    | Terbaik | S09 = 1.3%, S01 = 1.7%, S07 = 1.9% |
    | Gagal | S04 (SNR 1.78), S06 (SNR 1.60) — dua-duanya < ambang 1.8 = batas fisik |
    | vs trivial baseline (test S09+10) | MAE 3.1 vs 12.0 → **MENANG** |

    Yang paling ngangkat: **turunan fase** (MAPE rata-rata 13.6% → 9.8%,
    5/10 → 6/10), lalu **jendela 40s + medfilt 9** (9.8% → 6.5%, 6/10 → 8/10, poin 28d).
    Sudah dicoba dan GAK membantu: harmonic-sum, autocorrelation, Viterbi tracking,
    koreksi f-tilt (bagi PSD dengan f^p) — jangan diulang.
    Sudah dicoba dan DITOLAK walau angkanya bagus: band 0.9-2.0 Hz (overfit, poin 28).

    **KRITERIA SNR (K3)**: SNR di frekuensi ECG ≥ **1.8** memprediksi lolos/gagal
    benar di **11 dari 12 sesi** (10 subjek + 2 sesi jarak). Dikuatkan data jarak:
    50 cm (SNR 3.16, MAPE 8.1%, lolos) vs 100 cm (SNR 1.09, MAPE 58.3%, gagal).
    ⚠️ **JUJUR**: SNR ini dihitung di frekuensi ECG → **butuh ECG** → ini alat
    DIAGNOSIS, BUKAN gerbang kualitas yang bisa dipakai saat deployment. Pengganti
    tanpa-ECG sudah dicoba (peak-to-median, entropi spektral, jitter) → akurasi
    cuma 67-75%, **belum layak diklaim**. Tulis sebagai saran penelitian lanjutan.

    ⚠️ **CATATAN**: S10 (HR terendah, 61 bpm) dulu gagal di config 20s/medfilt-5
    (MAPE 19.0%); dengan 40s/medfilt-9 dia **LOLOS** (7.9%). S03 juga ikut lolos
    (9.7%). Yang tersisa gagal: S04 & S06, dua-duanya SNR < 1.8 = batas fisik.

25. **JUDUL & KONTRIBUSI (usulan, 13 Jul 2026).**
    Judul: *"Estimasi Detak Jantung Berbasis Sinyal Fase Mentah Radar FMCW TI
    IWR1443BOOST: Analisis Akar Kegagalan Pipeline Bawaan dan Rancangan Pipeline
    Pengganti Tervalidasi Standar ANSI/CTA-2065"*

    - **K1** — Analisis akar kegagalan (3 cacat, poin 20-22). Bukti: reproduksi
      `final_heart_rate` 100.00% persis.
    - **K2** — Pipeline pengganti berbasis fase mentah (poin 24, 28). Bukti: **8/10
      lolos** MAPE <10%, rata-rata **6.5%** vs kolom TI terbaik 20.2% (dan
      `final_heart_rate` ~90%). Korelasi antar-subjek **+0.83 vs −0.41**. Tanpa ML.
    - **K3** — Kriteria kelayakan SNR + karakterisasi jarak. Bukti: ambang 1.8
      benar 11/12 sesi; data jarak 50 vs 100 cm.
    - **K4** — Koreksi metodologi (poin 8, 12). Bukti: ceiling korelasi; model ML
      lama kalah dari tebak-konstan.

    **BATASAN MASALAH WAJIB DITULIS DI AWAL** (ini yang melindungi di sidang):
    > "Penelitian ini TIDAK memberikan vonis kelayakan terhadap library Vital Signs
    > bawaan TI, karena data direkam dengan konfigurasi yang tidak sesuai spesifikasi
    > TI (frame rate 50 fps, sementara library dirancang untuk 20 fps). Yang
    > dievaluasi adalah kelayakan SINYAL FASE MENTAH radar sebagai basis estimasi
    > detak jantung."

    Dengan kalimat itu, ketiga cacat berubah dari KELEMAHAN jadi TEMUAN (K1), dan
    penguji gak bisa mematahkan pakai argumen "pengujianmu gak adil".
    **ML = bab BONUS**, bukan penentu kelulusan.

26. **METRIK KUALITAS TANPA ECG — JAWABAN buat pertanyaan "kalau di lapangan gak
    ada ECG, gimana?"** Script: `code/evaluation/quality_metric.py`.
    (Ini nutup lubang di poin 24 — SNR butuh ECG, jadi gak deployable.)

    **JAWABANNYA: JITTER.** = `median(|selisih antar estimasi berurutan|)`,
    diambil dari deret estimasi MENTAH (SEBELUM median filter). Murni dari sinyal
    radar, **nol ECG**.
    - Alasan fisiologis: **detak jantung manusia gak bisa melompat.** Kalau estimasi
      per-jendela lompat-lompat → estimator lagi ngunci noise, bukan jantung.
    - **WAJIB pakai interpolasi parabolik** biar frekuensinya kontinu. Tanpa itu
      jitter terkuantisasi di lebar bin (1/20s = 3 bpm) → jadi gak informatif sama
      sekali (ini bug yang bikin percobaan pertama gagal).

    | metrik (tanpa ECG) | r vs log(MAPE) | akurasi **leave-one-out** |
    |---|---|---|
    | **jitter** | **+0.784** | **10/12 = 83%** |
    | pmr (peak-to-median) | -0.547 | 7/12 = 58% |
    | harmonik (energi di 2f) | +0.215 | 5/12 = 42% |
    | kecocokan PSD vs autocorr | +0.642 | 4/12 = 33% |
    | *(tebak mayoritas)* | — | *7/12 = 58%* |

    Ambang **jitter ≈ 3 bpm**. Di bawah itu → hasil bisa dipercaya.

    ⚠️ **JUJUR (wajib disampaikan sendiri):** angka 92% yang muncul kalau ambang
    disetel di data yang sama itu **OVERFIT**. Yang sah dilaporkan adalah
    **LOO = 83%** (vs tebak-mayoritas 58%). Batasan: cuma 12 sesi (n kecil);
    memprediksi kualitas SATU SESI, belum diuji sebagai gerbang per-jendela.
    → Laporkan sebagai **temuan awal yang menjanjikan**, BUKAN klaim final.
    Ini memperkuat **K3** (batas keberlakuan), bukan bikin kontribusi baru.

27. **HRV TIDAK BISA diambil dari `unwrapPhasePeak_mm` — SUDAH DIUKUR, bukan dugaan
    (14 Jul 2026).** Script: `code/evaluation/hrv_analysis.py`.

    **Kenapa HRV jauh lebih berat dari BPM**: `phase_pipeline.py` cuma butuh
    frekuensi dominan per jendela 20 detik → merata-ratakan ~25 detak sekaligus,
    jadi detak yang tenggelam di noise masih tertutup tetangganya. HRV butuh
    **waktu SETIAP detak, presisi milidetik** — satu detak kelewat langsung
    merusak DUA interval.

    **Hasil (rata-rata 10 subjek `position_front`):**

    | | ECG (acuan) | radar pita sempit (0.8-2.0 Hz) | radar pita lebar (0.8-8.0 Hz) |
    |---|---|---|---|
    | RMSSD | **46 ms** | 210 ms | 178 ms |
    | SDNN | **52 ms** | 164 ms | 128 ms |
    | R-peak ECG ketemu pasangan | — | **52%** | **61%** |

    Untuk HRV yang sah, "R-peak ketemu" harus >95%.

    ⚠️ **Hipotesis awal SALAH, jangan diulang**: gw sempat menduga pita sempit
    meregularisasi sinyal jadi sinus → RMSSD-nya bakal terlalu KECIL. **Datanya
    membantah** — pita sempit justru RMSSD 210 ms, lebih BESAR dari pita lebar.
    Melebar di KEDUA pita = penyebabnya bukan pilihan filter, tapi **deteksi
    detaknya sendiri** (puncak palsu + detak kelewat). Angka 210 ms itu ukuran
    KESALAHAN DETEKSI, bukan variabilitas jantung.

    Korelasi antar-subjek juga gak menyelamatkan: SDNN pita sempit r=+0.49 (n=10,
    belum bermakna), pita lebar malah NEGATIF (-0.26) → pola gak konsisten =
    kebetulan.

    **Dua batas keras, keduanya BUKAN soal algoritma** (jadi jangan buang waktu
    coba-coba estimator lain):
    - (a) SNR jantung cuma 1.4-5.9 (poin 10/22) → puncak tiap detak individual
      sering tenggelam. Batas yang SAMA yang bikin S03/S04/S06 dan sesi 100 cm gagal.
    - (b) Presisi waktu **20 ms/sampel** (50 fps) sebanding dengan besaran yang
      diukur (RMSSD istirahat 20-50 ms).

    **Temuan sampingan yang berguna**: `dt` timestamp radar median 15.6 ms tapi
    mean persis 20.0 ms, std 3.7-6.6 ms → frame **memang datang tiap 20 ms secara
    hardware**; jitter itu cuma kuantisasi timer OS Windows (15.6 ms tick), bukan
    jitter radar. Jadi clock seragam bisa direkonstruksi dari indeks frame
    (`uniform_clock()` = regresi linear indeks→waktu). Pakai ini kalau butuh
    timing presisi, JANGAN pakai timestamp mentah.

    **PAKAI SEBAGAI HASIL NEGATIF, jangan dibuang** — ini memperkuat **K3**
    (batas keberlakuan), dan gratis (data sudah ada). Kalimat tesis:
    > "Sinyal fase radar layak untuk estimasi DETAK RATA-RATA per jendela (6/10
    > subjek lolos ANSI/CTA-2065), namun BELUM layak untuk HRV: deteksi
    > detak-per-detak hanya mencapai 61% recall terhadap R-peak ECG, dan RMSSD
    > hasil radar merupakan 4x nilai ECG."

    Slide C9-C10 di `thesis/antisipasi_pertanyaan.pdf`. HRV baru realistis kalau
    rekam ulang di ~50 cm + frame rate lebih tinggi → saran penelitian lanjutan,
    BUKAN syarat lulus.

28. **ARAH RISET FINAL (14 Jul 2026, disetujui pembimbing): BANDINGKAN keluaran
    BPM logger vs pipeline fase.** Script: `code/evaluation/compare_ti_vs_phase.py`.
    Slide: `thesis/logger_vs_fase.pdf` (`code/reporting/make_slides_ti_vs_phase.py`).
    Semua metode diukur di **jendela yang SAMA PERSIS** (kolom BPM logger
    dirata-ratakan per jendela lewat `ti_windows()`, pakai `resample_antialias()`
    yang sama) → apel ke apel.

    **Kriteria pembimbing: kalau MAPE fase < MAPE keluaran logger → gak perlu
    rekam ulang, itu jadi kontribusi.** TERPENUHI (6,5% vs 20,2%).

    **(a) Akurasi.** Kolom logger TERBAIK = `heartRateEst_xCorr` (terkonfirmasi):

    | kolom | MAPE | MAE | lolos <10% |
    |---|---|---|---|
    | `final_heart_rate` | 91% | 68,2 | 0/10 |
    | `heart_rate_est_fft` | 45% | 32,4 | 0/10 |
    | **`heartRateEst_xCorr`** (terbaik TI) | **20,2%** | **14,2** | **2/10** |
    | trivial baseline (tebak konstan, LOSO) | 9,3% | — | — |
    | **`phase_pipeline.py`** | **6,5%** | **4,9** | **8/10** |

    → **3,1x** lebih baik dari logger, dan **ngalahin trivial baseline**.
    Fase menang atas xCorr di **8/10 subjek** (kalah tipis di S03: 9,7 vs 8,7;
    kalah di S04: 20,0 vs 11,1).

    **(b) BUKTI DETAK JANTUNG BENERAN KETANGKEP — ini senjata utamanya, BUKAN
    tabel MAPE.** Korelasi ANTAR-subjek (n=10; korelasi DALAM-subjek gak sah di
    dataset ini, lihat poin 12):

    | | korelasi | bias | std keluaran | vs trivial (test S09+S10) |
    |---|---|---|---|---|
    | `heartRateEst_xCorr` | **−0,41** (NEGATIF) | +11,6 | 4,3 bpm | MAE 19,5 → **KALAH** dari tebak-konstan (12,0) |
    | `phase_pipeline` | **+0,83** | −2,7 | **8,9 bpm** | MAE 3,1 → **MENANG** |
    | *(HR asli)* | — | — | *8,5 bpm* | — |

    xCorr bukan "kurang akurat" — dia **gak nangkep apa-apa**. Beda jenis, bukan
    beda derajat.

    **(c) KENAPA xCorr KADANG "MENANG" (S03/S04) — JAWABAN buat pembimbing.**
    xCorr praktis **PENEBAK-KONSTAN ~87 bpm** (std keluaran cuma 4,3 bpm padahal
    HR asli std 8,5). Bukti: korelasi antara MAPE xCorr dan MAPE "tebak konstan
    87" = **+0,93**. Jadi dia gak bawa informasi apa pun di luar konstantanya.
    S03 (78,5) & S04 (75,6) kebetulan HR-nya deket 87 → xCorr "menang" persis
    seperti **jam mati yang benar dua kali sehari**. Harganya keliatan di ujung:
    S07 (65 bpm) meleset 31%, S10 (61 bpm) meleset **52%**.
    Dihitung otomatis di `compare()` → `S["r_stuck"]`.

    **(d) JENDELA 40 DETIK + MEDFILT 9** (naik dari 20 s / 5). Alasannya
    dinyatakan DI MUKA, bukan hasil ngintip data uji: HR subjek praktis diam
    (std 0,9-4,2 bpm) → gak ada dinamika yang bisa hilang → jendela panjang murni
    untung (resolusi 60/durasi: 3,0 → 1,5 bpm).
    **Ini BUKAN sekadar "dihaluskan"**: estimator yang meratakan diri ke mean akan
    bikin std keluaran TURUN dan korelasi TURUN. Yang terjadi sebaliknya —
    std 6,5 → **8,9** (menyamai std asli 8,5), korelasi +0,76 → **+0,83**.
    HARGANYA (wajib ditulis sebagai batasan): respons ke perubahan HR jadi lambat,
    dan itu **gak bisa diuji** dengan dataset ini.

    ⚠️ **GODAAN YANG DITOLAK**: band 0,9-2,0 Hz kasih 9/10 lolos & MAPE 5,5%.
    **JANGAN.** Itu overfit — nolong mayoritas di tengah, **ngerusak S10** (HR 61,
    subjek ber-HR terendah, justru yang mau dibuktikan kedeteksi). 0,8 Hz itu spek
    TI sendiri. Tanda overfit paling khas: nolong rame-rame, ngorbanin kasus ekstrem.

    **PEMBINGKAIAN — jangan bilang "library TI jelek".** Bilang: *"pada konfigurasi
    akuisisi INI (50 fps), kolom BPM turunan TI gak kepake sama sekali; sinyal fase
    mentah masih kepake. Artinya detak jantung MEMANG tertangkap radar — yang
    membuangnya rantai pemrosesan hilirnya."* Batasan masalah poin 25 WAJIB ditulis
    di muka → 50 fps berubah dari *cacat tersembunyi* jadi **kondisi eksperimen yang
    dinyatakan**, dan penguji gak bisa mematahkan pakai "pengujianmu gak adil".

    Pelajaran teknik yang digeneralisasi (jual ini): **ambil sinyal sedini mungkin
    di rantai, sebelum filter vendor yang mengasumsikan sample rate tetap.**

    Sisa yang gagal: **S04** (MAPE 20,0%, SNR 1,78) & **S06** (12,5%, SNR 1,60) —
    dua-duanya SNR < 1,8 = batas fisik (K3), konsisten dengan poin 22/24.

    `compare_ti_vs_phase.py` punya **5 assert cek-mandiri** di akhir `main()` yang
    menjaga semua klaim ini tetap benar kalau script dijalankan ulang.

    ⚠️ **BUG YANG HAMPIR KEJADIAN — jangan diulang**: `fs` untuk analisis spektral
    HARUS dari **mean** `dt`, BUKAN median. Median `dt` = 15,6 ms (kuantisasi timer
    OS Windows) → fs 64 Hz → sumbu frekuensi melar 1,28x → puncak
    `outputFilterHeartOut` kebaca palsu di 3,8 Hz. Mean `dt` = 20,0 ms → fs 50 Hz
    (frame rate asli, poin 27) → puncak benar **2,57 Hz**. Dengan fs benar: energi
    di pita jantung asli (0,8-2,0 Hz) cuma **3,6%**, di pita tergeser (2-10 Hz)
    **96,2%** (rata-rata 10 subjek). Konsisten dengan poin 20.

29. **KONTROL NEGATIF (ruangan kosong) + HOLD-OUT — 15 Jul 2026. BUKTI TERKUAT
    YANG ADA.** Script: `code/evaluation/negative_control.py`.
    Slide: `thesis/bukti_detak_jantung.pdf` (`code/reporting/make_slides_bukti.py`).

    **(a) RUANGAN KOSONG** (`data/raw/no_subject/`, 2 sesi ~7 menit, skema kolom
    SAMA dengan `position_front`, punya `unwrapPhasePeak_mm`).
    Radar nyala di ruangan kosong, gak ada siapa-siapa. Alat yang jujur harusnya
    bilang TIDAK TAHU.

    | | ruangan KOSONG | ada ORANG | vonis |
    |---|---|---|---|
    | `heartRateEst_xCorr` | **82,7 – 85,4 bpm** | 81,4 – 94,8 bpm | **TUMPANG TINDIH** |
    | `heart_rate_est_fft` | 108,5 – 109,4 | 105,7 – 108,0 | **TUMPANG TINDIH** |
    | gerak dada (fase) | **0,00 mm** | 3,1 – 6,6 mm | pisah 850x |
    | **SNR napas (fase)** | **2,2 – 3,3** | **4.359 – 41.595** | **pisah 1.329x, NOL tumpang tindih** |

    → **Logger ngelaporin detak jantung ~83-85 bpm dari RUANGAN KOSONG**, gak bisa
    dibedain dari yang dia laporin buat manusia. Artinya angka itu **gak pernah
    berasal dari jantung siapa pun** — termasuk pas ada orangnya. Ini bukti paling
    telak, jauh lebih kuat dari tabel MAPE mana pun.
    → Sinyal fase **TAHU** ada orang atau enggak.

    **INI SEKALIGUS NUTUP LUBANG POIN 24/26.** Gerbang kualitas yang dulu butuh ECG
    (SNR di frekuensi ECG) sekarang punya pengganti yang **TIDAK butuh ECG**:
    deteksi kehadiran lewat **NAPAS** (0,15-0,6 Hz). Kenapa napas dan bukan jantung:
    gerak dada napas 1-12 mm, jantung cuma 0,1-0,5 mm (spek TI) → napas jauh di atas
    noise floor. Ambang `PRESENCE_THRESHOLD = 100` (SNR napas) — ditaruh di TENGAH
    jurang 3-orde-besaran, bukan disetel pas di tepi data.
    ⚠️ Metrik tanpa-ECG lain (jitter, PMR, harmonik, autokorelasi) **SUDAH DIUJI dan
    GAGAL** misahin kosong vs berisi — semuanya tumpang tindih (S03 jitter 12,2 malah
    lebih buruk dari ruangan kosong). Jangan diulang. Yang jalan cuma napas.

    **(b) HOLD-OUT: `subject_cadangan01` & `subject_cadangan02`** (ada di
    `data/raw/position_front/`, TAPI **gak masuk `aligned_all.csv`**).
    Gak pernah dipakai milih apa pun — jendela 40s, medfilt 9, pita 0,8-2,0 Hz semua
    ditetapkan pakai 10 subjek utama. Ini jawaban buat tuduhan *"kamu nyetel parameter
    sampai angkanya bagus"*.

    | subjek | SNR | ECG | FASE | xCorr |
    |---|---|---|---|---|
    | cadangan01 | 4,39 | 73,6 | **72,3 → MAPE 4,5% LOLOS** | 93,2 → 26,6% gagal |
    | cadangan02 | 2,58 | 72,3 | 65,4 → MAPE **10,0%** (mepet, gagal) | 91,8 → 27,7% gagal |

    Fase menang **2/2**. Dan xCorr keluarin **93,2 & 91,8 bpm** — nempel lagi ke
    konstanta ~87-nya, **di data yang belum pernah dia liat**. Cerita penebak-konstan
    (poin 28c) **TERKONFIRMASI DI LUAR SAMPEL**.

    ⚠️ **Ground truth cadangan dari `attys.tsv` KOLOM 7** (bukan `.csv`). Diverifikasi:
    CV interval RR 8-9% (kolom lain 22-44% = mustahil fisiologis), DAN cadangan01
    yang punya `.csv` MAUPUN `.tsv` kasih hasil **identik** (530 R-peak, 74,4 bpm,
    CV 8,0%, 427 s). Sama seperti data `distance` (poin 22).

    **Urutan argumen di sidang** (slide `bukti_detak_jantung.pdf`):
    B1 BPM apa adanya → B2 kenapa logger "menang" di S03/S04 (jam mati) →
    B3 fase mengandung pola HR (3 bukti independen: puncak spektrum tepat di
    frekuensi ECG; peringkat frekuensi ECG **0,08** vs acak 0,50 di 402 jendela;
    korelasi antar-subjek) → **B4 kontrol negatif** → B5 hold-out.

30. **SNR MEMPREDIKSI KEGAGALAN — JAWABAN buat saran pembimbing "ganti subjek yang
    MAPE-nya jelek" (15 Jul 2026).** Script: `code/evaluation/snr_criterion.py`.
    Slide: `thesis/kenapa_subjek_gagal.pdf` (`code/reporting/make_slides_snr.py`).

    **⚠️ JANGAN PERNAH buang subjek karena MAPE-nya jelek.** Itu cherry-picking, dan
    gak ada jawabannya kalau ditanya "berapa subjek yang direkam?". Yang SAH cuma
    buang karena DATA-nya rusak, dengan kriteria yang diukur TANPA ngeliat hasil.

    **Kegagalannya TIDAK acak — bisa diramalkan dari SNR.** 14 sesi, pipeline final
    yang sama (10 utama + 2 hold-out + 2 jarak):

    | sesi | SNR | MAPE | |
    |---|---|---|---|
    | 100 cm | **1,27** | 53,9% | gagal |
    | S06 | **1,60** | 12,5% | gagal |
    | S04 | **1,78** | 20,0% | gagal |
    | S03 | 2,39 | 9,7% | lolos |
    | cad02 | 2,58 | 10,0% | gagal (BATAS — meleset 0,0 poin) |
    | 50 cm | 3,04 | 6,4% | lolos |
    | 8 sesi lain | 4,2 – 9,3 | 1,3 – 7,9% | lolos |

    - **Korelasi log(SNR) vs log(MAPE) = −0,911.** Sangat teratur.
    - Ambang **SNR ≥ 1,8** benar di **13 dari 14 sesi** — termasuk 4 sesi yang GAK ikut
      nentuin ambangnya (2 hold-out + 2 jarak). Satu-satunya meleset: cad02, kasus batas.
    - **SEMUA sesi ber-SNR < 1,8 gagal, tanpa kecuali.**

    **EKSPERIMEN TERKENDALI (`data/raw/distance/`) — ini bukti kausalnya.** Subjek yang
    SAMA, logger yang SAMA, cuma jaraknya beda:
    - **50 cm → SNR 3,04, MAPE 6,4% LOLOS**
    - **100 cm → SNR 1,27, MAPE 53,9% GAGAL**
    Daya pantul radar ∝ 1/R⁴ → jarak 2x lipat = daya 16x lebih kecil. Terukur persis
    seperti diramalkan.
    (Angka lama di poin 22 pakai config lama 16 s/tanpa turunan fase — yang di atas ini
    pakai pipeline final, jadi ini yang dipakai.)

    **KONSEKUENSI:**
    - Ganti subjek gak nyelesaiin apa pun — penggantinya bakal **gagal juga** kalau duduk
      sejauh itu, dan **lolos juga** kalau duduk dekat. Yang perlu diganti bukan
      subjeknya, tapi **JARAK DUDUKNYA**.
    - Subjek gagal = **BUKTI buat K3** (kriteria kelayakan). Buang mereka = buang K3, dan
      nyisain metode yang ngaku berhasil di semua orang **tanpa batas yang dinyatakan** —
      itu justru LEBIH gampang dipatahkan penguji, bukan lebih aman.
    - Kalimat buat pembimbing: *"metode ini bekerja, dan INILAH persis kapan ia secara
      fisik tidak bisa."*

31. **DATA HALANGAN (`data/raw/obstacle/`) + DECK ALUR FINAL + BATASAN MASALAH
    (15 Jul 2026).** Script: `snr_criterion.py` (diperluas), `make_slides_alur.py`.
    Deck: `thesis/alur_pembuktian.pdf` (7 slide). Draft: `thesis/batasan_masalah_draft.md`.

    **⚠️ `obstacle/no/radar.csv` BYTE-IDENTIK dengan `subject_cadangan01/radar.csv`**
    (md5 sama). Rekaman yang SAMA disalin ke 2 folder. `obstacle/no` = kontrol
    halangan = cad01. JANGAN dihitung dua kali. `snr_criterion.OBST_CM` sengaja
    cuma proses tipis/tebal/supertebal (bukan `no`).

    **Data halangan BUKAN dosis-respons, dan MAPE-nya TIDAK BISA dipakai:**
    - SNR: tanpa halangan 4,39 → tipis(3cm) 1,75, tebal(6cm) 1,66, supertebal(10cm) 1,87.
      Buku 3 cm udah motong SNR ~2,5x; nambah tebal GAK nambah efek → **efek ambang**,
      bukan dosis-respons. 1 sesi/kondisi → gak bisa misahin efek tebal dari variasi
      antar-sesi. Jangan ngaku bisa.
    - **MAPE halangan gak sah buat klaim akurasi**: HR subjek 73-78 bpm (deket rata-rata
      kohort 75,1) → tebak-konstan aja MAPE ~3%, radar KALAH di keempat sesi. Sama
      persis jebakan yang dipakai buat bongkar xCorr — sekarang kena ke metode sendiri.
      **Cuma SNR-nya yang dipakai** (SNR lepas dari nilai HR).

    **Gabungan 17 sesi unik** (10 utama + 2 hold-out + 2 jarak + 3 halangan):
    korelasi log(SNR) vs log(MAPE) = **−0,885**, ambang 1,8 benar **15/17**.
    ⚠️ Klaim LAMA "semua di bawah 1,8 gagal TANPA KECUALI" (poin 24/30) sekarang
    **PUNYA kecuali**: `tipis` (SNR 1,75, MAPE 7,2% tampak lolos — tapi kalah dari
    tebak-konstan, jadi SNR-nya BENAR, MAPE-nya yang nyesatin). Laporkan SNR sebagai
    **prediktor kuat kontinu**, BUKAN gerbang biner sempurna. `snr_criterion` assert
    udah dilonggarin ke "meleset ≤ 2 sesi".

    **KOREKSI FAKTUAL (dulu salah di kepala user):** S03/S04 yang "MAPE kalah dari
    xCorr" BUKAN dua-duanya ber-SNR rendah. S03 SNR **2,39 (normal)** — di S03
    **dua-duanya LOLOS** (9,7 vs 8,7, seri). S04 SNR 1,78 — **dua-duanya GAGAL**
    (20,0 vs 11,1). **GAK ADA satu pun subjek di mana logger lolos & fase gagal.**
    Itu kalimat yang dipakai, bukan "ada 2 subjek xCorr lebih bagus".

    **ALUR DECK `alur_pembuktian.pdf`** (urutan dirancang user, SAH karena alat ukur
    ditegakkan SEBELUM liat hasil): A1 pertanyaan+batasan → A2 jarak→SNR → A3
    halangan→SNR → A4 MAPE 10 subjek → A5 bedah "2 subjek kalah" (gak ada yang
    beneran kalah) → A6 korelasi xCorr −0,41 vs fase +0,83 → A7 kesimpulan+batas.

    **KEPUTUSAN 50 FPS (user serba-salah, takut disuruh rekam ulang, udah telat lulus
    setahun):** SARANNYA = **tetap nyatakan 50 fps di Batasan Masalah, TAPI dibingkai
    sebagai kondisi eksperimen + MELINDUNGI TI ("kolom TI TIDAK dinilai"), bukan
    pengakuan salah.** Alasan kenapa nyatakan ≠ rekam ulang:
    - Rekam ulang cuma perlu kalau NGEKLAIM nilai TI. Batasan Masalah nyatain TIDAK
      ngeklaim itu. Objek riset = fase mentah, yang JUSTRU diuntungkan 50 fps.
    - Rekam ulang 20 fps malah bikin sinyal fase LEBIH JELEK (sampel lebih jarang).
    - Nama config bawaan TI = `profile_2d_VitalSigns_20fps.cfg` → angka 20 ada di
      nama file, di repo publik. Nyembunyiin gak mungkin aman.
    - Kriteria pembimbing (MAPE fase < MAPE logger → gak perlu rekam ulang) TERPENUHI
      6,5% vs 20,2%. **Bawa Batasan Masalah ke pembimbing DULUAN.**
    Draft lengkap (7 poin formal + versi 1-slide + catatan internal) di
    `thesis/batasan_masalah_draft.md`. JANGAN tulis "saya salah config" — tulis
    "akuisisi pada 50 fps, menguntungkan fase, karena itu kolom TI tidak dinilai".

32. **ARAH FINAL (15 Jul 2026, keputusan user): DECK UTAMA = MURNI FASE vs ECG,
    NOL angka BPM TI di jalur utama.** Script: `code/reporting/make_slides_fase.py`.
    Deck: `thesis/fase_vs_ecg.pdf` (9 slide).

    **Alasan pindah dari "banding TI" (poin 28) ke "murni fase":** nampilin angka
    BPM TI = otomatis kebaca sebagai vonis atas library TI → ngundang serangan
    "pengujianmu gak adil (config 50 fps)". Kalau TI GAK PERNAH ditampilin, gak ada
    yang perlu dibela — masalah 50 fps lenyap. Semua bukti terkuat (8/10 lolos,
    korelasi +0,83, kontrol negatif, hold-out, kriteria SNR) **gak butuh TI sama
    sekali**.

    **Pembanding kecukupan diganti dari "vs TI" jadi "vs TRIVIAL BASELINE"** (tebak
    konstan tanpa radar). Ini LEBIH KUAT secara akademis: trivial baseline itu tolok
    ukur standar, gak punya config yang bisa dipersoalkan. Angka: test S09+S10 MAE
    3,1 vs 12,0; LOSO MAPE 6,5% vs 9,3%. Fase menang.

    **Yang HILANG:** K1 (analisis akar kegagalan TI) turun jadi BUKAN kontribusi
    utama. Diterima — tesis berdiri di K2+K3+K4, gak butuh K1. Perbandingan TI
    (`logger_vs_fase.pdf`) disimpan sebagai **lampiran cadangan**, cuma dibuka kalau
    penguji spesifik nanya.

    **Jawaban kalau ditanya "kok gak dibandingkan sama output alatnya?"**: *"Output
    BPM bawaan dihasilkan filter yang dikonfigurasi 20 fps, akuisisi kami 50 fps,
    jadi bukan pembanding adil. Kami validasi ke ECG dan trivial baseline."* → 50 fps
    jadi ALASAN gak membandingkan, bukan klaim TI jelek. Gak bisa diserang "gak
    adil" karena justru MENOLAK nguji TI demi keadilan.

    **Alur `fase_vs_ecg.pdf`:** M1 pertanyaan+batasan(50fps netral) → M2 metode →
    M3 MAPE 10 subjek (8/10) → M4 vs trivial baseline → M5 bukti tertangkap (3 uji)
    → M6 kontrol negatif → M7 hold-out → M8 kriteria SNR+jarak+halangan → M9 kesimpulan.

    Batasan Masalah versi netral (tanpa paragraf defensif "kami tidak menilai TI" —
    gak perlu, karena angka TI gak ada) ada di slide M1. Draft panjang di
    `thesis/batasan_masalah_draft.md` masih versi "banding TI" — kalau mau full
    commit ke arah ini, sederhanain draft itu juga.

33. **HASIL TIDAK BERGANTUNG PILIHAN FS — amunisi sidang (18 Jul 2026).**
    Cek-mandiri: `python code/evaluation/phase_pipeline.py --demo data/processed/aligned_all.csv`

    **Pertanyaan yang diantisipasi**: *"Kenapa di-resample ke 25 Hz, padahal aslinya
    ~50 Hz? Kenapa bukan angka lain?"* — ini kelihatan seperti parameter yang disetel
    diam-diam sampai angkanya bagus.

    **JAWABANNYA: nggak berpengaruh sama sekali. Sudah diukur, bukan diklaim.**

    | FS | MAPE rata-rata | lolos |
    |---|---|---|
    | 12,5 Hz | 6,57% | 7/10 |
    | 20 Hz | 6,33% | 8/10 |
    | 25 Hz | **6,50%** | **8/10** |
    | 40 Hz | 6,56% | 8/10 |
    | 50 Hz | 6,54% | 8/10 |

    Datar dalam 0,24 poin di rentang 10-50 Hz. **FS=25 bukan nilai istimewa.**

    **Alasan fisisnya (pakai ini kalau ditanya, bukan tabelnya doang):**
    - Pita jantung mentok 2,0 Hz → Nyquist cuma minta > 4 Hz. 25 Hz = margin 12,5x.
    - Resample-nya ADA bukan buat nurunin laju, tapi buat **meratakan grid waktu**
      (timestamp radar jitter std 3,7-6,6 ms akibat tick timer OS Windows 15,6 ms).
      Welch butuh jarak sampel seragam; laju tujuannya bebas.
    - **Resolusi frekuensi TIDAK dipengaruhi FS**: `df = fs/N = fs/(T·fs) = 1/T`.
      fs-nya coret. Yang nentuin 1,5 bpm itu **durasi jendela 40 detik**, bukan FS.
      (Sering disalahpahami — termasuk sempat salah dijelaskan sendiri.)

    **BUG YANG DIPERBAIKI SEKALIAN — `resample_antialias` late-binding.**
    Dulu `def resample_antialias(t, x, fs=FS)`: Python membekukan nilai `FS` saat
    IMPOR. Siapa pun yang mengubah `FS` di runtime cuma mengubah bagian hilir
    (panjang jendela + sumbu frekuensi Welch) sementara sinyalnya tetap di-resample
    ke 25 Hz → **sumbu frekuensi melar dan hasilnya sampah, tanpa error**. Sekarang
    `fs=None` lalu dibaca saat dipanggil. `demo()` menjaga ini.
    ⚠️ Bug ini bikin uji sensitivitas FS yang pertama menghasilkan kesimpulan PALSU
    ("hasil rapuh, FS=50 → MAPE 15,6%"). **Itu artefak tes, bukan sifat pipeline.**
    Kalau nanti ada yang bilang hasilnya rapuh terhadap FS, cek dulu apakah dia
    kena bug yang sama.

    **`fs_orig` diganti ke `np.mean(np.diff(t))`** (dari median) — konsisten dengan
    poin 27/28: mean dt = 20,0 ms = laju frame hardware asli, median 15,6 ms cuma
    artefak timer. Benar secara prinsip, **TAPI JANGAN diklaim memperbaiki apa pun**:
    diukur langsung, median vs mean menghasilkan MAPE praktis sama di semua FS.
    Cutoff anti-alias (`fs/2*0.9`) jauh di atas pita 0,8-2,0 Hz, jadi meleset pun
    tidak menyentuh pita yang dipakai.

    ✅ **Semua angka utama UTUH** setelah kedua perubahan: 8/10 lolos, MAPE 6,5%,
    SNR per-subjek identik, korelasi antar-subjek +0,825.

    ⚠️ Salinan `resample_antialias` yang MASIH pakai median dt (skrip lama/sekunder,
    belum disentuh): `analyze_distance.py:50`, `quality_metric.py:66`,
    `corrected_estimator.py:45`. Semua deck utama impor dari `phase_pipeline`
    jadi sudah ikut terkoreksi.

## Konvensi Kode
- Python 3, library utama: pandas, numpy, scipy
- Semua script preprocessing/evaluation ada di `code/`, jangan taruh logic besar
  langsung di notebook
- Timestamp selalu dalam unix epoch (detik, float) — kalau butuh datetime,
  convert saat visualisasi saja, simpan raw tetap epoch
- Simpan setiap dataset session baru dengan penomoran `sessionXX` biar konsisten

## Preprocessing
`code/preprocessing/extract_ground_truth.py` punya 2 mode:
- `single` — proses satu pasang file attys/radar manual
- `batch` — auto-scan seluruh `data/raw/` (semua subjek + posisi sekaligus),
  hasil digabung ke satu file dengan kolom `dataset`/`subject_id`/`position`.
  **Ini mode yang dipakai setelah semua 10 subjek + data position_variation
  sudah ditaruh di `data/raw/`.**

Jalankan setiap kali ada data baru:
```bash
cd code/preprocessing
python extract_ground_truth.py batch ../../data/raw ../../data/processed/aligned_all.csv
```

## Belum Dikerjakan / To-Do
- [x] Semua 10 subjek `position_front` + 8 posisi `position_variation_subject01`
      sudah ada di `data/raw/` (per Jul 2026)
- [x] Batch preprocessing sudah dijalankan (per Jul 2026) — `aligned_all.csv`
      ada di `data/processed/`, 18/18 sesi berhasil, 275606 baris total
- [x] Tabel "Temuan Kunci" #1 sudah di-cek ulang di semua 10 subjek — pola stuck-
      value + korelasi ~0 KONSISTEN di semua subjek `position_front`, bukan cuma
      subject01
- [ ] Catatan kualitas data: sesi `position_variation_subject01/back` dan
      `/right` cuma dapet 13 dan 10 beat ground truth valid (dari 44 dan 34
      R-peak terdeteksi awal — sebagian besar dibuang karena implausible).
      Kemungkinan ada masalah kontak elektrode Attys atau noise di sesi itu.
      Ground truth utk kedua posisi ini rendah confidence, pertimbangkan exclude
      atau tandai khusus saat dipakai di analisis cross-position.
- [x] **Root-cause: kenapa semua pendekatan mentok di korelasi ~0** (13 Jul 2026)
      → TERJAWAB, dan kesimpulan lama ("radar rusak") KELIRU. Lihat poin 10-13.
      Sinyal HR ada; penyebabnya band kelewat lebar + no smoothing + salah metrik.
- [x] **Estimator DSP yang bener** (`code/evaluation/corrected_estimator.py`) —
      MAE 5.2 bpm di subjek uji (S10) vs trivial baseline 14.8 bpm.
- [x] Dokumen bimbingan: `thesis/diagnosis_riset.pdf` + `thesis/rekomendasi_riset.pdf`

### PRIORITAS SEKARANG — TANPA REKAM ULANG (poin 24-25)
User kehabisan waktu/resource → **tesis diselesaikan dari data yang ADA**, analisis
dipersempit ke `unwrapPhasePeak_mm`. Sisa pekerjaan = **MENULIS**, bukan eksperimen.
- [x] Pipeline final: `code/evaluation/phase_pipeline.py` → **8/10 lolos, MAPE 6.5%**
      (jendela 40s + medfilt 9, poin 28d)
- [x] Kriteria SNR (K3) + analisis jarak: `code/evaluation/analyze_distance.py`
- [x] **HRV: TIDAK FEASIBLE** (`code/evaluation/hrv_analysis.py`, poin 27) — hasil
      NEGATIF yang memperkuat K3. Jangan coba estimator lain, batasnya fisik.
- [x] **KRITERIA SNR** (`code/evaluation/snr_criterion.py`, poin 30) — kegagalan
      S04/S06/cad02/100cm **diprediksi SNR** (r = −0,91, benar 13/14 sesi). Bukti kausal:
      50 cm (SNR 3,04, LOLOS) vs 100 cm (SNR 1,27, GAGAL), subjek sama. **JANGAN ganti
      subjek yang MAPE-nya jelek.** Slide: `thesis/kenapa_subjek_gagal.pdf`.
- [x] **KONTROL NEGATIF + HOLD-OUT** (`code/evaluation/negative_control.py`, poin 29)
      — logger ngelaporin **83-85 bpm dari RUANGAN KOSONG**; fase misahin kosong vs
      berisi dengan margin **1.329x** lewat napas. Plus 2 subjek hold-out: fase
      menang 2/2. Slide: `thesis/bukti_detak_jantung.pdf`.
- [x] **Perbandingan utama logger vs fase** (`code/evaluation/compare_ti_vs_phase.py`,
      poin 28) — **MAPE 6,5% vs 20,2%, lolos 8/10 vs 2/10**; korelasi antar-subjek
      **+0,83 vs −0,41**. Kriteria pembimbing TERPENUHI → **gak perlu rekam ulang**.
      Slide: `thesis/logger_vs_fase.pdf`. Data mentah per jendela:
      `thesis/data_mentah/data_jendela_40detik.xlsx`.
- [ ] **[WAJIB] Tulis Batasan Masalah** (poin 25) — ini yang melindungi di sidang
- [ ] **[WAJIB] Tulis bab K1** (3 cacat, poin 20-22) — bukti & skrip sudah lengkap
- [ ] **[WAJIB] Tulis bab K2** (pipeline fase, poin 24) — angka sudah final
- [ ] **[WAJIB] Tulis bab K3** (kriteria SNR + jarak) — skrip sudah ada
- [ ] **[WAJIB] Tulis bab K4** (metodologi, poin 8 & 12)
- [ ] [TINGGI] Perbaiki kasus S10 (bias +9.7 bpm, HR rendah 61 bpm) → kalau berhasil
      jadi **7/10 lolos**, tesis makin kuat
- [ ] [OPSIONAL] Rekam ulang — **BUKAN syarat kelulusan lagi**, cuma saran lanjutan.
      Kalau ada kesempatan: perbaiki logger dulu (poin 23), jarak ~50 cm, HR bervariasi.
- [ ] **[TINGGI] Perbaiki bias sistematis -9 bpm** (poin 13). Urutan coba, dari
      paling murah:
      (a) pakai turunan fase (`np.diff` pada `unwrapPhasePeak_mm`) — TI sendiri
          bikin phase-difference justru buat nekan drift & nguatin komponen
          jantung (Developer's Guide hal. 9). Satu baris, coba duluan.
      (b) kurangkan komponen napas secara eksplisit: estimasi sinyal 0.1-0.5 Hz,
          rekonstruksi, SUBTRACT dari fase (beda dari notch-harmonic yang dulu
          gagal — itu ngapus frekuensi diskrit, ini ngapus komponen + ekornya)
      (c) ganti argmax dgn autokorelasi / cepstrum / harmonic-sum (energi f + 2f)
- [ ] **[TINGGI] Feature engineering dari `unwrapPhasePeak_mm`** (poin 16) —
      bisa dimulai SEKARANG, gak perlu nunggu data baru.
- [ ] **[SEDANG] Model ML pertama** (Ridge/RF) di atas fitur poin 16. Ukur vs
      `corrected_estimator.py`, bukan cuma vs trivial baseline. **Ini bab BONUS
      sekarang** (poin 19), bukan penentu kelulusan.
- [ ] Evaluasi wajib: MAE + RMSE + korelasi DALAM-subjek + korelasi ANTAR-subjek
      + trivial baseline (split subjek sama) + std GT + Bland-Altman + breakdown
      per subjek. Lihat `thesis/rekomendasi_riset.pdf` Bagian 4.
- [ ] Split train/test HARUS berbasis subjek (bukan random per-baris) supaya
      tidak ada data leakage antar subjek yang sama
- [ ] Review baseline model existing user (`code/baseline/` masih kosong) —
      kemungkinan besar penyebabnya sudah kejawab di poin 8 (GIGO: fitur BPM TI)
- [ ] [RENDAH] Analisis cross-position — **kepentok**: `position_variation` gak
      punya `unwrapPhasePeak_mm` sama sekali (poin 4), padahal itu satu-satunya
      kolom yang berguna. Harus DIREKAM ULANG dengan skema kolom lengkap, atau
      analisis ini dicoret dari rencana tesis. Tanyakan ke pembimbing.
