# Kerangka Bahasan — Laporan Tesis (Reframe: Analisis Murni Sinyal Fase Mentah)

> **Status:** kerangka untuk diedit manual. Belum dibuat ke Word.
>
> **Perubahan arah riset (penting):** fokus BUKAN lagi memperbaiki keluaran
> pustaka bawaan radar. Fokus sekarang **murni menganalisis seberapa akurat
> sinyal fase mentah (`unwrapPhasePeak_mm`) untuk estimasi detak jantung.**
>
> **Aturan penulisan yang WAJIB dipatuhi di seluruh dokumen:**
> 1. ❌ JANGAN menyebut pustaka/library bawaan radar (Vital Signs TI).
> 2. ❌ JANGAN menyebut filter internal / rantai pemrosesan bawaan.
> 3. ❌ JANGAN menyebut atau menampilkan nilai BPM keluaran bawaan.
> 4. ❌ JANGAN ada perbandingan terhadap keluaran alat / trivial baseline.
> 5. ❌ JANGAN ada validasi hold-out & subjek cadangan.
> 6. ✅ 50 fps disebut hanya sebagai konfigurasi akuisisi (tanpa pembelaan,
>    tanpa mengaitkan ke laju rancangan apa pun).
> 7. ✅ Objek tunggal = sinyal fase mentah `unwrapPhasePeak_mm`.
> 8. ✅ Data jarak, halangan, ruangan kosong = bukti tambahan bahwa radar
>    memang menangkap sinyal pergerakan dada.

---

## Judul (usulan)

**"Analisis Kelayakan Sinyal Fase Mentah Radar FMCW TI IWR1443BOOST untuk
Estimasi Detak Jantung Nirkontak Tervalidasi Standar ANSI/CTA-2065"**

Alternatif: *"Estimasi Detak Jantung Nirkontak Berbasis Sinyal Fase Mentah
Radar FMCW TI IWR1443BOOST dan Kriteria Kelayakannya"*

---

## Abstrak (poin-poin isi)

- Konteks: CVD penyebab kematian utama; perlu pemantauan HR nirkontak.
- Objek: sinyal fase mentah `unwrapPhasePeak_mm` dari radar FMCW IWR1443BOOST
  sebagai basis estimasi detak jantung.
- Data: 10 subjek kondisi istirahat; acuan EKG Attys 125 Hz (R-peak Pan-Tompkins);
  akuisisi 50 fps.
- Metode: pipeline pemrosesan sinyal (DSP) — turunan fase → reduksi napas →
  bandpass 0,8–2,0 Hz → Welch PSD jendela 40 s → median filter.
- Hasil: **8/10 subjek lolos ANSI/CTA-2065 (MAPE < 10%)**, rata-rata MAPE 6,5%,
  korelasi antar-subjek +0,83.
- Bukti sinyal fase menangkap gerak dada: 3 uji independen + kontrol negatif
  (ruangan kosong, pemisahan margin 1.329×) + eksperimen halangan.
- Kriteria kelayakan: kegagalan diprediksi SNR sinyal jantung (ambang ~1,8),
  dibuktikan kausal lewat eksperimen jarak terkendali (50 vs 100 cm).
- Kata kunci: radar FMCW, IWR1443BOOST, detak jantung, nirkontak, sinyal fase,
  ANSI/CTA-2065.

---

# BAB I — PENDAHULUAN

## 1.1 Latar Belakang
- Penyakit kardiovaskular (CVD): penyebab kematian utama global (17,9 juta/tahun).
- Pentingnya pemantauan HR akurat & berkelanjutan; keterbatasan EKG kontak
  (tidak nyaman, iritasi, kurang praktis untuk pemantauan kontinu).
- Radar FMCW sebagai solusi nirkontak: peka terhadap micromotion permukaan dada.
- **Inti pergeseran narasi:** sinyal fase mentah adalah representasi paling
  langsung dari perpindahan (displacement) permukaan dada. Menganalisis kelayakan
  sinyal ini menetapkan **kemampuan fundamental sensor** untuk estimasi HR,
  terlepas dari algoritma estimasi tertentu.
- Pertanyaan yang dijawab: seberapa akurat estimasi HR dari sinyal fase mentah,
  dan apa yang menentukan batas keberlakuannya.

## 1.2 Rumusan Masalah
1. Apakah sinyal fase mentah `unwrapPhasePeak_mm` benar mengandung informasi
   detak jantung yang dapat dibuktikan secara independen (bukan artefak/derau)?
2. Seberapa akurat estimasi detak jantung dari sinyal fase mentah menggunakan
   pipeline DSP, dibandingkan standar ANSI/CTA-2065?
3. Faktor fisik apa yang menentukan keberhasilan/kegagalan estimasi, dan dapatkah
   dijadikan kriteria kelayakan yang terukur?

## 1.3 Batasan Masalah
- **(a) Objek tunggal:** sinyal fase mentah `unwrapPhasePeak_mm` (perpindahan
  fase hasil phase-unwrapping pada bin jarak target).
- **(b) Konfigurasi akuisisi 50 fps.** Cukup untuk resolusi temporal rekonstruksi
  fase. *(Sebut ringkas saja — tanpa pembanding laju apa pun.)*
- **(c) Kondisi subjek: istirahat** (std HR dalam sesi 0,9–4,2 bpm) → yang
  dievaluasi ketepatan **HR rata-rata per jendela**, bukan pelacakan perubahan HR.
- **(d) Estimasi HR rata-rata, bukan HRV** (alasan fisik dinyatakan di 4.7).
- **(e) Acuan kebenaran:** EKG Attys 125 Hz, R-peak Pan-Tompkins.
- **(f) Standar penilaian:** ANSI/CTA-2065, MAPE < 10%.
- **(g) Batas fisik keberlakuan (SNR)** dilaporkan sebagai bagian dari hasil;
  tidak ada subjek yang dibuang.
- **(h) Dataset utama: 10 subjek** (`position_front`). Data jarak, halangan, dan
  ruangan kosong dipakai sebagai bukti tambahan penangkapan gerak dada.

## 1.4 Tujuan Penelitian
1. Membuktikan secara independen bahwa sinyal fase mentah mengandung informasi
   detak jantung yang valid.
2. Merancang pipeline DSP estimasi HR dari sinyal fase mentah & mengukur
   akurasinya terhadap ANSI/CTA-2065 pada 10 subjek.
3. Merumuskan kriteria kelayakan berbasis SNR yang menjelaskan & memprediksi
   keberhasilan/kegagalan secara fisik (dikuatkan eksperimen jarak).

## 1.5 Manfaat Penelitian
- **Akademis:** koreksi metodologi evaluasi HR pada kondisi HR nyaris konstan
  (ceiling korelasi dalam-subjek → pakai korelasi antar-subjek); metodologi
  validasi (kontrol negatif, uji sensitivitas) yang dapat diadopsi riset sensor
  nirkontak lain.
- **Praktis:** kriteria kelayakan SNR sebagai pedoman penempatan sensor (jarak,
  halangan) untuk aplikasi pemantauan HR nirkontak.

---

# BAB II — TINJAUAN PUSTAKA

## 2.1 Prinsip Kerja Radar
- Konsep dasar pancar–pantul–terima; Persamaan jarak `d = c·t/2`.
- (Gambar: prinsip kerja radar — sudah ada aset.)

## 2.2 Radar FMCW dan Sinyal Chirp
- Sinyal chirp (frekuensi naik linear), Persamaan chirp, sinyal IF.
- Hubungan fase–perpindahan: `φ(t) = 4π·d(t)/λ` → micromotion dada termanifestasi
  sebagai variasi fase. **Ini fondasi kenapa sinyal fase mentah dipakai.**
- (Gambar: sinyal chirp; blok pembangkitan/mixing FMCW — aset ada.)
- Tabel modul FMCW komersial (aset ada).

## 2.3 Modul FMCW IWR1443BOOST
- Spesifikasi hardware: 77 GHz, 3 Tx / 4 Rx, synthesizer, DSP internal.
- (Gambar: diagram blok modul IWR1443BOOST — aset ada.)
- **CATATAN: hanya deskripsi perangkat keras.** ❌ TIDAK ada sub-bab pemrosesan
  Vital Signs / rantai internal / estimasi BPM bawaan.

## 2.4 Ekstraksi Sinyal Fase Mentah (Perpindahan Dada)
- Deskripsikan generik & netral: Range FFT → identifikasi bin jarak target →
  ekstraksi fase → **phase unwrapping** → sinyal perpindahan `unwrapPhasePeak_mm`
  (mm) sebagai deret waktu gerak dada.
- Tekankan: sinyal ini adalah **pengukuran perpindahan fisik** yang menjadi input
  langsung analisis penelitian ini.
- ❌ JANGAN mengaitkan ke rantai pemrosesan HR bawaan / filter jantung–napas
  bawaan / keluaran BPM.

## 2.5 Deteksi Detak Jantung dari EKG (Pan–Tompkins)
- Kompleks QRS & R-peak sebagai penanda detak.
- Tahapan Pan-Tompkins: bandpass 5–15 Hz → derivative → squaring → moving window
  integration → peak detection (constraint fisiologis 40–180 bpm).

## 2.6 Standar ANSI/CTA-2065
- Kriteria validasi alat HR konsumer: MAPE < 10% terhadap EKG.
- Persamaan MAPE. Pada HR ~78 bpm, 10% ≈ 7,8 bpm.
- Sebagian studi pakai kriteria lebih ketat (±5%).

## 2.7 Penelitian Terkait
*(Semua dari library Zotero — relevan, tetap dipakai.)*
- Alizadeh dkk. (2019): radar TI 77 GHz + phase unwrapping, korelasi 80% —
  pemrosesan berbasis fase adalah pendekatan mapan; akurasi bergantung kondisi.
- Turppa dkk. (2020): MAE relatif 3,6% pada skenario tidur (jarak tetap, minim
  gerak) — kondisi terkontrol → akurasi tinggi.
- Wang dkk. (2020), Lv dkk. (2021): akurasi ~90–93% dengan penapisan adaptif +
  koreksi harmonik napas.
- Zhou dkk. (2023), Huang dkk. (2022): teknik DSP lanjutan (MAE <1 bpm; galat
  2,9%) — arah perbaikan bias (penelitian lanjutan).
- Xu dkk. (2025), Xia dkk. (2021): HRV/fidusial dari radar pada SNR tinggi &
  jarak dekat — konteks keterbatasan HRV penelitian ini.
- Kranjec dkk. (2014): tinjauan metode HR nirkontak.
- Posisi penelitian ini: memproses sinyal fase mentah secara langsung & menetapkan
  kriteria kelayakan fisik berbasis SNR.

---

# BAB III — METODOLOGI PENELITIAN

## 3.1 Desain Eksperimen & Akuisisi Data
- Perangkat: radar IWR1443BOOST (uji) + EKG Attys 125 Hz (acuan), rekam bersamaan,
  timestamp epoch independen → perlu alignment. Akuisisi 50 fps.
- Ringkasan dataset (Tabel 3.1):

  | Dataset | Jumlah | Peran |
  |---|---|---|
  | position_front | 10 subjek | Utama: penentuan parameter & evaluasi akurasi |
  | distance (50 & 100 cm) | 2 sesi | Bukti kausal SNR↔akurasi; penangkapan gerak dada |
  | obstacle | (beberapa) | Bukti tambahan penangkapan gerak dada (efek halangan) |
  | no_subject | 2 sesi (~7 mnt) | Kontrol negatif: radar tanpa subjek |

  ❌ TIDAK ada baris subjek cadangan.

## 3.2 Prapemrosesan & Ekstraksi Ground Truth
- R-peak Pan-Tompkins pada EKG → interpolasi ke timestamp radar (`gt_heart_rate`).
- Alasan pakai R-peak, bukan spektral langsung (energi QRS di frekuensi lebih
  tinggi → spektral langsung tidak stabil).

## 3.3 Kriteria SNR (definisi operasional)
- SNR = energi PSD sinyal fase di pita target ÷ noise floor (median PSD sekitar),
  per jendela.
- **SNR jantung** (0,8–2,0 Hz) → prediktor keberhasilan estimasi HR.
- **SNR napas** (0,15–0,6 Hz) → deteksi kehadiran subjek.
- Dasar fisik: gerak dada napas 1–12 mm ≫ jantung 0,1–0,5 mm.

## 3.4 Pembuktian Radar Menangkap Gerak Dada
*(Ditegakkan SEBELUM hasil akurasi.)*
- **Kontrol negatif** (ruangan kosong): radar seharusnya melaporkan "tidak ada".
- **Eksperimen jarak** (50 vs 100 cm): jarak makin jauh → sinyal makin lemah.
- **Eksperimen halangan** (buku): halangan → sinyal melemah → menguatkan bahwa
  yang terukur pantulan dari dada.

## 3.5 Pipeline Pemrosesan Sinyal Fase (DSP, tanpa ML)
Tahapan (`phase_pipeline.py`):
1. Resample anti-alias (grid waktu seragam; hasil tak sensitif terhadap laju
   resampling 10–50 Hz).
2. Turunan fase (`np.diff`) — menekan drift, menguatkan komponen dinamis.
3. Reduksi komponen napas (0,15–0,6 Hz).
4. Bandpass 0,8–2,0 Hz.
5. Welch PSD jendela 40 detik → argmax (interpolasi parabolik).
6. Median filter (window 9).
- Alasan jendela 40 s (dinyatakan di muka): subjek diam → tak ada dinamika HR yang
  hilang → jendela panjang murni menguntungkan resolusi frekuensi. Konsekuensi
  (respons lambat ke perubahan HR) dinyatakan sebagai batasan di 4.7.

## 3.6 Skema Evaluasi
- Metrik utama: **MAPE & MAE** terhadap EKG; ambang ANSI/CTA-2065 (MAPE < 10%).
- **Korelasi antar-subjek** (bukan dalam-subjek): karena HR nyaris konstan,
  korelasi dalam-subjek tidak sah (ceiling teoretis rendah). Korelasi antar-subjek
  menguji apakah radar bisa membedakan orang ber-HR tinggi vs rendah.
- **SNR sebagai prediktor kegagalan** (diukur tanpa melihat MAPE dulu),
  dikuatkan eksperimen jarak terkendali.
- ❌ TIDAK ada trivial baseline. ❌ TIDAK ada hold-out.

---

# BAB IV — HASIL DAN PEMBAHASAN

## 4.1 Ilustrasi Estimasi (EKG vs Estimasi Radar)
- **[GAMBAR SUDAH DIHITUNG ULANG, poin 4]** → **subject09** dipilih (MAPE 1,3%,
  MAE 1,1 bpm, SNR 6,70) — tracking paling mulus, estimasi radar menempel ke EKG
  sepanjang sesi. File: `scratchpad/.../estimasi_subject09.png`.
  - Catatan: gambar lama drop di ujung karena memakai **subject01** (tracking
    tengahnya bagus tapi jendela terakhir jatuh — artefak tepi). subject07 punya
    spike di tengah. subject09 paling bersih.
- Narasi: tiap jendela 40 s → satu estimasi (puncak PSD); deret dihaluskan median
  filter → kurva estimasi mengikuti EKG.

## 4.2 Akurasi terhadap Standar ANSI/CTA-2065
- **8/10 subjek lolos** (MAPE < 10%), rata-rata MAPE 6,5%.
- **[TABEL 4.1 — poin 5: MAPE + SNR berdampingan, urut lolos dulu lalu gagal.
  ANGKA RIIL sudah dihitung dari `snr_criterion.py` + `phase_pipeline.py`]**

  | Subjek | HR EKG (bpm) | SNR jantung | MAPE (%) | Status |
  |---|---|---|---|---|
  | S09 | 84 | 6,70 | 1,3 | Lolos |
  | S01 | 89 | 9,32 | 1,7 | Lolos |
  | S07 | 65 | 9,01 | 1,9 | Lolos |
  | S08 | 75 | 5,66 | 2,3 | Lolos |
  | S02 | 77 | 7,41 | 3,4 | Lolos |
  | S05 | 68 | 4,21 | 4,4 | Lolos |
  | S10 | 61 | 5,88 | 7,9 | Lolos |
  | S03 | 79 | 2,38 | 9,7 | Lolos |
  | **S06** | 76 | **1,60** | **12,5** | **Gagal** |
  | **S04** | 76 | **1,72** | **20,0** | **Gagal** |

  → Dengan SNR di kolom sebelah, **langsung terlihat: dua yang gagal (S04, S06)
  justru ber-SNR terendah** (< 1,8). Semua yang lolos ber-SNR ≥ 2,38. Pemisahan
  bersih pada 10 subjek utama. Rata-rata MAPE = **6,5%**.
- (Gambar batang MAPE per subjek vs ambang 10% — aset ada / bisa dibuat ulang.)

## 4.3 Bukti Sinyal Fase Mengandung Detak Jantung (3 Uji Independen)
- Uji 1: puncak spektrum PSD radar jatuh tepat di frekuensi EKG.
- Uji 2: pada 402 jendela (10 subjek), peringkat ternormalisasi frekuensi EKG di
  spektrum radar median **0,08** (acak = 0,50) → frekuensi benar memang menonjol.
- Uji 3: **korelasi antar-subjek +0,83** — radar bisa membedakan individu ber-HR
  tinggi vs rendah.
- (Gambar tiga-panel — aset ada; **panel korelasi jangan menyertakan referensi
  trivial baseline**.)

## 4.4 Kontrol Negatif: Ruangan Kosong
- Gerak dada: 0,00 mm (kosong) vs 3,1–6,6 mm (ada orang).
- SNR napas: 2,2–3,3 (kosong) vs 4.359–41.595 (ada orang) → **pisah 1.329×, nol
  tumpang tindih**.
- Kesimpulan: sinyal fase **tahu** ada/tidak orang → bukti kuat penangkapan gerak
  dada. Deteksi kehadiran via napas tidak butuh EKG.
- (Gambar kontrol negatif — aset ada.)

## 4.5 Bukti Tambahan: Eksperimen Halangan
- Halangan (buku) → SNR turun (efek ambang: 3 cm sudah memotong SNR signifikan).
- Menguatkan bahwa yang terukur adalah pantulan dari dada, bukan derau.
- ⚠️ Hanya nilai SNR yang dipakai (bukan MAPE) — HR subjek dekat rata-rata kohort.
- ❌ Tidak masuk ke tabel SNR–MAPE (poin 7).

## 4.6 Kriteria Kelayakan SNR & Eksperimen Jarak
- Kegagalan **tidak acak** — diprediksi SNR sinyal jantung.
- Korelasi log(SNR) vs log(MAPE) kuat & negatif; ambang **SNR ≈ 1,8** memisahkan
  lolos/gagal.
- **Bukti kausal — eksperimen jarak terkendali** (subjek & setup sama, hanya
  jarak berubah):

  **[TABEL 4.2 — poin 7: subjek + baris jarak, TANPA halangan. ANGKA RIIL]**

  | Sesi | SNR jantung | MAPE (%) | Status |
  |---|---|---|---|
  | 10 subjek utama (ringkas Tabel 4.1) | 1,60 – 9,32 | 1,3 – 20,0 | 8 lolos / 2 gagal |
  | Jarak 50 cm | 2,99 | 6,4 | Lolos |
  | Jarak 100 cm | 1,36 | 56,1 | Gagal |

  → Daya pantul ∝ 1/R⁴: jarak 2× → daya 16× lebih kecil → SNR & MAPE anjlok,
  persis seperti diramalkan. **Subjek & setup sama, hanya jarak berubah** →
  bukti sebabnya fisik, bukan kebetulan subjek.
- (Gambar scatter SNR vs MAPE dengan garis ambang 1,8 — aset ada / buat ulang;
  **buang penanda halangan & hold-out**.)
- **Alternatif penyajian (kalau tak mau 2 tabel):** cukup Tabel 4.1 (poin 5)
  ditambah 2 baris jarak di bawahnya → satu tabel melayani poin 5 & 7 sekaligus.
  *(Rekomendasi: pakai alternatif ini agar tidak redundan.)*

## 4.7 Keterbatasan
- **(a) Tidak ada variasi HR** (subjek diam) → pelacakan perubahan HR tak dapat
  dibuktikan/dibantah dengan data ini.
- **(b) HRV belum layak:** deteksi detak-per-detak recall ~61% vs R-peak EKG;
  RMSSD radar ~4× nilai EKG. Batas fisik (SNR + presisi waktu 20 ms/sampel),
  bukan algoritmik → hasil negatif yang memperkuat kriteria kelayakan.
- **(c) SNR jantung butuh EKG** → alat diagnosis, belum gerbang kualitas lapangan.
  Deteksi kehadiran via napas (4.4) sudah tak butuh EKG; gerbang kualitas
  per-estimasi tanpa EKG = penelitian lanjutan.

---

# BAB V — KESIMPULAN & PENELITIAN LANJUTAN

## 5.1 Kesimpulan (menjawab 3 rumusan)
1. Sinyal fase mentah **terbukti mengandung** detak jantung (3 uji independen +
   kontrol negatif + eksperimen jarak/halangan).
2. Pipeline DSP mencapai **rata-rata MAPE 6,5%, 8/10 lolos ANSI/CTA-2065**,
   korelasi antar-subjek +0,83.
3. Keberhasilan/kegagalan **diprediksi SNR** (ambang ~1,8), dibuktikan kausal
   lewat eksperimen jarak.

**Kontribusi:**
- **K1** — Pipeline DSP estimasi HR dari sinyal fase mentah, tervalidasi
  ANSI/CTA-2065 (8/10 lolos, MAPE 6,5%).
- **K2** — Kriteria kelayakan berbasis SNR + karakterisasi jarak/halangan.
- **K3** — Koreksi metodologi evaluasi (ceiling korelasi dalam-subjek pada HR
  nyaris konstan → gunakan korelasi antar-subjek).

*(Catatan: penomoran kontribusi diringkas jadi K1–K3; tidak ada kontribusi
analisis pustaka bawaan.)*

## 5.2 Penelitian Lanjutan
1. Perekaman ulang dengan **variasi HR** (protokol istirahat–beban–recovery) untuk
   menguji pelacakan perubahan HR.
2. **Koreksi bias** via teknik DSP lanjutan (mengacu Zhou dkk. 2023, Huang dkk.
   2022): matched filtering, VME, kanselasi harmonik napas, LPC.
3. **Gerbang kualitas tanpa EKG** untuk melengkapi deteksi kehadiran via napas.
4. **HRV** pada kondisi SNR tinggi & jarak dekat (mengacu Xu dkk. 2025; Xia dkk.
   2021).
5. **Model pembelajaran mesin** di atas fitur turunan sinyal fase (sebagai
   pelengkap pipeline DSP).
6. Evaluasi **cross-position** (perlu skema kolom fase lengkap pada dataset variasi
   posisi).

---

# LAMPIRAN (usulan)
- **Lampiran A — Peta skrip analisis → tahap** (reproduksibilitas):
  extract_ground_truth.py, snr_criterion.py, negative_control.py,
  phase_pipeline.py, analyze_distance.py, hrv_analysis.py.
- ❌ TIDAK ada lampiran analisis pustaka bawaan.

---

## Daftar tugas hitung-ulang (sebelum ke Word)
- [ ] **Poin 4:** recompute gambar EKG vs estimasi radar — pilih subjek terbaik.
- [ ] **Poin 5:** isi kolom SNR & HR per subjek di Tabel 4.1 (`snr_criterion.py`).
- [ ] **Poin 7:** rakit Tabel 4.2 (subjek + 50 cm + 100 cm) — atau gabung ke 4.1.
- [ ] Regenerasi gambar batang MAPE & scatter SNR tanpa penanda hold-out/halangan.
