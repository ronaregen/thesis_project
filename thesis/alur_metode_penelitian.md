# Alur Metode Penelitian

**Judul kerja:** *Estimasi Detak Jantung Berbasis Sinyal Fase Mentah Radar FMCW
TI IWR1443BOOST: Rancangan Pipeline Pemrosesan Sinyal Tervalidasi Standar
ANSI/CTA-2065*

**Objek yang diuji:** sinyal fase mentah `unwrapPhasePeak_mm` dari radar — BUKAN
output BPM bawaan library TI (lihat Batasan Masalah, poin B di bawah).

**Baku ukur (gold standard):** ECG dari perekam Attys @125 Hz, ground truth BPM
diambil via deteksi R-peak (Pan-Tompkins).

**Standar kelayakan:** ANSI/CTA-2065 (CTA, 2018) — alat HR VALID kalau MAPE
terhadap ECG < 10%.

---

## Ringkasan alur (satu tarikan napas)

> Radar bisa "melihat" gerak dada? → **buktikan pakai kontrol negatif (ruangan
> kosong)**. → Kalau bisa, seberapa akurat estimasi BPM-nya dari sinyal fase? →
> **ukur MAPE 10 subjek vs standar 10%**. → Kalau ada yang gagal, itu acak atau
> bisa diramalkan? → **jelaskan pakai SNR + eksperimen jarak terkendali**. →
> Semua parameter ketahuan disetel biar bagus? → **buktikan ndak, pakai hold-out
> + uji sensitivitas**.

Urutan ini penting: **alat ukur (SNR, kontrol negatif) ditegakkan DULU, sebelum
melihat hasil MAPE.** Itu yang bikin metode ini tahan serangan "kamu nyetel
parameter sampai angkanya bagus".

---

## Tahap 0 — Batasan Masalah (ditulis di MUKA, ini yang melindungi di sidang)

Wajib dinyatakan sebelum hasil apa pun:

1. Penelitian **tidak memberikan vonis kelayakan terhadap library Vital Signs
   bawaan TI.** Alasan: data direkam pada 50 fps, sedangkan library TI dirancang
   untuk 20 fps. Jadi kolom BPM turunan TI bukan pembanding yang adil dan
   **tidak dievaluasi**.
2. Yang dievaluasi adalah **kelayakan sinyal fase mentah** (`unwrapPhasePeak_mm`)
   sebagai basis estimasi detak jantung. Kolom ini diambil SEBELUM filter TI,
   jadi tidak terpengaruh mismatch frame rate.
3. Pembanding kecukupan = **trivial baseline** (tebak konstan tanpa radar) dan
   **ECG**, bukan output alat. Keduanya tidak punya konfigurasi yang bisa
   dipersoalkan.

> Dengan batasan ini, fakta "50 fps" berubah dari *cacat tersembunyi* menjadi
> *kondisi eksperimen yang dinyatakan* — penguji tidak bisa mematahkan dengan
> "pengujianmu tidak adil". (Draft lengkap: `thesis/batasan_masalah_draft.md`.)

---

## Tahap 1 — Landasan: apa itu SNR & kenapa jadi tulang punggung

**Definisi operasional:** SNR = energi PSD sinyal fase di pita frekuensi target
÷ noise floor (median PSD sekitarnya). Dihitung per jendela.

- **SNR jantung** (pita 0.8–2.0 Hz) → seberapa kuat komponen detak jantung
  muncul di sinyal fase. Dipakai untuk **memprediksi lolos/gagal MAPE**.
- **SNR napas** (pita 0.15–0.6 Hz) → seberapa kuat gerak dada. Dipakai untuk
  **deteksi kehadiran** (ada orang / ruangan kosong).

Dasar fisik yang wajib disebut: gerak dada akibat **napas 1–12 mm**, akibat
**jantung hanya 0.1–0.5 mm** (spek TI). Karena itu napas jauh di atas noise
floor dan jadi penanda kehadiran yang paling andal; jantung jauh lebih halus,
karena itu SNR jantung yang menentukan bisa/tidaknya BPM diestimasi.

Script: `code/evaluation/snr_criterion.py`. Penjelasan hitung: `thesis/cara_hitung_snr.pdf`.

---

## Tahap 2 — Pembuktian: radar BENAR membaca gerak dada (bukan menebak)

**Pertanyaan:** apakah angka yang keluar dari radar benar berasal dari dada
subjek, atau cuma artefak algoritma?

**Uji 2a — Kontrol negatif (ruangan kosong).** Data `data/raw/no_subject/`
(2 sesi ~7 menit, skema kolom sama, punya `unwrapPhasePeak_mm`). Radar nyala di
ruangan kosong. Alat jujur harus bilang "tidak ada".

| Metrik | Ruangan KOSONG | Ada ORANG | Vonis |
|---|---|---|---|
| Gerak dada (fase) | 0.00 mm | 3.1–6.6 mm | pisah 850x |
| SNR napas (fase) | 2.2–3.3 | 4.359–41.595 | **pisah 1.329x, nol tumpang tindih** |

→ Sinyal fase **tahu** ada orang atau tidak. (Sekaligus: gerbang kehadiran ini
**tidak butuh ECG** → deployable, menutup keterbatasan bahwa SNR jantung butuh ECG.)

**Uji 2b — Data halangan (dinding/buku).** Data `data/raw/obstacle/`. SNR turun
begitu ada penghalang (efek ambang: buku 3 cm sudah memotong SNR ~2.5x), yang
memperkuat bahwa yang diukur memang pantulan dari dada, bukan derau.

> ⚠️ **Batas kejujuran (wajib):** data halangan hanya dipakai untuk **SNR**, BUKAN
> untuk klaim akurasi MAPE (HR subjek dekat rata-rata kohort → tebak-konstan
> menang, jadi MAPE menyesatkan). 1 sesi/kondisi → tidak bisa mengklaim
> dosis-respons tebal penghalang; laporkan sebagai **efek ambang**, poin 31.
> `obstacle/no` byte-identik dengan `subject_cadangan01` — jangan dihitung dua kali.

Script: `code/evaluation/negative_control.py`. Slide: `thesis/bukti_detak_jantung.pdf`.

---

## Tahap 3 — Pipeline pengganti & akurasi 10 subjek

**Pipeline final** (`code/evaluation/phase_pipeline.py`, murni DSP, tanpa ML):
resample anti-alias → **turunan fase (`np.diff`)** → kurangi komponen napas
(0.15–0.6 Hz) → bandpass 0.8–2.0 Hz → Welch PSD jendela **40 s** → argmax
(interpolasi parabolik) → median filter **9**.

**Hasil (10 subjek `position_front`):**

| | Hasil |
|---|---|
| **Lolos MAPE < 10%** | **8 / 10 subjek** |
| MAPE rata-rata | **6.5%** |
| Terbaik | S09 = 1.3%, S01 = 1.7% |
| vs trivial baseline (test S09+S10) | MAE 3.1 vs 12.0 → **menang** |

**Bukti detak jantung benar tertangkap** (ini senjata utama, bukan tabel MAPE):
korelasi **antar-subjek +0.83** — radar bisa membedakan orang ber-HR tinggi vs
rendah. (Korelasi *dalam-subjek* TIDAK sah dipakai di dataset ini: semua subjek
duduk diam, std HR cuma 0.9–4.2 bpm → korelasi mendekati 0 sekalipun estimator
sempurna. Ini koreksi metodologi K4, poin 12.)

Script: `code/evaluation/phase_pipeline.py`. Slide: `thesis/fase_vs_ecg.pdf`.

---

## Tahap 4 — Kenapa 2 subjek gagal: SNR meramalkan kegagalan

**Pertanyaan penguji yang diantisipasi:** "kok ada yang gagal? ganti aja
subjeknya." → **JANGAN.** Itu cherry-picking dan tak terjawab kalau ditanya
"berapa subjek yang direkam?". Yang sah: jelaskan kegagalan pakai kriteria yang
diukur **tanpa melihat hasil**.

**Kegagalan tidak acak — bisa diramalkan dari SNR.** Gabungan 17 sesi:

- Korelasi **log(SNR) vs log(MAPE) = −0.885** (sangat teratur).
- Ambang **SNR ≥ 1.8** benar di 15/17 sesi (termasuk sesi hold-out & jarak yang
  tak ikut menentukan ambang).
- Dua subjek gagal (S04 SNR 1.78, S06 SNR 1.60) dua-duanya di bawah ambang =
  **batas fisik**, bukan cacat metode.

**Bukti kausal (eksperimen terkendali `data/raw/distance/`)** — subjek SAMA,
logger SAMA, hanya jarak beda:

| Jarak | SNR | MAPE | |
|---|---|---|---|
| 50 cm | 3.04 | 6.4% | lolos |
| 100 cm | 1.27 | 53.9% | gagal |

Daya pantul radar ∝ 1/R⁴ → jarak 2x = daya 16x lebih kecil. Terukur persis
seperti diramalkan.

> **Kalimat kunci untuk pembimbing:** *"Metode ini bekerja, dan INI persis kapan
> ia secara fisik tidak bisa."* Subjek gagal = **bukti untuk kriteria kelayakan
> (K3)**, bukan kelemahan.

Script: `code/evaluation/snr_criterion.py`. Slide: `thesis/kenapa_subjek_gagal.pdf`.

---

## Tahap 5 — Anti-tuduhan overfit: hold-out & uji sensitivitas

**Uji 5a — Hold-out.** `subject_cadangan01` & `cadangan02` TIDAK masuk
`aligned_all.csv`, tidak pernah dipakai memilih parameter apa pun (jendela 40s,
medfilt 9, pita 0.8–2.0 Hz semua ditetapkan dari 10 subjek utama).
→ cad01 MAPE 4.5% (lolos), cad02 10.0% (batas). Fase menang 2/2. Bukti di luar sampel.

**Uji 5b — Sensitivitas frekuensi sampling.** MAPE datar dalam 0.24 poin di
rentang FS 10–50 Hz (poin 33) → FS=25 Hz bukan angka istimewa, hasil bukan
artefak pilihan parameter. `python code/evaluation/phase_pipeline.py --demo ...`

---

## Tahap 6 — Keterbatasan (ditulis sendiri, bukan menunggu ditanya)

Menyatakan batas sendiri = memperkuat kredibilitas, bukan melemahkan:

1. **Tidak ada variasi HR.** Semua subjek duduk diam (std HR ~2.8 bpm) →
   kemampuan melacak PERUBAHAN HR tidak bisa dibuktikan maupun dibantah dengan
   data ini. Butuh rekam ulang protokol beban+recovery (saran lanjutan, poin 15).
2. **HRV tidak feasible.** Deteksi detak-per-detak cuma 61% recall vs R-peak
   ECG, RMSSD radar 4x nilai ECG. Batas fisik (SNR + presisi waktu 20 ms), bukan
   pilihan algoritma. Hasil NEGATIF yang memperkuat K3 (poin 27).
3. **SNR jantung butuh ECG** → alat diagnosis, belum gerbang kualitas lapangan.
   Pengganti tanpa-ECG (deteksi kehadiran via napas, Tahap 2) sudah ada untuk
   presence; gerbang kualitas per-estimasi tanpa ECG = saran lanjutan (poin 26).

---

## Kontribusi (yang dipertahankan di sidang)

| | Klaim | Bukti |
|---|---|---|
| **K2** | Pipeline fase mentah layak estimasi BPM | 8/10 lolos, MAPE 6.5%, korelasi +0.83, vs trivial baseline menang |
| **K3** | Kriteria kelayakan SNR + karakterisasi jarak/halangan | log(SNR)~log(MAPE) r=−0.89, ambang 1.8 benar 15/17, eksperimen 50 vs 100 cm |
| **K4** | Koreksi metodologi validasi | korelasi dalam-subjek tak sah di HR konstan; MAE telanjang menyesatkan tanpa trivial baseline |

*(K1 — analisis akar kegagalan library TI — disimpan sebagai lampiran cadangan
`thesis/logger_vs_fase.pdf`, dibuka hanya bila penguji spesifik bertanya. Tesis
berdiri di K2+K3+K4 tanpa menyentuh output TI.)*

---

## Peta script → tahap (buat nulis bab metodologi)

| Tahap | Script | Output/Slide |
|---|---|---|
| 0 Batasan | — | `thesis/batasan_masalah_draft.md` |
| 1 SNR | `snr_criterion.py` | `thesis/cara_hitung_snr.pdf` |
| 2 Kontrol negatif | `negative_control.py` | `thesis/bukti_detak_jantung.pdf` |
| 3 Pipeline+MAPE | `phase_pipeline.py` | `thesis/fase_vs_ecg.pdf` |
| 4 SNR→gagal | `snr_criterion.py`, `analyze_distance.py` | `thesis/kenapa_subjek_gagal.pdf` |
| 5 Sensitivitas | `phase_pipeline.py --demo` | — |
| 6 Keterbatasan | `hrv_analysis.py`, `quality_metric.py` | `thesis/antisipasi_pertanyaan.pdf` |

Ground truth semua tahap dari deteksi R-peak Attys via
`code/preprocessing/extract_ground_truth.py`.
