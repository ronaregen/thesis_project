# Penjelasan Tambahan — Slide Bimbingan 4 Juli 2026

Catatan pendamping `bimbingan_2026-07-04.pptx`, buat istilah/angka yang gak
jelas cuma dari slide-nya.

## Slide 4 — Empat Pendekatan Dicoba, Semua Mentok

Chart di slide ini bandingin korelasi 4 cara ekstrak HR dari radar (semakin
tinggi semakin bagus, target realistis 0.3–0.5):

| Pendekatan | Korelasi |
|---|---|
| Baseline TI (xCorr) | -0.036 |
| Fase mentah (raw) | 0.091 |
| Fase + notch napas | 0.077 |
| Filter confidence top 2% | 0.043 |

**Fase mentah (raw)**
Kolom `unwrapPhasePeak_mm` — pergerakan dada hasil baca fase radar (satuan
mm) — diolah spektral langsung, SKIP total algoritma HR bawaan TI. Caranya:
windowed-PSD (Power Spectral Density per potongan waktu), cari frekuensi
dominan di band 0.7–3.3 Hz (rentang HR manusia), itu jadi estimasi HR.
Script: `code/evaluation/analyze_phase_signal.py`.

**Fase mentah + notch napas**
Sama kayak di atas, tapi sebelum cari frekuensi dominan, dulu deteksi dulu
frekuensi napas (0.1–0.5 Hz) per window, terus "notch" (buang/nolkan)
frekuensi fundamental napas + 4 harmonic-nya yang kebetulan jatuh di band HR.
Alasan coba ini: napas amplitudonya jauh lebih gede dari micro-motion
jantung, dugaannya harmonic napas nyasar masuk band HR dan dominasi hasil
pilih frekuensi. Hasil: GAK BANTU, korelasi malah turun dikit (0.091 → 0.077).

**Filter confidence top 2%**
Kolom `confidence_heart` dari TI harusnya nunjukin seberapa yakin algoritma
sama hasil HR-nya. Dicoba filter cuma ambil 2% data dengan confidence
tertinggi (threshold ≥2.0) di dataset `position_front`, asumsi: sinyal
confidence tinggi = kualitas radar lebih bagus. Hasil: gak beda jauh dari
pakai semua data tanpa filter — kesimpulan, `confidence_heart` bukan
indikator kualitas yang valid (kemungkinan artefak internal algoritma,
kayak `final_heart_rate` yang stuck).

Kesimpulan slide: keempatnya korelasi <0.1, jauh dari ambang 0.3–0.5 yang
berarti "ada sinyal HR beneran yang bisa diekstrak".

## Slide 5 — Model ML Existing vs Baseline Trivial

Chart bandingin MAE (bpm, makin kecil makin bagus):

| | MAE (bpm) |
|---|---|
| Model ML existing | 15.0 |
| Baseline trivial (tebak rata-rata) | 11.9 |
| Target pembimbing | 5.0 |

**Trivial baseline** = model paling bego yang mungkin: prediksi angka
KONSTAN = rata-rata HR ground truth di data training, TANPA lihat fitur
radar sama sekali (gak pakai radar.csv, cuma nebak angka tetap). Ini dipakai
sbg pembanding minimum wajib — kalau model ML gak ngalahin trivial ini
secara meyakinkan, artinya model gak belajar pola HR beneran.

Fitur yang dipakai model ML existing: `heart_rate_est_fft`,
`heart_rate_est_fft_4hz`, `heartRateEst_xCorr`, `heart_rate_est_peak`,
`confidence_heart`. Split: train subjek 1-8, test subjek 9-10.

**Kenapa ini kritis**: MAE model existing (15.0) malah LEBIH JELEK dari
trivial baseline (11.9). Model yang "belajar" dari fitur radar kalah sama
model yang cuma nebak angka tetap tanpa data radar sama sekali. Ini bukti
kuat model existing gak nangkep sinyal HR apapun — konsisten sama temuan
slide 3 & 4 kalau fitur-fitur itu emang korelasinya ~0 ke ground truth.

## Slide 6 — Kenapa Ini Bisa Terjadi (GIGO)

GIGO = Garbage In, Garbage Out. Penjelasannya:

Fitur yang dipakai model ML existing (lihat slide 5) itu PERSIS kolom yang
di slide 3 kebukti korelasinya ~0 sama ground truth ECG. Karena input
gak ada sinyal HR asli, model regresi pas ditrain cuma nyerah — konvergen
ke arah prediksi rata-rata target training (mirip persis trivial baseline).
MAE hasilnya keliatan "masuk akal" (10-19 bpm, gak keliatan aneh/rusak),
padahal itu bukan bukti model belajar pola detak jantung — cuma efek alami
regresi ke mean pas fitur inputnya gak informatif.

**Poin penting buat kedepannya**: MAE doang gak cukup validasi model.
Wajib selalu sertain (a) korelasi/R², (b) banding ke trivial baseline
dengan split subjek yang sama — biar gak ketipu MAE "keliatan bagus" padahal
model gak belajar apa-apa.
