"""
Uji kelayakan HRV dari unwrapPhasePeak_mm, dibandingkan HRV dari ECG Attys.

HRV butuh waktu SETIAP DETAK (beat-to-beat), bukan cuma rata-rata BPM per
jendela seperti phase_pipeline.py. Jadi ini pertanyaan yang beda dan jauh
lebih berat: pipeline BPM cuma perlu frekuensi dominan per 20 detik, HRV perlu
posisi tiap puncak dengan presisi milidetik.

Yang diuji:
  1. Berapa presisi waktu yang secara fisik MUNGKIN dari radar 50 fps?
  2. Kalau puncak detak dicari langsung di sinyal fase, berapa banyak yang
     benar-benar cocok dengan R-peak ECG?
  3. SDNN/RMSSD radar vs SDNN/RMSSD ECG.

Dua pita diuji, karena ada dilema mendasar:
  - pita SEMPIT (0.8-2.0 Hz, dipakai phase_pipeline) -> bersih, tapi hasilnya
    mendekati sinus murni. Sinus punya interval antar-puncak yang nyaris tetap,
    jadi VARIASI antar-detak (yaitu HRV itu sendiri) ikut terfilter habis.
  - pita LEBAR (0.8-8.0 Hz) -> bentuk detak lebih tajam, variasi terjaga, tapi
    noise ikut masuk.

Pemakaian:
    python code/evaluation/hrv_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from extract_ground_truth import detect_r_peaks  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FRONT = ROOT / "data" / "raw" / "position_front"

FS = 50.0                 # frame rate radar sebenarnya (poin 20)
BAND_NARROW = (0.8, 2.0)  # pita phase_pipeline
BAND_WIDE = (0.8, 8.0)    # pita lebar, pertahankan bentuk detak
MATCH_TOL = 0.15          # detik; toleransi cocok radar-beat vs R-peak


def uniform_clock(ts: np.ndarray):
    """Timestamp host terkuantisasi timer OS (15.6 ms). Frame radar sendiri
    datang tiap 20 ms. Regresi linear indeks->waktu buang kuantisasi itu."""
    n = np.arange(len(ts))
    slope, intercept = np.polyfit(n, ts, 1)
    return intercept + slope * n, 1.0 / slope


def beat_times(phase_mm, t_uni, fs, band):
    """Puncak detak dari sinyal fase -> daftar waktu detak (unix epoch)."""
    x = np.diff(phase_mm, prepend=phase_mm[0])        # turunan fase (poin 24)
    sos = signal.butter(4, band, "bandpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, x)

    # jarak minimum antar detak: 180 bpm -> 0.33 s
    peaks, _ = signal.find_peaks(x, distance=int(0.33 * fs))
    if len(peaks) < 3:
        return np.array([]), x

    # interpolasi parabolik -> presisi sub-sampel (di bawah 20 ms)
    p = peaks[(peaks > 0) & (peaks < len(x) - 1)]
    a, b, c = x[p - 1], x[p], x[p + 1]
    denom = a - 2 * b + c
    shift = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / np.where(denom == 0, 1, denom), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    return t_uni[p] + shift / fs, x


def ibi_stats(times):
    """SDNN & RMSSD (ms) dari deret waktu detak."""
    ibi = np.diff(times) * 1000.0
    ibi = ibi[(ibi > 333) & (ibi < 1500)]            # 40-180 bpm
    if len(ibi) < 5:
        return dict(n=len(ibi), mean=np.nan, sdnn=np.nan, rmssd=np.nan)
    return dict(
        n=len(ibi),
        mean=ibi.mean(),
        sdnn=ibi.std(ddof=1),
        rmssd=np.sqrt(np.mean(np.diff(ibi) ** 2)),
    )


def match_beats(t_radar, t_ecg, tol=MATCH_TOL):
    """Cocokkan tiap detak radar ke R-peak ECG terdekat. Balikin selisih waktu
    (detik) untuk yang cocok, dan proporsi R-peak yang ketemu pasangannya."""
    if len(t_radar) == 0 or len(t_ecg) == 0:
        return np.array([]), 0.0
    idx = np.searchsorted(t_ecg, t_radar)
    idx = np.clip(idx, 1, len(t_ecg) - 1)
    left, right = t_ecg[idx - 1], t_ecg[idx]
    nearest = np.where(np.abs(t_radar - left) < np.abs(t_radar - right), left, right)
    err = t_radar - nearest
    ok = np.abs(err) <= tol
    # berapa % R-peak ECG yang ke-cover
    covered = len(np.unique(nearest[ok])) / len(t_ecg)
    return err[ok], covered


def main():
    rows = []
    for d in sorted(FRONT.iterdir()):
        radar = pd.read_csv(d / "radar.csv")
        attys = pd.read_csv(d / "attys.csv")

        t_uni, fs = uniform_clock(radar["Timestamp"].values)
        phase = radar["unwrapPhasePeak_mm"].values.astype(float)

        fs_ecg = 1.0 / np.median(np.diff(attys["timestamp"].values))
        t_ecg = detect_r_peaks(attys["value"].values, attys["timestamp"].values, fs_ecg)

        ecg = ibi_stats(t_ecg)
        rec = dict(subj=d.name, fs=fs, ecg=ecg)
        for name, band in [("narrow", BAND_NARROW), ("wide", BAND_WIDE)]:
            tb, _ = beat_times(phase, t_uni, fs, band)
            err, cov = match_beats(tb, t_ecg)
            rec[name] = dict(**ibi_stats(tb), cov=cov,
                             err_ms=np.median(np.abs(err)) * 1000 if len(err) else np.nan)
        rows.append(rec)

    # ---------- laporan ----------
    W = 78
    print("=" * W)
    print("UJI KELAYAKAN HRV DARI unwrapPhasePeak_mm".center(W))
    print("=" * W)

    print(f"\nBATAS FISIK: radar {np.mean([r['fs'] for r in rows]):.1f} fps "
          f"-> 1 sampel = {1000/np.mean([r['fs'] for r in rows]):.0f} ms.")
    print("HRV istirahat (RMSSD) biasanya 20-50 ms -> sebanding 1-2 sampel radar.")
    print("Interpolasi parabolik dipakai untuk tembus di bawah batas sampel.")

    print("\n" + "-" * W)
    print("HRV DARI ECG (acuan) vs HRV DARI RADAR")
    print("-" * W)
    print(f"{'subjek':<12}{'--- ECG ---':>18}{'-- radar sempit --':>22}{'-- radar lebar --':>22}")
    print(f"{'':<12}{'SDNN':>8}{'RMSSD':>10}{'SDNN':>10}{'RMSSD':>10}{'cocok%':>8}"
          f"{'SDNN':>10}{'RMSSD':>10}{'cocok%':>8}")
    for r in rows:
        e, n, w = r["ecg"], r["narrow"], r["wide"]
        print(f"{r['subj']:<12}{e['sdnn']:>8.0f}{e['rmssd']:>10.0f}"
              f"{n['sdnn']:>10.0f}{n['rmssd']:>10.0f}{n['cov']*100:>8.0f}"
              f"{w['sdnn']:>10.0f}{w['rmssd']:>10.0f}{w['cov']*100:>8.0f}")

    def col(k, sub):
        return np.array([r[sub][k] for r in rows], float)

    print("-" * W)
    print(f"{'RATA-RATA':<12}{np.nanmean(col('sdnn','ecg')):>8.0f}{np.nanmean(col('rmssd','ecg')):>10.0f}"
          f"{np.nanmean(col('sdnn','narrow')):>10.0f}{np.nanmean(col('rmssd','narrow')):>10.0f}"
          f"{np.nanmean(col('cov','narrow'))*100:>8.0f}"
          f"{np.nanmean(col('sdnn','wide')):>10.0f}{np.nanmean(col('rmssd','wide')):>10.0f}"
          f"{np.nanmean(col('cov','wide'))*100:>8.0f}")
    print("(semua satuan ms, kecuali cocok% = R-peak ECG yang ketemu pasangan di radar)")

    # korelasi SDNN/RMSSD antar-subjek: bisa gak radar bedain orang ber-HRV
    # tinggi vs rendah, walau nilai absolutnya meleset?
    print("\n" + "-" * W)
    print("KORELASI ANTAR-SUBJEK (n=10): radar bisa bedain HRV tinggi vs rendah?")
    print("-" * W)
    for sub in ["narrow", "wide"]:
        for k in ["sdnn", "rmssd"]:
            x, y = col(k, "ecg"), col(k, sub)
            m = np.isfinite(x) & np.isfinite(y)
            r = np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 2 else np.nan
            print(f"  {sub:<8} {k.upper():<6} r = {r:+.3f}")

    # ---- vonis ----
    print("\n" + "=" * W)
    print("VONIS")
    print("=" * W)
    print(f"""
1. DETEKSI DETAK GAGAL. Cuma {np.nanmean(col('cov','narrow'))*100:.0f}% (pita sempit) / {np.nanmean(col('cov','wide'))*100:.0f}% (pita lebar)
   R-peak ECG yang ketemu pasangan di radar. Untuk HRV yang sah angka ini
   harus >95%: satu detak kelewat langsung menggabungkan dua interval jadi
   satu, dan RMSSD meledak.

2. RMSSD RADAR 4x TERLALU BESAR, di KEDUA pita:
       ECG          {np.nanmean(col('rmssd','ecg')):>5.0f} ms   (nilai fisiologis wajar saat istirahat)
       pita sempit  {np.nanmean(col('rmssd','narrow')):>5.0f} ms
       pita lebar   {np.nanmean(col('rmssd','wide')):>5.0f} ms
   Melebar di kedua pita = penyebabnya BUKAN pilihan filter, tapi deteksi
   detaknya sendiri: puncak palsu dan detak kelewat. Angka {np.nanmean(col('rmssd','narrow')):.0f} ms itu
   ukuran KESALAHAN DETEKSI, bukan ukuran variabilitas jantung.

3. Korelasi antar-subjek juga tidak menyelamatkan: SDNN r=+0.49 pada n=10
   belum bermakna secara statistik, dan pita lebar malah negatif (-0.26)
   -- pola yang tidak konsisten = kebetulan, bukan sinyal.

KESIMPULAN: HRV TIDAK BISA diklaim dari dataset ini. Dua batas keras, dan
keduanya BUKAN soal algoritma:
  (a) SNR jantung cuma 1.4-5.9 (poin 10/22) -> puncak tiap detak individual
      sering tenggelam di noise. Estimasi BPM per jendela 20 detik selamat
      karena merata-ratakan ~25 detak sekaligus; HRV harus benar di SETIAP
      detak, jadi tidak punya kemewahan itu.
  (b) Presisi waktu {1000/np.mean([r['fs'] for r in rows]):.0f} ms/sampel sebanding dengan besaran yang diukur
      (RMSSD istirahat 20-50 ms).

Ini hasil NEGATIF yang layak ditulis di tesis sebagai perluasan K3 (batas
keberlakuan): "estimasi BPM rata-rata layak, HRV belum" -- lengkap dengan
angka di atas sebagai buktinya.
""".rstrip())


if __name__ == "__main__":
    main()
