"""
PERBANDINGAN UTAMA TESIS: keluaran BPM logger (TI) vs pipeline fase mentah.

Pertanyaan riset (disetujui pembimbing, 14 Jul 2026):
    "Pada konfigurasi akuisisi yang dipakai (50 fps), seberapa besar perbaikan
     yang didapat dengan mengolah `unwrapPhasePeak_mm` sendiri, dibanding
     memakai kolom BPM yang sudah jadi dari logger?"

BATASAN (WAJIB ditulis di tesis - lihat CLAUDE.md poin 25):
    Ini BUKAN vonis atas library Vital Signs TI. Data direkam pada 50 fps
    sementara library TI dirancang untuk 20 fps, sehingga rantai filternya
    mulur 2.5x dan justru MEMBUANG detak jantung manusia (CLAUDE.md poin 20).
    Yang dibandingkan adalah: pada kondisi akuisisi INI, sumber sinyal mana
    yang masih bisa dipakai.

Kenapa perbandingan ini tetap sah dan menarik: `unwrapPhasePeak_mm` diambil
SEBELUM rantai filter TI, jadi dia kebal terhadap salah-konfigurasi itu.
Kolom BPM turunan TI tidak. Itu temuan teknik yang bisa digeneralisasi:
ambil sinyal sedini mungkin, sebelum filter vendor yang mengasumsikan
sample rate tetap.

Tiga hal yang diukur, di jendela yang PERSIS SAMA untuk semua metode
(panjangnya = phase_pipeline.WINDOW_SEC):
  1. MAPE/MAE per subjek        -> seberapa akurat
  2. korelasi ANTAR-subjek      -> bisakah metode ini membedakan orang ber-HR
                                   tinggi dari yang rendah? (bukti "detak
                                   jantung benar-benar tertangkap")
  3. vs trivial baseline        -> apakah metode ini mengalahkan menebak satu
                                   angka konstan tanpa radar sama sekali?

Usage:
    python compare_ti_vs_phase.py ../../data/processed/aligned_all.csv
"""

import sys

import numpy as np
import pandas as pd

from phase_pipeline import (FS, MAPE_STANDARD, OVERLAP, TEST, TRAIN,
                            WINDOW_SEC, estimate, resample_antialias)

# Semua kolom BPM yang dikeluarkan logger untuk position_front.
TI_COLS = [
    "final_heart_rate",
    "heart_rate_est_peak",
    "heart_rate_est_fft",
    "heart_rate_est_fft_4hz",
    "heartRateEst_xCorr",
]


def ti_windows(t, col, gt):
    """Rata-ratakan kolom BPM logger di jendela yang IDENTIK dengan
    phase_pipeline.estimate() -- supaya perbandingannya apel-ke-apel."""
    tu, xu, order = resample_antialias(t, col)
    gu = np.interp(tu, np.sort(t), gt[order])
    w = int(WINDOW_SEC * FS)
    step = max(1, int(w * (1 - OVERLAP)))
    est, ref = [], []
    for s in range(0, len(xu) - w, step):
        est.append(float(np.nanmean(xu[s:s + w])))
        ref.append(float(gu[s:s + w].mean()))
    return np.array(est), np.array(ref)


def mae(e, r):
    return float(np.nanmean(np.abs(e - r)))


def mape(e, r):
    return 100 * float(np.nanmean(np.abs(e - r) / r))


def compare(front):
    """Balikin DataFrame per subjek + dict ringkasan. Dipakai juga oleh slide."""
    rows = []
    for sid, sub in front.groupby("subject_id"):
        t = sub.Timestamp.values
        gt = sub.gt_heart_rate.values
        est, ref, snr = estimate(t, sub.unwrapPhasePeak_mm.values, gt)

        rec = {"subj": sid, "hr_ecg": float(ref.mean()), "std_ecg": float(ref.std()),
               "snr": snr, "fase_mae": mae(est, ref), "fase_mape": mape(est, ref),
               "fase_bpm": float(est.mean())}
        for c in TI_COLS:
            e, r = ti_windows(t, sub[c].values.astype(float), gt)
            rec[f"{c}_mae"] = mae(e, r)
            rec[f"{c}_mape"] = mape(e, r)
            rec[f"{c}_bpm"] = float(np.nanmean(e))
        rows.append(rec)

    R = pd.DataFrame(rows).sort_values("subj").reset_index(drop=True)
    best = min(TI_COLS, key=lambda c: R[f"{c}_mape"].mean())

    # Korelasi ANTAR-subjek: sumbu-x = HR asli tiap orang, sumbu-y = HR estimasi.
    # Ini metrik yang SAH untuk dataset ini; korelasi DALAM-subjek tidak bermakna
    # karena HR tiap subjek nyaris konstan (std 0.9-4.2 bpm, CLAUDE.md poin 12).
    r_fase = float(np.corrcoef(R.hr_ecg, R.fase_bpm)[0, 1])
    r_best = float(np.corrcoef(R.hr_ecg, R[f"{best}_bpm"])[0, 1])

    # BUKTI bahwa kolom logger cuma PENEBAK-KONSTAN.
    # Ambil konstanta = rata-rata keluaran kolom itu sendiri, lalu hitung MAPE-nya
    # kalau kita cuma menebak konstanta itu untuk semua orang. Kalau MAPE penebak-
    # konstan ini berkorelasi ~+1 dengan MAPE kolom aslinya, kolom itu TIDAK
    # membawa informasi apa pun di luar konstantanya.
    # (Ini yang menjelaskan kenapa kolom logger kadang "menang" di subjek yang
    #  HR-nya kebetulan dekat konstanta itu -- jam mati benar dua kali sehari.)
    c = R[f"{best}_bpm"].mean()
    mape_stuck = (c - R.hr_ecg).abs() / R.hr_ecg * 100
    r_stuck = float(np.corrcoef(R[f"{best}_mape"], mape_stuck)[0, 1])

    # trivial baseline per subjek, JUJUR (konstanta = rata-rata 9 subjek LAIN)
    tri = np.array([abs(R.hr_ecg.drop(i).mean() - v) / v * 100
                    for i, v in R.hr_ecg.items()])
    R["trivial_mape"] = tri

    return R, {
        "best": best,
        "r_fase": r_fase,
        "r_best": r_best,
        "bias_fase": float((R.fase_bpm - R.hr_ecg).mean()),
        "bias_best": float((R[f"{best}_bpm"] - R.hr_ecg).mean()),
        "stuck_const": float(c),
        "r_stuck": r_stuck,
        "trivial_mape": float(tri.mean()),
    }


def baseline(front):
    """Pembanding wajib (CLAUDE.md poin 8): apakah kita mengalahkan tebak-konstan?
    Split BERBASIS SUBJEK: train S01-08, test S09-10."""
    const = front[front.subject_id.isin(TRAIN)].gt_heart_rate.mean()
    out = {"const": float(const)}
    E, R_, X = [], [], []
    for s in TEST:
        sub = front[front.subject_id == s]
        t, gt = sub.Timestamp.values, sub.gt_heart_rate.values
        e, r, _ = estimate(t, sub.unwrapPhasePeak_mm.values, gt)
        x, _ = ti_windows(t, sub.heartRateEst_xCorr.values.astype(float), gt)
        E.append(e)
        R_.append(r)
        X.append(x)
    e, r, x = np.concatenate(E), np.concatenate(R_), np.concatenate(X)
    c = np.full_like(r, const)
    for k, v in [("triv", c), ("xcorr", x), ("fase", e)]:
        out[f"{k}_mae"] = mae(v, r)
        out[f"{k}_mape"] = mape(v, r)
    return out


def main(path):
    df = pd.read_csv(path)
    front = df[(df.dataset == "position_front") & df.unwrapPhasePeak_mm.notna()]
    R, S = compare(front)
    B = baseline(front)
    best = S["best"]

    W = 104
    print("=" * W)
    print("KELUARAN BPM LOGGER (TI)  vs  PIPELINE FASE MENTAH".center(W))
    print(f"jendela {WINDOW_SEC:.0f} detik yang SAMA PERSIS untuk semua metode".center(W))
    print("=" * W)
    print("\nBATASAN: data direkam 50 fps, library TI dirancang 20 fps -> rantai filter TI")
    print("mulur 2.5x dan membuang detak jantung. Ini BUKAN vonis atas library TI, tapi")
    print("perbandingan sumber sinyal mana yang masih terpakai PADA KONFIGURASI INI.\n")

    print("-" * W)
    print("1. AKURASI -- MAPE (%) per subjek")
    print("-" * W)
    hdr = f"{'subjek':<10}{'ECG':>6}{'SNR':>6} |"
    for c in TI_COLS:
        hdr += f"{c[:13]:>14}"
    print(hdr + f"{'FASE':>10}{'trivial':>11}")
    print("-" * W)
    for _, x in R.iterrows():
        line = f"{x.subj:<10}{x.hr_ecg:>6.0f}{x.snr:>6.2f} |"
        for c in TI_COLS:
            line += f"{x[f'{c}_mape']:>13.0f}%"
        print(line + f"{x.fase_mape:>9.1f}%{x.trivial_mape:>10.1f}%")
    print("-" * W)
    line = f"{'RATA2':<10}{R.hr_ecg.mean():>6.0f}{R.snr.mean():>6.2f} |"
    for c in TI_COLS:
        line += f"{R[f'{c}_mape'].mean():>13.0f}%"
    print(line + f"{R.fase_mape.mean():>9.1f}%{S['trivial_mape']:>10.1f}%")
    line = f"{'LOLOS<10%':<10}{'':>12} |"
    for c in TI_COLS:
        line += f"{(R[f'{c}_mape'] < MAPE_STANDARD).sum():>10d}/10 "
    print(line + f"{(R.fase_mape < MAPE_STANDARD).sum():>7d}/10")

    print(f"\nKolom logger TERBAIK: {best}")
    print(f"   {best:<24s} MAPE {R[f'{best}_mape'].mean():>5.1f}%   MAE {R[f'{best}_mae'].mean():>5.1f} bpm")
    print(f"   {'pipeline fase':<24s} MAPE {R.fase_mape.mean():>5.1f}%   MAE {R.fase_mae.mean():>5.1f} bpm")
    print(f"   -> perbaikan {R[f'{best}_mape'].mean() / R.fase_mape.mean():.1f}x (MAPE), "
          f"{R[f'{best}_mae'].mean() / R.fase_mae.mean():.1f}x (MAE)")

    print("\n" + "-" * W)
    print("2. BUKTI DETAK JANTUNG BENAR-BENAR TERTANGKAP")
    print("   Korelasi ANTAR-subjek (n=10): bisakah metode membedakan orang")
    print("   ber-HR tinggi dari yang rendah?")
    print("-" * W)
    print(f"{'subjek':<10}{'ECG':>8}{'FASE':>8}{best[:12]:>14}")
    for _, x in R.iterrows():
        print(f"{x.subj:<10}{x.hr_ecg:>8.1f}{x.fase_bpm:>8.1f}{x[f'{best}_bpm']:>14.1f}")
    print("-" * W)
    print(f"   pipeline fase : r = {S['r_fase']:+.3f}   bias {S['bias_fase']:+.1f} bpm")
    print(f"   {best:<14s}: r = {S['r_best']:+.3f}   bias {S['bias_best']:+.1f} bpm")
    print("\n   Korelasi NEGATIF pada kolom logger = bukan sekadar 'kurang akurat',")
    print("   melainkan TIDAK menangkap apa pun. Pipeline fase r positif kuat.")
    print(f"\n   std keluaran: {best} = {R[f'{best}_bpm'].std():.1f} bpm, "
          f"fase = {R.fase_bpm.std():.1f} bpm, HR asli = {R.hr_ecg.std():.1f} bpm")
    print(f"   -> {best} nyaris tidak berani berbeda antar orang.")
    print(f"\n   BUKTI dia cuma PENEBAK-KONSTAN {S['stuck_const']:.0f} bpm:")
    print(f"   korelasi MAPE-nya dengan MAPE penebak-konstan = {S['r_stuck']:+.3f}")
    print("   -> mendekati +1 = kolom itu tidak membawa informasi di luar konstantanya.")
    print("   Itu sebabnya dia kadang 'menang' di subjek yang HR-nya kebetulan dekat")
    print("   konstanta itu -- jam mati pun benar dua kali sehari.")

    print("\n" + "-" * W)
    print(f"3. PEMBANDING WAJIB: trivial baseline (train S01-08 / test S09-10)")
    print("-" * W)
    print(f"{'metode':<36}{'MAE':>8}{'MAPE':>8}")
    triv_name = f"Trivial (tebak konstan {B['const']:.0f} bpm)"
    print(f"{triv_name:<36}{B['triv_mae']:>8.1f}{B['triv_mape']:>7.1f}%")
    print(f"{'heartRateEst_xCorr (logger)':<36}{B['xcorr_mae']:>8.1f}{B['xcorr_mape']:>7.1f}%")
    print(f"{'Pipeline fase (kita)':<36}{B['fase_mae']:>8.1f}{B['fase_mape']:>7.1f}%")
    print(f"\n   xCorr {'KALAH DARI' if B['xcorr_mae'] > B['triv_mae'] else 'mengalahkan'} tebak-konstan"
          " -> radar TIDAK memberi informasi apa pun lewat kolom itu.")
    print(f"   Fase  {'MENGALAHKAN' if B['fase_mae'] < B['triv_mae'] else 'kalah dari'} tebak-konstan"
          " -> radar MEMANG membawa informasi detak jantung.")

    # cek-mandiri: klaim inti tesis harus tetap benar kalau skrip ini dijalankan lagi
    assert R.fase_mape.mean() < R[f"{best}_mape"].mean(), "pipeline fase harus mengalahkan kolom TI terbaik"
    assert S["r_fase"] > 0 > S["r_best"], "korelasi antar-subjek: fase positif, TI negatif"
    assert B["fase_mae"] < B["triv_mae"] < B["xcorr_mae"], "urutan vs trivial baseline berubah"
    assert S["r_stuck"] > 0.8, "kolom TI terbaik harus terbukti setara penebak-konstan"
    assert R.fase_mape.mean() < S["trivial_mape"], "fase harus mengalahkan trivial baseline (MAPE rata-rata)"
    print("\n(cek-mandiri: 5 klaim inti terverifikasi)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
