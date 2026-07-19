"""
Root-cause diagnostik: KENAPA semua pendekatan ekstraksi HR mentok di korelasi ~0.

Beda dari script lain: ini gak nyoba bikin estimator HR baru. Ini ngetes apakah
informasi HR-nya EMANG ADA di sinyal radar. Kalau gak ada, gak ada algoritma
apapun (termasuk ML) yang bisa mengekstraknya.

Tes yang dijalankan:
  1. Frame rate aktual radar vs spek TI (20 Hz slow-time sampling)
  2. Stabilitas range-bin target (TI mengasumsikan subjek tetap di range-bin sama)
  3. Amplitudo chest displacement: band napas (0.1-0.5Hz) vs band HR (0.8-2.0Hz),
     dibanding spek TI (napas 1-12mm, HR 0.1-0.5mm)
  4. TES DESISIF: per window, ranking bin frekuensi ground-truth di dalam PSD
     band HR. Kalau sinyal HR beneran ada, bin GT harusnya sering jadi peak
     (rank 1) atau dekat. Kalau rank-nya terdistribusi UNIFORM = sinyal HR tidak
     ada sama sekali di fase radar -> masalah akuisisi, bukan algoritma.
  5. SNR HR-band: tinggi peak di frekuensi GT vs median PSD sekitarnya

Usage:
    python diagnose_root_cause.py <aligned_csv> [outdir]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

RESAMPLE_FS = 20.0          # Hz, sesuai spek slow-time sampling TI
HR_BAND = (0.8, 2.0)        # Hz, band HR sesuai spek TI (48-120 bpm)
BREATH_BAND = (0.1, 0.5)    # Hz
WINDOW_SEC = 16.0           # detik, sesuai window processing TI
OVERLAP = 0.5
STUCK_VALUE = 5.859375      # bpm, nilai stuck final_heart_rate/heart_rate_est_peak


def band_rms(x, fs, band):
    """RMS amplitudo sinyal di dalam band frekuensi tertentu."""
    sos = signal.butter(3, band, btype="bandpass", fs=fs, output="sos")
    return float(np.sqrt(np.mean(signal.sosfiltfilt(sos, x) ** 2)))


def gt_bin_rank(seg, fs, gt_bpm):
    """
    Ranking bin frekuensi ground-truth di PSD band HR.
    rank 1 = bin GT adalah peak tertinggi (sinyal HR kuat & benar).
    Return (rank, n_bins, snr) atau None kalau GT di luar band.
    """
    gt_hz = gt_bpm / 60.0
    if not (HR_BAND[0] <= gt_hz <= HR_BAND[1]):
        return None

    freqs, psd = signal.welch(seg, fs=fs, nperseg=len(seg))
    mask = (freqs >= HR_BAND[0]) & (freqs <= HR_BAND[1])
    f_band, p_band = freqs[mask], psd[mask]
    if len(f_band) < 3:
        return None

    gt_idx = int(np.argmin(np.abs(f_band - gt_hz)))
    # rank: berapa banyak bin yang power-nya LEBIH TINGGI dari bin GT, +1
    rank = int(np.sum(p_band > p_band[gt_idx])) + 1
    snr = float(p_band[gt_idx] / np.median(p_band))
    return rank, len(f_band), snr


def analyze_subject(sub_df):
    t = sub_df["Timestamp"].values
    order = np.argsort(t)
    t = t[order]
    phase = sub_df["unwrapPhasePeak_mm"].values[order]
    gt = sub_df["gt_heart_rate"].values[order]

    dt = np.diff(t)
    dt = dt[(dt > 0) & (dt < 1.0)]

    t_u = np.arange(t[0], t[-1], 1.0 / RESAMPLE_FS)
    phase_u = np.interp(t_u, t, phase)
    gt_u = np.interp(t_u, t, gt)

    win = int(WINDOW_SEC * RESAMPLE_FS)
    step = max(1, int(win * (1 - OVERLAP)))

    ranks, norm_ranks, snrs, breath_amp, hr_amp = [], [], [], [], []
    for s in range(0, len(phase_u) - win, step):
        seg = phase_u[s:s + win]
        seg = signal.detrend(seg)
        gt_w = float(gt_u[s:s + win].mean())

        breath_amp.append(band_rms(seg, RESAMPLE_FS, BREATH_BAND))
        hr_amp.append(band_rms(seg, RESAMPLE_FS, HR_BAND))

        r = gt_bin_rank(seg, RESAMPLE_FS, gt_w)
        if r is not None:
            rank, n_bins, snr = r
            ranks.append(rank)
            norm_ranks.append((rank - 1) / (n_bins - 1))  # 0=peak, 1=terlemah
            snrs.append(snr)

    return {
        "n_rows": int(len(sub_df)),
        "fs_median": float(1.0 / np.median(dt)) if len(dt) else float("nan"),
        "fs_mean": float(1.0 / np.mean(dt)) if len(dt) else float("nan"),
        "n_windows": len(ranks),
        "ranks": ranks,
        "norm_ranks": norm_ranks,
        "snrs": snrs,
        "breath_amp_mm": float(np.median(breath_amp)) if breath_amp else float("nan"),
        "hr_amp_mm": float(np.median(hr_amp)) if hr_amp else float("nan"),
        "gt_mean": float(np.mean(gt)),
        "gt_std": float(np.std(gt)),
    }


def main(aligned_path, outdir="."):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(aligned_path)
    front = df[df["dataset"] == "position_front"].copy()

    out = {"per_subject": {}}

    # --- Tes 1-3 + 4-5 per subjek ---
    all_norm_ranks, all_snrs = [], []
    for sid, sub in front.groupby("subject_id"):
        sub = sub[sub["unwrapPhasePeak_mm"].notna()]
        if len(sub) < 1000:
            continue
        res = analyze_subject(sub)
        all_norm_ranks.extend(res["norm_ranks"])
        all_snrs.extend(res["snrs"])
        out["per_subject"][sid] = {k: v for k, v in res.items()
                                   if k not in ("ranks", "norm_ranks", "snrs")}
        out["per_subject"][sid]["median_norm_rank"] = (
            float(np.median(res["norm_ranks"])) if res["norm_ranks"] else float("nan"))
        out["per_subject"][sid]["median_snr"] = (
            float(np.median(res["snrs"])) if res["snrs"] else float("nan"))
        out["per_subject"][sid]["pct_rank_top3"] = (
            100 * float(np.mean(np.array(res["ranks"]) <= 3)) if res["ranks"] else float("nan"))

    out["aggregate"] = {
        "n_windows": len(all_norm_ranks),
        "median_norm_rank": float(np.median(all_norm_ranks)),
        "mean_norm_rank": float(np.mean(all_norm_ranks)),
        "median_snr": float(np.median(all_snrs)),
        "pct_snr_above_2": 100 * float(np.mean(np.array(all_snrs) > 2.0)),
    }
    out["_norm_ranks"] = all_norm_ranks
    out["_snrs"] = all_snrs

    # --- Tes 2: stabilitas range bin ---
    rb = {}
    for sid, sub in front.groupby("subject_id"):
        idx = sub["rangeBinPhaseIndex"].dropna().values
        if len(idx) < 10:
            continue
        rb[sid] = {
            "n_unique_bins": int(len(np.unique(idx))),
            "pct_changed": 100 * float(np.mean(np.diff(idx) != 0)),
            "mode_bin": float(pd.Series(idx).mode().iloc[0]),
            "pct_at_mode": 100 * float(np.mean(idx == pd.Series(idx).mode().iloc[0])),
        }
    out["range_bin"] = rb

    # --- Tes: stuck value ---
    stuck = {}
    for col in ["final_heart_rate", "heart_rate_est_peak"]:
        v = front[col].dropna().values
        if len(v):
            stuck[col] = {
                "pct_stuck": 100 * float(np.mean(v == STUCK_VALUE)),
                "n_unique": int(len(np.unique(v))),
            }
    out["stuck"] = stuck
    out["stuck_value_arithmetic"] = {
        "value_bpm": STUCK_VALUE,
        "value_hz": STUCK_VALUE / 60.0,
        "fft_bin_at_20hz_1024pt": (STUCK_VALUE / 60.0) / (20.0 / 1024),
    }

    with open(outdir / "root_cause.json", "w") as f:
        json.dump({k: v for k, v in out.items() if not k.startswith("_")}, f, indent=2)

    # --- print ringkas ---
    print("=== FRAME RATE AKTUAL vs SPEK TI (20 Hz) ===")
    for sid, r in out["per_subject"].items():
        print(f"  {sid}: median {r['fs_median']:.1f} Hz, mean {r['fs_mean']:.1f} Hz")

    print("\n=== AMPLITUDO CHEST DISPLACEMENT (spek TI: napas 1-12mm, HR 0.1-0.5mm) ===")
    print(f"  {'subjek':<12s} {'napas (mm)':>12s} {'HR band (mm)':>14s} {'rasio':>8s}")
    for sid, r in out["per_subject"].items():
        ratio = r["breath_amp_mm"] / r["hr_amp_mm"] if r["hr_amp_mm"] else float("nan")
        print(f"  {sid:<12s} {r['breath_amp_mm']:>12.3f} {r['hr_amp_mm']:>14.3f} {ratio:>8.1f}x")

    print("\n=== STABILITAS RANGE BIN (TI asumsi: subjek tetap di bin sama) ===")
    print(f"  {'subjek':<12s} {'n_bin unik':>11s} {'% berubah':>10s} {'% di modus':>11s}")
    for sid, r in rb.items():
        print(f"  {sid:<12s} {r['n_unique_bins']:>11d} {r['pct_changed']:>9.1f}% {r['pct_at_mode']:>10.1f}%")

    print("\n=== TES DESISIF: RANK FREKUENSI GROUND-TRUTH DI PSD BAND HR ===")
    print("  (0.0 = bin GT selalu jadi peak tertinggi = sinyal HR kuat)")
    print("  (0.5 = bin GT rata-rata di tengah = TIDAK ADA sinyal HR, murni acak)")
    print(f"  {'subjek':<12s} {'n_win':>6s} {'median rank ternorm':>21s} {'% rank<=3':>11s} {'median SNR':>11s}")
    for sid, r in out["per_subject"].items():
        print(f"  {sid:<12s} {r['n_windows']:>6d} {r['median_norm_rank']:>21.3f} "
              f"{r['pct_rank_top3']:>10.1f}% {r['median_snr']:>11.2f}")
    a = out["aggregate"]
    print(f"  {'-'*66}")
    print(f"  {'SEMUA':<12s} {a['n_windows']:>6d} {a['median_norm_rank']:>21.3f} "
          f"{'':>11s} {a['median_snr']:>11.2f}")
    print(f"\n  Median rank ternormalisasi agregat: {a['median_norm_rank']:.3f}")
    print(f"  Nilai acak murni (chance level)   : 0.500")
    print(f"  Median SNR di frekuensi GT        : {a['median_snr']:.2f} (1.0 = setara noise)")

    print("\n=== NILAI STUCK ===")
    print(f"  {STUCK_VALUE} bpm = {STUCK_VALUE/60:.8f} Hz")
    print(f"  = persis bin ke-{out['stuck_value_arithmetic']['fft_bin_at_20hz_1024pt']:.0f} "
          f"dari FFT 1024-titik @ 20 Hz (df = {20/1024:.8f} Hz)")
    for col, s in stuck.items():
        print(f"  {col}: {s['pct_stuck']:.1f}% stuck, cuma {s['n_unique']} nilai unik")

    return out


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else ".")
