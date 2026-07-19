"""
Diagnostik: apakah sinyal fase mentah radar (`unwrapPhasePeak_mm`) punya
komponen periodik yang match ground truth heart rate (ECG), TERLEPAS dari
output BPM "jadi" milik algoritma TI (yang sudah terbukti tidak reliable,
lihat CLAUDE.md poin 1).

Kalau kolom bawaan TI gagal, belum tentu radarnya gagal menangkap sinyal --
bisa jadi algoritma DSP bawaan TI yang jelek. Script ini cek langsung sinyal
fase mentah pakai spectral estimation (Welch PSD per window), tanpa lewat
algoritma TI sama sekali.

Dua mode dibandingkan per window:
  - RAW    : dominant-frequency picking langsung di band HR (0.7-3.3 Hz)
  - NOTCH  : dulu deteksi frekuensi napas (0.1-0.5 Hz) dari sinyal mentah per
             window, notch fundamental+harmonic-nya kalau nyasar masuk band
             HR, baru pilih dominant-frequency. Rasional: amplitudo napas
             jauh lebih besar dari micro-motion jantung, harmonic-nya bisa
             mendominasi band HR dan bikin dominant-frequency salah pilih.

Catatan: hanya jalan di dataset `position_front` -- `position_variation`
TIDAK PUNYA kolom `unwrapPhasePeak_mm` sama sekali (skema kolom beda, lihat
CLAUDE.md poin 4), jadi otomatis di-skip.

Usage:
    python analyze_phase_signal.py <aligned_csv>

Contoh:
    python analyze_phase_signal.py ../../data/processed/aligned_all.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy import signal

RESAMPLE_FS = 10.0  # Hz, grid seragam untuk analisis (radar asli ~64Hz non-uniform)
HR_BAND = (0.7, 3.3)  # Hz, setara ~42-200 bpm
BREATH_BAND = (0.1, 0.5)  # Hz, setara ~6-30 napas/menit
N_HARMONICS = 4
NOTCH_Q = 15.0
WINDOW_SEC = 12.0
OVERLAP = 0.5


def estimate_breathing_freq(seg_raw: np.ndarray, fs: float):
    freqs, psd = signal.welch(seg_raw, fs=fs, nperseg=len(seg_raw))
    mask = (freqs >= BREATH_BAND[0]) & (freqs <= BREATH_BAND[1])
    if not mask.any():
        return None
    return freqs[mask][np.argmax(psd[mask])]


def suppress_breathing_harmonics(seg: np.ndarray, breath_freq: float, fs: float):
    out = seg.copy()
    for k in range(1, N_HARMONICS + 1):
        f0 = breath_freq * k
        if f0 <= HR_BAND[0] or f0 >= HR_BAND[1]:
            continue
        b, a = signal.iirnotch(f0, NOTCH_Q, fs)
        out = signal.filtfilt(b, a, out)
    return out


def pick_peak_freq(seg: np.ndarray, fs: float):
    freqs, psd = signal.welch(seg, fs=fs, nperseg=len(seg))
    mask = (freqs >= HR_BAND[0]) & (freqs <= HR_BAND[1])
    if not mask.any():
        return None
    return freqs[mask][np.argmax(psd[mask])]


def analyze_session(t: np.ndarray, phase: np.ndarray, gt: np.ndarray):
    """Windowed spectral HR estimate dari sinyal fase mentah, mode RAW vs NOTCH."""
    order = np.argsort(t)
    t, phase, gt = t[order], phase[order], gt[order]

    t_uniform = np.arange(t[0], t[-1], 1.0 / RESAMPLE_FS)
    phase_uniform = np.interp(t_uniform, t, phase)
    gt_uniform = np.interp(t_uniform, t, gt)

    sos = signal.butter(3, HR_BAND, btype="bandpass", fs=RESAMPLE_FS, output="sos")
    phase_hr = signal.sosfiltfilt(sos, phase_uniform)

    win_samples = int(WINDOW_SEC * RESAMPLE_FS)
    step = max(1, int(win_samples * (1 - OVERLAP)))

    raw_est, notch_est, gt_hr = [], [], []
    for start in range(0, len(phase_hr) - win_samples, step):
        seg_hr = phase_hr[start:start + win_samples]
        seg_raw = phase_uniform[start:start + win_samples]

        raw_freq = pick_peak_freq(seg_hr, RESAMPLE_FS)
        if raw_freq is None:
            continue

        breath_freq = estimate_breathing_freq(seg_raw, RESAMPLE_FS)
        if breath_freq is not None:
            seg_notched = suppress_breathing_harmonics(seg_hr, breath_freq, RESAMPLE_FS)
            notch_freq = pick_peak_freq(seg_notched, RESAMPLE_FS)
        else:
            notch_freq = raw_freq

        raw_est.append(raw_freq * 60.0)
        notch_est.append(notch_freq * 60.0)
        gt_hr.append(gt_uniform[start:start + win_samples].mean())

    return np.array(raw_est), np.array(notch_est), np.array(gt_hr)


def mae_corr(est, gt):
    mae = np.mean(np.abs(est - gt))
    corr = np.corrcoef(est, gt)[0, 1] if np.std(est) > 0 else float("nan")
    return mae, corr


def main(aligned_path: str):
    df = pd.read_csv(aligned_path)

    if "unwrapPhasePeak_mm" not in df.columns:
        print("Kolom 'unwrapPhasePeak_mm' tidak ada di file input. Berhenti.")
        sys.exit(1)

    df = df[(df["dataset"] == "position_front") & df["unwrapPhasePeak_mm"].notna()]
    if df.empty:
        print("Tidak ada baris position_front dengan unwrapPhasePeak_mm valid.")
        sys.exit(1)

    print(f"{'Subjek':<12s} {'n_win':>6s} {'MAE raw':>9s} {'corr raw':>9s} "
          f"{'MAE notch':>10s} {'corr notch':>11s}")
    print("-" * 60)

    all_raw, all_notch, all_gt = [], [], []
    for subject_id, sub_df in df.groupby("subject_id"):
        raw_est, notch_est, gt_hr = analyze_session(
            sub_df["Timestamp"].values,
            sub_df["unwrapPhasePeak_mm"].values,
            sub_df["gt_heart_rate"].values,
        )
        if len(raw_est) == 0:
            print(f"{subject_id:<12s} {'0':>6s} {'n/a':>9s} {'n/a':>9s} {'n/a':>10s} {'n/a':>11s}")
            continue
        mae_raw, corr_raw = mae_corr(raw_est, gt_hr)
        mae_notch, corr_notch = mae_corr(notch_est, gt_hr)
        print(f"{subject_id:<12s} {len(raw_est):>6d} {mae_raw:>9.1f} {corr_raw:>9.3f} "
              f"{mae_notch:>10.1f} {corr_notch:>11.3f}")
        all_raw.append(raw_est)
        all_notch.append(notch_est)
        all_gt.append(gt_hr)

    if all_raw:
        all_raw = np.concatenate(all_raw)
        all_notch = np.concatenate(all_notch)
        all_gt = np.concatenate(all_gt)
        mae_raw, corr_raw = mae_corr(all_raw, all_gt)
        mae_notch, corr_notch = mae_corr(all_notch, all_gt)
        print("-" * 60)
        print(f"{'SEMUA':<12s} {len(all_raw):>6d} {mae_raw:>9.1f} {corr_raw:>9.3f} "
              f"{mae_notch:>10.1f} {corr_notch:>11.3f}")

    print("\nPembanding (dari compare_baseline.py, agregat position_front):")
    print("  heart_rate_est_peak : MAE ~69 bpm, korelasi ~0.42 (tapi 75% stuck di modus)")
    print("  heart_rate_est_fft  : MAE ~31 bpm, korelasi ~-0.35")
    print("\nInterpretasi: kalau kolom 'notch' jelas lebih baik dari 'raw' (korelasi")
    print("naik ke 0.3-0.5+), breathing harmonic memang sumber masalah utama --")
    print("lanjut ke feature engineering + model ML pakai teknik ini. Kalau notch")
    print("gak beda jauh dari raw, breathing harmonic bukan penyebab utama --")
    print("kemungkinan besar masalah ada di akuisisi radar, bukan cuma software.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
