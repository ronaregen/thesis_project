"""
Estimator HR yang DIPERBAIKI dari sinyal fase mentah, plus analisis kenapa
korelasi bukan metrik yang tepat buat dataset ini.

Perbaikan vs `analyze_phase_signal.py` (yang dapet korelasi ~0):
  1. BAND: 0.8-2.0 Hz (48-120 bpm, sesuai spek TI) -- BUKAN 0.7-3.3 Hz.
     Band lebar bikin argmax kepilih harmonic napas / motion, padahal peak HR
     asli ADA (lihat diagnose_root_cause.py: median rank ternorm 0.158).
  2. DECIMATE: pakai anti-alias filter sebelum turun ke 20 Hz -- BUKAN np.interp
     langsung dari 64 Hz ke 10 Hz (itu aliasing: konten >5 Hz kelipat balik
     masuk band HR).
  3. DETREND per window (buang drift fase).
  4. SMOOTHING: median filter di deret estimasi (HR fisiologis gak lompat).

Juga menghitung:
  - Trivial baseline (prediksi konstan = mean GT training) buat pembanding wajib.
  - CEILING KORELASI: karena std GT dalam sesi cuma 2-7 bpm (subjek diam,
    istirahat), korelasi secara matematis dibatasi rendah walaupun estimator
    bagus. Simulasi: estimator sempurna + noise X bpm -> korelasi jadi berapa.

Usage:
    python corrected_estimator.py <aligned_csv>
"""

import sys

import numpy as np
import pandas as pd
from scipy import signal

FS = 20.0
HR_BAND = (0.8, 2.0)
WINDOW_SEC = 16.0
OVERLAP = 0.75
MEDFILT_K = 5

TRAIN_SUBJECTS = [f"subject{i:02d}" for i in range(1, 9)]
TEST_SUBJECTS = ["subject09", "subject10"]


def resample_uniform(t, x, fs=FS):
    """Resample ke grid seragam dengan anti-alias (bukan interp telanjang)."""
    order = np.argsort(t)
    t, x = t[order], x[order]
    fs_orig = 1.0 / np.median(np.diff(t))
    # low-pass di bawah Nyquist target dulu, baru interpolasi
    sos = signal.butter(6, fs / 2 * 0.9, btype="low", fs=fs_orig, output="sos")
    x_lp = signal.sosfiltfilt(sos, x)
    t_u = np.arange(t[0], t[-1], 1.0 / fs)
    return t_u, np.interp(t_u, t, x_lp)


def estimate_hr(t, phase, gt):
    t_u, phase_u = resample_uniform(t, phase)
    gt_u = np.interp(t_u, np.sort(t), gt[np.argsort(t)])

    sos = signal.butter(4, HR_BAND, btype="bandpass", fs=FS, output="sos")
    hr_sig = signal.sosfiltfilt(sos, signal.detrend(phase_u))

    win = int(WINDOW_SEC * FS)
    step = max(1, int(win * (1 - OVERLAP)))

    est, ref = [], []
    for s in range(0, len(hr_sig) - win, step):
        seg = signal.detrend(hr_sig[s:s + win])
        freqs, psd = signal.welch(seg, fs=FS, nperseg=win, noverlap=win // 2)
        m = (freqs >= HR_BAND[0]) & (freqs <= HR_BAND[1])
        if not m.any():
            continue
        f_band, p_band = freqs[m], psd[m]
        pk = int(np.argmax(p_band))
        # interpolasi parabolik biar resolusi gak dibatasi grid FFT
        if 0 < pk < len(p_band) - 1:
            a, b, c = np.log(p_band[pk - 1:pk + 2] + 1e-20)
            offset = 0.5 * (a - c) / (a - 2 * b + c)
            f_pk = f_band[pk] + offset * (f_band[1] - f_band[0])
        else:
            f_pk = f_band[pk]
        est.append(f_pk * 60.0)
        ref.append(float(gt_u[s:s + win].mean()))

    est = np.array(est)
    ref = np.array(ref)
    if len(est) >= MEDFILT_K:
        est = signal.medfilt(est, MEDFILT_K)
    return est, ref


def stats(est, gt):
    mae = float(np.mean(np.abs(est - gt)))
    corr = float(np.corrcoef(est, gt)[0, 1]) if np.std(est) > 0 else float("nan")
    return mae, corr


def correlation_ceiling(gt, noise_bpm):
    """Estimator SEMPURNA + noise gaussian -> korelasi teoretis yang bisa dicapai."""
    rng = np.random.default_rng(0)
    est = gt + rng.normal(0, noise_bpm, len(gt))
    return float(np.corrcoef(est, gt)[0, 1])


def main(path):
    df = pd.read_csv(path)
    front = df[(df["dataset"] == "position_front") & df["unwrapPhasePeak_mm"].notna()]

    print("=== ESTIMATOR DIPERBAIKI (band 0.8-2.0 Hz, anti-alias, detrend, medfilt) ===")
    print(f"{'subjek':<12s} {'n_win':>6s} {'MAE':>7s} {'korelasi':>9s} {'std GT':>8s} "
          f"{'ceiling korelasi*':>18s}")
    print("-" * 68)

    per_subj, all_est, all_gt = {}, [], []
    for sid, sub in front.groupby("subject_id"):
        est, gt = estimate_hr(sub["Timestamp"].values,
                              sub["unwrapPhasePeak_mm"].values,
                              sub["gt_heart_rate"].values)
        if len(est) < 10:
            continue
        mae, corr = stats(est, gt)
        ceil = correlation_ceiling(gt, mae)
        per_subj[sid] = (mae, corr, float(np.std(gt)), ceil)
        all_est.append(est)
        all_gt.append(gt)
        print(f"{sid:<12s} {len(est):>6d} {mae:>7.1f} {corr:>9.3f} {np.std(gt):>8.1f} "
              f"{ceil:>18.3f}")

    all_est = np.concatenate(all_est)
    all_gt = np.concatenate(all_gt)
    mae_a, corr_a = stats(all_est, all_gt)
    print("-" * 68)
    print(f"{'SEMUA':<12s} {len(all_est):>6d} {mae_a:>7.1f} {corr_a:>9.3f} "
          f"{np.std(all_gt):>8.1f} {correlation_ceiling(all_gt, mae_a):>18.3f}")
    print("\n* ceiling = korelasi yang didapat estimator SEMPURNA (unbiased) dengan")
    print("  noise sebesar MAE-nya sendiri, pada GT sesi ini. Ini batas atas realistis:")
    print("  kalau std GT cuma 3-4 bpm, korelasi TIDAK BISA tinggi walau estimator bagus.")

    # --- per-subject bias: apakah error-nya offset konstan (bisa dikalibrasi)? ---
    print("\n=== BIAS PER SUBJEK (error = est - GT) ===")
    print("Kalau bias-nya konsisten per subjek, itu offset yang bisa dikalibrasi,")
    print("bukan kegagalan mendeteksi HR.")
    print(f"{'subjek':<12s} {'MAE':>7s} {'MAE setelah buang bias':>24s}")
    print("-" * 46)
    for sid, sub in front.groupby("subject_id"):
        est, gt = estimate_hr(sub["Timestamp"].values,
                              sub["unwrapPhasePeak_mm"].values,
                              sub["gt_heart_rate"].values)
        if len(est) < 10:
            continue
        bias = np.median(est - gt)
        mae_raw = np.mean(np.abs(est - gt))
        mae_deb = np.mean(np.abs((est - bias) - gt))
        print(f"{sid:<12s} {mae_raw:>7.1f} {mae_deb:>24.1f}")

    # --- trivial baseline pembanding wajib ---
    print("\n=== PEMBANDING WAJIB: TRIVIAL BASELINE (split subjek 1-8 / 9-10) ===")
    tr = front[front["subject_id"].isin(TRAIN_SUBJECTS)]["gt_heart_rate"]
    te = front[front["subject_id"].isin(TEST_SUBJECTS)]["gt_heart_rate"]
    const = tr.mean()
    mae_triv = float(np.mean(np.abs(te - const)))
    print(f"Prediksi konstan = {const:.1f} bpm (mean GT training, TANPA fitur radar)")
    print(f"MAE trivial baseline di test set : {mae_triv:.1f} bpm")
    print(f"Korelasi trivial baseline        : 0.000 (konstan, per definisi)")

    est_te, gt_te = [], []
    for sid in TEST_SUBJECTS:
        sub = front[front["subject_id"] == sid]
        e, g = estimate_hr(sub["Timestamp"].values,
                           sub["unwrapPhasePeak_mm"].values,
                           sub["gt_heart_rate"].values)
        est_te.append(e)
        gt_te.append(g)
    est_te = np.concatenate(est_te)
    gt_te = np.concatenate(gt_te)
    mae_te, corr_te = stats(est_te, gt_te)
    print(f"\nEstimator fase diperbaiki di test set (subject09+10):")
    print(f"  MAE      : {mae_te:.1f} bpm  (trivial: {mae_triv:.1f} bpm)")
    print(f"  Korelasi : {corr_te:.3f}  (trivial: 0.000)")
    verdict = "MENGALAHKAN" if mae_te < mae_triv else "KALAH DARI"
    print(f"  -> {verdict} trivial baseline")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
