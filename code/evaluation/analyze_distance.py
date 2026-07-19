"""
Analisis pengaruh JARAK subjek ke radar (data/raw/distance: 50 cm dan 100 cm).

Kenapa penting: konfigurasi TI membatasi pencarian target di 0.3-1.0 meter
(`vitalSignsCfg 0.3 1.0 ...`), dan daya pantul radar turun kira-kira 1/R^4.
Skrip ini mengukur apakah jarak benar-benar menentukan kualitas sinyal jantung.

Catatan format: data jarak memakai `attys.tsv` (BUKAN attys.csv seperti dataset
lain). Formatnya:
  - baris pertama: '# <unix_epoch> <MAC>'  -> waktu mulai perekaman
  - kolom 0        : waktu relatif (detik, 125 Hz)
  - kolom 1,2,3    : akselerometer x,y,z
  - kolom 4,5,6    : magnetometer
  - kolom 7        : ECG  <-- ini yang dipakai (diverifikasi lewat keteraturan
                             interval RR: CV ~9.7%, wajar untuk ECG asli;
                             kolom lain CV >40% = bukan ECG)
  - kolom 8        : kanal ADC kedua (bukan ECG)

Usage:
    python analyze_distance.py [distance_dir]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from extract_ground_truth import detect_r_peaks, rr_to_instantaneous_hr  # noqa: E402

ECG_COL = 7
ATTYS_FS = 125.0
FS = 20.0
HR_BAND = (0.8, 2.0)
WINDOW_SEC = 16.0


def load_attys(path: Path):
    with open(path) as f:
        t0 = float(f.readline().split()[1])
    df = pd.read_csv(path, sep="\t", comment="#", header=None)
    return t0 + df[0].values, df[ECG_COL].values


def estimate_hr(t, phase, gt):
    order = np.argsort(t)
    t, phase, gt = t[order], phase[order], gt[order]
    fs_orig = 1.0 / np.median(np.diff(t))
    phase = signal.sosfiltfilt(
        signal.butter(6, FS / 2 * 0.9, "low", fs=fs_orig, output="sos"), phase)
    tu = np.arange(t[0], t[-1], 1.0 / FS)
    xu = np.interp(tu, t, phase)
    gu = np.interp(tu, t, gt)
    sig = signal.sosfiltfilt(
        signal.butter(4, HR_BAND, "bandpass", fs=FS, output="sos"), signal.detrend(xu))

    w = int(WINDOW_SEC * FS)
    est, ref, snrs = [], [], []
    for s in range(0, len(sig) - w, w // 4):
        seg = signal.detrend(sig[s:s + w])
        f, p = signal.welch(seg, fs=FS, nperseg=w)
        m = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
        est.append(f[m][np.argmax(p[m])] * 60)
        ref.append(gu[s:s + w].mean())

        # SNR di frekuensi ground-truth, dari sinyal MENTAH (tanpa bandpass)
        raw = signal.detrend(xu[s:s + w])
        g_hz = gu[s:s + w].mean() / 60
        if HR_BAND[0] <= g_hz <= HR_BAND[1]:
            fr, pr = signal.welch(raw, fs=FS, nperseg=w)
            mk = (fr >= HR_BAND[0]) & (fr <= HR_BAND[1])
            fb, pb = fr[mk], pr[mk]
            snrs.append(pb[int(np.argmin(np.abs(fb - g_hz)))] / np.median(pb))

    est = signal.medfilt(np.array(est), 5)
    return est, np.array(ref), np.array(snrs)


def main(root="data/raw/distance"):
    root = Path(root)
    dists = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: int(d.name))

    print(f"{'jarak':<8s} {'n_win':>6s} {'HR ECG':>7s} {'MAE':>7s} {'MAPE':>7s} "
          f"{'SNR':>6s} {'vonis (standar 10%)'}")
    print("-" * 68)
    for d in dists:
        ts, ecg = load_attys(d / "attys.tsv")
        peaks = detect_r_peaks(ecg, ts, ATTYS_FS)
        pt, hr = rr_to_instantaneous_hr(peaks)

        radar = pd.read_csv(d / "radar.csv")
        rt = radar["Timestamp"].values
        m = (rt >= pt.min()) & (rt <= pt.max())
        if m.sum() < 2000:
            print(f"{d.name + ' cm':<8s} overlap waktu tidak cukup ({m.sum()} baris)")
            continue

        gt = np.interp(rt[m], pt, hr)
        est, ref, snrs = estimate_hr(rt[m], radar["unwrapPhasePeak_mm"].values[m], gt)
        mae = np.mean(np.abs(est - ref))
        mape = 100 * np.mean(np.abs(est - ref) / ref)
        snr = np.median(snrs) if len(snrs) else float("nan")
        print(f"{d.name + ' cm':<8s} {len(est):>6d} {ref.mean():>7.1f} {mae:>7.1f} "
              f"{mape:>6.1f}% {snr:>6.2f}   {'LOLOS' if mape < 10 else 'GAGAL'}")

    print("\nSNR ~1.0 berarti sinyal jantung setara noise floor -- tidak ada yang bisa")
    print("diekstrak algoritma apapun. Daya pantul radar turun ~1/R^4, jadi jarak")
    print("subjek adalah parameter akuisisi yang KRITIS, bukan detail sepele.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/distance")
