"""
KONTROL NEGATIF + HOLD-OUT: dua uji yang membuat klaim tesis tidak bisa dipatahkan.

=== UJI 1: RUANGAN KOSONG (data/raw/no_subject) ===
Radar dinyalakan di ruangan kosong, tanpa siapa pun. Pertanyaannya sederhana:
apa yang dilaporkan tiap metode ketika TIDAK ADA detak jantung untuk dilaporkan?

Sebuah alat ukur yang jujur harus BILANG TIDAK TAHU. Kalau ia tetap mengeluarkan
angka yang terlihat masuk akal, angka itu tidak pernah berasal dari jantung siapa
pun -- termasuk ketika ada orangnya.

Hasil: kolom BPM logger tetap melaporkan ~83-85 bpm dari ruangan KOSONG, tak
terbedakan dari yang ia laporkan untuk manusia sungguhan. Sinyal fase mentah,
sebaliknya, memisahkan kosong vs berisi dengan margin TIGA ORDE BESARAN -- lewat
NAPAS, yang amplitudonya 4-9x lebih besar dari jantung (spek TI: napas 1-12 mm,
jantung 0.1-0.5 mm) sehingga jauh lebih mudah dideteksi.

Ini sekaligus MENUTUP lubang di CLAUDE.md poin 24/26: gerbang kualitas yang
selama ini butuh ECG (SNR di frekuensi ECG) kini punya pengganti yang TIDAK
butuh ECG sama sekali dan bisa dipakai saat deployment.

=== UJI 2: DUA SUBJEK CADANGAN (hold-out) ===
subject_cadangan01 & 02 TIDAK PERNAH dipakai untuk memilih apa pun -- tidak untuk
jendela 40 s, medfilt 9, maupun pita 0.8-2.0 Hz. Semua itu ditetapkan memakai 10
subjek utama. Jadi dua subjek ini menjawab tuduhan paling berbahaya di sidang:
"jangan-jangan kamu menyetel parameter sampai angkanya bagus."

Catatan: ground truth keduanya dari attys.tsv KOLOM 7 (bukan .csv). Kolom itu
diverifikasi sebagai ECG lewat CV interval RR (8-9%, sementara kolom lain 22-44%
= tidak mungkin fisiologis), dan untuk cadangan01 -- yang punya .csv MAUPUN .tsv
-- keduanya memberi hasil identik (530 R-peak, 74.4 bpm, CV 8.0%).

Usage:
    python negative_control.py ../../data/processed/aligned_all.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from extract_ground_truth import (detect_r_peaks,  # noqa: E402
                                  rr_to_instantaneous_hr)

from compare_ti_vs_phase import mae, mape, ti_windows  # noqa: E402
from phase_pipeline import (FS, HR_BAND, MAPE_STANDARD,  # noqa: E402
                            estimate, resample_antialias)

ROOT = Path(__file__).resolve().parents[2]
EMPTY = ROOT / "data" / "raw" / "no_subject"
FRONT = ROOT / "data" / "raw" / "position_front"
HOLDOUT = ["subject_cadangan01", "subject_cadangan02"]

BR_BAND = (0.15, 0.6)     # 9-36 napas/menit
NOISE_BAND = (3.0, 8.0)   # di atas napas & jantung -> noise murni
ECG_COL = 7               # kolom ECG di attys.tsv (diverifikasi, lihat docstring)

# Ambang KEHADIRAN. Dipilih di tengah jurang 3-orde-besaran antara kosong dan
# berisi (kosong <= 3.3, berisi >= 4359) -- bukan disetel pas di tepi data.
PRESENCE_THRESHOLD = 100.0


def presence(t, phase):
    """Deteksi KEHADIRAN manusia dari NAPAS. Tidak butuh ECG.

    Kenapa napas dan bukan jantung: gerak dada akibat napas 1-12 mm, akibat
    jantung hanya 0.1-0.5 mm (spek TI). Napas jauh di atas noise floor, jadi
    keberadaan manusia bisa dipastikan JAUH sebelum detaknya bisa diukur.
    """
    tu, xu, _ = resample_antialias(t, phase)
    xu = signal.detrend(xu)
    f, p = signal.welch(xu, fs=FS, nperseg=int(60 * FS))
    br = (f >= BR_BAND[0]) & (f <= BR_BAND[1])
    nz = (f >= NOISE_BAND[0]) & (f <= NOISE_BAND[1])
    return {
        "gerak_mm": float(np.std(xu)),
        "napas_per_menit": float(f[br][np.argmax(p[br])] * 60),
        "snr_napas": float(p[br].max() / np.median(p[nz])),
    }


def phase_bpm(t, phase):
    """Estimasi BPM tanpa ECG -- untuk sesi yang memang tak punya ground truth."""
    est, _, _ = estimate(t, phase, np.zeros_like(t))
    return est


def load_tsv(path, col=ECG_COL):
    """attys.tsv: baris-1 '# <unix_epoch> <MAC>', kolom-0 waktu relatif 125 Hz."""
    with open(path) as fh:
        t0 = float(fh.readline().split()[1])
    a = np.loadtxt(path, comments="#")
    return t0 + a[:, 0], a[:, col]


def holdout_session(d):
    """Satu subjek cadangan -> (est fase, est xCorr, ground truth, SNR)."""
    t_ecg, v = load_tsv(d / "attys.tsv")
    pk = detect_r_peaks(v, t_ecg, 1.0 / np.median(np.diff(t_ecg)))
    t_hr, hr = rr_to_instantaneous_hr(pk)

    r = pd.read_csv(d / "radar.csv").sort_values("Timestamp")
    t = r.Timestamp.values
    lo, hi = max(t[0], t_hr[0]), min(t[-1], t_hr[-1])   # irisan waktu kedua alat
    m = (t >= lo) & (t <= hi)
    r, t = r[m], t[m]
    gt = np.interp(t, t_hr, hr)

    est, ref, snr = estimate(t, r.unwrapPhasePeak_mm.values, gt)
    xc, _ = ti_windows(t, r.heartRateEst_xCorr.values.astype(float), gt)
    return est, xc, ref, snr


def main(path):
    df = pd.read_csv(path)
    front = df[(df.dataset == "position_front") & df.unwrapPhasePeak_mm.notna()]

    W = 92
    print("=" * W)
    print("KONTROL NEGATIF: APA YANG DILAPORKAN TIAP METODE DARI RUANGAN KOSONG?".center(W))
    print("=" * W)
    print("\nAlat ukur yang jujur harus BILANG TIDAK TAHU. Kalau ia tetap mengeluarkan angka")
    print("yang terlihat wajar, angka itu tidak pernah berasal dari jantung siapa pun.\n")

    rows = []
    for f in sorted(EMPTY.glob("*.csv")):
        r = pd.read_csv(f).sort_values("Timestamp")
        rows.append(("KOSONG", f.stem, r, presence(r.Timestamp.values,
                                                   r.unwrapPhasePeak_mm.values)))
    for sid, sub in front.groupby("subject_id"):
        sub = sub.sort_values("Timestamp")
        rows.append(("orang", sid, sub, presence(sub.Timestamp.values,
                                                 sub.unwrapPhasePeak_mm.values)))

    print(f"{'':8}{'sesi':<12}{'xCorr':>8}{'est_fft':>9}  |{'gerak dada':>12}{'SNR napas':>11}")
    print(f"{'':8}{'':12}{'(logger)':>8}{'(logger)':>9}  |{'(fase)':>12}{'(fase)':>11}")
    print("-" * W)
    for k, n, r, pr in rows:
        print(f"{k:<8}{n:<12}{r.heartRateEst_xCorr.mean():>8.1f}"
              f"{r.heart_rate_est_fft.mean():>9.1f}  |"
              f"{pr['gerak_mm']:>10.2f}mm{pr['snr_napas']:>11.0f}")

    ko = [pr for k, _, _, pr in rows if k == "KOSONG"]
    og = [pr for k, _, _, pr in rows if k == "orang"]
    x_ko = [r.heartRateEst_xCorr.mean() for k, _, r, _ in rows if k == "KOSONG"]
    x_og = [r.heartRateEst_xCorr.mean() for k, _, r, _ in rows if k == "orang"]

    print("-" * W)
    print("\nVONIS:")
    print(f"  LOGGER  xCorr di ruangan kosong : {min(x_ko):.1f} - {max(x_ko):.1f} bpm")
    print(f"          xCorr pada manusia      : {min(x_og):.1f} - {max(x_og):.1f} bpm")
    print("          -> TUMPANG TINDIH. Logger melaporkan detak jantung yang terlihat")
    print("             wajar dari ruangan KOSONG. Ia tidak mengukur apa pun.\n")

    gap = min(p["snr_napas"] for p in og) / max(p["snr_napas"] for p in ko)
    print(f"  FASE    SNR napas ruangan kosong: {min(p['snr_napas'] for p in ko):.1f} - "
          f"{max(p['snr_napas'] for p in ko):.1f}")
    print(f"          SNR napas pada manusia  : {min(p['snr_napas'] for p in og):.0f} - "
          f"{max(p['snr_napas'] for p in og):.0f}")
    print(f"          -> PISAH BERSIH, margin {gap:.0f}x. NOL tumpang tindih.")
    print(f"          Gerak dada: kosong {max(p['gerak_mm'] for p in ko):.2f} mm, "
          f"manusia {min(p['gerak_mm'] for p in og):.1f}-{max(p['gerak_mm'] for p in og):.1f} mm.\n")
    print("  Sinyal fase TAHU ada orang atau tidak. Kolom BPM logger TIDAK.")
    print("  Ini juga menutup lubang di poin 24/26: gerbang kualitas yang dulu butuh ECG")
    print("  kini punya pengganti yang TIDAK butuh ECG -- bisa dipakai saat deployment.")

    # ---------------- UJI 2: hold-out ----------------
    print("\n" + "=" * W)
    print("HOLD-OUT: DUA SUBJEK YANG TIDAK PERNAH DIPAKAI MENYETEL APA PUN".center(W))
    print("=" * W)
    print("\nJendela 40 s, medfilt 9, pita 0.8-2.0 Hz -- semuanya ditetapkan memakai 10")
    print("subjek utama. Dua subjek ini menjawab tuduhan 'kamu menyetel sampai bagus'.\n")
    print(f"{'subjek':<22}{'SNR':>6}{'ECG':>7}  |{'FASE':>7}{'MAPE':>7}{'vonis':>8}  |"
          f"{'xCorr':>7}{'MAPE':>7}{'vonis':>8}")
    print("-" * W)
    hold = []
    for name in HOLDOUT:
        est, xc, ref, snr = holdout_session(FRONT / name)
        mf, mx = mape(est, ref), mape(xc, ref)
        hold.append((name, snr, ref.mean(), est.mean(), mf, np.nanmean(xc), mx))
        print(f"{name:<22}{snr:>6.2f}{ref.mean():>7.1f}  |{est.mean():>7.1f}{mf:>6.1f}%"
              f"{'LOLOS' if mf < MAPE_STANDARD else 'gagal':>8}  |"
              f"{np.nanmean(xc):>7.1f}{mx:>6.1f}%{'LOLOS' if mx < MAPE_STANDARD else 'gagal':>8}")
    print("-" * W)
    print(f"\n  Fase menang di {sum(1 for h in hold if h[4] < h[6])}/2 subjek hold-out.")
    print(f"  xCorr mengeluarkan {hold[0][5]:.1f} dan {hold[1][5]:.1f} bpm -- menempel lagi ke")
    print("  konstantanya (~87), pada data yang belum pernah ia lihat. Cerita 'penebak-")
    print("  konstan' TERKONFIRMASI DI LUAR SAMPEL.")

    # ---- cek-mandiri ----
    assert max(p["snr_napas"] for p in ko) < PRESENCE_THRESHOLD < min(p["snr_napas"] for p in og), \
        "ambang kehadiran harus memisahkan kosong vs berisi dengan bersih"
    assert min(x_ko) > 60, "xCorr di ruangan kosong harus tetap melaporkan angka 'wajar'"
    assert all(h[4] < h[6] for h in hold), "fase harus mengalahkan xCorr di kedua hold-out"
    print("\n(cek-mandiri: 3 klaim terverifikasi)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
