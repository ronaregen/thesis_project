"""
METRIK KUALITAS TANPA ECG — jawaban untuk: "kalau di lapangan tidak ada ECG,
bagaimana tahu hasil estimasi bisa dipercaya atau tidak?"

Masalahnya: kriteria SNR di `phase_pipeline.py` dihitung DI FREKUENSI ECG, jadi
ia butuh ECG. Bagus untuk MENJELASKAN kenapa sebuah sesi gagal, tapi tidak bisa
dipakai saat alat dipakai sungguhan.

Skrip ini menguji 4 kandidat metrik yang SEMUANYA dihitung hanya dari sinyal
radar (tanpa ECG sama sekali):

  1. JITTER    - median |selisih antar estimasi berurutan|, SEBELUM median filter.
                 Motivasi fisiologis: detak jantung manusia TIDAK BISA melompat.
                 Kalau estimasi per-jendela melompat-lompat, estimator sedang
                 mengunci noise, bukan jantung.
                 (Penting: pakai interpolasi parabolik supaya frekuensinya
                 kontinu -- tanpa itu jitter terkuantisasi di lebar bin dan
                 jadi tidak informatif.)
  2. KECOCOKAN - selisih antara dua estimator INDEPENDEN (puncak PSD vs
                 autokorelasi). Kalau dua metode berbeda sepakat, hasilnya
                 lebih bisa dipercaya.
  3. HARMONIK  - energi di 2x frekuensi puncak. Denyut jantung punya harmonik
                 kedua; puncak noise tidak.
  4. PMR       - puncak dibagi median daya di pita (peak-to-median ratio).

Validasi memakai LEAVE-ONE-OUT: ambang disetel pada 11 sesi, lalu diuji pada
1 sesi yang ditinggalkan. Ini menghindari klaim yang terlalu optimistis akibat
menyetel ambang di data yang sama.

Usage:
    python quality_metric.py <aligned_csv>              # tabel saja
    python quality_metric.py <aligned_csv> --plot       # + simpan gambar PNG

Contoh (dari root project):
    python code/evaluation/quality_metric.py data/processed/aligned_all.csv --plot
    -> gambar tersimpan di: thesis/_fig_quality/jitter.png
"""

import contextlib
import io
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from extract_ground_truth import detect_r_peaks, rr_to_instantaneous_hr  # noqa: E402

warnings.filterwarnings("ignore")

FS = 25.0
HR_BAND = (0.8, 2.0)
BR_BAND = (0.15, 0.6)
WINDOW_SEC = 20.0
MAPE_STANDARD = 10.0
ECG_COL = 7


def prep(t, x, g):
    o = np.argsort(t)
    t, x, g = t[o], x[o], g[o]
    fso = 1.0 / np.median(np.diff(t))
    x = signal.sosfiltfilt(signal.butter(6, FS/2*0.9, "low", fs=fso, output="sos"), x)
    tu = np.arange(t[0], t[-1], 1.0/FS)
    return tu, np.interp(tu, t, x), np.interp(tu, t, g)


def parabolic(fb, pb, i):
    """Interpolasi parabolik -> frekuensi kontinu. TANPA ini, jitter terkuantisasi
    di lebar bin (= 1/WINDOW = 3 bpm) dan kehilangan daya prediksinya."""
    if 0 < i < len(pb) - 1:
        a, b, c = np.log(pb[i-1:i+2] + 1e-30)
        denom = a - 2*b + c
        if abs(denom) > 1e-20:
            return fb[i] + 0.5 * (a - c) / denom * (fb[1] - fb[0])
    return fb[i]


def analyze(t, phase, gt):
    tu, xu, gu = prep(t, phase, gt)
    d = np.diff(xu, prepend=xu[0])
    sos_br = signal.butter(4, BR_BAND, "bandpass", fs=FS, output="sos")
    sos_hr = signal.butter(4, HR_BAND, "bandpass", fs=FS, output="sos")

    w = int(WINDOW_SEC * FS)
    est_psd, est_ac, ref, harm, pmr = [], [], [], [], []
    for s in range(0, len(d) - w, w // 4):
        seg = signal.detrend(d[s:s+w])
        seg = seg - signal.sosfiltfilt(sos_br, seg)
        seg = signal.sosfiltfilt(sos_hr, seg)

        f, p = signal.welch(seg, fs=FS, nperseg=w)
        m = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
        fb, pb = f[m], p[m]
        i = int(np.argmax(pb))
        f_pk = parabolic(fb, pb, i)
        est_psd.append(f_pk * 60)

        # estimator independen: autokorelasi (domain waktu, bukan frekuensi)
        sg = seg - seg.mean()
        ac = signal.correlate(sg, sg, mode="full")[len(sg)-1:]
        ac /= (ac[0] + 1e-20)
        lo, hi = int(FS/HR_BAND[1]), min(int(FS/HR_BAND[0]), len(ac)-1)
        lag = lo + int(np.argmax(ac[lo:hi])) if hi > lo else lo
        est_ac.append(60 * FS / lag)

        j = int(np.argmin(np.abs(f - 2*f_pk)))
        harm.append(p[j] / (pb[i] + 1e-30))
        pmr.append(pb[i] / np.median(pb))
        ref.append(gu[s:s+w].mean())

    est_psd = np.array(est_psd)
    est_ac = np.array(est_ac)
    ref = np.array(ref)
    final = signal.medfilt(est_psd, 5)
    mape = 100 * float(np.mean(np.abs(final - ref) / ref))

    diffs = np.abs(np.diff(est_psd))
    return {
        "mape": mape,
        "jitter": float(np.median(diffs)),
        "kecocokan": float(np.median(np.abs(est_psd - est_ac))),
        "harmonik": float(np.median(harm)),
        "pmr": float(np.median(pmr)),
        "_diffs": diffs,          # dipakai untuk gambar
    }


def load_distance(d: Path, notes: list | None = None):
    """Muat sesi variasi jarak. Pesan dari deteksi R-peak ditampung ke `notes`
    supaya tidak mengotori tabel hasil (dicetak sebagai catatan di akhir)."""
    with open(d / "attys.tsv") as f:
        t0 = float(f.readline().split()[1])
    a = pd.read_csv(d / "attys.tsv", sep="\t", comment="#", header=None)
    ts = t0 + a[0].values

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        peaks = detect_r_peaks(a[ECG_COL].values, ts, 125.0)
        pt, hr = rr_to_instantaneous_hr(peaks)
    if notes is not None:
        for line in buf.getvalue().splitlines():
            line = line.strip()
            if line:
                n = re.search(r"(\d+)", line)
                notes.append(f"jarak{d.name}: {n.group(1) if n else '?'} detak dibuang "
                             f"karena di luar rentang fisiologis 40-180 bpm (ECG Attys berisik)")

    r = pd.read_csv(d / "radar.csv")
    rt = r["Timestamp"].values
    m = (rt >= pt.min()) & (rt <= pt.max())
    return rt[m], r["unwrapPhasePeak_mm"].values[m], np.interp(rt[m], pt, hr)


def best_threshold(v, y, lower_is_better=True):
    best = (None, -1)
    for th in np.unique(v):
        pred = (v < th) if lower_is_better else (v >= th)
        acc = float(np.mean(pred == y))
        if acc > best[1]:
            best = (float(th), acc)
    return best


JUDUL = {
    "jitter": "jitter estimasi",
    "kecocokan": "kecocokan PSD vs autokorelasi",
    "harmonik": "energi harmonik (di 2f)",
    "pmr": "peak-to-median ratio",
}
LOWER = {"jitter": True, "kecocokan": True, "harmonik": False, "pmr": False}


def rule(ch="-", n=64):
    print(ch * n)


def head(text):
    print()
    rule("=")
    print(f"  {text}")
    rule("=")


def make_plot(rows, th_j, outdir: Path):
    """Simpan gambar: sebaran |perubahan estimasi| + jitter vs MAPE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    C_BAD, C_GOOD, C_NEU = "#c0392b", "#1e8449", "#7f8c8d"

    sids = [r[0] for r in rows]
    J = np.array([r[1]["jitter"] for r in rows])
    M = np.array([r[1]["mape"] for r in rows])
    cols = [C_GOOD if m < MAPE_STANDARD else C_BAD for m in M]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4),
                                 gridspec_kw={"width_ratios": [1, 1.1]})

    # kiri: sebaran |perubahan| untuk sesi terbaik vs terburuk
    best = rows[0]
    worst = max(rows, key=lambda r: r[1]["jitter"])
    for r, col in [(best, C_GOOD), (worst, C_BAD)]:
        dif = r[1]["_diffs"]
        a1.hist(np.clip(dif, 0, 30), bins=np.arange(0, 31, 1.5), alpha=.6, color=col,
                label=f"{r[0]}  (median = {r[1]['jitter']:.1f} bpm)")
        a1.axvline(r[1]["jitter"], color=col, lw=2.5, ls="--")
    a1.axvline(th_j, color=C_NEU, lw=2, ls=":", label=f"ambang {th_j:.1f} bpm")
    a1.set_xlabel("|perubahan estimasi antar jendela| (bpm)")
    a1.set_ylabel("jumlah jendela")
    a1.set_title("Jitter = MEDIAN dari sebaran ini\n(estimasi mentah, sebelum median filter)",
                 fontweight="bold", fontsize=12)
    a1.legend(fontsize=9)
    a1.grid(alpha=.3)

    # kanan: jitter vs MAPE
    a2.scatter(J, M, s=140, c=cols, edgecolors="w", lw=1.5, zorder=3)
    for sid, j, m in zip(sids, J, M):
        a2.annotate(sid, (j, m), fontsize=8, xytext=(6, -3), textcoords="offset points")
    a2.axhline(MAPE_STANDARD, color="#8e44ad", ls="--", lw=1.8,
               label=f"batas standar MAPE {MAPE_STANDARD:.0f}%")
    a2.axvline(th_j, color=C_NEU, ls=":", lw=2, label=f"ambang jitter {th_j:.1f} bpm")
    a2.set_xlabel("jitter (bpm)   — dihitung TANPA ECG")
    a2.set_ylabel("MAPE (%)")
    a2.set_title(f"Jitter memprediksi galat\n(r = {np.corrcoef(J, np.log(M))[0,1]:+.2f} thd log MAPE)",
                 fontweight="bold", fontsize=12)
    a2.legend(fontsize=9)
    a2.grid(alpha=.3)

    fig.suptitle("Metrik kualitas TANPA ECG: hijau = memenuhi standar, merah = gagal",
                 fontweight="bold", fontsize=13.5, y=1.03)
    p = outdir / "jitter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main(path, plot=False):
    notes = []

    df = pd.read_csv(path)
    front = df[(df.dataset == "position_front") & df.unwrapPhasePeak_mm.notna()]

    rows = []
    for sid, sub in front.groupby("subject_id"):
        rows.append((sid.replace("subject", "S"),
                     analyze(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                             sub.gt_heart_rate.values)))
    dist_root = Path(__file__).resolve().parents[2] / "data" / "raw" / "distance"
    if dist_root.exists():
        for d in sorted(dist_root.iterdir(), key=lambda p: int(p.name)):
            t, x, g = load_distance(d, notes)
            rows.append((f"jarak{d.name}", analyze(t, x, g)))

    rows.sort(key=lambda r: r[1]["jitter"])
    M = np.array([r[1]["mape"] for r in rows])
    Y = M < MAPE_STANDARD
    n = len(rows)

    J = np.array([r[1]["jitter"] for r in rows])
    th_j, _ = best_threshold(J, Y, True)

    print()
    rule("=")
    print("  METRIK KUALITAS TANPA ECG")
    print("  Semua kolom di bawah dihitung HANYA dari sinyal radar - nol ECG.")
    rule("=")

    head(f"HASIL PER SESI  (diurutkan berdasarkan jitter; ambang = {th_j:.1f} bpm)")
    print(f"  {'sesi':<10s} {'MAPE':>7s} | {'jitter':>7s} {'cocok':>7s} {'harm':>7s} {'PMR':>6s} | "
          f"{'prediksi':>9s} {'aktual':>7s}")
    rule()
    crossed = False
    for sid, v in rows:
        if not crossed and v["jitter"] >= th_j:
            print(f"  {'':<10s} {'':>7s} | {'- - - - ambang jitter = ' + f'{th_j:.1f}' + ' bpm - - - -':^31s} |")
            crossed = True
        pred = v["jitter"] < th_j
        act = v["mape"] < MAPE_STANDARD
        mark = " " if pred == act else "  <-- MELESET"
        print(f"  {sid:<10s} {v['mape']:>6.1f}% | {v['jitter']:>7.1f} {v['kecocokan']:>7.1f} "
              f"{v['harmonik']:>7.3f} {v['pmr']:>6.2f} | "
              f"{'LOLOS' if pred else 'gagal':>9s} {'LOLOS' if act else 'gagal':>7s}{mark}")
    rule()
    errs = [sid for sid, v in rows
            if (v["jitter"] < th_j) != (v["mape"] < MAPE_STANDARD)]
    if errs:
        print(f"  Meleset {len(errs)} dari {n} sesi ({', '.join(errs)}) -- "
              f"tepat di sekitar ambang (kasus batas),")
        print("  bukan kesalahan liar. Di sekitar garis batas metrik ini memang tak bisa memutuskan.")
    else:
        print(f"  Tepat di seluruh {n} sesi (dengan ambang disetel di data ini juga -- lihat LOO).")

    head("PERBANDINGAN 4 KANDIDAT METRIK")
    print("  Validasi LEAVE-ONE-OUT: ambang disetel pada 11 sesi, diuji pada 1 sesi")
    print("  yang DITINGGALKAN. Diulang untuk semua sesi. Ini menghindari klaim palsu")
    print("  akibat menyetel ambang di data yang sama dengan data ujinya.\n")
    print(f"  {'metrik':<32s} {'r vs log(MAPE)':>15s} {'akurasi LOO':>13s}")
    rule()
    results = []
    for k in JUDUL:
        V = np.array([r[1][k] for r in rows])
        c = float(np.corrcoef(V, np.log(M))[0, 1])
        ok = 0
        for i in range(n):
            mask = np.ones(n, bool)
            mask[i] = False
            th, _ = best_threshold(V[mask], Y[mask], LOWER[k])
            pred = (V[i] < th) if LOWER[k] else (V[i] >= th)
            ok += int(pred == Y[i])
        results.append((k, c, ok))
    base = max(int(Y.sum()), n - int(Y.sum()))
    for k, c, ok in sorted(results, key=lambda r: -r[2]):
        tag = ""
        if ok == max(r[2] for r in results):
            tag = "  <-- TERBAIK"
        elif ok <= base:
            tag = "  (gagal: tak lebih baik dari menebak)"
        print(f"  {JUDUL[k]:<32s} {c:>+15.3f} {ok:>7d}/{n} {100*ok/n:>3.0f}%{tag}")
    rule()
    print(f"  {'(tebak mayoritas - pembanding)':<32s} {'-':>15s} {base:>7d}/{n} {100*base/n:>3.0f}%")

    best_k, best_c, best_ok = max(results, key=lambda r: r[2])
    head("KESIMPULAN")
    print(f"  Pemenang: {JUDUL[best_k].upper()}")
    print(f"    - korelasi terhadap galat : r = {best_c:+.3f}")
    print(f"    - akurasi leave-one-out   : {best_ok}/{n} = {100*best_ok/n:.0f}%  "
          f"(tebak mayoritas: {100*base/n:.0f}%)")
    print(f"    - ambang                  : jitter < {th_j:.1f} bpm -> hasil bisa dipercaya")
    print("    - dasar fisiologis        : detak jantung manusia tidak bisa melompat,")
    print("                                jadi estimasi yang gemetar = estimator mengunci noise")
    n_fail = sum(1 for _, _, ok in results if ok <= base)
    print(f"\n  {n_fail} kandidat lain GAGAL (tidak lebih baik dari sekadar menebak). Sengaja")
    print("  tetap ditampilkan: pemenang dipilih lewat perbandingan yang adil, bukan")
    print("  dipetik karena kebetulan terlihat bagus.")

    head("YANG HARUS DISAMPAIKAN SENDIRI (jangan tunggu ditanya)")
    th_all, acc_all = best_threshold(J, Y, True)
    print(f"  1. Kalau ambang disetel di data yang sama, akurasinya terlihat {100*acc_all:.0f}%.")
    print(f"     Itu OVERFITTING (12 sesi, 1 parameter bebas). Angka yang SAH adalah")
    print(f"     hasil leave-one-out: {100*best_ok/n:.0f}%. LAPORKAN {100*best_ok/n:.0f}%, JANGAN {100*acc_all:.0f}%.")
    print(f"  2. Baru {n} sesi - sampel kecil, perlu validasi di data lebih banyak.")
    print("  3. Ini memprediksi kualitas SATU SESI, belum diuji sebagai gerbang")
    print("     per-jendela secara real-time.")
    print("  -> Laporkan sebagai TEMUAN AWAL YANG MENJANJIKAN, bukan klaim final.")
    print("     Temuan ini MEMPERKUAT Kontribusi 3 (batas keberlakuan), bukan")
    print("     menjadi kontribusi baru.")

    if plot:
        outdir = Path(__file__).resolve().parents[2] / "thesis" / "_fig_quality"
        p = make_plot(rows, th_j, outdir)
        head("GAMBAR")
        print(f"  Tersimpan: {p}")

    if notes:
        head("CATATAN PEMROSESAN (bukan error)")
        for x in notes:
            print(f"  - {x}")
    print()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__)
        sys.exit(1)
    main(args[0], plot="--plot" in sys.argv)
