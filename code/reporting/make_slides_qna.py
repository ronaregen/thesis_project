"""
Slide CADANGAN (backup) - antisipasi 3 pertanyaan pembimbing:
  1. SNR itu ngitungnya gimana?
  2. Turunan fase itu sinyalnya diapain?
  3. Kontribusi tesisnya di mana? Cuma ngetes alat?

Usage:
    python make_slides_qna.py <aligned_csv> <outdir>
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from phase_pipeline import (FS, HR_BAND, BR_BAND, MAPE_STANDARD,  # noqa: E402
                            WINDOW_SEC, estimate, metrics)

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": .3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

NAVY, RED, GREEN, GREY, PURPLE = (28, 42, 66), (192, 57, 43), (30, 132, 73), (110, 118, 128), (110, 60, 140)
AMBER = (176, 122, 12)
C_BAD, C_GOOD, C_GT, C_NEU = "#c0392b", "#1e8449", "#2c3e50", "#7f8c8d"
W, H = 338.7, 190.5


def fig_snr(front, out):
    """Q1: SNR = daya di f_ECG dibagi median daya di band."""
    sub = front[front.subject_id == "subject01"].sort_values("Timestamp")
    t = sub.Timestamp.values
    o = np.argsort(t)
    t = t[o]
    x = sub.unwrapPhasePeak_mm.values[o]
    g = sub.gt_heart_rate.values[o]
    fso = 1 / np.median(np.diff(t))
    xl = signal.sosfiltfilt(signal.butter(6, FS/2*.9, "low", fs=fso, output="sos"), x)
    tu = np.arange(t[0], t[-1], 1/FS)
    xu = np.interp(tu, t, xl)
    gu = np.interp(tu, t, g)

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.4))
    for ax, sid, lab in [(axs[0], "subject01", "SNR TINGGI — sinyal jantung jelas"),
                         (axs[1], "subject03", "SNR RENDAH — tenggelam di noise")]:
        s2 = front[front.subject_id == sid].sort_values("Timestamp")
        t2 = s2.Timestamp.values
        o2 = np.argsort(t2)
        t2 = t2[o2]
        x2 = s2.unwrapPhasePeak_mm.values[o2]
        g2 = s2.gt_heart_rate.values[o2]
        f2 = 1 / np.median(np.diff(t2))
        xl2 = signal.sosfiltfilt(signal.butter(6, FS/2*.9, "low", fs=f2, output="sos"), x2)
        tu2 = np.arange(t2[0], t2[-1], 1/FS)
        xu2 = np.interp(tu2, t2, xl2)
        gu2 = np.interp(tu2, t2, g2)

        w = int(20*FS)
        s = len(xu2)//2
        seg = signal.detrend(xu2[s:s+w])
        f, p = signal.welch(seg, fs=FS, nperseg=w)
        m = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
        fb, pb = f[m], p[m]
        gt = gu2[s:s+w].mean() / 60
        gi = int(np.argmin(np.abs(fb - gt)))
        med = np.median(pb)
        snr = pb[gi] / med

        ax.plot(fb*60, pb, color=C_GT, lw=1.8)
        ax.axhline(med, color=C_NEU, ls="--", lw=1.8,
                   label=f"median daya di band\n= lantai noise")
        ax.plot(fb[gi]*60, pb[gi], "o", color=C_BAD, ms=13, zorder=5,
                label=f"daya di frekuensi ECG\n({gt*60:.0f} bpm)")
        ax.annotate("", xy=(fb[gi]*60, pb[gi]), xytext=(fb[gi]*60, med),
                    arrowprops=dict(arrowstyle="<->", lw=2.2, color=C_BAD))
        ax.text(fb[gi]*60 + 3, np.sqrt(pb[gi]*med), f" SNR = {snr:.2f}\n ({10*np.log10(snr):+.1f} dB)",
                fontsize=12, fontweight="bold", color=C_BAD, va="center")
        ax.set_xlabel("detak jantung (bpm)")
        ax.set_ylabel("daya (PSD)")
        ax.set_title(f"{sid} — {lab}", fontweight="bold", fontsize=12)
        ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("SNR = daya tepat di frekuensi ECG  dibagi  lantai noise (median daya di band)",
                 fontweight="bold", fontsize=14, y=1.04)
    fig.savefig(out / "q1_snr.png")
    plt.close(fig)


def fig_jitter(front, out):
    """Q1c: jitter - metrik kualitas TANPA ECG."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
    from quality_metric import analyze, load_distance

    rows = []
    for sid, sub in front.groupby("subject_id"):
        rows.append((sid.replace("subject", "S"),
                     analyze(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                             sub.gt_heart_rate.values)))
    droot = Path(__file__).resolve().parents[2] / "data" / "raw" / "distance"
    if droot.exists():
        for d in sorted(droot.iterdir(), key=lambda p: int(p.name)):
            t, x, g = load_distance(d)
            rows.append((f"jarak{d.name}", analyze(t, x, g)))

    J = np.array([r[1]["jitter"] for r in rows])
    M = np.array([r[1]["mape"] for r in rows])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.3),
                                 gridspec_kw={"width_ratios": [1.15, 1]})

    # kiri: sebaran |perubahan antar estimasi| -> median-nya = jitter
    from scipy import signal as sg
    for sid, col, lab in [("subject01", C_GOOD, "SNR baik"),
                          ("subject06", C_BAD, "SNR rendah")]:
        sub = front[front.subject_id == sid].sort_values("Timestamp")
        t = sub.Timestamp.values
        o = np.argsort(t)
        t = t[o]
        x = sub.unwrapPhasePeak_mm.values[o]
        fso = 1/np.median(np.diff(t))
        xl = sg.sosfiltfilt(sg.butter(6, FS/2*.9, "low", fs=fso, output="sos"), x)
        tu = np.arange(t[0], t[-1], 1/FS)
        xu = np.interp(tu, t, xl)
        d = np.diff(xu, prepend=xu[0])
        sb = sg.butter(4, BR_BAND, "bandpass", fs=FS, output="sos")
        sh = sg.butter(4, HR_BAND, "bandpass", fs=FS, output="sos")
        w = int(20*FS)
        est = []
        for s2 in range(0, len(d)-w, w//4):
            seq = sg.detrend(d[s2:s2+w])
            seq = seq - sg.sosfiltfilt(sb, seq)
            seq = sg.sosfiltfilt(sh, seq)
            f, p = sg.welch(seq, fs=FS, nperseg=w)
            m = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
            fb, pb = f[m], p[m]
            i = int(np.argmax(pb))
            if 0 < i < len(pb)-1:
                a, b, c = np.log(pb[i-1:i+2]+1e-30)
                dd = a - 2*b + c
                fpk = fb[i] + (0.5*(a-c)/dd*(fb[1]-fb[0]) if abs(dd) > 1e-20 else 0)
            else:
                fpk = fb[i]
            est.append(fpk*60)
        est = np.array(est)
        dif = np.abs(np.diff(est))
        jit = np.median(dif)
        a1.hist(np.clip(dif, 0, 30), bins=np.arange(0, 31, 1.5), alpha=.6,
                color=col, label=f"{sid} ({lab})\nmedian = {jit:.1f} bpm")
        a1.axvline(jit, color=col, lw=2.5, ls="--")
    a1.axvline(3.1, color=C_NEU, lw=2, ls=":", label="ambang 3 bpm")
    a1.set_xlabel("|perubahan estimasi antar jendela| (bpm)")
    a1.set_ylabel("jumlah jendela")
    a1.set_title("Jitter = MEDIAN dari sebaran ini\n(estimasi mentah, sebelum median filter)",
                 fontweight="bold", fontsize=12)
    a1.legend(fontsize=8)

    # kanan: jitter vs MAPE
    cols = [C_GOOD if m < 10 else C_BAD for m in M]
    a2.scatter(J, M, s=130, c=cols, edgecolors="w", lw=1.5, zorder=3)
    for sid, j, m in zip([r[0] for r in rows], J, M):
        a2.annotate(sid, (j, m), fontsize=7.5, xytext=(5, -3),
                    textcoords="offset points")
    a2.axhline(10, color="#8e44ad", ls="--", lw=1.8, label="batas standar MAPE 10%")
    a2.axvline(3.1, color=C_NEU, ls=":", lw=2, label="ambang jitter = 3 bpm")
    a2.set_xlabel("jitter (bpm)  -- dihitung TANPA ECG")
    a2.set_ylabel("MAPE (%)")
    a2.legend(fontsize=8.5)
    a2.set_title(f"Jitter memprediksi galat\n(r = {np.corrcoef(J, np.log(M))[0,1]:+.2f} thd log MAPE)",
                 fontweight="bold", fontsize=12)
    fig.suptitle("Jitter: metrik kualitas yang tidak butuh ECG sama sekali",
                 fontweight="bold", fontsize=14, y=1.04)
    fig.savefig(out / "q1c_jitter.png")
    plt.close(fig)


def fig_hrv(out):
    """Q4: bisa dapat HRV? Kiri = detak radar vs R-peak ECG. Kanan = RMSSD."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
    import hrv_analysis as hv

    rows = []
    for d in sorted(hv.FRONT.iterdir()):
        radar = pd.read_csv(d / "radar.csv")
        attys = pd.read_csv(d / "attys.csv")
        tu, fs = hv.uniform_clock(radar["Timestamp"].values)
        ph = radar["unwrapPhasePeak_mm"].values.astype(float)
        fse = 1.0 / np.median(np.diff(attys["timestamp"].values))
        te = hv.detect_r_peaks(attys["value"].values, attys["timestamp"].values, fse)
        rec = dict(sid=d.name, ecg=hv.ibi_stats(te), te=te, tu=tu, ph=ph, fs=fs)
        for nm, bd in [("narrow", hv.BAND_NARROW), ("wide", hv.BAND_WIDE)]:
            tb, sg_ = hv.beat_times(ph, tu, fs, bd)
            _, cov = hv.match_beats(tb, te)
            rec[nm] = dict(**hv.ibi_stats(tb), cov=cov, tb=tb, sig=sg_)
        rows.append(rec)

    r0 = next(r for r in rows if r["sid"] == "subject01")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.3),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    # kiri: 8 detik sinyal, detak radar vs R-peak ECG
    sig, tu, tb, te = r0["wide"]["sig"], r0["tu"], r0["wide"]["tb"], r0["te"]
    t0 = tu[len(tu) // 2]
    m = (tu >= t0) & (tu <= t0 + 8)
    sn = sig[m] / np.std(sig[m])
    a1.plot(tu[m] - t0, sn, color=C_NEU, lw=1.4,
            label="sinyal fase (turunan, pita lebar)")
    for k, tt in enumerate(te[(te >= t0) & (te <= t0 + 8)]):
        a1.axvline(tt - t0, color=C_GOOD, lw=2.4, alpha=.75,
                   label="R-peak ECG (kebenaran)" if k == 0 else None)
    sel = tb[(tb >= t0) & (tb <= t0 + 8)]
    ys = np.interp(sel, tu[m], sn)
    a1.plot(sel - t0, ys + .35, "v", color=C_BAD, ms=10, zorder=5,
            label=f"detak terdeteksi radar ({len(sel)} buah)")
    a1.set_xlabel("detik")
    a1.set_ylabel("amplitudo (ternormalisasi)")
    a1.set_ylim(sn.min() - .5, sn.max() + 1.6)
    a1.legend(fontsize=8.5, loc="upper left", ncol=1, framealpha=.95)
    a1.set_title("subject01 (SNR TERBAIK) - detak radar tetap meleset dari R-peak\n"
                 f"hanya {r0['wide']['cov']*100:.0f}% R-peak ketemu pasangan",
                 fontweight="bold", fontsize=11.5)

    # kanan: RMSSD
    sids = [r["sid"].replace("subject", "S") for r in rows]
    e = [r["ecg"]["rmssd"] for r in rows]
    n = [r["narrow"]["rmssd"] for r in rows]
    w = [r["wide"]["rmssd"] for r in rows]
    x = np.arange(len(rows))
    a2.bar(x - .27, e, .27, color=C_GOOD, label=f"ECG (acuan), rata2 {np.mean(e):.0f} ms")
    a2.bar(x, n, .27, color="#e67e22", label=f"radar pita sempit, {np.mean(n):.0f} ms")
    a2.bar(x + .27, w, .27, color=C_BAD, label=f"radar pita lebar, {np.mean(w):.0f} ms")
    a2.set_xticks(x)
    a2.set_xticklabels(sids, fontsize=8)
    a2.set_ylabel("RMSSD (ms)")
    a2.legend(fontsize=8.5)
    a2.set_title("RMSSD radar ~4x nilai ECG, di KEDUA pita\n"
                 "= ukuran kesalahan deteksi, bukan variabilitas jantung",
                 fontweight="bold", fontsize=11.5)

    fig.suptitle("HRV butuh waktu SETIAP detak. Di SNR 1.4-5.9, detak individual tenggelam.",
                 fontweight="bold", fontsize=14, y=1.05)
    fig.savefig(out / "q4_hrv.png")
    plt.close(fig)


def fig_deriv(front, out):
    """Q2: turunan fase - respons frekuensi + efek nyata."""
    f = np.linspace(0.01, 3.0, 500)
    Hh = 2 * np.abs(np.sin(np.pi * f / FS))     # |1 - e^-jw|

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))

    axs[0].plot(f, Hh / Hh.max(), color=C_GT, lw=2.4)
    axs[0].axvspan(BR_BAND[0], BR_BAND[1], color="#2980b9", alpha=.22)
    axs[0].axvspan(HR_BAND[0], HR_BAND[1], color=C_BAD, alpha=.18)
    gb = 2*np.abs(np.sin(np.pi*0.30/FS))
    gh = 2*np.abs(np.sin(np.pi*1.20/FS))
    axs[0].plot([0.30], [gb/Hh.max()], "o", color="#2980b9", ms=10)
    axs[0].plot([1.20], [gh/Hh.max()], "o", color=C_BAD, ms=10)
    axs[0].text(0.36, gb/Hh.max()+.07, "napas\n0.3 Hz", fontsize=9, color="#2980b9",
                fontweight="bold")
    axs[0].text(1.30, gh/Hh.max()+.06, f"jantung 1.2 Hz\n{gh/gb:.1f}x lebih kuat",
                fontsize=9, color=C_BAD, fontweight="bold")
    axs[0].set_xlabel("frekuensi (Hz)")
    axs[0].set_ylabel("penguatan (ternormalisasi)")
    axs[0].set_title("Turunan = filter yang penguatannya\nnaik sebanding frekuensi",
                     fontweight="bold", fontsize=11.5)
    axs[0].set_xlim(0, 3)

    # sinyal nyata sebelum/sesudah
    sub = front[front.subject_id == "subject01"].sort_values("Timestamp")
    t = sub.Timestamp.values
    o = np.argsort(t)
    t = t[o]
    x = sub.unwrapPhasePeak_mm.values[o]
    fso = 1/np.median(np.diff(t))
    xl = signal.sosfiltfilt(signal.butter(6, FS/2*.9, "low", fs=fso, output="sos"), x)
    tu = np.arange(t[0], t[-1], 1/FS)
    xu = np.interp(tu, t, xl)
    d = np.diff(xu, prepend=xu[0])
    s, w = 2000, int(30*FS)
    tt = np.arange(w)/FS

    axs[1].plot(tt, xu[s:s+w] - xu[s:s+w].mean(), color=C_GT, lw=1.6)
    axs[1].set_title("SEBELUM: perpindahan dada (mm)\nnapas mendominasi",
                     fontweight="bold", fontsize=11.5)
    axs[1].set_xlabel("detik"); axs[1].set_ylabel("mm")

    axs[2].plot(tt, d[s:s+w], color="#e67e22", lw=1.3)
    axs[2].set_title("SESUDAH: kecepatan dada (mm/sampel)\ndenyut jantung menonjol",
                     fontweight="bold", fontsize=11.5)
    axs[2].set_xlabel("detik"); axs[2].set_ylabel("mm/sampel")

    fig.suptitle("Turunan fase: d[n] = x[n] - x[n-1]   →   dari PERPINDAHAN menjadi KECEPATAN dada",
                 fontweight="bold", fontsize=14, y=1.05)
    fig.savefig(out / "q2_deriv.png")
    plt.close(fig)


def fig_ratio(front, out):
    """Q2b: rasio napas:jantung sebelum vs sesudah turunan."""
    sids, b4, af = [], [], []
    for sid, sub in front.groupby("subject_id"):
        t = sub.Timestamp.values
        o = np.argsort(t)
        t = t[o]
        x = sub.unwrapPhasePeak_mm.values[o]
        fso = 1/np.median(np.diff(t))
        xl = signal.sosfiltfilt(signal.butter(6, FS/2*.9, "low", fs=fso, output="sos"), x)
        tu = np.arange(t[0], t[-1], 1/FS)
        xu = np.interp(tu, t, xl)
        d = np.diff(xu, prepend=xu[0])

        def ratio(sig):
            sb = signal.sosfiltfilt(signal.butter(4, BR_BAND, "bandpass", fs=FS, output="sos"), sig)
            sh = signal.sosfiltfilt(signal.butter(4, HR_BAND, "bandpass", fs=FS, output="sos"), sig)
            return np.std(sb)/np.std(sh)
        sids.append(sid.replace("subject", "S"))
        b4.append(ratio(xu))
        af.append(ratio(d))

    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.bar(x - .2, b4, .4, color="#2980b9", alpha=.9, label="SEBELUM turunan")
    ax.bar(x + .2, af, .4, color="#e67e22", alpha=.9, label="SESUDAH turunan")
    ax.axhline(1.0, color=C_GOOD, ls="--", lw=1.8, label="setara (napas = jantung)")
    ax.set_xticks(x); ax.set_xticklabels(sids)
    ax.set_ylabel("napas : jantung")
    ax.legend(fontsize=9)
    ax.set_title(f"Dominasi napas turun dari rata-rata {np.mean(b4):.1f}x menjadi "
                 f"{np.mean(af):.1f}x  (teori memprediksi 4.0x, terukur {np.mean(b4)/np.mean(af):.1f}x)",
                 fontweight="bold", fontsize=12)
    fig.savefig(out / "q2b_ratio.png")
    plt.close(fig)


class Slides(FPDF):
    def __init__(self):
        super().__init__(orientation="L", format=(H, W))
        self.set_auto_page_break(False)
        self.set_margins(16, 14, 16)

    def slide(self, n, title, sub=None):
        self.add_page()
        self.set_xy(16, 11)
        self.set_font("Helvetica", "B", 19)
        self.set_text_color(*NAVY)
        self.cell(0, 9, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if sub:
            self.set_x(16)
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*GREY)
            self.cell(0, 6, sub, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*RED)
        self.set_line_width(1.1)
        y = self.get_y() + 1.5
        self.line(16, y, 44, y)
        self.set_xy(W - 26, H - 11)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*GREY)
        self.cell(10, 5, f"C{n}", align="R")
        self.set_xy(16, y + 3)

    def box(self, x, y, w, text, title=None, color=NAVY, size=10):
        self.set_xy(x + 4, y + 3)
        if title:
            self.set_font("Helvetica", "B", size + .5)
            self.set_text_color(*color)
            self.multi_cell(w - 8, 5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_x(x + 4)
        self.set_font("Helvetica", "", size)
        self.set_text_color(40, 48, 58)
        self.multi_cell(w - 8, 4.8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        y2 = self.get_y() + 3
        self.set_fill_color(*color)
        self.rect(x, y, 1.6, y2 - y, style="F")
        self.set_draw_color(226, 229, 233)
        self.set_line_width(.2)
        self.rect(x, y, w, y2 - y)
        return y2


def build(fig, out, K):
    p = Slides()

    # ---- C1: SNR
    p.slide(1, "Pertanyaan 1: SNR itu menghitungnya bagaimana?",
            "Jawaban singkat: seberapa menonjol daya di frekuensi ECG dibanding lantai noise di sekitarnya.")
    y = p.get_y()
    p.box(16, y, 306,
          f"Untuk tiap jendela {WINDOW_SEC:.0f} detik, dari sinyal fase MENTAH (sebelum turunan):\n\n"
          "   Langkah 1.  Hitung spektrum daya (Welch PSD).\n"
          "   Langkah 2.  Ambil hanya pita detak jantung manusia: 0.8 - 2.0 Hz (48 - 120 bpm).\n"
          "   Langkah 3.  Cari bin frekuensi yang paling dekat dengan detak jantung SEBENARNYA dari ECG:  f_ECG = HR_ECG / 60.\n"
          "   Langkah 4.  SNR = daya di f_ECG  /  MEDIAN daya seluruh pita tersebut.\n"
          "   Langkah 5.  Ambil median SNR dari semua jendela dalam satu sesi.\n\n"
          "Kenapa MEDIAN, bukan rata-rata? Karena median tahan terhadap puncak itu sendiri dan terhadap pencilan, "
          "sehingga ia menjadi penaksir LANTAI NOISE yang jujur.\n\n"
          "CATATAN: grafik di bawah menampilkan SATU jendela contoh. Nilai SNR yang dilaporkan per sesi adalah "
          "MEDIAN dari seluruh jendela (subject01 = 6.41, subject03 = 1.67), jadi wajar bila berbeda dari angka "
          "di satu jendela.",
          "Definisi operasional", NAVY, 10)
    p.image(str(fig / "q1_snr.png"), x=40, y=p.get_y() + 1, w=258)

    # ---- C1b: tafsir
    p.slide(2, "Pertanyaan 1 (lanjutan): cara membacanya",
            "SNR adalah rasio DAYA (linear). Kalau ditanya dalam dB: dB = 10 log10(SNR).")
    y = p.get_y()
    p.box(16, y, 150,
          "SNR = 1.00  ->   0.0 dB   daya jantung SAMA dengan noise = tak terdeteksi\n"
          "SNR = 1.09  ->  +0.4 dB   sesi jarak 100 cm  (GAGAL)\n"
          "SNR = 1.67  ->  +2.2 dB   subject03           (GAGAL)\n"
          "SNR = 1.80  ->  +2.6 dB   AMBANG KELAYAKAN\n"
          "SNR = 3.16  ->  +5.0 dB   sesi jarak 50 cm    (LOLOS)\n"
          "SNR = 6.41  ->  +8.1 dB   subject01           (LOLOS)",
          "Nilai SNR dan artinya", NAVY, 10)
    p.box(172, y, 150,
          "Ambang SNR >= 1.8 memisahkan sesi yang LOLOS dan GAGAL standar dengan benar "
          "pada 11 dari 12 sesi (10 subjek + 2 sesi variasi jarak).\n\n"
          "Artinya: kita bisa memastikan sebelumnya apakah sebuah sesi layak dianalisis "
          "atau memang mustahil.",
          "Kegunaannya", GREEN, 10)
    y2 = y + 42
    p.box(16, y2, 306,
          "SNR ini dihitung DI FREKUENSI ECG, sehingga ia MEMBUTUHKAN ECG sebagai acuan. "
          "Konsekuensinya: ia adalah ALAT DIAGNOSIS untuk menjelaskan mengapa sebuah sesi gagal, "
          "BUKAN penyaring kualitas yang bisa dipakai saat alat dipakai sungguhan (di lapangan tidak ada ECG).\n\n"
          "Pengganti tanpa-ECG sudah diuji (peak-to-median, entropi spektral, jitter estimasi) namun "
          "akurasinya baru 67-75% - belum layak diklaim, dan dicantumkan sebagai saran penelitian lanjutan. "
          "Kejujuran ini sengaja ditulis di bab keterbatasan.",
          "BATASAN yang harus disampaikan sendiri sebelum ditanya", AMBER, 10)

    # ---- C2b: JITTER (metrik tanpa ECG)
    p.slide(3, "Kalau di lapangan tidak ada ECG, pakai metrik apa?",
            "Jawaban: JITTER. Dihitung sepenuhnya dari sinyal radar - nol ECG.")
    y = p.get_y()
    p.box(16, y, 306,
          "JITTER  =  median ( |selisih antar estimasi berurutan| ),  diambil dari deret estimasi MENTAH "
          "(sebelum median filter).\n\n"
          "Alasannya fisiologis dan sederhana: DETAK JANTUNG MANUSIA TIDAK BISA MELOMPAT. Kalau estimasi "
          "per-jendela melompat-lompat, itu tanda estimator sedang mengunci NOISE, bukan jantung. Jadi "
          "ketidakstabilan estimasi ITU SENDIRI adalah alarm bahwa hasilnya tak bisa dipercaya.\n\n"
          "Ambang: jitter < 3 bpm  ->  hasil bisa dipercaya.",
          "Definisi dan alasannya", NAVY, 10)
    p.image(str(fig / "q1c_jitter.png"), x=24, y=p.get_y() + 1, w=290)

    p.slide(4, "Jitter (lanjutan): seberapa bisa dipercaya?",
            "Diuji terhadap 3 kandidat metrik lain, dengan validasi leave-one-out.")
    y = p.get_y()
    p.box(16, y, 150,
          "jitter                    r = +0.78     LOO = 83%\n"
          "peak-to-median            r = -0.55     LOO = 58%\n"
          "harmonik (energi di 2f)   r = +0.22     LOO = 42%\n"
          "kecocokan PSD vs autocorr r = +0.64     LOO = 33%\n"
          "-------------------------------------------------\n"
          "(tebak mayoritas)                       LOO = 58%\n\n"
          "r = korelasi terhadap log(MAPE)\n"
          "LOO = akurasi leave-one-out",
          "Empat kandidat diuji, jitter menang telak", GREEN, 9.5)
    p.box(172, y, 150,
          "Ambang disetel pada 11 sesi, lalu diuji pada 1 sesi yang DITINGGALKAN. "
          "Diulang untuk semua 12 sesi.\n\n"
          "Ini menghindari klaim palsu akibat menyetel ambang di data yang sama "
          "dengan data ujinya.",
          "Kenapa leave-one-out", NAVY, 9.5)
    y2 = y + 52
    y2 = p.box(16, y2, 306,
               "Kalau ambang disetel di data yang sama, akurasinya terlihat 92%. Itu OVERFITTING - hanya ada "
               "12 sesi dan satu parameter bebas. Angka yang SAH dilaporkan adalah hasil leave-one-out: 83%, "
               "dibanding tebak-mayoritas 58%.\n\n"
               "Sampaikan 83%, JANGAN 92%. Kalau penguji yang menemukan sendiri bahwa ambang disetel di data uji, "
               "kredibilitas runtuh. Kalau kita yang mengucapkannya duluan, itu justru menunjukkan kedewasaan "
               "metodologis - dan itu persis semangat Kontribusi 4.",
               "KEJUJURAN yang harus disampaikan SENDIRI, sebelum ditanya", RED, 9.8)
    p.box(16, y2 + 2, 306,
          "Baru 12 sesi - sampelnya kecil, perlu validasi di data lebih banyak. Dan ini memprediksi kualitas "
          "SATU SESI, belum diuji sebagai gerbang per-jendela secara real-time. Karena itu dilaporkan sebagai "
          "TEMUAN AWAL YANG MENJANJIKAN, bukan klaim final. Temuan ini memperkuat Kontribusi 3 (batas "
          "keberlakuan metode), bukan menjadi kontribusi baru.",
          "Batasan yang diakui terbuka", AMBER, 9.8)

    # ---- C2: turunan fase
    p.slide(5, "Pertanyaan 2: turunan fase itu sinyalnya diapakan?",
            "Jawaban singkat: selisih antar sampel. Dari PERPINDAHAN dada menjadi KECEPATAN dada.")
    y = p.get_y()
    p.box(16, y, 306,
          "Rumusnya cuma satu baris:      d[n] = x[n] - x[n-1]        (np.diff)\n\n"
          "Secara fisik: x adalah PERPINDAHAN dinding dada (mm). Selisihnya adalah KECEPATAN dada (mm per sampel).\n\n"
          "Secara frekuensi: operasi selisih adalah filter dengan penguatan |H(f)| = 2 |sin(pi f / fs)|, "
          "yang untuk frekuensi rendah naik SEBANDING dengan f. Dua akibatnya:\n"
          "   (a) di f = 0 penguatannya NOL  ->  drift / pergeseran pelan fase terhapus total\n"
          "   (b) jantung (1.2 Hz) diperkuat 4.0x lebih besar daripada napas (0.3 Hz)",
          "Apa yang terjadi", NAVY, 10)
    p.image(str(fig / "q2_deriv.png"), x=22, y=p.get_y() + 1, w=294)

    # ---- C2b
    p.slide(6, "Pertanyaan 2 (lanjutan): buktinya di data",
            "Prediksi teori 4.0x. Terukur di data 3.9x. Cocok.")
    y = p.get_y()
    p.image(str(fig / "q2b_ratio.png"), x=22, y=y + 2, w=294)
    y2 = y + 96
    p.box(16, y2, 150,
          "Sebelum turunan, napas rata-rata 4.9x lebih besar daripada jantung - "
          "inilah kenapa pencarian puncak sering salah tertarik ke napas.\n\n"
          "Sesudah turunan, tinggal 1.3x - keduanya nyaris setara, sehingga puncak "
          "jantung bisa terpilih dengan benar.",
          "Efeknya", GREEN, 10)
    p.box(172, y2, 150,
          "Langkah ini BUKAN karangan sendiri. TI sendiri melakukannya di dalam "
          "pipeline-nya - Developer's Guide halaman 9:\n\n"
          "\"Phase Difference: ... performed on the unwrapped phase by subtracting "
          "successive phase values. This helps in enhancing the heart-beat signal "
          "and removing any phase drifts.\"\n\n"
          "Kita menerapkannya sendiri karena kita MELEWATI pipeline TI yang rusak.",
          "Didukung dokumentasi resmi TI", PURPLE, 9.5)

    # ---- C3: kontribusi
    p.slide(7, "Pertanyaan 3: kontribusinya di mana? Cuma menguji alat?",
            "Jawaban singkat: 'menguji alat' berhenti di angka. Ini melampaui itu di tiga tingkat.")
    y = p.get_y()
    p.box(16, y, 150,
          "Menekan tombol, membaca keluaran alat, melaporkan angkanya akurat atau tidak.\n\n"
          "Berhenti di situ. Kalau angkanya salah, tidak tahu kenapa, dan tidak ada yang "
          "bisa diperbuat.",
          "Yang DISEBUT 'sekadar menguji alat'", GREY, 10)
    p.box(172, y, 150,
          "Angka keluaran alat ternyata SALAH. Lalu ditelusuri KENAPA sampai ke tingkat "
          "byte dan konfigurasi. Lalu DIBANGUN penggantinya. Lalu ditetapkan KAPAN "
          "pengganti itu bisa dipercaya.\n\n"
          "Itu bukan pengujian. Itu rekayasa.",
          "Yang SEBENARNYA dikerjakan", GREEN, 10)
    y2 = y + 40
    y2 = p.box(16, y2, 306,
               "Keluaran bawaan alat meleset ~90% dan penyebabnya ditelusuri sampai akar: frame rate 2.5x "
               "terlalu cepat membuat filter internal justru MEMBUANG detak jantung (terbukti: 94.3% energi "
               "di pita yang salah), dan perekam salah membaca 4 kolom sehingga kolom 'detak jantung' berisi "
               "LAJU NAPAS (terbukti: reproduksi 100.00% persis). Ini jebakan nyata yang bisa menimpa peneliti "
               "lain, dan belum terdokumentasi sebaik ini.",
               "KONTRIBUSI 1 - Analisis akar kegagalan (bukan 'alatnya jelek', tapi 'INI persis kenapa')",
               RED, 9.8)
    y2 = p.box(16, y2 + 1.5, 306,
               "Dibangun rantai pemrosesan sinyal pengganti yang hanya memakai satu kolom yang selamat "
               f"(unwrapPhasePeak_mm). Hasilnya: {K['npass']} dari {K['n']} subjek MEMENUHI standar "
               f"internasional ANSI/CTA-2065 (MAPE < {MAPE_STANDARD:.0f}%), terbaik MAE {K['best_mae']:.1f} bpm / "
               f"MAPE {K['best_mape']:.1f}% - dibanding keluaran bawaan yang NOL subjek lolos. "
               "Tanpa machine learning, tanpa mengganti perangkat keras.",
               "KONTRIBUSI 2 - Pipeline pengganti, divalidasi standar yang diakui (bukan target karangan sendiri)",
               GREEN, 9.8)
    y2 = p.box(16, y2 + 1.5, 306,
               "Ditetapkan KAPAN metode ini bisa dipakai dan kapan secara fisik mustahil: ambang SNR 1.8 "
               "memisahkan berhasil/gagal dengan benar di 11 dari 12 sesi, dikuatkan eksperimen variasi jarak "
               "(50 cm lolos, 100 cm sinyalnya setara noise). Kebanyakan paper radar vital-sign hanya melaporkan "
               "kasus yang berhasil; menetapkan BATAS KEBERLAKUAN justru lebih jujur dan lebih berguna.",
               "KONTRIBUSI 3 - Batas keberlakuan metode (ini yang paling bernilai ilmiah)",
               NAVY, 9.8)
    p.box(16, y2 + 1.5, 306,
          "Dibongkar bahwa model ML terdahulu sebenarnya KALAH dari menebak angka konstan, dan bahwa korelasi "
          "adalah metrik yang keliru untuk dataset dengan detak jantung nyaris statis. Setiap klaim akurasi "
          "kini wajib disertai pembanding trivial baseline.",
          "KONTRIBUSI 4 - Koreksi metodologi evaluasi", PURPLE, 9.8)

    # ---- C3b: kalimat penutup
    p.slide(8, "Pertanyaan 3 (lanjutan): satu kalimat untuk menjawabnya",
            None)
    y = p.get_y() + 6
    p.set_xy(24, y)
    p.set_font("Helvetica", "B", 15)
    p.set_text_color(*NAVY)
    p.multi_cell(292, 9,
                 "\"Kalau tesis ini sekadar menguji alat, hasilnya akan berhenti di kalimat\n"
                 "'alat ini tidak akurat'. Yang saya lakukan adalah menemukan MENGAPA ia tidak\n"
                 "akurat sampai ke tingkat byte dan konfigurasi, MEMBANGUN pipeline pengganti\n"
                 "yang memenuhi standar internasional, dan MENETAPKAN batas kapan pipeline itu\n"
                 "bisa dipercaya. Ketiganya dibuktikan secara kuantitatif dan dapat direproduksi.\"")
    p.ln(6)
    y = p.get_y()
    p.box(24, y, 292,
          "Perangkatnya sendiri TIDAK bersalah - sensornya menangkap detak jantung dengan baik. "
          "Yang gagal adalah rantai pemrosesan dan konfigurasi pengambilan datanya. Justru karena "
          "itulah tesis ini punya sesuatu untuk disumbangkan: ia tidak berhenti pada vonis, "
          "melainkan menjelaskan, memperbaiki, dan menetapkan batasnya.",
          "Inti pertahanannya", GREEN, 11)
    y = p.get_y() + 4
    p.box(24, y, 292,
          "\"Penelitian ini TIDAK memberikan vonis kelayakan terhadap library Vital Signs bawaan TI, "
          "karena data direkam dengan konfigurasi yang tidak sesuai spesifikasi TI. Yang dievaluasi "
          "adalah kelayakan SINYAL FASE MENTAH radar sebagai basis estimasi detak jantung.\"\n\n"
          "Sampaikan ini SENDIRI di awal. Dengan begitu ketiga cacat berubah dari kelemahan menjadi "
          "TEMUAN, dan penguji tidak bisa mematahkan tesis dengan argumen 'pengujianmu tidak adil'.",
          "Kalimat WAJIB di Batasan Masalah - ini yang melindungi Anda", AMBER, 10.5)

    # ---- C9: HRV
    p.slide(9, "Pertanyaan 4: dari sinyal fase ini bisa dapat HRV juga?",
            "Jawaban singkat: TIDAK, dan ini sudah diukur - bukan dugaan. Alasannya bukan algoritma, tapi batas fisik.")
    y = p.get_y()
    p.box(16, y, 306,
          "BPM dan HRV terlihat mirip, padahal bebannya jauh berbeda:\n\n"
          f"   BPM  -> cukup frekuensi dominan per jendela {WINDOW_SEC:.0f} detik. Itu MERATA-RATAKAN puluhan detak "
          "sekaligus, sehingga beberapa detak yang tenggelam di noise masih tertutupi tetangganya.\n"
          "   HRV  -> butuh WAKTU SETIAP DETAK, satu per satu, dengan presisi milidetik. Tidak ada "
          "perata-rataan yang menyelamatkan: satu detak terlewat langsung merusak DUA interval sekaligus.",
          "Kenapa HRV jauh lebih berat daripada BPM", NAVY, 10)
    p.image(str(fig / "q4_hrv.png"), x=24, y=p.get_y() + 1, w=290)

    p.slide(10, "Pertanyaan 4 (lanjutan): angkanya, dan mengapa ini justru menguatkan tesis",
            "Skrip: code/evaluation/hrv_analysis.py - dijalankan pada 10 subjek position_front.")
    y = p.get_y()
    p.box(16, y, 150,
          "                     ECG   sempit   lebar\n"
          "RMSSD (ms)            46      210     178\n"
          "SDNN  (ms)            52      164     128\n"
          "R-peak ketemu (%)      -       52      61\n\n"
          "Untuk HRV yang sah, 'R-peak ketemu' harus > 95%.",
          "Hasil terukur (rata-rata 10 subjek)", RED, 9.8)
    p.box(172, y, 150,
          "RMSSD radar melebar di KEDUA pita. Kalau penyebabnya pilihan filter, pita sempit "
          "dan pita lebar akan berbeda arah.\n\n"
          "Karena keduanya sama-sama meleset ~4x, penyebabnya adalah DETEKSI DETAKNYA: puncak "
          "palsu dan detak terlewat. Angka 210 ms itu ukuran KESALAHAN, bukan variabilitas jantung.",
          "Cara membacanya", NAVY, 9.8)
    y2 = y + 46
    y2 = p.box(16, y2, 306,
               "   (a) SNR jantung hanya 1.4 - 5.9  ->  puncak tiap detak individual sering tenggelam di noise. "
               "Ini batas yang SAMA yang membuat subject03/04/06 gagal, dan yang membuat sesi jarak 100 cm gagal total.\n"
               "   (b) Presisi waktu 20 ms per sampel (50 fps) sebanding dengan besaran yang hendak diukur "
               "(RMSSD saat istirahat hanya 20 - 50 ms). Seperti mengukur benda 40 mm dengan penggaris berskala 20 mm.\n\n"
               "Keduanya TIDAK bisa diperbaiki dengan algoritma - hanya dengan rekam ulang (jarak ~50 cm, frame rate lebih tinggi).",
               "Dua batas keras, dan keduanya BUKAN soal algoritma", AMBER, 9.8)
    p.box(16, y2 + 2, 306,
          f"\"Sinyal fase radar terbukti layak untuk estimasi DETAK RATA-RATA per jendela ({K['npass']} dari {K['n']} subjek memenuhi "
          "ANSI/CTA-2065), namun BELUM layak untuk HRV: deteksi detak-per-detak hanya mencapai 61% terhadap R-peak ECG, "
          "dan RMSSD hasil radar merupakan 4x nilai ECG.\"\n\n"
          "Ini hasil NEGATIF, dan justru memperkuat KONTRIBUSI 3 (batas keberlakuan). Menetapkan dengan angka SAMPAI MANA "
          "sebuah metode berlaku lebih jujur - dan lebih berguna - daripada hanya melaporkan yang berhasil.",
          "Kalimat untuk tesis - jangan dibuang, ini justru nilai tambah", GREEN, 9.8)

    p.output(str(out / "antisipasi_pertanyaan.pdf"))
    print("ok:", out / "antisipasi_pertanyaan.pdf")


def main(csv, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "_fig_qna"
    figdir.mkdir(exist_ok=True)
    df = pd.read_csv(csv)
    front = df[(df.dataset == "position_front") & df.unwrapPhasePeak_mm.notna()].copy()
    fig_snr(front, figdir)
    fig_jitter(front, figdir)
    fig_hrv(figdir)
    fig_deriv(front, figdir)
    fig_ratio(front, figdir)
    # angka kunci DITURUNKAN dari pipeline -- jangan pernah diketik tangan
    mp = []
    for _, sub in front.groupby("subject_id"):
        e, r, _ = estimate(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                           sub.gt_heart_rate.values)
        mae, mape, _ = metrics(e, r)
        mp.append((mape, mae))
    K = dict(n=len(mp),
             npass=sum(1 for m, _ in mp if m < MAPE_STANDARD),
             best_mape=min(m for m, _ in mp),
             best_mae=min(a for _, a in mp))
    print("figure ok")
    build(figdir, out, K)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
