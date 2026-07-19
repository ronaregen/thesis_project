"""
Slide: BAGAIMANA MAE & MAPE DIPEROLEH - langkah demi langkah.

Dua jalur diproses terpisah lalu dipertemukan:
  jalur A (ECG Attys)  -> detak jantung SEBENARNYA (ground truth)
  jalur B (radar)      -> detak jantung ESTIMASI
  keduanya bertemu     -> selisihnya = MAE dan MAPE

Semua grafik memakai sinyal ASLI dari data, bukan ilustrasi.

Usage:
    python make_slides_pipeline.py <aligned_csv> <outdir>
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from phase_pipeline import (FS, HR_BAND, BR_BAND, WINDOW_SEC, OVERLAP,  # noqa: E402
                            MEDFILT_K, MAPE_STANDARD, estimate, metrics)
from extract_ground_truth import detect_r_peaks  # noqa: E402

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": .3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

NAVY, RED, GREEN, GREY, PURPLE = (28, 42, 66), (192, 57, 43), (30, 132, 73), (110, 118, 128), (110, 60, 140)
AMBER = (176, 122, 12)
C_BAD, C_GOOD, C_GT, C_NEU = "#c0392b", "#1e8449", "#2c3e50", "#7f8c8d"
C_ECG, C_RAD = "#1e8449", "#e67e22"
W, H = 338.7, 190.5

ROOT = Path(__file__).resolve().parents[2]
DEMO = "subject01"


# ---------------------------------------------------------------- flowchart
def fig_flow(out):
    fig, ax = plt.subplots(figsize=(14, 7.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, txt, fc, ec, fs=9.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=fc, ec=ec, lw=1.6))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.4)

    def arrow(x1, y1, x2, y2, c="#555"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=15, lw=1.6, color=c))

    TOP, GAP, BH = 7.55, .74, .52

    def chain(cx, x0, items, fc, ec):
        y = TOP
        for i, t in enumerate(items):
            box(x0, y - BH, 5.2, BH, t, fc, ec, 8.6, i == 0)
            if i < len(items) - 1:
                arrow(cx, y - BH, cx, y - GAP)
            y -= GAP + (BH - .52)
        return y - BH + .52

    # jalur A: ECG
    ax.text(3.0, 8.0, "JALUR A  --  ECG Attys  ->  detak SEBENARNYA", ha="center",
            fontsize=11.5, fontweight="bold", color=C_ECG)
    ya = chain(3.0, 0.4, [
        "attys.csv\n125 Hz, sinyal ECG mentah",
        "Bandpass 5-15 Hz\n(isolasi kompleks QRS)",
        "Turunan -> kuadrat ->\nrata-rata bergerak 150 ms",
        "find_peaks -> waktu tiap R-peak",
        "RR = selisih antar R-peak\nHR = 60 / RR",
        "buang HR di luar 40-180 bpm",
    ], "#eaf6ee", C_ECG)

    # jalur B: radar
    ax.text(11.0, 8.0, "JALUR B  --  radar  ->  detak ESTIMASI", ha="center",
            fontsize=11.5, fontweight="bold", color=C_RAD)
    yb = chain(11.0, 8.4, [
        "radar.csv -> unwrapPhasePeak_mm\n~50 Hz, tak seragam",
        "Resample 25 Hz + anti-alias",
        "TURUNAN fase: d[n] = x[n] - x[n-1]",
        "Kurangi komponen napas 0.15-0.6 Hz",
        "Bandpass jantung 0.8-2.0 Hz",
        f"Welch PSD per jendela {WINDOW_SEC:.0f} s -> puncak",
        f"Median filter {MEDFILT_K} titik",
    ], "#fdf0e3", C_RAD)

    # pertemuan
    ym = min(ya, yb) - 1.05
    box(3.9, ym, 6.2, .62,
        f"PERTEMUAN: untuk tiap jendela {WINDOW_SEC:.0f} detik, ambil RATA-RATA HR ECG di jendela itu",
        "#eef1f5", C_GT, 9, True)
    arrow(3.0, ya, 4.4, ym + .62, C_ECG)
    arrow(11.0, yb, 9.6, ym + .62, C_RAD)

    box(3.9, ym - 1.0, 6.2, .68,
        "MAE  = rata-rata |estimasi - sebenarnya|        (bpm)\n"
        "MAPE = rata-rata |estimasi - sebenarnya| / sebenarnya   (%)",
        "#fdecea", C_BAD, 9.5, True)
    arrow(7.0, ym, 7.0, ym - .32, "#555")

    fig.savefig(out / "p1_flow.png")
    plt.close(fig)


# ---------------------------------------------------------------- jalur ECG
def fig_ecg(out):
    attys = pd.read_csv(ROOT / "data" / "raw" / "position_front" / DEMO / "attys.csv")
    t, x = attys["timestamp"].values, attys["value"].values
    fs = 1.0 / np.median(np.diff(t))

    s = len(t) // 2
    n = int(5 * fs)                      # 5 detik
    sl = slice(s, s + n)
    tt = t[sl] - t[s]

    sos = signal.butter(3, [5, 15], "bandpass", fs=fs, output="sos")
    xb = signal.sosfiltfilt(sos, x)
    xd = np.diff(xb, prepend=xb[0])
    xs = xd ** 2
    win = max(1, int(0.15 * fs))
    xi = np.convolve(xs, np.ones(win) / win, mode="same")
    thr = np.mean(xi) + 0.5 * np.std(xi)
    pk, _ = signal.find_peaks(xi, distance=max(1, int(0.27 * fs)), height=thr)

    fig, axs = plt.subplots(2, 2, figsize=(13.6, 5.6))
    axs = axs.ravel()

    axs[0].plot(tt, x[sl] * 1e6, color=C_NEU, lw=1.1)
    axs[0].set_title("1. ECG MENTAH dari Attys (125 Hz)", fontweight="bold", fontsize=11)
    axs[0].set_ylabel("mikrovolt")

    axs[1].plot(tt, xb[sl] * 1e6, color=C_ECG, lw=1.2)
    axs[1].set_title("2. Bandpass 5-15 Hz  ->  kompleks QRS menonjol,\n"
                     "drift & noise otot hilang", fontweight="bold", fontsize=11)
    axs[1].set_ylabel("mikrovolt")

    axs[2].plot(tt, xi[sl] / xi[sl].max(), color=PURPLE_HEX, lw=1.4)
    axs[2].axhline(thr / xi[sl].max(), color=C_BAD, ls="--", lw=1.6, label="ambang")
    axs[2].set_title("3. Turunan -> kuadrat -> rata-rata bergerak 150 ms\n"
                     "(tiap QRS jadi satu gundukan tunggal)", fontweight="bold", fontsize=11)
    axs[2].set_xlabel("detik"); axs[2].set_ylabel("energi (ternorm.)")
    axs[2].legend(fontsize=8.5)

    # tampilkan WAKTU R-peak (itu yang dipakai), bukan amplitudonya
    axs[3].plot(tt, x[sl] * 1e6, color=C_NEU, lw=1.0, alpha=.7, label="ECG mentah")
    inw = pk[(pk >= s) & (pk < s + n)]
    top = x[sl].max() * 1e6
    for k, i in enumerate(inw):
        axs[3].axvline(t[i] - t[s], color=C_BAD, lw=1.8, alpha=.85,
                       label="waktu R-peak terdeteksi" if k == 0 else None)
    for i in range(len(inw) - 1):
        x0, x1 = t[inw[i]] - t[s], t[inw[i + 1]] - t[s]
        yv = top * 1.28
        axs[3].annotate("", xy=(x1, yv), xytext=(x0, yv),
                        arrowprops=dict(arrowstyle="<->", lw=1.4, color=C_GT))
        axs[3].text((x0 + x1) / 2, yv * 1.05, f"{(x1-x0)*1000:.0f}",
                    ha="center", fontsize=7.5, color=C_GT, fontweight="bold")
    hr = 60.0 / np.mean(np.diff(t[inw]))
    axs[3].set_title(f"4. Waktu R-peak -> interval RR (ms) -> HR = 60 / RR\n"
                     f"rata-rata jendela ini = {hr:.0f} bpm",
                     fontweight="bold", fontsize=11)
    axs[3].set_xlabel("detik"); axs[3].set_ylabel("mikrovolt")
    axs[3].legend(fontsize=8, loc="lower right")
    axs[3].set_ylim(top=top * 1.5)

    fig.suptitle(f"JALUR A - dari ECG mentah menjadi detak jantung SEBENARNYA  ({DEMO}, Pan-Tompkins)",
                 fontweight="bold", fontsize=13.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "p2_ecg.png")
    plt.close(fig)


PURPLE_HEX = "#6e3c8c"


# ---------------------------------------------------------------- jalur radar
def fig_radar(front, out):
    sub = front[front.subject_id == DEMO].sort_values("Timestamp")
    t = sub.Timestamp.values
    x = sub.unwrapPhasePeak_mm.values
    o = np.argsort(t)
    t, x = t[o], x[o]

    fso = 1 / np.median(np.diff(t))
    xl = signal.sosfiltfilt(signal.butter(6, FS / 2 * .9, "low", fs=fso, output="sos"), x)
    tu = np.arange(t[0], t[-1], 1 / FS)
    xu = np.interp(tu, t, xl)
    d = np.diff(xu, prepend=xu[0])

    w = int(WINDOW_SEC * FS)
    step = max(1, int(w * (1 - OVERLAP)))
    gu = np.interp(tu, t, sub.gt_heart_rate.values[o])

    sb = signal.butter(4, BR_BAND, "bandpass", fs=FS, output="sos")
    sh = signal.butter(4, HR_BAND, "bandpass", fs=FS, output="sos")

    def est_win(s2):
        g = signal.detrend(d[s2:s2 + w])
        g = g - signal.sosfiltfilt(sb, g)
        g = signal.sosfiltfilt(sh, g)
        f2, p2 = signal.welch(g, fs=FS, nperseg=w)
        m2 = (f2 >= HR_BAND[0]) & (f2 <= HR_BAND[1])
        return f2[m2][np.argmax(p2[m2])] * 60

    # pilih jendela TIPIKAL: galatnya = median galat seluruh sesi.
    # (bukan yang terbaik - itu cherry-picking; bukan yang di tengah rekaman -
    #  itu kebetulan jendela gagal dan menyesatkan sebagai contoh cara kerja)
    starts = list(range(0, len(d) - w, step))
    errs = np.array([abs(est_win(s2) - gu[s2:s2 + w].mean()) for s2 in starts])
    s = starts[int(np.argsort(errs)[len(errs) // 2])]

    seg0 = signal.detrend(d[s:s + w])
    seg1 = seg0 - signal.sosfiltfilt(sb, seg0)
    seg2 = signal.sosfiltfilt(sh, seg1)
    tt = np.arange(w) / FS

    fig, axs = plt.subplots(2, 3, figsize=(15, 5.8))
    axs = axs.ravel()

    axs[0].plot(tt, xu[s:s + w] - xu[s:s + w].mean(), color=C_NEU, lw=1.3)
    axs[0].set_title("1. unwrapPhasePeak_mm (setelah resample)\nPERPINDAHAN dada - napas mendominasi",
                     fontweight="bold", fontsize=10)
    axs[0].set_ylabel("mm")

    axs[1].plot(tt, seg0, color=C_RAD, lw=1.1)
    axs[1].set_title("2. TURUNAN fase: d[n] = x[n] - x[n-1]\nKECEPATAN dada - drift hilang",
                     fontweight="bold", fontsize=10)
    axs[1].set_ylabel("mm/sampel")

    axs[2].plot(tt, seg1, color="#2980b9", lw=1.1)
    axs[2].set_title("3. Kurangi komponen napas (0.15-0.6 Hz)",
                     fontweight="bold", fontsize=10)

    axs[3].plot(tt, seg2, color=C_BAD, lw=1.3)
    axs[3].set_title("4. Bandpass jantung 0.8-2.0 Hz\n(= 48-120 bpm)", fontweight="bold", fontsize=10)
    axs[3].set_xlabel("detik"); axs[3].set_ylabel("amplitudo")

    f, p = signal.welch(seg2, fs=FS, nperseg=w)
    m = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    fb, pb = f[m], p[m]
    i = int(np.argmax(pb))
    gt = float(gu[s:s + w].mean())
    axs[4].plot(fb * 60, pb, color=C_GT, lw=1.8)
    axs[4].plot(fb[i] * 60, pb[i], "o", color=C_BAD, ms=12, zorder=5,
                label=f"puncak = ESTIMASI\n{fb[i]*60:.1f} bpm")
    axs[4].axvline(gt, color=C_ECG, lw=2, ls="--", label=f"ECG = {gt:.1f} bpm")
    axs[4].set_title(f"5. Welch PSD -> frekuensi puncak\n(jendela TIPIKAL, galat {abs(fb[i]*60-gt):.1f} bpm)",
                     fontweight="bold", fontsize=10)
    axs[4].set_xlabel("bpm"); axs[4].set_ylabel("daya")
    axs[4].legend(fontsize=8)

    est, ref, _ = estimate(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                           sub.gt_heart_rate.values)
    raw = np.array([est_win(s2) for s2 in starts])
    tw = np.arange(len(raw)) * step / FS

    axs[5].plot(tw, raw, color=C_NEU, lw=1, alpha=.6, label="estimasi mentah")
    axs[5].plot(tw[:len(est)], est, color=C_BAD, lw=2, label=f"setelah median filter {MEDFILT_K}")
    axs[5].plot(tw[:len(ref)], ref, color=C_ECG, lw=2, ls="--", label="ECG (sebenarnya)")
    axs[5].set_title("6. Median filter 5 titik\n(detak jantung tak bisa melompat)",
                     fontweight="bold", fontsize=10)
    axs[5].set_xlabel("detik"); axs[5].set_ylabel("bpm")
    axs[5].legend(fontsize=7.5)

    fig.suptitle(f"JALUR B - dari sinyal fase radar menjadi detak jantung ESTIMASI  ({DEMO})",
                 fontweight="bold", fontsize=13.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "p3_radar.png")
    plt.close(fig)


# ---------------------------------------------------------------- MAE/MAPE
def fig_mae(front, out):
    sub = front[front.subject_id == DEMO]
    est, ref, _ = estimate(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                           sub.gt_heart_rate.values)
    mae, mape, bias = metrics(est, ref)
    err = np.abs(est - ref)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 4.2),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    tw = np.arange(len(est)) * WINDOW_SEC * (1 - OVERLAP)
    a1.plot(tw, ref, color=C_ECG, lw=2.2, ls="--", label="ECG (sebenarnya)")
    a1.plot(tw, est, color=C_BAD, lw=2, label="radar (estimasi)")
    a1.fill_between(tw, ref, est, color=C_BAD, alpha=.16, label="selisih = galat")
    a1.set_xlabel("detik"); a1.set_ylabel("bpm")
    a1.legend(fontsize=9)
    a1.set_title(f"{DEMO}: tiap jendela menghasilkan SATU pasang angka\n"
                 f"({len(est)} jendela)", fontweight="bold", fontsize=11.5)

    a2.hist(err, bins=18, color=C_BAD, alpha=.75)
    a2.axvline(mae, color=C_GT, lw=2.5, ls="--", label=f"MAE = {mae:.1f} bpm")
    a2.set_xlabel("|galat| tiap jendela (bpm)"); a2.set_ylabel("jumlah jendela")
    a2.legend(fontsize=9)
    a2.set_title(f"MAE = rata-rata dari galat ini\nMAPE = {mape:.1f}%  (galat relatif)",
                 fontweight="bold", fontsize=11.5)

    fig.suptitle("MAE dan MAPE dihitung PER JENDELA, lalu dirata-ratakan",
                 fontweight="bold", fontsize=13.5, y=1.04)
    fig.savefig(out / "p4_mae.png")
    plt.close(fig)


# ---------------------------------------------------------------- PDF
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
        self.cell(10, 5, f"P{n}", align="R")
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


def build(fig, out, res, agg):
    p = Slides()

    # P1 flowchart
    p.slide(1, "Bagaimana MAE dan MAPE diperoleh - peta keseluruhan",
            "Dua jalur diproses TERPISAH, lalu dipertemukan. Selisih keduanya = galat.")
    p.image(str(fig / "p1_flow.png"), x=26, y=p.get_y() + 1, w=286)

    # P2 ECG
    p.slide(2, "Jalur A: ECG Attys  ->  detak jantung SEBENARNYA",
            "Metode Pan-Tompkins. Tujuannya: menemukan waktu tiap R-peak, bukan mengukur frekuensi.")
    p.image(str(fig / "p2_ecg.png"), x=28, y=p.get_y() + 1, w=282)
    y = p.get_y() + 1
    p.box(16, y, 306,
          "Kenapa TIDAK memakai FFT langsung pada ECG? Sudah dicoba dan hasilnya kacau (90-180 bpm naik-turun "
          "tak masuk akal). Sebabnya: energi ECG yang penting (kompleks QRS) tersebar di frekuensi TINGGI "
          "(5-15 Hz), bukan di pita detak jantung (0.8-2.0 Hz). Jadi kita mencari LOKASI WAKTU tiap detak, "
          "lalu HR dihitung dari jarak antar detak: HR = 60 / RR.",
          "Catatan penting", AMBER, 9.8)

    # P3 radar
    p.slide(3, "Jalur B: sinyal fase radar  ->  detak jantung ESTIMASI",
            "Enam langkah. Setiap grafik di bawah adalah sinyal ASLI subject01, bukan ilustrasi.")
    p.image(str(fig / "p3_radar.png"), x=20, y=p.get_y() + 1, w=298)

    # P4 pertemuan
    p.slide(4, "Pertemuan kedua jalur: menyamakan satuan waktu",
            f"Masalahnya: ECG memberi satu nilai per DETAK, radar memberi satu nilai per JENDELA {WINDOW_SEC:.0f} detik.")
    y = p.get_y()
    p.box(16, y, 150,
          f"1. Radar dipotong menjadi jendela {WINDOW_SEC:.0f} detik, bergeser {WINDOW_SEC*(1-OVERLAP):.0f} detik "
          f"(tumpang tindih {OVERLAP*100:.0f}%).\n\n"
          "2. Tiap jendela menghasilkan SATU estimasi bpm.\n\n"
          "3. Untuk jendela yang sama, HR ECG di dalamnya DIRATA-RATAKAN menjadi satu angka.\n\n"
          "4. Jadi tiap jendela punya satu pasang: (estimasi, sebenarnya).",
          "Cara menyamakannya", NAVY, 9.8)
    p.box(172, y, 150,
          f"Kenapa jendela {WINDOW_SEC:.0f} detik? Karena resolusi frekuensi = 60 / durasi.\n\n"
          f"   20 detik  ->  3.0 bpm\n"
          f"   {WINDOW_SEC:.0f} detik  ->  {60/WINDOW_SEC:.1f} bpm   <- yang dipakai\n\n"
          "Makin panjang jendela, makin halus resolusinya - tetapi makin lamban mengikuti "
          "perubahan HR. Di dataset ini HR subjek praktis DIAM (std 0.9-4.2 bpm), jadi tidak "
          "ada dinamika yang hilang: jendela panjang murni untung.\n\n"
          "Batasannya: pada subjek yang HR-nya berubah cepat, jendela ini akan terlambat "
          "mengikuti - dan itu tidak bisa diuji dengan dataset ini.",
          f"Kenapa {WINDOW_SEC:.0f} detik", PURPLE, 9.8)
    y2 = y + 48
    y2 = p.box(16, y2, 306,
               "   MAE  = rata-rata ( | estimasi - sebenarnya | )                     satuan: bpm\n"
               "   MAPE = rata-rata ( | estimasi - sebenarnya | / sebenarnya ) x 100  satuan: %\n\n"
               "Bedanya: MAE adalah galat MUTLAK. Galat 8 bpm sama beratnya, entah HR-nya 50 atau 150.\n"
               "MAPE adalah galat RELATIF. Galat 8 bpm pada HR 50 (16%) lebih parah daripada pada HR 150 (5%).\n\n"
               "Standar ANSI/CTA-2065 memakai MAPE, bukan MAE - karena itulah kita melaporkan MAPE sebagai vonis.",
               "Rumusnya", RED, 9.8)
    e0, r0 = agg["ex"]
    p.box(16, y2 + 2, 306,
          f"Satu jendela NYATA dari {DEMO} (jendela dengan galat tipikal):\n"
          f"   radar mengestimasi {e0:.1f} bpm, ECG menyatakan {r0:.1f} bpm\n"
          f"   galat         = |{e0:.1f} - {r0:.1f}|              = {abs(e0-r0):.1f} bpm\n"
          f"   galat relatif = {abs(e0-r0):.1f} / {r0:.1f} x 100  = {100*abs(e0-r0)/r0:.1f}%\n"
          "Ulangi untuk SEMUA jendela, lalu rata-ratakan. Itulah MAE dan MAPE satu subjek.",
          "Contoh satu jendela", GREEN, 9.8)

    # P5 hasil
    p.slide(5, "Hasilnya: MAE dan MAPE tiap subjek",
            f"Vonis memakai MAPE terhadap ambang ANSI/CTA-2065 (< {MAPE_STANDARD:.0f}%).")
    p.image(str(fig / "p4_mae.png"), x=30, y=p.get_y() + 1, w=278)
    y = p.get_y() + 1

    hdr = f"{'subjek':<11}{'MAE':>7}{'MAPE':>8}{'SNR':>7}   vonis"
    lines = [hdr, "-" * 44]
    for sid in sorted(res):
        mae, mape, bias, snr = res[sid]
        lines.append(f"{sid.replace('subject','S'):<11}{mae:>6.1f} {mape:>6.1f}% {snr:>6.2f}   "
                     f"{'LOLOS' if mape < MAPE_STANDARD else 'gagal'}")
    lines += ["-" * 44,
              f"{'RATA2':<11}{agg['mae']:>6.1f} {agg['mape']:>6.1f}%"]
    p.box(16, y, 150, "\n".join(lines), None, NAVY, 8.2)
    p.box(172, y, 150,
          f"LOLOS standar: {agg['npass']} dari 10 subjek.\n"
          f"Keluaran bawaan TI: 0 dari 10 (MAPE ~90%).\n\n"
          "Yang gagal (S03/S04/S06) semuanya ber-SNR < 1.8 - sinyal jantungnya memang "
          "setara noise, jadi bukan kekurangan metode melainkan batas fisik akuisisi.\n\n"
          "S10 adalah pengecualian jujur: SNR-nya baik namun tetap gagal (HR-nya paling "
          "rendah di kohort, 61 bpm).",
          "Membacanya", GREEN, 9.5)

    p.output(str(out / "cara_hitung_mae_mape.pdf"))
    print("ok:", out / "cara_hitung_mae_mape.pdf")


def main(csv, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "_fig_pipe"
    figdir.mkdir(exist_ok=True)

    df = pd.read_csv(csv)
    front = df[(df.dataset == "position_front") & df.unwrapPhasePeak_mm.notna()].copy()

    res, ex = {}, None
    for sid, sub in front.groupby("subject_id"):
        est, ref, snr = estimate(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                                 sub.gt_heart_rate.values)
        mae, mape, bias = metrics(est, ref)
        res[sid] = (mae, mape, bias, snr)
        if sid == DEMO:   # jendela ber-galat tipikal (median), bukan yang terbaik
            k = int(np.argsort(np.abs(est - ref))[len(est) // 2])
            ex = (float(est[k]), float(ref[k]))
    agg = dict(mae=np.mean([v[0] for v in res.values()]),
               mape=np.mean([v[1] for v in res.values()]),
               npass=sum(1 for v in res.values() if v[1] < MAPE_STANDARD),
               ex=ex)

    fig_flow(figdir)
    fig_ecg(figdir)
    fig_radar(front, figdir)
    fig_mae(front, figdir)
    print("figure ok")
    build(figdir, out, res, agg)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
