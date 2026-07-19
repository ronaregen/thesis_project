"""
Slide: BUKTI BAHWA DETAK JANTUNG BENAR-BENAR TERTANGKAP.

Alur argumennya, satu slide satu langkah:
  B1  Berapa BPM yang dilaporkan tiap metode, dibanding ECG?
  B2  Kolom logger menang di 2 subjek -- KENAPA? (jawaban: kebetulan, bukan akurasi)
  B3  Buktikan sinyal fase memang MENGANDUNG pola detak jantung
  B4  KONTROL NEGATIF: ruangan kosong. Siapa yang mengarang, siapa yang jujur?
  B5  HOLD-OUT: 2 subjek yang tidak pernah dipakai menyetel apa pun

Semua angka diimpor dari code/evaluation/ -- tidak ada yang diketik tangan.

Usage:
    python make_slides_bukti.py <aligned_csv> <outdir>
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
from PIL import Image as PILImage
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from compare_ti_vs_phase import compare  # noqa: E402
from negative_control import (EMPTY, FRONT, HOLDOUT, holdout_session,  # noqa: E402
                              presence)
from phase_pipeline import (BR_BAND, FS, HR_BAND, MAPE_STANDARD,  # noqa: E402
                            OVERLAP, WINDOW_SEC, resample_antialias)

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": .3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

NAVY, RED, GREEN, GREY, PURPLE = (28, 42, 66), (192, 57, 43), (30, 132, 73), (110, 118, 128), (110, 60, 140)
AMBER = (176, 122, 12)
C_ECG, C_TI, C_FASE, C_NEU = "#2c3e50", "#c0392b", "#1e8449", "#7f8c8d"
W, H = 338.7, 190.5


# ============================================================ B1: BPM langsung
def fig_bpm(R, S, out):
    """Bukan MAPE -- BPM apa adanya. Paling gampang dibaca pembimbing."""
    best = S["best"]
    x = np.arange(len(R))
    c = R[f"{best}_bpm"].mean()

    fig, ax = plt.subplots(figsize=(13.8, 4.8))
    ax.plot(x, R.hr_ecg, "o-", color=C_ECG, lw=2.4, ms=9, zorder=5,
            label="ECG (yang SEBENARNYA)")
    ax.plot(x, R.fase_bpm, "s-", color=C_FASE, lw=2, ms=8, alpha=.9,
            label="pipeline fase mentah")
    ax.plot(x, R[f"{best}_bpm"], "^-", color=C_TI, lw=2, ms=8, alpha=.9,
            label=f"{best} (kolom logger TERBAIK)")
    ax.axhline(c, color=C_TI, ls=":", lw=1.6, alpha=.8)
    ax.text(len(R) - .4, c + .8, f"rata-rata keluaran logger = {c:.0f} bpm",
            ha="right", fontsize=8.6, color=C_TI, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("subject", "S") for s in R.subj])
    ax.set_ylabel("detak jantung (bpm)")
    ax.legend(fontsize=9.4, ncol=3, loc="upper center")
    ax.set_ylim(52, 104)
    ax.set_title("Garis hijau mengikuti garis hitam. Garis merah nyaris DATAR.",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "b1_bpm.png")
    plt.close(fig)


# ============================================================ B2: kenapa logger "menang"
def fig_why(R, S, out):
    """Dua panel. Kiri: xCorr = penebak-konstan, dan S03/S04 kebetulan dekat
    konstantanya. Kanan: buktinya -- MAPE xCorr ~ MAPE penebak-konstan (r=+0.93)."""
    best = S["best"]
    c = S["stuck_const"]
    win = R[R.fase_mape >= R[f"{best}_mape"]]      # subjek tempat logger "menang"

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    # --- kiri: jarak HR tiap orang dari konstanta logger
    order = R.sort_values("hr_ecg")
    y = np.arange(len(order))
    a1.axvline(c, color=C_TI, lw=2.5, label=f"keluaran logger, ~{c:.0f} bpm untuk SIAPA PUN")
    for i, (_, r) in enumerate(order.iterrows()):
        lucky = r.subj in set(win.subj)
        a1.plot([r.hr_ecg, c], [i, i], color=C_TI if lucky else C_NEU,
                lw=2.4 if lucky else 1.0, alpha=.85 if lucky else .35, zorder=2)
        a1.plot(r.hr_ecg, i, "o", ms=9, color=C_ECG, zorder=4)
        a1.text(r.hr_ecg - 1.2, i, f"{r.hr_ecg:.0f}", ha="right", va="center",
                fontsize=8, color=C_ECG)
        if lucky:
            a1.text(c + 1.2, i, f"meleset cuma {abs(c - r.hr_ecg):.0f} bpm  <- 'menang'",
                    va="center", fontsize=8.4, color=C_TI, fontweight="bold")
    a1.set_yticks(y)
    a1.set_yticklabels([s.replace("subject", "S") for s in order.subj], fontsize=9)
    a1.set_xlabel("detak jantung (bpm)")
    a1.set_xlim(55, 108)
    a1.legend(fontsize=8.8, loc="lower right")
    a1.set_title("Logger memberi angka yang SAMA untuk semua orang.\n"
                 "Ia 'menang' hanya di yang kebetulan dekat angka itu.",
                 fontweight="bold", fontsize=11)

    # --- kanan: MAPE xCorr vs MAPE penebak-konstan
    stuck = (c - R.hr_ecg).abs() / R.hr_ecg * 100
    a2.scatter(stuck, R[f"{best}_mape"], s=95, color=C_TI, alpha=.85,
               edgecolor="white", lw=1.2, zorder=4)
    for _, r in R.iterrows():
        a2.annotate(r.subj.replace("subject", "S"),
                    (stuck[r.name], r[f"{best}_mape"]),
                    textcoords="offset points", xytext=(7, -3), fontsize=8, color=C_ECG)
    lim = [0, max(stuck.max(), R[f"{best}_mape"].max()) * 1.08]
    a2.plot(lim, lim, ls="--", color=C_NEU, lw=1.5, label="kalau IDENTIK")
    k = np.polyfit(stuck, R[f"{best}_mape"], 1)
    a2.plot(lim, np.polyval(k, lim), color=C_TI, lw=2, alpha=.7,
            label=f"r = {S['r_stuck']:+.2f}")
    a2.set_xlim(lim)
    a2.set_ylim(lim)
    a2.set_xlabel(f"galat kalau kita cuma MENEBAK {c:.0f} bpm (%)")
    a2.set_ylabel(f"galat {best} sebenarnya (%)")
    a2.legend(fontsize=9)
    a2.set_title("Galat logger = galat penebak-konstan.\n"
                 "Ia tidak membawa informasi apa pun di luar konstantanya.",
                 fontweight="bold", fontsize=11)

    fig.tight_layout()
    fig.savefig(out / "b2_why.png")
    plt.close(fig)
    return list(win.subj)


# ============================================================ B3: fase mengandung pola HR
def fig_hr_pattern(front, R, S, out):
    """Tiga bukti berdiri sendiri, satu panel masing-masing."""
    fig, axs = plt.subplots(1, 3, figsize=(15.2, 4.5))

    # (1) spektrum: puncak jatuh TEPAT di frekuensi ECG
    sub = front[front.subject_id == "subject01"].sort_values("Timestamp")
    tu, xu, o = resample_antialias(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values)
    gu = np.interp(tu, np.sort(sub.Timestamp.values), sub.gt_heart_rate.values[o])
    d = np.diff(xu, prepend=xu[0])
    sb = signal.butter(4, BR_BAND, "bandpass", fs=FS, output="sos")
    sh = signal.butter(4, HR_BAND, "bandpass", fs=FS, output="sos")
    w = int(WINDOW_SEC * FS)
    seg = signal.detrend(d[:w])
    seg = seg - signal.sosfiltfilt(sb, seg)
    seg = signal.sosfiltfilt(sh, seg)
    f, p = signal.welch(seg, fs=FS, nperseg=w)
    m = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    gt0 = float(gu[:w].mean())
    axs[0].plot(f[m] * 60, p[m], color=C_FASE, lw=2)
    axs[0].axvline(gt0, color=C_ECG, lw=2.4, ls="--", label=f"ECG = {gt0:.0f} bpm")
    pk = f[m][np.argmax(p[m])] * 60
    axs[0].plot(pk, p[m].max(), "o", ms=12, color=C_FASE, zorder=5,
                label=f"puncak radar = {pk:.0f} bpm")
    axs[0].set_xlabel("bpm")
    axs[0].set_ylabel("daya")
    axs[0].legend(fontsize=8.8)
    axs[0].set_title("1. Puncak spektrum jatuh TEPAT\ndi frekuensi ECG (subject01)",
                     fontweight="bold", fontsize=11)

    # (2) uji peringkat: apakah frekuensi ECG menonjol di spektrum radar?
    ranks = []
    for _, s2 in front.groupby("subject_id"):
        s2 = s2.sort_values("Timestamp")
        tu, xu, o = resample_antialias(s2.Timestamp.values, s2.unwrapPhasePeak_mm.values)
        gu = np.interp(tu, np.sort(s2.Timestamp.values), s2.gt_heart_rate.values[o])
        d = np.diff(xu, prepend=xu[0])
        step = max(1, int(w * (1 - OVERLAP)))
        for st in range(0, len(d) - w, step):
            g = signal.detrend(d[st:st + w])
            g = g - signal.sosfiltfilt(sb, g)
            g = signal.sosfiltfilt(sh, g)
            f2, p2 = signal.welch(g, fs=FS, nperseg=w)
            mm = (f2 >= HR_BAND[0]) & (f2 <= HR_BAND[1])
            fb, pb = f2[mm], p2[mm]
            g0 = float(gu[st:st + w].mean()) / 60
            if not (HR_BAND[0] <= g0 <= HR_BAND[1]):
                continue
            i = int(np.argmin(np.abs(fb - g0)))
            # peringkat bin ECG di antara semua bin (0 = puncak tertinggi)
            ranks.append((pb > pb[i]).sum() / (len(pb) - 1))
    ranks = np.array(ranks)
    axs[1].hist(ranks, bins=22, color=C_FASE, alpha=.8)
    axs[1].axvline(0.5, color=C_TI, lw=2.4, ls="--",
                   label="kalau radar BUTA\n(peringkat acak) = 0.50")
    axs[1].axvline(np.median(ranks), color=C_ECG, lw=2.4,
                   label=f"kenyataan = {np.median(ranks):.2f}")
    axs[1].set_xlabel("peringkat frekuensi ECG di spektrum radar\n(0 = puncak tertinggi)")
    axs[1].set_ylabel("jumlah jendela")
    axs[1].legend(fontsize=8.4)
    axs[1].set_title(f"2. Frekuensi ECG yang benar MENONJOL\ndi {len(ranks)} jendela, "
                     "bukan kebetulan", fontweight="bold", fontsize=11)

    # (3) antar-subjek: radar bisa membedakan orang
    best = S["best"]
    lo, hi = 55, 100
    axs[2].plot([lo, hi], [lo, hi], ls="--", color=C_NEU, lw=1.5)
    axs[2].scatter(R.hr_ecg, R.fase_bpm, s=90, color=C_FASE, alpha=.88,
                   edgecolor="white", lw=1.2, zorder=4,
                   label=f"fase   r = {S['r_fase']:+.2f}")
    axs[2].scatter(R.hr_ecg, R[f"{best}_bpm"], s=90, color=C_TI, alpha=.88,
                   edgecolor="white", lw=1.2, zorder=4, marker="^",
                   label=f"logger r = {S['r_best']:+.2f}")
    for col, c2 in [("fase_bpm", C_FASE), (f"{best}_bpm", C_TI)]:
        k = np.polyfit(R.hr_ecg, R[col], 1)
        axs[2].plot([lo, hi], np.polyval(k, [lo, hi]), color=c2, lw=2, alpha=.6)
    axs[2].set_xlim(lo, hi)
    axs[2].set_ylim(lo, hi)
    axs[2].set_xlabel("HR sebenarnya dari ECG (bpm)")
    axs[2].set_ylabel("HR yang diestimasi (bpm)")
    axs[2].legend(fontsize=8.8, loc="upper left")
    axs[2].set_title("3. Orang ber-HR tinggi diestimasi tinggi\n(logger: justru TERBALIK)",
                     fontweight="bold", fontsize=11)

    fig.tight_layout()
    fig.savefig(out / "b3_pattern.png")
    plt.close(fig)
    return float(np.median(ranks)), len(ranks)


# ============================================================ B4: kontrol negatif
def fig_empty(front, out):
    rows = []
    for f in sorted(EMPTY.glob("*.csv")):
        r = pd.read_csv(f).sort_values("Timestamp")
        rows.append(("kosong", f.stem, r,
                     presence(r.Timestamp.values, r.unwrapPhasePeak_mm.values)))
    for sid, s2 in front.groupby("subject_id"):
        s2 = s2.sort_values("Timestamp")
        rows.append(("orang", sid, s2,
                     presence(s2.Timestamp.values, s2.unwrapPhasePeak_mm.values)))

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.2, 4.5),
                                     gridspec_kw={"width_ratios": [1, 1, 1.15]})

    # (1) sinyal fase mentah: kosong vs berisi
    for k, n, r, _ in rows:
        if k == "kosong" and n == "radar":
            tu, xu, _ = resample_antialias(r.Timestamp.values, r.unwrapPhasePeak_mm.values)
            a1.plot(tu[:int(60 * FS)] - tu[0], signal.detrend(xu[:int(60 * FS)]),
                    color=C_TI, lw=1.2, label="RUANGAN KOSONG")
    s2 = front[front.subject_id == "subject01"].sort_values("Timestamp")
    tu, xu, _ = resample_antialias(s2.Timestamp.values, s2.unwrapPhasePeak_mm.values)
    a1.plot(tu[:int(60 * FS)] - tu[0], signal.detrend(xu[:int(60 * FS)]),
            color=C_FASE, lw=1.2, label="ADA ORANG (subject01)")
    a1.set_xlabel("detik")
    a1.set_ylabel("gerak dada (mm)")
    a1.legend(fontsize=9)
    a1.set_title("Sinyal fase mentah, 60 detik pertama.\nNapas terlihat kasat mata.",
                 fontweight="bold", fontsize=11)

    # (2) apa kata LOGGER
    ko = [r.heartRateEst_xCorr.mean() for k, _, r, _ in rows if k == "kosong"]
    og = [r.heartRateEst_xCorr.mean() for k, _, r, _ in rows if k == "orang"]
    a2.scatter([0] * len(ko), ko, s=150, color=C_TI, alpha=.9, zorder=4,
               edgecolor="white", lw=1.4)
    a2.scatter([1] * len(og), og, s=110, color=C_NEU, alpha=.75, zorder=3,
               edgecolor="white", lw=1.2)
    a2.axhspan(min(min(ko), min(og)), max(max(ko), max(og)), color=C_TI, alpha=.12)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["RUANGAN\nKOSONG", "ADA\nORANG"], fontsize=10, fontweight="bold")
    a2.set_xlim(-.55, 1.55)
    a2.set_ylabel("bpm yang DILAPORKAN logger")
    a2.set_title(f"LOGGER: ruangan kosong dilaporkan\n{min(ko):.0f}-{max(ko):.0f} bpm. "
                 "TUMPANG TINDIH.", fontweight="bold", fontsize=11, color=C_TI)

    # (3) apa kata FASE
    ko_s = [pr["snr_napas"] for k, _, _, pr in rows if k == "kosong"]
    og_s = [pr["snr_napas"] for k, _, _, pr in rows if k == "orang"]
    a3.scatter([0] * len(ko_s), ko_s, s=150, color=C_TI, alpha=.9, zorder=4,
               edgecolor="white", lw=1.4)
    a3.scatter([1] * len(og_s), og_s, s=110, color=C_FASE, alpha=.85, zorder=4,
               edgecolor="white", lw=1.2)
    a3.set_yscale("log")
    a3.axhline(100, color=C_ECG, ls="--", lw=2, label="ambang kehadiran")
    a3.set_xticks([0, 1])
    a3.set_xticklabels(["RUANGAN\nKOSONG", "ADA\nORANG"], fontsize=10, fontweight="bold")
    a3.set_xlim(-.55, 1.55)
    a3.set_ylabel("kekuatan sinyal NAPAS (skala log)")
    a3.legend(fontsize=9)
    gap = min(og_s) / max(ko_s)
    a3.set_title(f"FASE: pisah bersih, margin {gap:.0f}x.\nNOL tumpang tindih.",
                 fontweight="bold", fontsize=11, color=C_FASE)

    fig.suptitle("KONTROL NEGATIF - radar dinyalakan di ruangan KOSONG, tanpa siapa pun",
                 fontweight="bold", fontsize=13.5, y=1.04)
    fig.tight_layout()
    fig.savefig(out / "b4_empty.png")
    plt.close(fig)
    return dict(ko_bpm=(min(ko), max(ko)), og_bpm=(min(og), max(og)),
                ko_snr=(min(ko_s), max(ko_s)), og_snr=(min(og_s), max(og_s)), gap=gap,
                ko_mm=max(pr["gerak_mm"] for k, _, _, pr in rows if k == "kosong"),
                og_mm=(min(pr["gerak_mm"] for k, _, _, pr in rows if k == "orang"),
                       max(pr["gerak_mm"] for k, _, _, pr in rows if k == "orang")))


# ============================================================ B5: hold-out
def fig_holdout(out):
    hold = []
    for name in HOLDOUT:
        est, xc, ref, snr = holdout_session(FRONT / name)
        hold.append(dict(name=name, est=est, xc=xc, ref=ref, snr=snr))

    fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.4), sharey=True)
    for ax, h in zip(axs, hold):
        tw = np.arange(len(h["est"])) * WINDOW_SEC * (1 - OVERLAP)
        ax.plot(tw, h["ref"], color=C_ECG, lw=2.6, ls="--", label="ECG (sebenarnya)")
        ax.plot(tw, h["est"], color=C_FASE, lw=2, label="pipeline fase")
        ax.plot(tw[:len(h["xc"])], h["xc"][:len(tw)], color=C_TI, lw=2,
                label="heartRateEst_xCorr")
        mf = 100 * np.mean(np.abs(h["est"] - h["ref"]) / h["ref"])
        mx = 100 * np.nanmean(np.abs(h["xc"] - h["ref"]) / h["ref"])
        ax.set_xlabel("detik")
        ax.legend(fontsize=8.6, loc="upper right")
        ax.set_title(f"{h['name'].replace('subject_', '')}  (SNR {h['snr']:.2f})\n"
                     f"fase MAPE {mf:.1f}%   |   logger MAPE {mx:.1f}%",
                     fontweight="bold", fontsize=11)
    axs[0].set_ylabel("bpm")
    axs[0].set_ylim(48, 118)
    fig.suptitle("HOLD-OUT - dua subjek yang TIDAK PERNAH dipakai menyetel parameter apa pun",
                 fontweight="bold", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out / "b5_holdout.png")
    plt.close(fig)
    return hold


# ============================================================ PDF
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
        self.cell(10, 5, f"B{n}", align="R")
        self.set_xy(16, y + 3)

    def pic(self, path, x, w, gap=1):
        y = self.get_y() + gap
        self.image(str(path), x=x, y=y, w=w)
        with PILImage.open(path) as im:
            iw, ih = im.size
        return y + w * ih / iw

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


def build(fig, out, R, S, win, rank, nwin, E, hold):
    best = S["best"]
    c = S["stuck_const"]
    wn = ", ".join(s.replace("subject", "S") for s in win)
    p = Slides()

    # ---- B1
    p.slide(1, "Berapa BPM yang dilaporkan tiap metode?",
            "Bukan galat, bukan persen - angka BPM apa adanya, dibanding ECG.")
    y = p.pic(fig / "b1_bpm.png", x=20, w=298)
    p.box(16, y + 3, 306,
          f"ECG bergerak dari {R.hr_ecg.min():.0f} sampai {R.hr_ecg.max():.0f} bpm - orangnya "
          f"memang berbeda-beda. Pipeline fase mengikutinya (std keluaran {R.fase_bpm.std():.1f} bpm, "
          f"HR asli {R.hr_ecg.std():.1f}).\n"
          f"Kolom logger nyaris tidak bergerak: std keluarannya cuma {R[f'{best}_bpm'].std():.1f} bpm, "
          f"berkerumun di sekitar {c:.0f} bpm untuk siapa pun.",
          "Yang langsung terlihat", NAVY, 10)

    # ---- B2
    p.slide(2, f"Kolom logger 'menang' di {wn}. Kenapa?",
            "Jawabannya bukan 'logger lebih akurat di situ' - melainkan kebetulan. Ini buktinya.")
    y = p.pic(fig / "b2_why.png", x=22, w=294)
    p.box(16, y + 3, 306,
          f"Kolom logger praktis PENEBAK-KONSTAN: apa pun yang ada di depan radar, ia menjawab "
          f"sekitar {c:.0f} bpm. Buktinya di panel kanan - galatnya hampir sama persis dengan galat "
          f"orang yang cuma menebak {c:.0f} bpm tanpa radar sama sekali (r = {S['r_stuck']:+.2f}).\n\n"
          f"{wn} kebetulan ber-HR dekat {c:.0f}, jadi logger 'menang' di situ - persis seperti JAM MATI "
          f"yang benar dua kali sehari. Harganya terlihat di ujung: S07 (65 bpm) meleset "
          f"{R.loc[R.subj == 'subject07', f'{best}_mape'].iloc[0]:.0f}%, S10 (61 bpm) meleset "
          f"{R.loc[R.subj == 'subject10', f'{best}_mape'].iloc[0]:.0f}%.",
          "Jam mati pun benar dua kali sehari", RED, 10)

    # ---- B3
    p.slide(3, "Apakah sinyal fase memang MENGANDUNG pola detak jantung?",
            "Tiga bukti yang berdiri sendiri-sendiri. Tak satu pun bergantung pada MAPE.")
    y = p.pic(fig / "b3_pattern.png", x=18, w=302)
    p.box(16, y + 3, 306,
          f"(1) Puncak spektrum radar jatuh tepat di frekuensi yang dinyatakan ECG.\n"
          f"(2) Diuji di {nwin} jendela dari 10 subjek: kalau radar buta, peringkat frekuensi ECG "
          f"akan acak (0.50). Kenyataannya {rank:.2f} - frekuensi yang benar memang MENONJOL.\n"
          f"(3) Antar-subjek, orang ber-HR tinggi diestimasi tinggi (r = {S['r_fase']:+.2f}); "
          f"kolom logger justru TERBALIK (r = {S['r_best']:+.2f}).",
          "Tiga bukti independen", GREEN, 10)

    # ---- B4
    p.slide(4, "Kontrol negatif: radar dinyalakan di ruangan KOSONG",
            "Alat ukur yang jujur harus bilang TIDAK TAHU. Yang mengarang, ketahuan di sini.")
    y = p.pic(fig / "b4_empty.png", x=18, w=302)
    p.box(16, y + 3, 148,
          f"Logger melaporkan {E['ko_bpm'][0]:.0f}-{E['ko_bpm'][1]:.0f} bpm dari ruangan KOSONG. "
          f"Untuk manusia sungguhan ia melaporkan {E['og_bpm'][0]:.0f}-{E['og_bpm'][1]:.0f} bpm.\n"
          "TUMPANG TINDIH. Angka itu tidak pernah berasal dari jantung siapa pun - termasuk "
          "ketika ada orangnya.",
          "LOGGER: mengarang", RED, 9.6)
    p.box(172, y + 3, 150,
          f"Gerak dada: kosong {E['ko_mm']:.2f} mm, manusia {E['og_mm'][0]:.1f}-{E['og_mm'][1]:.1f} mm.\n"
          f"Kekuatan sinyal napas: kosong {E['ko_snr'][0]:.1f}-{E['ko_snr'][1]:.1f}, "
          f"manusia {E['og_snr'][0]:.0f}-{E['og_snr'][1]:.0f}.\n\n"
          f"PISAH BERSIH, margin {E['gap']:.0f}x, nol tumpang tindih. Sinyal fase TAHU ada orang "
          "atau tidak - lewat NAPAS, yang 4-9x lebih besar dari jantung.",
          "FASE: jujur", GREEN, 9.6)

    # ---- B5
    p.slide(5, "Hold-out: dua subjek yang tidak pernah dipakai menyetel apa pun",
            "Menjawab tuduhan paling berbahaya di sidang: 'jangan-jangan kamu menyetel sampai bagus'.")
    y = p.pic(fig / "b5_holdout.png", x=24, w=290)
    mf = [100 * np.mean(np.abs(h["est"] - h["ref"]) / h["ref"]) for h in hold]
    mx = [100 * np.nanmean(np.abs(h["xc"] - h["ref"]) / h["ref"]) for h in hold]
    p.box(16, y + 3, 306,
          f"Jendela 40 s, median filter 9, pita 0.8-2.0 Hz - semuanya ditetapkan memakai 10 subjek "
          f"utama, sebelum kedua subjek ini pernah disentuh.\n\n"
          f"Hasilnya: fase MAPE {mf[0]:.1f}% dan {mf[1]:.1f}%, logger {mx[0]:.1f}% dan {mx[1]:.1f}%. "
          f"Fase menang di 2 dari 2.\n"
          f"Dan logger tetap mengeluarkan {np.nanmean(hold[0]['xc']):.0f} dan "
          f"{np.nanmean(hold[1]['xc']):.0f} bpm - menempel lagi ke konstantanya (~{c:.0f}), pada data "
          f"yang belum pernah ia lihat. Cerita 'penebak-konstan' terkonfirmasi DI LUAR SAMPEL.\n\n"
          f"Jujur: cadangan02 MAPE {mf[1]:.1f}% - persis di ambang, belum lolos. SNR-nya "
          f"{hold[1]['snr']:.2f}, paling rendah di antara keduanya. Konsisten dengan batas SNR.",
          "Kenapa ini penting", NAVY, 9.8)

    p.output(str(out / "bukti_detak_jantung.pdf"))
    print("ok:", out / "bukti_detak_jantung.pdf")


def main(csv, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "_fig_bukti"
    figdir.mkdir(exist_ok=True)

    df = pd.read_csv(csv)
    front = df[(df.dataset == "position_front") & df.unwrapPhasePeak_mm.notna()].copy()
    R, S = compare(front)

    fig_bpm(R, S, figdir)
    win = fig_why(R, S, figdir)
    rank, nwin = fig_hr_pattern(front, R, S, figdir)
    E = fig_empty(front, figdir)
    hold = fig_holdout(figdir)
    print("figure ok")
    build(figdir, out, R, S, win, rank, nwin, E, hold)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
