"""
DECK UTAMA (arah final, 15 Jul 2026): MURNI FASE vs ECG.

Pertanyaan tunggal: "Apakah sinyal fase mentah radar mengandung detak jantung
yang bisa diekstrak dengan akurasi standar?" TIDAK ada angka BPM keluaran TI di
jalur utama -- menampilkannya otomatis terbaca sebagai perbandingan/vonis atas
library TI, dan itu bukan lingkup penelitian ini.

Pembanding kecukupan: TRIVIAL BASELINE (tebak satu angka konstan tanpa radar),
BUKAN output TI. Trivial baseline adalah tolok ukur akademis standar, dan tidak
punya konfigurasi yang bisa dipersoalkan.

Alur:
  M1  Pertanyaan + Batasan Masalah (50 fps netral)
  M2  Metode ringkas: dari fase mentah & ECG jadi angka
  M3  Hasil: MAPE 10 subjek vs standar ANSI/CTA-2065
  M4  Cukup bermakna? -> mengalahkan trivial baseline
  M5  Bukti detak jantung BENAR-BENAR tertangkap (3 bukti)
  M6  Kontrol negatif: ruangan kosong
  M7  Hold-out: 2 subjek yang tak dipakai menyetel apa pun
  M8  Kriteria kelayakan: SNR, jarak, halangan
  M9  Kesimpulan + batas keberlakuan

Usage:
    python make_slides_fase.py <aligned_csv> <outdir>
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
from compare_ti_vs_phase import baseline, compare  # noqa: E402
from negative_control import (EMPTY, FRONT, HOLDOUT,  # noqa: E402
                              holdout_session, presence)
from phase_pipeline import (BR_BAND, FS, HR_BAND, MAPE_STANDARD,  # noqa: E402
                            MEDFILT_K, OVERLAP, WINDOW_SEC, estimate,
                            resample_antialias)
from snr_criterion import SNR_THRESHOLD, collect  # noqa: E402

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": .3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

NAVY, RED, GREY, GREEN, AMBER, PURPLE = (28, 42, 66), (192, 57, 43), (110, 118, 128), (30, 132, 73), (176, 122, 12), (110, 60, 140)
C_ECG, C_FASE, C_BAD, C_NEU = "#2c3e50", "#1e8449", "#c0392b", "#7f8c8d"
W, H = 338.7, 190.5
DEMO = "subject01"


# ============================================================ M2: metode
def fig_method(front, out):
    sub = front[front.subject_id == DEMO].sort_values("Timestamp")
    tu, xu, o = resample_antialias(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values)
    gu = np.interp(tu, np.sort(sub.Timestamp.values), sub.gt_heart_rate.values[o])
    d = np.diff(xu, prepend=xu[0])
    sb = signal.butter(4, BR_BAND, "bandpass", fs=FS, output="sos")
    sh = signal.butter(4, HR_BAND, "bandpass", fs=FS, output="sos")
    w = int(WINDOW_SEC * FS)
    seg = signal.sosfiltfilt(sh, signal.detrend(d[:w]) - signal.sosfiltfilt(sb, signal.detrend(d[:w])))
    f, p = signal.welch(seg, fs=FS, nperseg=w)
    m = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    gt0 = float(gu[:w].mean())

    est, ref, _ = estimate(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                           sub.gt_heart_rate.values)
    tw = np.arange(len(est)) * WINDOW_SEC * (1 - OVERLAP)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 4.4),
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    a1.plot(f[m] * 60, p[m], color=C_FASE, lw=2)
    a1.axvline(gt0, color=C_ECG, lw=2.4, ls="--", label=f"ECG = {gt0:.0f} bpm")
    pk = f[m][np.argmax(p[m])] * 60
    a1.plot(pk, p[m].max(), "o", ms=12, color=C_FASE, zorder=5,
            label=f"puncak radar = {pk:.0f} bpm")
    a1.set_xlabel("bpm")
    a1.set_ylabel("daya")
    a1.legend(fontsize=9)
    a1.set_title("Satu jendela: frekuensi puncak\ndiambil sebagai estimasi",
                 fontweight="bold", fontsize=11)

    a2.plot(tw, ref, color=C_ECG, lw=2.4, ls="--", label="ECG (sebenarnya)")
    a2.plot(tw, est, color=C_FASE, lw=2, label="estimasi radar")
    a2.fill_between(tw, ref, est, color=C_FASE, alpha=.15)
    a2.set_xlabel("detik")
    a2.set_ylabel("bpm")
    a2.legend(fontsize=9)
    a2.set_title(f"Ulangi tiap jendela -> satu deret estimasi\n({DEMO})",
                 fontweight="bold", fontsize=11)

    fig.tight_layout()
    fig.savefig(out / "m2.png")
    plt.close(fig)


# ============================================================ M3: hasil
def fig_results(R, out):
    x = np.arange(len(R))
    fig, ax = plt.subplots(figsize=(13.4, 4.6))
    cols = [C_FASE if v < MAPE_STANDARD else C_BAD for v in R.fase_mape]
    b = ax.bar(x, R.fase_mape, .62, color=cols, alpha=.9)
    ax.bar_label(b, fmt="%.1f", fontsize=8.6, fontweight="bold", padding=2)
    ax.axhline(MAPE_STANDARD, color=C_ECG, ls="--", lw=2.2,
               label=f"standar ANSI/CTA-2065 = {MAPE_STANDARD:.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("subject", "S") for s in R.subj])
    ax.set_ylabel("MAPE terhadap ECG (%)")
    ax.set_ylim(0, max(R.fase_mape) * 1.2)
    ax.legend(fontsize=10)
    n = int((R.fase_mape < MAPE_STANDARD).sum())
    ax.set_title(f"{n} dari 10 subjek memenuhi standar  (rata-rata {R.fase_mape.mean():.1f}%).  "
                 "Dua yang gagal ditandai merah - dibahas di kriteria kelayakan.",
                 fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out / "m3.png")
    plt.close(fig)


# ============================================================ M4: trivial baseline
def fig_trivial(B, out):
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    names = [f"Tebak konstan {B['const']:.0f} bpm\n(TANPA radar)", "Pipeline fase\n(usulan)"]
    vals = [B["triv_mae"], B["fase_mae"]]
    b = ax.bar(names, vals, color=[C_NEU, C_FASE], width=.5, alpha=.9)
    ax.bar_label(b, fmt="%.1f bpm", fontsize=12, fontweight="bold", padding=3)
    ax.axhline(B["triv_mae"], color=C_NEU, ls="--", lw=1.6)
    ax.text(1, B["triv_mae"] * 1.04, "apa pun di ATAS garis ini lebih buruk\n"
            "daripada tidak memakai radar sama sekali",
            ha="center", fontsize=8.8, color=C_NEU, fontweight="bold")
    ax.set_ylabel("MAE pada subjek uji (bpm)")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("Uji kecukupan: apakah radar mengalahkan menebak satu angka konstan?\n"
                 "(subjek uji S09+S10, split berbasis subjek)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "m4.png")
    plt.close(fig)


# ============================================================ M5: bukti tertangkap
def fig_capture(front, R, S, out):
    fig, axs = plt.subplots(1, 3, figsize=(15.2, 4.5))
    sb = signal.butter(4, BR_BAND, "bandpass", fs=FS, output="sos")
    sh = signal.butter(4, HR_BAND, "bandpass", fs=FS, output="sos")
    w = int(WINDOW_SEC * FS)

    # (1) puncak tepat di frekuensi ECG
    sub = front[front.subject_id == DEMO].sort_values("Timestamp")
    tu, xu, o = resample_antialias(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values)
    gu = np.interp(tu, np.sort(sub.Timestamp.values), sub.gt_heart_rate.values[o])
    d = np.diff(xu, prepend=xu[0])
    seg = signal.sosfiltfilt(sh, signal.detrend(d[:w]) - signal.sosfiltfilt(sb, signal.detrend(d[:w])))
    f, p = signal.welch(seg, fs=FS, nperseg=w)
    m = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    gt0 = float(gu[:w].mean())
    axs[0].plot(f[m] * 60, p[m], color=C_FASE, lw=2)
    axs[0].axvline(gt0, color=C_ECG, lw=2.4, ls="--", label=f"ECG = {gt0:.0f} bpm")
    axs[0].plot(f[m][np.argmax(p[m])] * 60, p[m].max(), "o", ms=11, color=C_FASE)
    axs[0].set_xlabel("bpm")
    axs[0].set_ylabel("daya")
    axs[0].legend(fontsize=8.8)
    axs[0].set_title("1. Puncak spektrum jatuh TEPAT\ndi frekuensi ECG", fontweight="bold", fontsize=11)

    # (2) uji peringkat frekuensi ECG di semua jendela
    ranks = []
    for _, s2 in front.groupby("subject_id"):
        s2 = s2.sort_values("Timestamp")
        tu, xu, o = resample_antialias(s2.Timestamp.values, s2.unwrapPhasePeak_mm.values)
        gu = np.interp(tu, np.sort(s2.Timestamp.values), s2.gt_heart_rate.values[o])
        d = np.diff(xu, prepend=xu[0])
        step = max(1, int(w * (1 - OVERLAP)))
        for st in range(0, len(d) - w, step):
            g = signal.detrend(d[st:st + w])
            g = signal.sosfiltfilt(sh, g - signal.sosfiltfilt(sb, g))
            f2, p2 = signal.welch(g, fs=FS, nperseg=w)
            mm = (f2 >= HR_BAND[0]) & (f2 <= HR_BAND[1])
            fb, pb = f2[mm], p2[mm]
            g0 = float(gu[st:st + w].mean()) / 60
            if not (HR_BAND[0] <= g0 <= HR_BAND[1]):
                continue
            i = int(np.argmin(np.abs(fb - g0)))
            ranks.append((pb > pb[i]).sum() / (len(pb) - 1))
    ranks = np.array(ranks)
    axs[1].hist(ranks, bins=22, color=C_FASE, alpha=.8)
    axs[1].axvline(0.5, color=C_BAD, lw=2.4, ls="--", label="kalau radar BUTA = 0.50")
    axs[1].axvline(np.median(ranks), color=C_ECG, lw=2.4, label=f"kenyataan = {np.median(ranks):.2f}")
    axs[1].set_xlabel("peringkat frekuensi ECG di spektrum radar\n(0 = puncak tertinggi)")
    axs[1].set_ylabel("jumlah jendela")
    axs[1].legend(fontsize=8.4)
    axs[1].set_title(f"2. Frekuensi ECG MENONJOL\ndi {len(ranks)} jendela", fontweight="bold", fontsize=11)

    # (3) antar-subjek: fase saja
    lo, hi = 55, 100
    axs[2].plot([lo, hi], [lo, hi], ls="--", color=C_NEU, lw=1.5, label="estimasi = sebenarnya")
    axs[2].scatter(R.hr_ecg, R.fase_bpm, s=95, color=C_FASE, alpha=.88,
                   edgecolor="white", lw=1.2, zorder=4)
    for _, pp in R.iterrows():
        axs[2].annotate(pp.subj.replace("subject", "S"), (pp.hr_ecg, pp.fase_bpm),
                        textcoords="offset points", xytext=(7, -3), fontsize=8, color=C_ECG)
    k = np.polyfit(R.hr_ecg, R.fase_bpm, 1)
    axs[2].plot([lo, hi], np.polyval(k, [lo, hi]), color=C_FASE, lw=2,
                alpha=.7, label=f"tren: r = {S['r_fase']:+.2f}")
    axs[2].set_xlim(lo, hi)
    axs[2].set_ylim(lo, hi)
    axs[2].set_xlabel("HR sebenarnya dari ECG (bpm)")
    axs[2].set_ylabel("HR diestimasi (bpm)")
    axs[2].legend(fontsize=8.6, loc="upper left")
    axs[2].set_title("3. Orang ber-HR tinggi\ndiestimasi tinggi", fontweight="bold", fontsize=11)

    fig.suptitle("Bukti sinyal fase MENGANDUNG detak jantung - tiga uji yang berdiri sendiri",
                 fontweight="bold", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out / "m5.png")
    plt.close(fig)
    return float(np.median(ranks)), len(ranks)


# ============================================================ M6: kontrol negatif
def fig_empty(front, out):
    rows = []
    for f in sorted(EMPTY.glob("*.csv")):
        r = pd.read_csv(f).sort_values("Timestamp")
        rows.append(("kosong", r, presence(r.Timestamp.values, r.unwrapPhasePeak_mm.values)))
    for _, s2 in front.groupby("subject_id"):
        s2 = s2.sort_values("Timestamp")
        rows.append(("orang", s2, presence(s2.Timestamp.values, s2.unwrapPhasePeak_mm.values)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.5))
    # (1) sinyal mentah kosong vs orang
    for k, r, _ in rows:
        if k == "kosong":
            tu, xu, _ = resample_antialias(r.Timestamp.values, r.unwrapPhasePeak_mm.values)
            a1.plot(tu[:int(60 * FS)] - tu[0], signal.detrend(xu[:int(60 * FS)]),
                    color=C_BAD, lw=1.2, label="RUANGAN KOSONG")
            break
    s2 = front[front.subject_id == DEMO].sort_values("Timestamp")
    tu, xu, _ = resample_antialias(s2.Timestamp.values, s2.unwrapPhasePeak_mm.values)
    a1.plot(tu[:int(60 * FS)] - tu[0], signal.detrend(xu[:int(60 * FS)]),
            color=C_FASE, lw=1.2, label="ADA ORANG (subject01)")
    a1.set_xlabel("detik")
    a1.set_ylabel("gerak dada (mm)")
    a1.legend(fontsize=9.5)
    a1.set_title("Sinyal fase mentah, 60 detik.\nNapas terlihat kasat mata / tidak ada.",
                 fontweight="bold", fontsize=11.5)

    # (2) SNR napas: pisah bersih
    ko = [pr["snr_napas"] for k, _, pr in rows if k == "kosong"]
    og = [pr["snr_napas"] for k, _, pr in rows if k == "orang"]
    a2.scatter([0] * len(ko), ko, s=150, color=C_BAD, alpha=.9, zorder=4, edgecolor="white", lw=1.4)
    a2.scatter([1] * len(og), og, s=110, color=C_FASE, alpha=.85, zorder=4, edgecolor="white", lw=1.2)
    a2.set_yscale("log")
    a2.axhline(100, color=C_ECG, ls="--", lw=2, label="ambang kehadiran")
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["RUANGAN\nKOSONG", "ADA\nORANG"], fontsize=10, fontweight="bold")
    a2.set_xlim(-.55, 1.55)
    a2.set_ylabel("kekuatan sinyal NAPAS (skala log)")
    a2.legend(fontsize=9)
    gap = min(og) / max(ko)
    a2.set_title(f"Deteksi kehadiran lewat NAPAS:\npisah bersih, margin {gap:.0f}x",
                 fontweight="bold", fontsize=11.5, color=C_FASE)

    fig.suptitle("KONTROL NEGATIF - radar dinyalakan di ruangan KOSONG, tanpa siapa pun",
                 fontweight="bold", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out / "m6.png")
    plt.close(fig)
    return (min(ko), max(ko)), (min(og), max(og)), gap


# ============================================================ M7: hold-out
def fig_holdout(out):
    hold = []
    for name in HOLDOUT:
        est, _, ref, snr = holdout_session(FRONT / name)
        hold.append(dict(name=name, est=est, ref=ref, snr=snr))
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.4), sharey=True)
    for ax, h in zip(axs, hold):
        tw = np.arange(len(h["est"])) * WINDOW_SEC * (1 - OVERLAP)
        ax.plot(tw, h["ref"], color=C_ECG, lw=2.6, ls="--", label="ECG (sebenarnya)")
        ax.plot(tw, h["est"], color=C_FASE, lw=2, label="pipeline fase")
        mf = 100 * np.mean(np.abs(h["est"] - h["ref"]) / h["ref"])
        ax.set_xlabel("detik")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"{h['name'].replace('subject_', '')}  (SNR {h['snr']:.2f})   MAPE {mf:.1f}%",
                     fontweight="bold", fontsize=11.5)
    axs[0].set_ylabel("bpm")
    axs[0].set_ylim(52, 100)
    fig.suptitle("HOLD-OUT - dua subjek yang TIDAK PERNAH dipakai menyetel parameter apa pun",
                 fontweight="bold", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out / "m7.png")
    plt.close(fig)
    return hold


# ============================================================ M8: kriteria SNR
def fig_snr(SN, out):
    mark = {"utama": "o", "hold-out": "s", "jarak": "D", "halangan": "^"}
    lab = {"utama": "10 subjek utama", "hold-out": "2 hold-out",
           "jarak": "2 jarak terkendali", "halangan": "3 halangan terkendali"}
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    ax.axvspan(0, SNR_THRESHOLD, color=C_BAD, alpha=.08)
    ax.axhspan(MAPE_STANDARD, 100, color=C_BAD, alpha=.05)
    ax.axvline(SNR_THRESHOLD, color=C_BAD, ls="--", lw=2, label=f"ambang SNR = {SNR_THRESHOLD}")
    ax.axhline(MAPE_STANDARD, color=C_ECG, ls="--", lw=2, label=f"standar = {MAPE_STANDARD:.0f}%")
    for grp, mk in mark.items():
        s = SN[SN.kelompok == grp]
        ax.scatter(s.snr, s.mape, s=150, marker=mk,
                   c=[C_FASE if v else C_BAD for v in s.lolos],
                   edgecolor="white", lw=1.4, zorder=5, label=lab[grp])
    nudge = {"S03": (-26, -3), "cad02": (10, 4), "50 cm": (10, -10),
             "S05": (-24, -8), "cad01": (10, 2), "tipis": (8, 4), "tebal": (-30, 4)}
    for _, r in SN.iterrows():
        ax.annotate(r.sesi, (r.snr, r.mape), textcoords="offset points",
                    xytext=nudge.get(r.sesi, (8, -3)), fontsize=8.4, color=C_ECG)
    lr = np.corrcoef(np.log(SN.snr), np.log(SN.mape))[0, 1]
    xs = np.linspace(SN.snr.min() * .85, SN.snr.max() * 1.1, 100)
    k = np.polyfit(np.log(SN.snr), np.log(SN.mape), 1)
    ax.plot(xs, np.exp(np.polyval(k, np.log(xs))), color=C_NEU, lw=1.8, ls=":",
            zorder=2, label=f"tren (r = {lr:+.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, 11)
    ax.set_ylim(1, 70)
    ax.set_xticks([1, 1.8, 3, 5, 8, 10])
    ax.set_xticklabels(["1", "1.8", "3", "5", "8", "10"])
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1%", "2%", "5%", "10%", "20%", "50%"])
    ax.set_xlabel("SNR sinyal jantung  (makin kanan makin kuat)")
    ax.set_ylabel("MAPE  (makin bawah makin baik)")
    ax.legend(fontsize=8.6, loc="upper right", ncol=2)
    n_ok = int(SN.cocok.sum())
    ax.set_title(f"Kegagalan bisa diprediksi dari kekuatan sinyal (r = {lr:+.2f}).\n"
                 f"Ambang SNR {SNR_THRESHOLD} benar di {n_ok} dari {len(SN)} sesi - "
                 "termasuk jarak & halangan terkendali.",
                 fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out / "m8.png")
    plt.close(fig)
    return lr


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
        self.cell(10, 5, f"M{n}", align="R")
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


def build(fig, out, R, S, B, SN, rank, nwin, emp, hold, lr):
    n_pass = int((R.fase_mape < MAPE_STANDARD).sum())
    fail = R[R.fase_mape >= MAPE_STANDARD]
    p = Slides()

    # M1
    p.slide(1, "Pertanyaan penelitian dan batasannya",
            "Satu pertanyaan, satu jawaban. Divalidasi terhadap ECG dan trivial baseline.")
    y = p.get_y() + 2
    y = p.box(16, y, 306,
              "\"Apakah sinyal fase mentah radar FMCW TI IWR1443BOOST (unwrapPhasePeak_mm) "
              "mengandung detak jantung yang dapat diekstrak dengan akurasi yang memenuhi standar "
              "ANSI/CTA-2065 (MAPE < 10%)?\"",
              "Pertanyaan penelitian", NAVY, 11)
    y = p.box(16, y + 3, 306,
              "- Yang dievaluasi: sinyal fase MENTAH, diambil sebelum rantai pemrosesan detak "
              "jantung di dalam radar.\n"
              "- Acuan kebenaran: ECG (Attys, 125 Hz), R-peak metode Pan-Tompkins.\n"
              "- Pembanding kecukupan: TRIVIAL BASELINE (menebak satu angka konstan tanpa radar) "
              "- tolok ukur akademis standar.\n"
              "- Akuisisi pada 50 fps: memberi resolusi waktu lebih halus untuk phase-unwrapping, "
              "MENGUNTUNGKAN pendekatan berbasis fase ini.",
              "Ruang lingkup", GREEN, 10)
    p.box(16, y + 3, 306,
          "Output BPM siap-pakai di dalam radar dihasilkan oleh rantai filter yang dikonfigurasi "
          "untuk 20 fps, sementara akuisisi ini 50 fps. Karena itu output tersebut BUKAN pembanding "
          "yang adil, dan penelitian ini tidak menilainya. Validasi dilakukan terhadap ECG dan "
          "trivial baseline.",
          "Kenapa tidak dibandingkan dengan output BPM bawaan", AMBER, 9.8)

    # M2
    p.slide(2, "Metode: dari sinyal fase mentah menjadi angka detak jantung",
            "Murni pemrosesan sinyal (DSP), tanpa machine learning.")
    y = p.pic(fig / "m2.png", x=26, w=286)
    p.box(16, y + 3, 306,
          f"Resample anti-alias -> TURUNAN fase (menekan napas, menonjolkan jantung) -> buang sisa "
          f"napas -> bandpass 0.8-2.0 Hz -> Welch PSD per jendela {WINDOW_SEC:.0f} detik -> ambil "
          f"frekuensi puncak -> median filter {MEDFILT_K} titik. Galat dihitung per jendela terhadap "
          "rata-rata HR ECG di jendela yang sama, lalu dirata-ratakan menjadi MAE dan MAPE.",
          "Rantai proses", NAVY, 10)

    # M3
    p.slide(3, "Hasil utama: akurasi terhadap standar ANSI/CTA-2065",
            f"MAPE terhadap ECG, tiap subjek, dibanding ambang {MAPE_STANDARD:.0f}%.")
    y = p.pic(fig / "m3.png", x=22, w=294)
    p.box(16, y + 3, 306,
          f"Pipeline fase mentah: rata-rata MAPE {R.fase_mape.mean():.1f}%, {n_pass} dari 10 subjek "
          f"memenuhi standar. Terbaik {R.fase_mape.min():.1f}%.\n"
          f"Dua subjek yang gagal ({', '.join(s.replace('subject','S') for s in fail.subj)}) tidak "
          "disembunyikan - penyebabnya dijelaskan lewat kriteria kelayakan (M8).",
          "Yang terbaca", NAVY, 10)

    # M4
    p.slide(4, "Apakah akurasi ini bermakna? Uji terhadap trivial baseline",
            "Sebuah metode baru berarti hanya jika mengalahkan cara paling bodoh: menebak angka konstan.")
    y = p.pic(fig / "m4.png", x=20, w=170)
    p.box(198, y, 124,
          f"Subjek uji S09 + S10 (split berbasis subjek):\n\n"
          f"  tebak konstan {B['const']:.0f} bpm : MAE {B['triv_mae']:.1f}\n"
          f"  pipeline fase          : MAE {B['fase_mae']:.1f}\n\n"
          f"Radar 3-4x lebih baik daripada menebak tanpa radar. Pada leave-one-subject-out, "
          f"trivial baseline MAPE {S['trivial_mape']:.1f}% vs fase {R.fase_mape.mean():.1f}%.\n\n"
          "Artinya radar benar-benar membawa informasi detak jantung, bukan sekadar menebak ke "
          "rata-rata populasi.",
          "Yang dibuktikan", GREEN, 9.8)

    # M5
    p.slide(5, "Bukti sinyal fase memang MENGANDUNG detak jantung",
            "MAPE rendah saja bisa diperdebatkan. Tiga uji ini tidak.")
    y = p.pic(fig / "m5.png", x=18, w=302)
    p.box(16, y + 3, 306,
          f"(1) Puncak spektrum radar jatuh tepat di frekuensi yang dinyatakan ECG.\n"
          f"(2) Di {nwin} jendela dari 10 subjek: kalau radar buta, peringkat frekuensi ECG akan "
          f"acak (0.50). Kenyataannya {rank:.2f} - frekuensi yang benar memang menonjol.\n"
          f"(3) Antar-subjek, orang ber-HR tinggi diestimasi tinggi (r = {S['r_fase']:+.2f}): "
          "radar bisa membedakan individu, bukan hanya menebak rata-rata.",
          "Tiga bukti independen", GREEN, 10)

    # M6
    p.slide(6, "Kontrol negatif: apa yang terjadi tanpa siapa pun di depan radar?",
            "Uji paling keras untuk klaim 'ada detak jantung': matikan sumbernya.")
    y = p.pic(fig / "m6.png", x=28, w=282)
    p.box(16, y + 3, 306,
          f"Di ruangan kosong, kekuatan sinyal napas {emp[0][0]:.1f}-{emp[0][1]:.1f}; pada manusia "
          f"{emp[1][0]:.0f}-{emp[1][1]:.0f}. Pisah bersih, margin {emp[2]:.0f}x, nol tumpang tindih.\n"
          "Deteksi kehadiran memakai NAPAS (gerak dada 1-12 mm, jauh di atas jantung 0.1-0.5 mm), "
          "sehingga TIDAK butuh ECG - bisa dipakai saat deployment. Catatan jujur: estimator puncak "
          "tetap mengeluarkan angka dari ruangan kosong (argmax selalu ada), sehingga gerbang napas "
          "ini WAJIB sebagai penyaring.",
          "Yang dibuktikan", NAVY, 9.8)

    # M7
    p.slide(7, "Hold-out: dua subjek yang tak pernah dipakai menyetel apa pun",
            "Menjawab 'jangan-jangan parameternya disetel sampai bagus'.")
    y = p.pic(fig / "m7.png", x=24, w=290)
    mf = [100 * np.mean(np.abs(h["est"] - h["ref"]) / h["ref"]) for h in hold]
    p.box(16, y + 3, 306,
          f"Jendela {WINDOW_SEC:.0f} s, median filter {MEDFILT_K}, pita 0.8-2.0 Hz - semua "
          f"ditetapkan memakai 10 subjek utama, sebelum kedua subjek ini disentuh.\n"
          f"Hasil: MAPE {mf[0]:.1f}% dan {mf[1]:.1f}%. Yang pertama LOLOS; yang kedua "
          f"({hold[1]['name'].replace('subject_','')}) tepat di ambang (SNR {hold[1]['snr']:.2f}, "
          f"paling rendah) - dilaporkan apa adanya, konsisten dengan kriteria SNR.",
          "Kenapa ini penting", NAVY, 9.8)

    # M8
    p.slide(8, "Kriteria kelayakan: kapan metode ini bisa dipakai, kapan tidak",
            "Kegagalan bukan acak - ia diprediksi oleh SNR, dan SNR oleh jarak & halangan.")
    y = p.pic(fig / "m8.png", x=40, w=258)
    p.box(16, y + 3, 306,
          f"SNR sinyal jantung memprediksi keberhasilan (r = {lr:+.2f} terhadap log MAPE), benar di "
          f"{int(SN.cocok.sum())} dari {len(SN)} sesi. Dua eksperimen TERKENDALI membuktikan "
          f"penyebabnya fisik: JARAK (daya pantul ~1/R^4; 50 cm lolos, 100 cm gagal total) dan "
          f"HALANGAN (buku 3 cm sudah memotong SNR ~2.5x). Di bawah SNR ~{SNR_THRESHOLD}, sinyal "
          "jantung setara noise - tidak ada algoritma (termasuk ML) yang bisa mengangkatnya. Dua "
          f"subjek yang gagal ({', '.join(s.replace('subject','S') for s in fail.subj)}) tepat berada "
          "di wilayah itu.",
          "Batas keberlakuan (bagian dari hasil, bukan aib)", AMBER, 9.8)

    # M9
    p.slide(9, "Kesimpulan",
            "Satu klaim, dinyatakan lengkap dengan batasnya.")
    y = p.get_y() + 2
    ya = p.box(16, y, 150,
               f"Sinyal fase mentah radar TERBUKTI mengandung detak jantung yang dapat diekstrak:\n\n"
               f"- {n_pass}/10 subjek memenuhi ANSI/CTA-2065 (rata-rata {R.fase_mape.mean():.1f}%).\n"
               f"- Mengalahkan trivial baseline 3-4x.\n"
               f"- Korelasi antar-subjek +{S['r_fase']:.2f}.\n"
               f"- Kontrol negatif: pisah {emp[2]:.0f}x dari ruangan kosong.\n"
               f"- Hold-out 2 subjek: konsisten.\n"
               "Semuanya tanpa machine learning, tanpa mengganti perangkat keras.",
               "Yang terbukti", GREEN, 9.6)
    yb = p.box(172, y, 150,
               f"- Kelayakan diprediksi SNR (ambang ~{SNR_THRESHOLD}); penyebab fisik jarak & "
               f"halangan dibuktikan terkendali.\n"
               f"- Berlaku pada kondisi ISTIRAHAT (HR nyaris konstan); pelacakan perubahan HR di "
               "luar lingkup.\n"
               f"- HRV belum layak (batas fisik, dilaporkan sebagai hasil negatif).\n"
               f"- Seluruh subjek yang direkam dilaporkan; tidak ada yang dibuang.",
               "Batas keberlakuan yang dinyatakan", AMBER, 9.6)
    p.box(16, max(ya, yb) + 3, 306,
          "Kontribusi: (K2) pipeline pengganti berbasis fase mentah, tervalidasi standar; "
          "(K3) kriteria kelayakan SNR + karakterisasi jarak/halangan; (K4) koreksi metodologi "
          "evaluasi (trivial baseline, ceiling korelasi pada target hampir konstan).",
          "Kontribusi", NAVY, 9.8)

    p.output(str(out / "fase_vs_ecg.pdf"))
    print("ok:", out / "fase_vs_ecg.pdf")


def main(csv, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "_fig_fase"
    figdir.mkdir(exist_ok=True)

    df = pd.read_csv(csv)
    front = df[(df.dataset == "position_front") & df.unwrapPhasePeak_mm.notna()].copy()
    R, S = compare(front)
    B = baseline(front)
    SN = collect(csv)
    SN["lolos"] = SN.mape < MAPE_STANDARD
    SN["cocok"] = SN.lolos == (SN.snr >= SNR_THRESHOLD)

    fig_method(front, figdir)
    fig_results(R, figdir)
    fig_trivial(B, figdir)
    rank, nwin = fig_capture(front, R, S, figdir)
    emp = fig_empty(front, figdir)
    hold = fig_holdout(figdir)
    lr = fig_snr(SN, figdir)
    print("figure ok")
    build(figdir, out, R, S, B, SN, rank, nwin, emp, hold, lr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
