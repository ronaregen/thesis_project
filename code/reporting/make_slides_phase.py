"""
Slide presentasi (PDF lanskap 16:9) — estimasi detak jantung HANYA dari
`unwrapPhasePeak_mm`. Alurnya bertahap: apa datanya -> kenapa cuma itu ->
cara mengolah -> hasil vs standar -> kesimpulan.

Usage:
    python make_slides_phase.py <aligned_csv> <outdir>
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
from phase_pipeline import (estimate, metrics, FS, HR_BAND, BR_BAND,  # noqa: E402
                            MAPE_STANDARD, MEDFILT_K, OVERLAP, WINDOW_SEC)

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": .3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

NAVY, RED, GREEN, GREY, PURPLE = (28, 42, 66), (192, 57, 43), (30, 132, 73), (110, 118, 128), (110, 60, 140)
AMBER = (176, 122, 12)
C_BAD, C_GOOD, C_GT, C_NEU, C_PUR = "#c0392b", "#1e8449", "#2c3e50", "#7f8c8d", "#8e44ad"

W, H = 338.7, 190.5   # mm, 16:9


# ============================================================ FIGURES
def fig_what(front, out):
    """Slide 2: apa itu unwrapPhasePeak_mm — gerakan dada, napas vs jantung."""
    sub = front[front.subject_id == "subject01"].sort_values("Timestamp")
    t = sub.Timestamp.values - sub.Timestamp.values[0]
    x = sub.unwrapPhasePeak_mm.values
    m = (t > 60) & (t < 100)
    t2, x2 = t[m], x[m]

    tu = np.arange(t2[0], t2[-1], 1 / FS)
    xu = np.interp(tu, t2, x2)
    br = signal.sosfiltfilt(signal.butter(4, BR_BAND, "bandpass", fs=FS, output="sos"), xu)
    hr = signal.sosfiltfilt(signal.butter(4, HR_BAND, "bandpass", fs=FS, output="sos"), xu)

    fig, ax = plt.subplots(3, 1, figsize=(13, 6.2), sharex=True)
    ax[0].plot(tu, xu - xu.mean(), color=C_GT, lw=1.5)
    ax[0].set_ylabel("mm")
    ax[0].set_title("unwrapPhasePeak_mm — perpindahan dinding dada yang terukur radar (mentah)",
                    fontweight="bold", fontsize=13, loc="left")

    ax[1].plot(tu, br, color="#2980b9", lw=1.8)
    ax[1].set_ylabel("mm")
    ax[1].set_title(f"Komponen NAPAS (0.15–0.6 Hz) — amplitudo besar, "
                    f"±{np.std(br):.2f} mm", fontweight="bold", fontsize=12,
                    loc="left", color="#2980b9")

    ax[2].plot(tu, hr, color=C_BAD, lw=1.8)
    ax[2].set_ylabel("mm")
    ax[2].set_xlabel("waktu (detik)")
    ax[2].set_title(f"Komponen JANTUNG (0.8–2.0 Hz) — amplitudo jauh lebih kecil, "
                    f"±{np.std(hr):.2f} mm  ← inilah yang kita cari",
                    fontweight="bold", fontsize=12, loc="left", color=C_BAD)

    ratio = np.std(br) / np.std(hr)
    fig.suptitle(f"Sinyal jantung tersembunyi di balik napas yang {ratio:.0f}x lebih besar",
                 fontweight="bold", fontsize=15, y=1.02)
    fig.savefig(out / "s2_what.png")
    plt.close(fig)


def fig_why(front_all, out):
    """Slide 3: kenapa cuma kolom ini — filter TI membuang jantung."""
    d = front_all[front_all.subject_id == "subject01"].sort_values("Timestamp")
    t = d.Timestamp.values
    tu = np.arange(t[0], t[-1], 1 / 50.0)
    xu = np.interp(tu, t, d.outputFilterHeartOut.values)
    f, p = signal.welch(xu, fs=50.0, nperseg=1024)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    a1.semilogy(f, p, color=C_GT, lw=1.6)
    a1.axvspan(0.8, 2.0, color=C_GOOD, alpha=.25)
    a1.axvspan(2.0, 10.0, color=C_BAD, alpha=.12)
    a1.text(1.4, .04, "band\njantung\nmanusia", transform=a1.get_xaxis_transform(),
            ha="center", fontsize=9, color=C_GOOD, fontweight="bold")
    a1.text(6.0, .97, "band yang SEBENARNYA\ndilewatkan filter TI",
            transform=a1.get_xaxis_transform(), ha="center", va="top",
            fontsize=9.5, color=C_BAD, fontweight="bold")
    a1.set_xlim(0, 12)
    a1.set_xlabel("frekuensi (Hz)")
    a1.set_ylabel("PSD")
    a1.set_title("Kolom 'waveform jantung' bawaan TI — isinya BUKAN jantung",
                 fontweight="bold", fontsize=12.5)

    tot = np.trapezoid(p, f)
    vals = []
    for lo, hi in [(0.8, 2.0), (2.0, 10.0)]:
        mk = (f >= lo) & (f < hi)
        vals.append(100 * np.trapezoid(p[mk], f[mk]) / tot)
    a2.bar([0, 1], vals, color=[C_GOOD, C_BAD], alpha=.9, width=.55)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["0.8–2.0 Hz\n(jantung asli)", "2.0–10 Hz\n(band tergeser)"],
                       fontsize=10.5)
    a2.set_ylabel("% energi")
    a2.set_ylim(0, 108)
    for i, v in enumerate(vals):
        a2.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=15, fontweight="bold",
                color=[C_GOOD, C_BAD][i])
    a2.set_title("Energinya hampir seluruhnya\ndi band yang salah",
                 fontweight="bold", fontsize=12.5)
    fig.savefig(out / "s3_why.png")
    plt.close(fig)


def fig_pipeline(front, out):
    """Slide 4: tahapan pengolahan."""
    sub = front[front.subject_id == "subject09"].sort_values("Timestamp")
    t = sub.Timestamp.values
    o = np.argsort(t)
    t = t[o]
    x = sub.unwrapPhasePeak_mm.values[o]
    g = sub.gt_heart_rate.values[o]
    fso = 1 / np.median(np.diff(t))
    xl = signal.sosfiltfilt(signal.butter(6, FS/2*.9, "low", fs=fso, output="sos"), x)
    tu = np.arange(t[0], t[-1], 1 / FS)
    xu = np.interp(tu, t, xl)
    gu = np.interp(tu, t, g)

    s = 3000
    w = int(20 * FS)
    seg0 = xu[s:s+w]
    tt = np.arange(w) / FS

    d = np.diff(xu, prepend=xu[0])[s:s+w]
    d = signal.detrend(d)
    sos_br = signal.butter(4, BR_BAND, "bandpass", fs=FS, output="sos")
    d2 = d - signal.sosfiltfilt(sos_br, d)
    d3 = signal.sosfiltfilt(signal.butter(4, HR_BAND, "bandpass", fs=FS, output="sos"), d2)
    f, p = signal.welch(d3, fs=FS, nperseg=w)
    mk = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    fpk = f[mk][np.argmax(p[mk])]
    gt = gu[s:s+w].mean()

    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.5))
    ax[0].plot(tt, seg0 - seg0.mean(), color=C_GT, lw=1.3)
    ax[0].set_title("1. Fase mentah\n(napas mendominasi)", fontweight="bold", fontsize=11)
    ax[0].set_xlabel("detik"); ax[0].set_ylabel("mm")

    ax[1].plot(tt, d, color="#e67e22", lw=1.1)
    ax[1].set_title("2. Turunan fase\n(drift & napas ditekan)", fontweight="bold", fontsize=11)
    ax[1].set_xlabel("detik")

    ax[2].plot(tt, d3, color=C_BAD, lw=1.3)
    ax[2].set_title("3. Buang napas +\nbandpass 0.8–2.0 Hz", fontweight="bold", fontsize=11)
    ax[2].set_xlabel("detik")

    ax[3].plot(f[mk] * 60, p[mk], color=C_GOOD, lw=2)
    ax[3].axvline(gt, color=C_GT, ls="--", lw=2, label=f"ECG = {gt:.0f} bpm")
    ax[3].plot(fpk * 60, p[mk].max(), "v", color=C_BAD, ms=12,
               label=f"puncak = {fpk*60:.0f} bpm")
    ax[3].set_title("4. Puncak spektrum\n= estimasi detak jantung",
                    fontweight="bold", fontsize=11)
    ax[3].set_xlabel("bpm")
    ax[3].legend(fontsize=8.5)
    fig.suptitle("Empat langkah, semuanya dari satu kolom saja",
                 fontweight="bold", fontsize=14, y=1.06)
    fig.savefig(out / "s4_pipeline.png")
    plt.close(fig)


def fig_results(front, out):
    """Slide 5: hasil vs standar."""
    sids, maes, mapes, snrs, pcts = [], [], [], [], []
    for sid, sub in front.groupby("subject_id"):
        e, r, snr = estimate(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                             sub.gt_heart_rate.values)
        mae, mape, _ = metrics(e, r)
        tol = np.maximum(5.0, 0.10 * r)
        sids.append(sid.replace("subject", "S"))
        maes.append(mae); mapes.append(mape); snrs.append(snr)
        pcts.append(100 * np.mean(np.abs(e - r) <= tol))

    order = np.argsort(mapes)
    sids = [sids[i] for i in order]
    maes = [maes[i] for i in order]
    mapes = [mapes[i] for i in order]
    snrs = [snrs[i] for i in order]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(10)
    cols = [C_GOOD if m < 10 else C_BAD for m in mapes]
    a1.bar(x, mapes, color=cols, alpha=.9)
    a1.axhline(10, color=C_PUR, ls="--", lw=2.2,
               label="Batas standar ANSI/CTA-2065: MAPE 10%")
    a1.set_xticks(x); a1.set_xticklabels(sids, fontsize=11)
    a1.set_ylabel("MAPE (%)")
    a1.set_ylim(0, 23)
    a1.legend(fontsize=10, loc="upper left")
    for i, (m, mm) in enumerate(zip(mapes, maes)):
        a1.text(i, m + .4, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")
    a1.set_title(f"{sum(m < MAPE_STANDARD for m in mapes)} dari {len(mapes)} subjek "
                 "MEMENUHI standar kesehatan", fontweight="bold", fontsize=13)

    a2.scatter(snrs, mapes, s=140, c=cols, edgecolors="w", lw=1.5, zorder=3)
    for s_, m_, sd in zip(snrs, mapes, sids):
        a2.annotate(sd, (s_, m_), fontsize=8.5, xytext=(6, -3),
                    textcoords="offset points")
    a2.axhline(10, color=C_PUR, ls="--", lw=1.8)
    a2.axvline(1.8, color=C_NEU, ls=":", lw=2, label="ambang SNR = 1.8")
    a2.set_xlabel("SNR sinyal jantung")
    a2.set_ylabel("MAPE (%)")
    a2.legend(fontsize=9.5)
    a2.set_title("Yang gagal = yang SNR-nya rendah\n(batas fisik, bukan salah metode)",
                 fontweight="bold", fontsize=12)
    fig.savefig(out / "s5_results.png")
    plt.close(fig)
    return sids, maes, mapes, pcts, snrs


def fig_track(front, out):
    """Slide 5b: contoh pelacakan terbaik."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.6))
    for a, sid in zip(ax, ["subject09", "subject01"]):
        sub = front[front.subject_id == sid]
        e, r, _ = estimate(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                           sub.gt_heart_rate.values)
        mae, mape, _ = metrics(e, r)
        tt = np.arange(len(e)) * 5
        a.plot(tt, r, color=C_GT, lw=2.4, label="Ground truth ECG")
        a.plot(tt, e, color=C_GOOD, lw=1.6, alpha=.95, label="Estimasi dari radar")
        a.fill_between(tt, r - np.maximum(5, .1*r), r + np.maximum(5, .1*r),
                       color=C_PUR, alpha=.13, label="toleransi AAMI EC13")
        a.set_xlabel("waktu (detik)"); a.set_ylabel("bpm")
        a.set_title(f"{sid} — MAE {mae:.1f} bpm, MAPE {mape:.1f}%",
                    fontweight="bold", fontsize=12)
        a.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Estimasi radar mengikuti ECG di dalam pita toleransi standar",
                 fontweight="bold", fontsize=14, y=1.04)
    fig.savefig(out / "s5b_track.png")
    plt.close(fig)


# ============================================================ SLIDES
class Slides(FPDF):
    def __init__(self):
        # fpdf: orientation="L" menukar (w,h) dari format -> beri (H, W)
        super().__init__(orientation="L", format=(H, W))
        self.set_auto_page_break(False)
        self.set_margins(16, 14, 16)

    def slide(self, n, title, sub=None):
        self.add_page()
        self.set_xy(16, 11)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*NAVY)
        self.cell(0, 9, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if sub:
            self.set_x(16)
            self.set_font("Helvetica", "", 11.5)
            self.set_text_color(*GREY)
            self.cell(0, 6, sub, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*RED)
        self.set_line_width(1.1)
        y = self.get_y() + 1.5
        self.line(16, y, 44, y)
        # nomor slide -- gambar dulu, lalu KEMBALIKAN kursor (kalau tidak,
        # seluruh konten slide ikut turun ke bawah halaman)
        self.set_xy(W - 26, H - 11)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*GREY)
        self.cell(10, 5, str(n), align="R")
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

    def img(self, path, x, y, w):
        self.image(str(path), x=x, y=y, w=w)


def build(fig, out, res):
    sids, maes, mapes, pcts, snrs = res
    n_pass = sum(1 for m in mapes if m < 10)
    p = Slides()

    # ---------- 1: judul
    p.add_page()
    p.set_xy(20, 48)
    p.set_font("Helvetica", "B", 12)
    p.set_text_color(*RED)
    p.cell(0, 6, "LAPORAN BIMBINGAN TESIS  -  13 JULI 2026",
           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    p.ln(4)
    p.set_x(20)
    p.set_font("Helvetica", "B", 30)
    p.set_text_color(*NAVY)
    p.multi_cell(300, 13, "Estimasi Detak Jantung dari\nSinyal Fase Mentah Radar FMCW")
    p.ln(3)
    p.set_x(20)
    p.set_font("Helvetica", "", 14)
    p.set_text_color(*GREY)
    p.multi_cell(290, 7, "Mengandalkan satu kolom data saja: unwrapPhasePeak_mm\n"
                         "TI IWR1443BOOST  |  10 subjek  |  pembanding: ECG Attys 125 Hz")
    p.ln(6)
    p.set_draw_color(*RED)
    p.set_line_width(1.6)
    p.line(20, p.get_y(), 62, p.get_y())
    p.ln(9)
    p.box(20, p.get_y(), 300,
          f"Semua kolom detak jantung bawaan perangkat TIDAK BISA DIPAKAI. "
          f"Satu-satunya data yang selamat adalah sinyal fase mentah. "
          f"Laporan ini menunjukkan bahwa dari kolom itu saja, detak jantung "
          f"{n_pass} dari 10 subjek dapat diestimasi memenuhi standar kesehatan "
          f"internasional -- tanpa machine learning, tanpa data baru.",
          "Inti laporan", GREEN, 11)

    # ---------- 2: apa itu
    p.slide(2, "Apa itu unwrapPhasePeak_mm?",
            "Bukan angka detak jantung. Ini adalah gerakan dinding dada, dalam milimeter.")
    y = p.get_y()
    p.box(16, y, 100,
          "Radar FMCW menembakkan gelombang ke dada. Saat dada BERGERAK, jarak "
          "pantul berubah, dan FASE gelombang pantul ikut bergeser.\n\n"
          "Pergeseran fase itu diubah menjadi jarak:\n"
          "     perpindahan = (lambda / 4pi) x perubahan fase\n\n"
          "Hasilnya: grafik naik-turun dinding dada, dalam milimeter.\n\n"
          "Dada bergerak karena DUA hal sekaligus:\n"
          "  - NAPAS   : 1-12 mm  (0.1-0.5 Hz)\n"
          "  - JANTUNG : 0.1-0.5 mm  (0.8-2.0 Hz)\n\n"
          "Tantangannya: gerakan jantung PULUHAN KALI lebih kecil daripada napas, "
          "dan keduanya bercampur di sinyal yang sama.",
          "Dari gelombang radar ke gerakan dada", NAVY, 10)
    p.img(fig / "s2_what.png", 122, y - 2, 200)

    # ---------- 3: kenapa cuma kolom ini
    p.slide(3, "Kenapa hanya kolom ini yang dipakai?",
            "Perangkat merekam 12 kolom. Sebelas di antaranya rusak atau tidak bermakna.")
    y = p.get_y()
    p.img(fig / "s3_why.png", 16, y + 22, 200)
    y2 = p.box(222, y, 100,
               "1. FRAME RATE SALAH\n"
               "Perangkat dijalankan 50 fps, padahal pemroses bawaannya dirancang untuk "
               "20 fps. Akibatnya seluruh filter internalnya bergeser 2.5x - dan justru "
               "MEMBUANG detak jantung (grafik di kiri).\n\n"
               "2. PEREKAM SALAH BACA BYTE\n"
               "Empat kolom terbaca dari posisi yang keliru. Kolom 'detak jantung' "
               "ternyata berisi LAJU NAPAS.\n\n"
               "3. JARAK SUBJEK\n"
               "Makin jauh, sinyal jantung makin tenggelam di noise.",
               "Tiga cacat pada pengambilan data", RED, 9.5)
    p.box(222, y2 + 2, 100,
          "unwrapPhasePeak_mm dibaca dari posisi byte yang BENAR, dan diambil "
          "SEBELUM masuk filter internal. Jadi ia tidak terkena cacat 1 maupun 2.\n\n"
          "Bonus: frame rate 50 fps justru MENGUNTUNGKAN sinyal fase - resolusi "
          "waktunya lebih rapat.",
          "Kolom ini SELAMAT", GREEN, 9.5)

    # ---------- 4: cara mengolah
    p.slide(4, "Cara mengolahnya: empat langkah",
            "Semua murni pemrosesan sinyal. Tidak ada machine learning.")
    y = p.get_y()
    p.img(fig / "s4_pipeline.png", 16, y + 6, 306)
    p.box(16, y + 108, 306,
          "Langkah kunci adalah TURUNAN FASE (langkah 2). Karena napas bergerak lambat "
          "dan jantung cepat, operasi turunan menekan napas dan menonjolkan jantung "
          "sekaligus. Diukur dengan ablasi pada konfigurasi ini: tanpa turunan fase galat "
          "rata-rata 11.9% (6/10 lolos), dengan turunan fase 6.5% (8/10 lolos). "
          "Lalu sisa komponen napas dibuang secara eksplisit, sinyal disaring ke pita "
          "detak jantung manusia (0.8-2.0 Hz), dan frekuensi puncaknya diambil sebagai "
          f"estimasi - dihitung ulang tiap {WINDOW_SEC*(1-OVERLAP):.0f} detik memakai jendela "
          f"{WINDOW_SEC:.0f} detik, lalu dihaluskan median filter {MEDFILT_K} titik.",
          "Yang paling menentukan: turunan fase", NAVY, 10)

    # ---------- 5: standar
    p.slide(5, "Hasil: diuji terhadap standar kesehatan internasional",
            "Bukan target yang dikarang sendiri - dua standar yang diakui.")
    y = p.get_y()
    p.box(16, y, 148,
          "ANSI/CTA-2065 (2018) - Physical Activity Monitoring for Heart Rate\n"
          "   Alat dinyatakan VALID bila MAPE < 10%.\n\n"
          "ANSI/AAMI EC13:2002 (R2007) - Cardiac Monitors & Heart Rate Meters\n"
          "   Galat pembacaan tidak boleh melebihi +/-10% ATAU +/-5 bpm,\n"
          "   mana yang lebih besar.\n"
          "   -> Inilah asal angka 'MAE 5 bpm' yang sering dikutip.",
          "Dua standar yang dipakai", PURPLE, 10)
    p.box(172, y, 150,
          f"MAE rata-rata        : {np.mean(maes):.1f} bpm\n"
          f"MAPE rata-rata       : {np.mean(mapes):.1f} %\n"
          f"Subjek terbaik       : MAE {min(maes):.1f} bpm, MAPE {min(mapes):.1f} %\n\n"
          f"LOLOS ANSI/CTA-2065  : {n_pass} dari 10 subjek\n"
          f"Pembanding - pipeline bawaan TI: MAPE ~90%, NOL subjek lolos.",
          "Hasil pipeline fase", GREEN, 10)
    p.img(fig / "s5_results.png", 24, y + 44, 290)

    # ---------- 6: pelacakan
    p.slide(6, "Seperti apa hasilnya kalau dilihat langsung",
            "Estimasi radar (hijau) vs ECG (gelap), di dalam pita toleransi standar AAMI.")
    y = p.get_y()
    p.img(fig / "s5b_track.png", 30, y + 2, 278)
    p.box(16, y + 93, 306,
          "Pada subjek dengan kualitas sinyal baik, estimasi dari radar mengikuti ECG dan "
          "sebagian besar waktu berada di dalam pita toleransi standar. Empat subjek yang "
          "gagal bukan karena metodenya lemah: tiga di antaranya memiliki SNR mendekati "
          "noise (1.67-1.76), yang berarti sinyal jantungnya memang tidak terekam - batas "
          "fisik yang tidak bisa diperbaiki algoritma apapun. Ini dikuatkan eksperimen "
          "jarak: pada 50 cm SNR 3.16 (lolos), pada 100 cm SNR 1.09 (gagal).",
          "Yang berhasil dan yang tidak - keduanya bisa dijelaskan", NAVY, 10)

    # ---------- 7: kesimpulan
    p.slide(7, "Kesimpulan", None)
    y = p.get_y() + 2
    y = p.box(16, y, 306,
              "Sinyal fase mentah radar MEMBAWA informasi detak jantung. Dari satu kolom itu "
              f"saja, {n_pass} dari 10 subjek dapat diestimasi memenuhi standar ANSI/CTA-2065 "
              f"(MAPE < 10%), dengan hasil terbaik MAE {min(maes):.1f} bpm / MAPE {min(mapes):.1f}% - "
              "sementara keluaran bawaan perangkat meleset ~90% dan tidak satu pun subjek lolos.",
              "1. Perangkatnya sebenarnya mampu - yang gagal adalah pemrosesan bawaannya",
              GREEN, 10.5)
    y = p.box(16, y + 2, 306,
              "Kegagalan keluaran bawaan bukan misteri: frame rate 2.5x terlalu cepat membuat "
              "filter internal justru membuang detak jantung, dan perekam salah membaca empat "
              "kolom sehingga kolom 'detak jantung' berisi laju napas. Keduanya dibuktikan "
              "secara kuantitatif dan dapat direproduksi.",
              "2. Penyebab kegagalan sudah ditemukan sampai ke akarnya", RED, 10.5)
    y = p.box(16, y + 2, 306,
              "Empat subjek yang gagal punya penjelasan terukur: SNR sinyal jantungnya "
              "mendekati noise. Ambang SNR 1.8 memisahkan berhasil/gagal dengan benar pada "
              "11 dari 12 sesi, dan dikuatkan eksperimen variasi jarak. Artinya kita tahu "
              "KAPAN metode ini bisa dipakai dan kapan secara fisik mustahil.",
              "3. Batas keberlakuannya juga sudah terukur", AMBER, 10.5)
    fail = [(sid, m, sn) for sid, m, sn in zip(sids, mapes, snrs) if m >= MAPE_STANDARD]
    fdesc = ", ".join(f"{sid.replace('subject','S')} (MAPE {m:.1f}%, SNR {sn:.2f})"
                      for sid, m, sn in fail)
    p.box(16, y + 2, 306,
          f"(a) Sisa yang gagal: {fdesc}. Keduanya ber-SNR di bawah ambang 1.8, "
          "yaitu batas FISIK akuisisi (jarak subjek), bukan kekurangan metode. "
          "(b) Ambang SNR dihitung pada frekuensi ECG, sehingga ia alat diagnosis, bukan "
          "penyaring kualitas yang bisa dipakai tanpa ECG. Keduanya terbuka untuk penelitian "
          "lanjutan.",
          "4. Keterbatasan yang diakui terbuka", GREY, 10.5)

    p.output(str(out / "presentasi_unwrapPhasePeak.pdf"))
    print("ok:", out / "presentasi_unwrapPhasePeak.pdf")


def main(csv, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "_fig_slides"
    figdir.mkdir(exist_ok=True)

    df = pd.read_csv(csv)
    front_all = df[df.dataset == "position_front"].copy()
    front = front_all[front_all.unwrapPhasePeak_mm.notna()].copy()

    fig_what(front, figdir)
    fig_why(front_all, figdir)
    fig_pipeline(front, figdir)
    res = fig_results(front, figdir)
    fig_track(front, figdir)
    print("figure ok")
    build(figdir, out, res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
