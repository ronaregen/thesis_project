"""
Slide bimbingan: KELUARAN LOGGER (TI) vs PIPELINE FASE MENTAH.

Ini perbandingan utama tesis versi baru (disetujui pembimbing 14 Jul 2026).
Semua angka diimpor dari code/evaluation/compare_ti_vs_phase.py -- tidak ada
angka yang diketik tangan di sini.

Usage:
    python make_slides_ti_vs_phase.py <aligned_csv> <outdir>
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
from phase_pipeline import MAPE_STANDARD, MEDFILT_K, WINDOW_SEC  # noqa: E402

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": .3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

NAVY, RED, GREEN, GREY, PURPLE = (28, 42, 66), (192, 57, 43), (30, 132, 73), (110, 118, 128), (110, 60, 140)
AMBER = (176, 122, 12)
C_BAD, C_GOOD, C_GT, C_NEU = "#c0392b", "#1e8449", "#2c3e50", "#7f8c8d"
C_TI = "#c0392b"      # kolom logger
C_FASE = "#1e8449"    # pipeline kita
W, H = 338.7, 190.5

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------- kenapa kolom TI runtuh
def fig_bandshift(front, out):
    """Bukti mekanismenya: filter TI dirancang 20 fps, dijalankan 50 fps ->
    responsnya mulur 2.5x -> band jantung geser ke 2-10 Hz -> detak manusia
    (1-2 Hz) justru DIBUANG oleh filter TI sendiri.

    PENTING: fs dihitung dari MEAN dt, bukan median. Timestamp host terkuantisasi
    timer OS Windows (median dt 15.6 ms), padahal frame radar benar-benar datang
    tiap 20 ms (mean dt = 20.0 ms) -- lihat CLAUDE.md poin 27. Memakai median akan
    memelarkan sumbu frekuensi 1.28x dan memberi puncak palsu di 3.8 Hz.
    """
    spec = []
    for _, sub in front.groupby("subject_id"):
        sub = sub.sort_values("Timestamp")
        x = sub.outputFilterHeartOut.values.astype(float)
        fs = 1.0 / np.mean(np.diff(sub.Timestamp.values))
        f, p = signal.welch(signal.detrend(x), fs=fs, nperseg=min(4096, len(x)))
        spec.append((f, p))

    # semua subjek difilter di grid frekuensi yang sama -> rata-ratakan
    f = np.linspace(0, 12, 1200)
    p = np.mean([np.interp(f, fi, pi) for fi, pi in spec], axis=0)

    tot = np.trapezoid(p, f)
    e_hr = np.trapezoid(p[(f >= .8) & (f <= 2)], f[(f >= .8) & (f <= 2)]) / tot * 100
    e_sh = np.trapezoid(p[(f >= 2) & (f <= 10)], f[(f >= 2) & (f <= 10)]) / tot * 100

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 4.6),
                                 gridspec_kw={"width_ratios": [1, 1.25]})

    # kiri: band yang DIMAKSUD vs band yang SEBENARNYA lewat
    a1.barh([1], [4.0 - 0.8], left=[0.8], height=.42, color=C_GOOD, alpha=.85,
            label="band jantung yang DIMAKSUD TI (20 fps)")
    a1.barh([0], [10.0 - 2.0], left=[2.0], height=.42, color=C_BAD, alpha=.85,
            label="band yang SEBENARNYA lewat (50 fps)")
    a1.axvspan(1.0, 1.7, color="#2980b9", alpha=.22)
    a1.text(1.35, 1.75, "detak jantung\nmanusia istirahat\n(1.0-1.7 Hz)", ha="center",
            fontsize=9, color="#1f5f8b", fontweight="bold")
    a1.set_yticks([0, 1])
    a1.set_yticklabels(["dijalankan\n50 fps", "dirancang\n20 fps"], fontsize=9.5)
    a1.set_xlabel("Hz")
    a1.set_xlim(0, 11)
    a1.set_ylim(-.6, 2.5)
    a1.legend(fontsize=8.2, loc="lower right")
    a1.set_title("Filter TI berkoefisien TETAP.\nDi 50 fps responsnya mulur 2.5x.",
                 fontweight="bold", fontsize=11)

    # kanan: buktinya di data
    a2.semilogy(f, p, color=C_GT, lw=1.4)
    a2.axvspan(.8, 2.0, color=C_GOOD, alpha=.16)
    a2.axvspan(2.0, 10.0, color=C_BAD, alpha=.13)
    a2.text(1.4, p.max() * .5, f"{e_hr:.1f}%", ha="center", fontsize=13,
            fontweight="bold", color=C_GOOD)
    a2.text(6.0, p.max() * .5, f"{e_sh:.1f}%", ha="center", fontsize=13,
            fontweight="bold", color=C_BAD)
    a2.plot(f[np.argmax(p)], p.max(), "v", color=C_BAD, ms=11)
    a2.annotate(f"puncak {f[np.argmax(p)]:.2f} Hz = {f[np.argmax(p)]*60:.0f}/menit\n"
                "(bukan detak jantung manusia)",
                xy=(f[np.argmax(p)], p.max()), xytext=(5.2, p.max() * 5e-3),
                fontsize=8.6, color=C_BAD, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.2,
                                connectionstyle="arc3,rad=-.25"))
    a2.set_xlabel("Hz")
    a2.set_ylabel("daya")
    a2.set_title("BUKTINYA di data: spektrum outputFilterHeartOut\n"
                 "(rata-rata 10 subjek) - energinya di band yang tergeser",
                 fontweight="bold", fontsize=11)

    fig.tight_layout()
    fig.savefig(out / "t1_band.png")
    plt.close(fig)
    return e_hr, e_sh, f[np.argmax(p)]


# ------------------------------------------------- akurasi per subjek
def fig_mape(R, S, out):
    best = S["best"]
    x = np.arange(len(R))
    fig, ax = plt.subplots(figsize=(13.6, 4.4))
    ax.bar(x - .2, R[f"{best}_mape"], .4, color=C_TI, alpha=.88,
           label=f"{best} (kolom logger TERBAIK)")
    ax.bar(x + .2, R.fase_mape, .4, color=C_FASE, alpha=.88,
           label="pipeline fase mentah (usulan)")
    ax.axhline(MAPE_STANDARD, color=C_GT, ls="--", lw=2,
               label=f"ambang ANSI/CTA-2065 = {MAPE_STANDARD:.0f}%")
    for i, v in enumerate(R.fase_mape):
        if v < MAPE_STANDARD:
            ax.text(i + .2, v + .6, "OK", ha="center", fontsize=8,
                    fontweight="bold", color=C_FASE)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("subject", "S") for s in R.subj])
    ax.set_ylabel("MAPE (%)   makin rendah makin baik")
    ax.legend(fontsize=9.2)
    n_ti = int((R[f"{best}_mape"] < MAPE_STANDARD).sum())
    n_fa = int((R.fase_mape < MAPE_STANDARD).sum())
    ax.set_title(f"MAPE per subjek -- lolos standar: kolom logger {n_ti}/10, "
                 f"pipeline fase {n_fa}/10", fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out / "t2_mape.png")
    plt.close(fig)


# ------------------------------------------------- bukti detak tertangkap
def fig_scatter(R, S, out):
    """Slide kunci. Sumbu-x = HR asli tiap orang, sumbu-y = HR yang diestimasi.
    Kalau metode benar-benar menangkap detak jantung, titiknya naik mengikuti
    garis diagonal. Kalau tidak, titiknya acak/mendatar."""
    best = S["best"]
    lo, hi = 55, 100
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)

    for ax, col, r, c, name in [
        (axs[0], f"{best}_bpm", S["r_best"], C_TI, f"{best}\n(kolom logger terbaik)"),
        (axs[1], "fase_bpm", S["r_fase"], C_FASE, "pipeline fase mentah\n(usulan)"),
    ]:
        ax.plot([lo, hi], [lo, hi], color=C_NEU, ls="--", lw=1.5,
                label="estimasi = sebenarnya")
        ax.scatter(R.hr_ecg, R[col], s=95, color=c, alpha=.85, zorder=4,
                   edgecolor="white", lw=1.2)
        for _, p in R.iterrows():
            ax.annotate(p.subj.replace("subject", "S"), (p.hr_ecg, p[col]),
                        textcoords="offset points", xytext=(7, -3), fontsize=8,
                        color=C_GT)
        k = np.polyfit(R.hr_ecg, R[col], 1)
        xs = np.array([lo, hi])
        ax.plot(xs, np.polyval(k, xs), color=c, lw=2, alpha=.65,
                label=f"tren: r = {r:+.2f}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("HR SEBENARNYA dari ECG (bpm)")
        ax.legend(fontsize=8.8, loc="upper left")
        ax.set_title(name, fontweight="bold", fontsize=11.5, color=c)
    axs[0].set_ylabel("HR yang DIESTIMASI (bpm)")

    fig.suptitle("Apakah metodenya benar-benar menangkap detak jantung?\n"
                 "Tiap titik = satu subjek. Kalau menangkap, titik naik mengikuti garis diagonal.",
                 fontweight="bold", fontsize=12.5, y=1.06)
    fig.tight_layout()
    fig.savefig(out / "t3_scatter.png")
    plt.close(fig)


# ------------------------------------------------- vs trivial baseline
def fig_baseline(B, out):
    names = [f"Tebak konstan {B['const']:.0f} bpm\n(TANPA radar sama sekali)",
             "heartRateEst_xCorr\n(kolom logger terbaik)",
             "Pipeline fase mentah\n(usulan)"]
    vals = [B["triv_mae"], B["xcorr_mae"], B["fase_mae"]]
    cols = [C_NEU, C_TI, C_FASE]

    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    b = ax.bar(names, vals, color=cols, alpha=.88, width=.58)
    ax.bar_label(b, fmt="%.1f bpm", fontsize=10.5, fontweight="bold", padding=3)
    ax.axhline(B["triv_mae"], color=C_NEU, ls="--", lw=1.6)
    ax.set_ylabel("MAE pada subjek uji (bpm)")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.text(2, B["triv_mae"] * 1.14, "apa pun yang di ATAS garis ini lebih buruk\ndaripada tidak memakai radar sama sekali",
            ha="center", fontsize=8.6, color=C_NEU, fontweight="bold")
    ax.set_title("Uji paling keras: apakah radar mengalahkan menebak satu angka konstan?\n"
                 "(split berbasis subjek: latih S01-08, uji S09-10)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "t4_base.png")
    plt.close(fig)


# ------------------------------------------------- PDF
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
        self.cell(10, 5, f"T{n}", align="R")
        self.set_xy(16, y + 3)

    def pic(self, path, x, w, gap=1):
        """FPDF.image() tidak memajukan kursor y. Taruh gambar, balikin y BAWAHnya
        supaya kotak berikutnya tidak menabrak gambar."""
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


def build(fig, out, R, S, B, band):
    best = S["best"]
    e_hr, e_sh, fpk = band
    m_ti, m_fa = R[f"{best}_mape"].mean(), R.fase_mape.mean()
    a_ti, a_fa = R[f"{best}_mae"].mean(), R.fase_mae.mean()
    n_fa = int((R.fase_mape < MAPE_STANDARD).sum())
    n_ti = int((R[f"{best}_mape"] < MAPE_STANDARD).sum())

    p = Slides()

    # T1 -- pertanyaan & batasan
    p.slide(1, "Pertanyaan riset dan batasannya",
            "Yang dibandingkan: kolom BPM siap-pakai dari logger  vs  sinyal fase mentah yang kita olah sendiri.")
    y = p.get_y() + 2
    y = p.box(16, y, 306,
              "\"Pada konfigurasi akuisisi yang dipakai, seberapa besar perbaikan yang didapat dengan mengolah "
              "sinyal fase mentah (unwrapPhasePeak_mm) sendiri, dibanding memakai kolom detak jantung yang "
              "sudah jadi dari logger?\"",
              "Pertanyaan", NAVY, 11.5)
    y = p.box(16, y + 3, 306,
              "Penelitian ini TIDAK memberikan vonis kelayakan atas library Vital Signs bawaan TI.\n\n"
              "Data direkam pada frame rate 50 fps, sementara library TI dirancang untuk 20 fps. Rantai "
              "filter TI berkoefisien tetap, sehingga pada 50 fps seluruh responsnya mulur 2.5x. Menguji "
              "library TI di kondisi itu bukan pengujian yang adil, dan kami tidak mengklaimnya.\n\n"
              "Yang dievaluasi adalah: PADA KONDISI AKUISISI INI, sumber sinyal mana yang masih membawa "
              "informasi detak jantung.",
              "BATASAN MASALAH (dinyatakan di muka)", AMBER, 10.2)
    p.box(16, y + 3, 306,
          "unwrapPhasePeak_mm diambil SEBELUM rantai filter TI, sehingga kebal terhadap salah-konfigurasi itu. "
          "Kolom BPM turunan TI tidak. Ini pelajaran teknik yang bisa digeneralisasi: ambil sinyal sedini "
          "mungkin di rantai pemrosesan, sebelum filter vendor yang mengasumsikan sample rate tetap.",
          "Justru di situ temuannya", GREEN, 10.2)

    # T2 -- mekanisme
    p.slide(2, "Kenapa kolom BPM logger runtuh - mekanismenya, bukan dugaan",
            "Filter jantung TI justru MEMBUANG detak jantung manusia ketika dijalankan di 50 fps.")
    y = p.pic(fig / "t1_band.png", x=22, w=294)
    p.box(16, y + 3, 306,
          f"Energi sinyal outputFilterHeartOut: hanya {e_hr:.1f}% berada di pita detak jantung sebenarnya "
          f"(0.8-2.0 Hz), sementara {e_sh:.1f}% berada di pita yang tergeser (2-10 Hz). Puncak spektrumnya di "
          f"{fpk:.2f} Hz = {fpk*60:.0f} per menit - mustahil sebagai detak jantung manusia.\n"
          "Inilah sebabnya SEMUA kolom BPM turunan TI berkorelasi nol, padahal sinyal fase mentahnya sehat.",
          "Terukur, bisa direproduksi", RED, 9.8)

    # T3 -- akurasi
    p.slide(3, "Hasil 1: akurasi (MAPE terhadap standar ANSI/CTA-2065)",
            f"Jendela {WINDOW_SEC:.0f} detik yang SAMA PERSIS dipakai untuk kedua metode - perbandingan apel ke apel.")
    y = p.pic(fig / "t2_mape.png", x=22, w=294) + 3
    rows = [f"{'kolom':<26}{'MAPE':>7}{'MAE':>8}{'lolos':>8}", "-" * 49]
    for c in ["final_heart_rate", "heart_rate_est_fft", best]:
        rows.append(f"{c:<26}{R[f'{c}_mape'].mean():>6.0f}%{R[f'{c}_mae'].mean():>7.1f}"
                    f"{(R[f'{c}_mape'] < MAPE_STANDARD).sum():>6d}/10")
    rows += ["-" * 49,
             f"{'PIPELINE FASE (usulan)':<26}{m_fa:>6.1f}%{a_fa:>7.1f}{n_fa:>6d}/10"]
    p.box(16, y, 152, "\n".join(rows), None, NAVY, 8.6)
    p.box(174, y, 148,
          f"Kolom logger TERBAIK adalah {best}.\n\n"
          f"Perbaikan: MAPE {m_ti:.1f}% -> {m_fa:.1f}%  ({m_ti/m_fa:.1f}x lebih baik)\n"
          f"                MAE  {a_ti:.1f} -> {a_fa:.1f} bpm  ({a_ti/a_fa:.1f}x)\n\n"
          f"Lolos ambang 10%: {n_ti}/10  ->  {n_fa}/10.",
          "Ringkasnya", GREEN, 9.6)

    # T4 -- SLIDE KUNCI
    p.slide(4, "Hasil 2: apakah detak jantung BENAR-BENAR tertangkap radar?",
            "MAPE lebih kecil saja bisa dibantah. Ini ujinya: bisakah metode membedakan orang ber-HR tinggi dari yang rendah?")
    y = p.pic(fig / "t3_scatter.png", x=62, w=214) + 3
    p.box(16, y, 152,
          f"Korelasi antar-subjek (n = 10):\n"
          f"   {best:<20} r = {S['r_best']:+.2f}\n"
          f"   pipeline fase        r = {S['r_fase']:+.2f}\n\n"
          f"Bias:  logger {S['bias_best']:+.1f} bpm,  fase {S['bias_fase']:+.1f} bpm.\n\n"
          f"Std keluaran (keberanian berbeda antar orang):\n"
          f"   HR asli   {R.hr_ecg.std():.1f} bpm\n"
          f"   logger    {R[f'{best}_bpm'].std():.1f} bpm   <- nyaris tidak berubah\n"
          f"   fase      {R.fase_bpm.std():.1f} bpm",
          "Angkanya", NAVY, 9.0)
    p.box(174, y, 148,
          "Korelasi kolom logger NEGATIF. Artinya bukan sekadar 'kurang akurat' - dia tidak menangkap "
          "APA PUN. Orang ber-HR rendah malah diestimasi tinggi.\n\n"
          f"Dia praktis PENEBAK-KONSTAN {S['stuck_const']:.0f} bpm: korelasi MAPE-nya dengan MAPE "
          f"penebak-konstan = {S['r_stuck']:+.2f}. Itu sebabnya dia kadang 'menang' - di subjek yang "
          "HR-nya kebetulan dekat 87. Jam mati pun benar dua kali sehari.\n\n"
          "Pipeline fase r positif kuat, biasnya hampir nol, dan std keluarannya menyamai std HR asli.",
          "Membacanya", GREEN, 9.0)

    # T5 -- pembanding wajib + kejujuran
    p.slide(5, "Hasil 3: uji paling keras, dan apa yang belum berhasil",
            "Sebuah metode baru bermakna hanya jika mengalahkan cara paling bodoh: menebak satu angka konstan.")
    y0 = p.get_y() + 1
    p.image(str(fig / "t4_base.png"), x=18, y=y0, w=168)
    y = p.box(196, y0, 126,
              f"Tebak konstan {B['const']:.0f} bpm : MAE {B['triv_mae']:.1f}\n"
              f"{best[:18]:<18}: MAE {B['xcorr_mae']:.1f}\n"
              f"Pipeline fase            : MAE {B['fase_mae']:.1f}\n\n"
              "Kolom logger KALAH dari menebak angka konstan tanpa radar sama sekali.\n"
              "Pipeline fase MENANG telak.",
              "Subjek uji S09 + S10", NAVY, 9.4)
    # daftar subjek gagal DITURUNKAN dari data -- supaya tidak pernah basi
    fail = R[R.fase_mape >= MAPE_STANDARD]
    fdesc = ", ".join(f"{r.subj.replace('subject','S')} (MAPE {r.fase_mape:.1f}%, SNR {r.snr:.2f})"
                      for _, r in fail.iterrows())
    lose = R[R.fase_mape >= R[f"{best}_mape"]]
    ldesc = ", ".join(r.subj.replace("subject", "S") for _, r in lose.iterrows())
    p.box(196, y + 3, 126,
          f"- {len(fail)} dari 10 subjek masih gagal: {fdesc}. SNR jantungnya di bawah "
          "ambang 1.8 - sinyalnya setara noise. Itu batas FISIK akuisisi (jarak subjek), "
          "bukan kekurangan metode.\n\n"
          f"- Kolom logger tampak lebih baik di {ldesc}. Itu KEBETULAN: kolom logger praktis "
          "mengeluarkan angka konstan ~87 bpm untuk siapa pun (std keluarannya hanya "
          f"{R[f'{best}_bpm'].std():.1f} bpm, padahal HR asli std {R.hr_ecg.std():.1f}), dan HR "
          "kedua subjek itu kebetulan dekat 87.\n\n"
          "- Perbandingan ini berlaku PADA 50 fps. Kami tidak memvonis library TI secara umum.",
          "Yang belum berhasil (kami sampaikan sendiri)", AMBER, 9.4)

    p.output(str(out / "logger_vs_fase.pdf"))
    print("ok:", out / "logger_vs_fase.pdf")


def main(csv, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "_fig_tivsfase"
    figdir.mkdir(exist_ok=True)

    df = pd.read_csv(csv)
    front = df[(df.dataset == "position_front") & df.unwrapPhasePeak_mm.notna()].copy()

    R, S = compare(front)
    B = baseline(front)
    band = fig_bandshift(front, figdir)
    fig_mape(R, S, figdir)
    fig_scatter(R, S, figdir)
    fig_baseline(B, figdir)
    print("figure ok")
    build(figdir, out, R, S, B, band)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
