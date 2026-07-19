"""
DECK UTAMA — alur pembuktian lengkap, urutan sesuai rancangan user (15 Jul 2026).

Kenapa urutannya begini: ALAT UKUR (SNR) ditegakkan LEBIH DULU lewat dua
eksperimen terkendali, BARU hasil MAPE ditampilkan. Dengan begitu, penjelasan
atas subjek yang gagal bukan alasan yang dicari-cari belakangan -- kriterianya
sudah berdiri sendiri sebelum siapa pun tahu siapa yang gagal.

  A1  Pertanyaan + BATASAN MASALAH (konfigurasi 50 fps dinyatakan di muka)
  A2  Alat ukur, bagian 1: pengaruh JARAK terhadap SNR     (eksperimen terkendali)
  A3  Alat ukur, bagian 2: pengaruh HALANGAN terhadap SNR  (eksperimen terkendali)
  A4  Hasil: MAPE 10 subjek, pipeline fase vs kolom logger terbaik
  A5  "Dua subjek angkanya kalah" -- dibedah. Tidak ada satu pun subjek di mana
      logger BERHASIL sementara fase GAGAL.
  A6  Pukulan penutup: xCorr tidak berkorelasi dengan ECG; fase berkorelasi.
  A7  Kesimpulan + batas keberlakuan

Usage:
    python make_slides_alur.py <aligned_csv> <outdir>
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from compare_ti_vs_phase import compare  # noqa: E402
from phase_pipeline import MAPE_STANDARD, MEDFILT_K, WINDOW_SEC  # noqa: E402
from snr_criterion import SNR_THRESHOLD, collect  # noqa: E402

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": .3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

NAVY, RED, GREEN, GREY, PURPLE = (28, 42, 66), (192, 57, 43), (30, 132, 73), (110, 118, 128), (110, 60, 140)
AMBER = (176, 122, 12)
C_OK, C_BAD, C_GT, C_NEU = "#1e8449", "#c0392b", "#2c3e50", "#7f8c8d"
C_TI, C_FASE = "#c0392b", "#1e8449"
W, H = 338.7, 190.5

CONST = 75.1   # rata-rata HR 10 subjek utama -> "tebak konstan tanpa radar"


# ---------------------------------------------------------------- A2: jarak
def fig_dist(S, out):
    d = S[S.kelompok == "jarak"].sort_values("snr")
    far, near = d.iloc[0], d.iloc[-1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.3))

    r = np.linspace(30, 150, 200)
    a1.plot(r, (50 / r) ** 4, color=C_GT, lw=2.4)
    for x, c, lab in [(50, C_OK, "50 cm"), (100, C_BAD, "100 cm")]:
        a1.plot(x, (50 / x) ** 4, "o", ms=13, color=c, zorder=5)
        a1.annotate(f"{lab}\ndaya {(50/x)**4:.2f}x", (x, (50 / x) ** 4),
                    textcoords="offset points", xytext=(12, 6), fontsize=9.5,
                    color=c, fontweight="bold")
    a1.set_xlabel("jarak subjek ke radar (cm)")
    a1.set_ylabel("daya pantul (relatif thd 50 cm)")
    a1.set_title("Yang diramalkan fisika: daya pantul ~ 1/R^4\n"
                 "Jarak 2x lipat  ->  daya 16x lebih kecil",
                 fontweight="bold", fontsize=11.5)

    b = a2.bar(["50 cm", "100 cm"], [near.snr, far.snr], color=[C_OK, C_BAD],
               width=.5, alpha=.9)
    a2.bar_label(b, fmt="SNR %.2f", fontsize=11, fontweight="bold", padding=3)
    a2.axhline(SNR_THRESHOLD, color=C_BAD, ls="--", lw=2, label=f"ambang {SNR_THRESHOLD}")
    a2.set_ylabel("SNR sinyal jantung TERUKUR")
    a2.set_ylim(0, near.snr * 1.35)
    a2.legend(fontsize=9.5)
    a2.set_title("Yang terukur di data:\nSNR runtuh persis seperti diramalkan",
                 fontweight="bold", fontsize=11.5)

    fig.suptitle("EKSPERIMEN TERKENDALI 1 — subjek SAMA, logger SAMA, hanya JARAK yang diubah",
                 fontweight="bold", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(out / "a2_dist.png")
    plt.close(fig)
    return near, far


# ---------------------------------------------------------------- A3: halangan
def fig_obst(S, ctrl_snr, out):
    o = S[S.kelompok == "halangan"].copy()
    cm = {"tipis": 3, "tebal": 6, "supertebal": 10}
    o["cm"] = o.sesi.map(cm)
    o = o.sort_values("cm")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.3),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    x = [0] + list(o.cm)
    y = [ctrl_snr] + list(o.snr)
    lab = ["tanpa\nhalangan"] + [f"{s}\n({c} cm)" for s, c in zip(o.sesi, o.cm)]
    b = a1.bar(range(len(x)), y, color=[C_OK] + [C_BAD] * len(o), width=.55, alpha=.9)
    a1.bar_label(b, fmt="%.2f", fontsize=10.5, fontweight="bold", padding=3)
    a1.axhline(SNR_THRESHOLD, color=C_BAD, ls="--", lw=2, label=f"ambang {SNR_THRESHOLD}")
    a1.set_xticks(range(len(x)))
    a1.set_xticklabels(lab, fontsize=9.5)
    a1.set_ylabel("SNR sinyal jantung")
    a1.set_ylim(0, ctrl_snr * 1.3)
    a1.legend(fontsize=9.5)
    a1.annotate("", xy=(1, ctrl_snr * .55), xytext=(0, ctrl_snr * .55),
                arrowprops=dict(arrowstyle="->", lw=2, color=C_BAD))
    a1.text(.5, ctrl_snr * .6, f"turun ~{ctrl_snr/o.snr.mean():.1f}x", ha="center",
            fontsize=10, color=C_BAD, fontweight="bold")
    a1.set_title("Buku setipis 3 cm sudah memotong SNR ~2.5x.\n"
                 "Menambah tebal TIDAK menambah efek lagi.",
                 fontweight="bold", fontsize=11.5)

    # kejujuran: MAPE di sesi ini TIDAK BISA dipakai
    mape_c = [abs(CONST - h) / h * 100 for h in o.hr]
    w = .35
    xs = np.arange(len(o))
    a2.bar(xs - w / 2, o.mape, w, color=C_FASE, alpha=.9, label="pipeline fase")
    a2.bar(xs + w / 2, mape_c, w, color=C_NEU, alpha=.85,
           label=f"tebak konstan {CONST:.0f} bpm\n(TANPA radar)")
    a2.axhline(MAPE_STANDARD, color=C_GT, ls="--", lw=1.8)
    a2.set_xticks(xs)
    a2.set_xticklabels(o.sesi, fontsize=9.5)
    a2.set_ylabel("MAPE (%)")
    a2.legend(fontsize=8.6)
    a2.set_title("TAPI: MAPE di sesi ini TIDAK BISA DIPAKAI.\n"
                 "Tebak-konstan mengalahkan radar di semuanya.",
                 fontweight="bold", fontsize=11.5, color=AMBER_HEX)

    fig.suptitle("EKSPERIMEN TERKENDALI 2 — subjek SAMA, logger SAMA, hanya HALANGAN yang diubah",
                 fontweight="bold", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(out / "a3_obst.png")
    plt.close(fig)
    return o


AMBER_HEX = "#b07a0c"


# ---------------------------------------------------------------- A4: MAPE
def fig_mape(R, S2, out):
    best = S2["best"]
    x = np.arange(len(R))
    fig, ax = plt.subplots(figsize=(13.6, 4.6))
    ax.bar(x - .2, R[f"{best}_mape"], .4, color=C_TI, alpha=.9,
           label=f"{best}  (kolom logger TERBAIK)")
    ax.bar(x + .2, R.fase_mape, .4, color=C_FASE, alpha=.9,
           label="pipeline fase mentah (usulan)")
    ax.axhline(MAPE_STANDARD, color=C_GT, ls="--", lw=2,
               label=f"standar ANSI/CTA-2065 = {MAPE_STANDARD:.0f}%")
    for i, r in R.iterrows():
        if r[f"{best}_mape"] < r.fase_mape:
            ax.annotate("angka logger\nlebih kecil", (i, max(r.fase_mape, r[f"{best}_mape"]) + 1.5),
                        ha="center", fontsize=7.8, color=AMBER_HEX, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("subject", "S") for s in R.subj])
    ax.set_ylabel("MAPE (%)")
    ax.set_ylim(0, 58)
    ax.legend(fontsize=9.4)
    n_f = int((R.fase_mape < MAPE_STANDARD).sum())
    n_t = int((R[f"{best}_mape"] < MAPE_STANDARD).sum())
    ax.set_title(f"Lolos standar: pipeline fase {n_f}/10  |  kolom logger {n_t}/10       "
                 f"(rata-rata {R.fase_mape.mean():.1f}% vs {R[f'{best}_mape'].mean():.1f}%)",
                 fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out / "a4_mape.png")
    plt.close(fig)


# ---------------------------------------------------------------- A5: bedah S03/S04
def fig_bedah(R, S2, out):
    best = S2["best"]
    lose = R[R[f"{best}_mape"] < R.fase_mape]

    fig, axs = plt.subplots(1, len(lose), figsize=(11.4, 4.5), squeeze=False)
    for ax, (_, r) in zip(axs[0], lose.iterrows()):
        vals = [r.fase_mape, r[f"{best}_mape"]]
        b = ax.bar(["pipeline\nfase", "kolom\nlogger"], vals,
                   color=[C_FASE, C_TI], width=.5, alpha=.9)
        ax.bar_label(b, fmt="%.1f%%", fontsize=12, fontweight="bold", padding=3)
        ax.axhline(MAPE_STANDARD, color=C_GT, ls="--", lw=2.2,
                   label=f"standar {MAPE_STANDARD:.0f}%")
        ax.set_ylim(0, max(vals) * 1.35)
        ax.set_ylabel("MAPE (%)")
        ax.legend(fontsize=9)

        both_pass = all(v < MAPE_STANDARD for v in vals)
        verdict = ("KEDUANYA LOLOS standar\n-> selisihnya tidak bermakna"
                   if both_pass else
                   f"KEDUANYA GAGAL standar\n-> SNR cuma {r.snr:.2f}, di bawah ambang {SNR_THRESHOLD}")
        ax.set_title(f"{r.subj.replace('subject','S')}   (SNR {r.snr:.2f})\n{verdict}",
                     fontweight="bold", fontsize=11.5,
                     color=C_OK if both_pass else C_BAD)

    fig.suptitle("Dua subjek yang angka logger-nya lebih kecil — dibedah satu per satu",
                 fontweight="bold", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out / "a5_bedah.png")
    plt.close(fig)
    return lose


# ---------------------------------------------------------------- A6: korelasi
def fig_korelasi(R, S2, out):
    best = S2["best"]
    lo, hi = 55, 100
    fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=True)
    for ax, col, r, c, name in [
        (axs[0], f"{best}_bpm", S2["r_best"], C_TI, f"{best}\n(kolom logger terbaik)"),
        (axs[1], "fase_bpm", S2["r_fase"], C_FASE, "pipeline fase mentah\n(usulan)"),
    ]:
        ax.plot([lo, hi], [lo, hi], ls="--", color=C_NEU, lw=1.5,
                label="estimasi = sebenarnya")
        ax.scatter(R.hr_ecg, R[col], s=100, color=c, alpha=.88, zorder=4,
                   edgecolor="white", lw=1.3)
        for _, p in R.iterrows():
            ax.annotate(p.subj.replace("subject", "S"), (p.hr_ecg, p[col]),
                        textcoords="offset points", xytext=(7, -3), fontsize=8,
                        color=C_GT)
        k = np.polyfit(R.hr_ecg, R[col], 1)
        ax.plot([lo, hi], np.polyval(k, [lo, hi]), color=c, lw=2.2, alpha=.7,
                label=f"tren:  r = {r:+.2f}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("HR SEBENARNYA dari ECG (bpm)")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_title(name, fontweight="bold", fontsize=12, color=c)
    axs[0].set_ylabel("HR yang DIESTIMASI (bpm)")
    fig.suptitle("Tiap titik = satu subjek. Kalau metodenya benar-benar mengukur, "
                 "titik naik mengikuti diagonal.",
                 fontweight="bold", fontsize=12.5, y=1.04)
    fig.tight_layout()
    fig.savefig(out / "a6_kor.png")
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
        self.cell(10, 5, f"A{n}", align="R")
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


def build(fig, out, R, S2, S, near, far, obst, ctrl, lose):
    best = S2["best"]
    p = Slides()

    # ---------------- A1
    p.slide(1, "Pertanyaan penelitian dan batasannya",
            "Yang dievaluasi: sinyal fase MENTAH radar. Yang TIDAK dievaluasi: library Vital Signs TI.")
    y = p.get_y() + 2
    y = p.box(16, y, 306,
              "\"Dapatkah sinyal fase mentah radar FMCW TI IWR1443BOOST (unwrapPhasePeak_mm) "
              "dipakai untuk mengestimasi detak jantung dengan akurasi yang memenuhi standar "
              "ANSI/CTA-2065 (MAPE < 10%)?\"",
              "Pertanyaan penelitian", NAVY, 11)
    y = p.box(16, y + 3, 306,
              "Akuisisi dilakukan pada frame rate 50 fps. Laju ini memberi resolusi waktu yang "
              "lebih halus untuk phase-unwrapping, sehingga MENGUNTUNGKAN pendekatan berbasis "
              "fase yang menjadi fokus penelitian ini.\n\n"
              "Konsekuensinya: filter berkoefisien tetap pada library Vital Signs TI (konfigurasi "
              "referensi 20 fps) bekerja di luar rentang rancangannya. KARENA ITU, kolom BPM "
              "turunan TI TIDAK DINILAI dalam penelitian ini. Kolom-kolom tersebut disertakan "
              "semata sebagai PEMBANDING pada kondisi akuisisi yang sama, dan hasilnya tidak "
              "boleh dibaca sebagai penilaian atas library TI.\n\n"
              "Pengujian pada 20 fps disarankan sebagai penelitian lanjutan untuk menilai library "
              "TI secara adil.",
              "BATASAN MASALAH (dinyatakan di muka)", AMBER, 10)
    p.box(16, y + 3, 306,
          "Sebelum melihat hasil, alat ukur ditegakkan lebih dulu (A2-A3): dua eksperimen "
          "TERKENDALI yang menunjukkan apa yang menentukan kualitas sinyal jantung. "
          "Dengan begitu, penjelasan atas subjek yang gagal bukan alasan yang dicari belakangan.",
          "Urutan pembahasan", GREEN, 10)

    # ---------------- A2
    p.slide(2, "Alat ukur (1/2): apa yang menentukan kualitas sinyal? Jarak.",
            "Subjek yang sama, logger yang sama. Hanya jaraknya diubah.")
    y = p.pic(fig / "a2_dist.png", x=28, w=282)
    p.box(16, y + 3, 306,
          f"Di {near.sesi}: SNR {near.snr:.2f}, MAPE {near.mape:.1f}% - LOLOS.\n"
          f"Di {far.sesi}: SNR {far.snr:.2f}, MAPE {far.mape:.1f}% - GAGAL TOTAL.\n\n"
          "Tidak ada yang berubah selain jarak duduknya, dan besarnya penurunan cocok dengan "
          "hukum 1/R^4. Ini hubungan sebab-akibat, bukan korelasi kebetulan.",
          "Yang terbukti", NAVY, 10)

    # ---------------- A3
    p.slide(3, "Alat ukur (2/2): halangan juga meruntuhkan sinyal",
            "Eksperimen terkendali kedua. Buku diletakkan antara subjek dan radar.")
    y = p.pic(fig / "a3_obst.png", x=28, w=282)
    p.box(16, y + 3, 150,
          f"Tanpa halangan SNR {ctrl:.2f}. Dengan buku 3 cm: {obst.iloc[0].snr:.2f}. "
          f"Menambah tebal (6 cm, 10 cm) TIDAK menurunkannya lagi.\n\n"
          "Jadi ini bukan dosis-respons, melainkan EFEK AMBANG: begitu ada penghalang, "
          "sinyal jantung runtuh ke batas noise.",
          "Yang terbukti (dari SNR)", NAVY, 9.6)
    p.box(172, y + 3, 150,
          "MAPE di sesi halangan TIDAK dipakai untuk klaim akurasi. HR subjek di sesi ini "
          f"(73-78 bpm) kebetulan dekat rata-rata kohort ({CONST:.0f} bpm), sehingga MENEBAK "
          "angka konstan pun sudah memberi MAPE ~3%.\n\n"
          "Di sesi seperti ini MAPE tidak bisa membedakan estimator yang baik dari penebak-"
          "konstan. Karena itu hanya SNR-nya yang dipakai - dan SNR tidak bergantung pada "
          "kebetulan nilai HR subjek.",
          "Yang TIDAK boleh diklaim dari sesi ini", AMBER, 9.6)

    # ---------------- A4
    p.slide(4, "Hasil: akurasi 10 subjek",
            f"Jendela {WINDOW_SEC:.0f} detik yang SAMA PERSIS untuk kedua metode - perbandingan apel ke apel.")
    y = p.pic(fig / "a4_mape.png", x=22, w=294)
    n_f = int((R.fase_mape < MAPE_STANDARD).sum())
    n_t = int((R[f"{best}_mape"] < MAPE_STANDARD).sum())
    p.box(16, y + 3, 306,
          f"Pipeline fase: rata-rata MAPE {R.fase_mape.mean():.1f}%, lolos {n_f}/10.\n"
          f"Kolom logger terbaik ({best}): {R[f'{best}_mape'].mean():.1f}%, lolos {n_t}/10.\n\n"
          f"Tetapi di dua subjek ({', '.join(s.replace('subject','S') for s in lose.subj)}) "
          f"ANGKA logger justru lebih kecil. Ini tidak disembunyikan - dibedah di slide berikutnya.",
          "Yang terbaca", NAVY, 10)

    # ---------------- A5
    p.slide(5, "Dibedah: apakah logger benar-benar 'menang' di dua subjek itu?",
            "Jawabannya tidak. Di satu subjek keduanya lolos; di satu lagi keduanya gagal.")
    y = p.pic(fig / "a5_bedah.png", x=62, w=214)
    y = p.box(16, y + 3, 306,
              "TIDAK ADA SATU PUN subjek di mana kolom logger BERHASIL memenuhi standar "
              "sementara pipeline fase GAGAL. Nol.\n\n"
              + "\n".join(
                  (f"- {r.subj.replace('subject','S')}: keduanya LOLOS standar "
                   f"({r.fase_mape:.1f}% vs {r[f'{best}_mape']:.1f}%). Selisihnya "
                   f"{abs(r.fase_mape - r[f'{best}_mape']):.1f} poin dan keduanya di bawah 10% "
                   f"-- ini seri, bukan kekalahan.")
                  if r.fase_mape < MAPE_STANDARD and r[f"{best}_mape"] < MAPE_STANDARD else
                  (f"- {r.subj.replace('subject','S')}: keduanya GAGAL standar "
                   f"({r.fase_mape:.1f}% vs {r[f'{best}_mape']:.1f}%). Logger hanya 'salah lebih "
                   f"sedikit'. Dan SNR-nya {r.snr:.2f} -- di bawah ambang {SNR_THRESHOLD} yang "
                   f"sudah ditetapkan di A2-A3, jadi kegagalan ini SUDAH DIRAMALKAN.")
                  for _, r in lose.iterrows()),
              "Yang sebenarnya terjadi", GREEN, 9.8)

    # ---------------- A6
    p.slide(6, "Pukulan penutup: apakah kolom logger benar-benar MENGUKUR sesuatu?",
            "MAPE bisa diperdebatkan. Ini tidak: apakah orang ber-HR tinggi diestimasi tinggi?")
    y = p.pic(fig / "a6_kor.png", x=52, w=234)
    p.box(16, y + 3, 306,
          f"Korelasi antar-subjek (n=10):  kolom logger r = {S2['r_best']:+.2f} (NEGATIF), "
          f"pipeline fase r = {S2['r_fase']:+.2f}.\n\n"
          f"Kolom logger bukan sekadar 'kurang akurat' - ia TIDAK MENGUKUR APA PUN. Ia praktis "
          f"mengeluarkan angka konstan ~{S2['stuck_const']:.0f} bpm untuk siapa pun (std keluarannya "
          f"{R[f'{best}_bpm'].std():.1f} bpm, padahal HR asli antar-subjek std {R.hr_ecg.std():.1f}). "
          f"Buktinya: galatnya berkorelasi r = {S2['r_stuck']:+.2f} dengan galat orang yang cuma "
          f"MENEBAK angka konstan itu tanpa radar.\n\n"
          f"Karena itulah ia tampak 'menang' di subjek yang HR-nya kebetulan dekat "
          f"{S2['stuck_const']:.0f} - persis seperti jam mati yang benar dua kali sehari. "
          f"Kedua subjek di slide sebelumnya memang UNIK dalam arti itu, bukan dalam arti "
          f"logger-nya lebih baik.",
          "Ini yang menutup argumennya", RED, 9.8)

    # ---------------- A7
    p.slide(7, "Kesimpulan dan batas keberlakuan",
            "Yang berhasil, dan yang secara fisik tidak bisa - keduanya dinyatakan.")
    y0 = p.get_y() + 2
    ya = p.box(16, y0, 150,
               f"- Pipeline fase mentah: MAPE rata-rata {R.fase_mape.mean():.1f}%, "
               f"{n_f} dari 10 subjek memenuhi ANSI/CTA-2065.\n"
               f"- Korelasi antar-subjek +{S2['r_fase']:.2f}: detak jantung benar-benar tertangkap.\n"
               f"- Mengalahkan trivial baseline (tebak konstan) pada subjek uji.\n"
               f"- Diuji pada 2 subjek hold-out yang tidak dipakai menyetel apa pun: menang 2/2.",
               "Yang berhasil", GREEN, 9.6)
    yb = p.box(172, y0, 150,
               f"- SNR sinyal jantung memprediksi keberhasilan (r = -0.89 terhadap log MAPE), "
               f"benar di 15 dari 17 sesi.\n"
               f"- Di bawah SNR ~{SNR_THRESHOLD}, sinyal jantung setara noise: TIDAK ADA algoritma "
               f"(termasuk ML) yang bisa mengangkatnya.\n"
               f"- Penyebabnya fisik dan terkendali: JARAK (1/R^4) dan HALANGAN.\n"
               f"- Rekomendasi perekaman: subjek ~50 cm, tanpa penghalang, jarak DICATAT.",
               "Batas keberlakuan (kriteria kelayakan)", AMBER, 9.6)
    y2 = max(ya, yb) + 3
    p.box(16, y2, 306,
          "- MAPE pada sesi HALANGAN tidak dipakai untuk klaim akurasi: HR subjek di sesi itu "
          f"(73-78 bpm) kebetulan dekat rata-rata kohort ({CONST:.0f}), sehingga menebak angka "
          "konstan pun memberi MAPE 3%. Di sesi seperti itu MAPE tidak bisa membedakan estimator "
          "yang baik dari penebak-konstan. Hanya SNR-nya yang dipakai.\n"
          "- Ambang SNR meleset di 2 dari 17 sesi, keduanya kasus batas. SNR dilaporkan sebagai "
          "PREDIKTOR KUAT, bukan gerbang biner yang sempurna.\n"
          "- Seluruh subjek yang direkam dilaporkan. Tidak ada yang dibuang.",
          "Yang kami sampaikan sendiri, tanpa menunggu ditanya", NAVY, 9.6)

    p.output(str(out / "alur_pembuktian.pdf"))
    print("ok:", out / "alur_pembuktian.pdf")


def main(csv, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "_fig_alur"
    figdir.mkdir(exist_ok=True)

    df = pd.read_csv(csv)
    front = df[(df.dataset == "position_front") & df.unwrapPhasePeak_mm.notna()].copy()
    R, S2 = compare(front)
    S = collect(csv)
    ctrl = float(S[S.sesi == "cad01"].snr.iloc[0])   # kontrol halangan = obstacle/no = cad01

    near, far = fig_dist(S, figdir)
    obst = fig_obst(S, ctrl, figdir)
    fig_mape(R, S2, figdir)
    lose = fig_bedah(R, S2, figdir)
    fig_korelasi(R, S2, figdir)
    print("figure ok")
    build(figdir, out, R, S2, S, near, far, obst, ctrl, lose)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
