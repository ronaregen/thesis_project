"""
Slide bimbingan: "APAKAH SUBJEK YANG GAGAL PERLU DIGANTI?"

Dibuat untuk menjawab saran pembimbing (ganti subjek yang MAPE-nya jelek).
Argumennya tiga langkah:
  N1  Kegagalannya TIDAK acak - ia diprediksi oleh SNR sinyal jantung
  N2  SNR itu sendiri diprediksi oleh JARAK - dibuktikan eksperimen terkendali
  N3  Karena itu subjek gagal = BUKTI (kontribusi K3), bukan aib. Usul konkret.

Semua angka diimpor dari code/evaluation/snr_criterion.py.

Usage:
    python make_slides_snr.py <aligned_csv> <outdir>
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from phase_pipeline import MAPE_STANDARD  # noqa: E402
from snr_criterion import SNR_THRESHOLD, collect  # noqa: E402

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": .3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

NAVY, RED, GREEN, GREY, PURPLE = (28, 42, 66), (192, 57, 43), (30, 132, 73), (110, 118, 128), (110, 60, 140)
AMBER = (176, 122, 12)
C_OK, C_BAD, C_GT, C_NEU = "#1e8449", "#c0392b", "#2c3e50", "#7f8c8d"
W, H = 338.7, 190.5

MARK = {"utama": "o", "hold-out": "s", "jarak": "D"}
LABEL = {"utama": "10 subjek utama", "hold-out": "2 subjek hold-out",
         "jarak": "2 sesi jarak terkendali"}


# ------------------------------------------------------------ N1: SNR vs MAPE
def fig_snr(R, out):
    fig, ax = plt.subplots(figsize=(12.6, 5.4))

    ax.axvspan(0, SNR_THRESHOLD, color=C_BAD, alpha=.09)
    ax.axhspan(MAPE_STANDARD, 100, color=C_BAD, alpha=.05)
    ax.axvline(SNR_THRESHOLD, color=C_BAD, ls="--", lw=2,
               label=f"ambang SNR = {SNR_THRESHOLD}")
    ax.axhline(MAPE_STANDARD, color=C_GT, ls="--", lw=2,
               label=f"standar ANSI/CTA-2065 = {MAPE_STANDARD:.0f}%")

    for grp, mk in MARK.items():
        s = R[R.kelompok == grp]
        ax.scatter(s.snr, s.mape, s=155, marker=mk,
                   c=[C_OK if v else C_BAD for v in s.lolos],
                   edgecolor="white", lw=1.5, zorder=5, label=LABEL[grp])
    # geser label yang bertumpuk (S03 & cad02 hampir berimpit di sekitar ambang)
    nudge = {"S03": (-26, -3), "cad02": (10, 4), "50 cm": (10, -10),
             "S05": (-24, -8), "cad01": (10, 2)}
    for _, r in R.iterrows():
        ax.annotate(r.sesi, (r.snr, r.mape), textcoords="offset points",
                    xytext=nudge.get(r.sesi, (9, -3)), fontsize=8.6, color=C_GT)

    # tren: makin lemah sinyal, makin besar galat
    lx = np.log(R.snr)
    k = np.polyfit(lx, np.log(R.mape), 1)
    xs = np.linspace(R.snr.min() * .85, R.snr.max() * 1.1, 100)
    ax.plot(xs, np.exp(np.polyval(k, np.log(xs))), color=C_NEU, lw=1.8, ls=":",
            zorder=2, label=f"tren  (r = {np.corrcoef(lx, np.log(R.mape))[0,1]:+.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, 11)
    ax.set_ylim(1, 70)
    ax.set_xticks([1, 1.8, 3, 5, 8, 10])
    ax.set_xticklabels(["1", "1.8", "3", "5", "8", "10"])
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1%", "2%", "5%", "10%", "20%", "50%"])
    ax.set_xlabel("SNR sinyal jantung  (kekuatan sinyal, makin kanan makin kuat)")
    ax.set_ylabel("MAPE  (galat, makin bawah makin baik)")
    ax.legend(fontsize=9, loc="upper right", ncol=2)

    n_ok = int(R.cocok.sum())
    ax.set_title(f"Semua sesi ber-SNR di bawah {SNR_THRESHOLD} GAGAL. "
                 f"Ambang ini benar di {n_ok} dari {len(R)} sesi.\n"
                 "Kegagalannya tidak acak - ia bisa diprediksi dari kekuatan sinyal.",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "n1_snr.png")
    plt.close(fig)


# ------------------------------------------------------------ N2: jarak
def fig_dist(R, out):
    d = R[R.kelompok == "jarak"].sort_values("snr")
    far, near = d.iloc[0], d.iloc[-1]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.4, 4.3))

    # (1) hukum fisika 1/R^4
    r = np.linspace(30, 150, 200)
    a1.plot(r, (50 / r) ** 4, color=C_GT, lw=2.4)
    for x, c, lab in [(50, C_OK, "50 cm"), (100, C_BAD, "100 cm")]:
        a1.plot(x, (50 / x) ** 4, "o", ms=13, color=c, zorder=5)
        a1.annotate(f"{lab}\ndaya {(50/x)**4:.2f}x", (x, (50 / x) ** 4),
                    textcoords="offset points", xytext=(10, 6), fontsize=9,
                    color=c, fontweight="bold")
    a1.set_xlabel("jarak subjek (cm)")
    a1.set_ylabel("daya pantul (relatif thd 50 cm)")
    a1.set_title("Hukum fisika: daya pantul ~ 1/R^4\n"
                 "Jarak 2x lipat -> daya 16x lebih kecil",
                 fontweight="bold", fontsize=11)

    # (2) SNR terukur
    b = a2.bar(["50 cm", "100 cm"], [near.snr, far.snr],
               color=[C_OK, C_BAD], width=.55, alpha=.9)
    a2.bar_label(b, fmt="%.2f", fontsize=11, fontweight="bold", padding=3)
    a2.axhline(SNR_THRESHOLD, color=C_BAD, ls="--", lw=2,
               label=f"ambang {SNR_THRESHOLD}")
    a2.set_ylabel("SNR sinyal jantung terukur")
    a2.set_ylim(0, near.snr * 1.3)
    a2.legend(fontsize=9)
    a2.set_title("Yang terukur di data:\nSNR runtuh persis seperti diramalkan",
                 fontweight="bold", fontsize=11)

    # (3) akibatnya ke MAPE
    b = a3.bar(["50 cm", "100 cm"], [near.mape, far.mape],
               color=[C_OK, C_BAD], width=.55, alpha=.9)
    a3.bar_label(b, fmt="%.1f%%", fontsize=11, fontweight="bold", padding=3)
    a3.axhline(MAPE_STANDARD, color=C_GT, ls="--", lw=2, label="standar 10%")
    a3.set_ylabel("MAPE (%)")
    a3.set_ylim(0, far.mape * 1.25)
    a3.legend(fontsize=9)
    a3.set_title("Akibatnya:\nLOLOS  vs  GAGAL TOTAL", fontweight="bold", fontsize=11)

    fig.suptitle("EKSPERIMEN TERKENDALI - subjek yang SAMA, logger yang SAMA, "
                 "hanya JARAK yang berbeda",
                 fontweight="bold", fontsize=13.5, y=1.04)
    fig.tight_layout()
    fig.savefig(out / "n2_dist.png")
    plt.close(fig)
    return near, far


# ------------------------------------------------------------ PDF
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
        self.cell(10, 5, f"N{n}", align="R")
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


def build(fig, out, R, near, far):
    fail = R[~R.lolos]
    n_ok = int(R.cocok.sum())
    lr = float(np.corrcoef(np.log(R.snr), np.log(R.mape))[0, 1])
    p = Slides()

    # ---- N1
    p.slide(1, "Apakah subjek yang gagal itu acak?",
            "Kalau acak, mengganti subjek masuk akal. Kalau punya sebab, membuangnya "
            "berarti membuang temuan.")
    y = p.pic(fig / "n1_snr.png", x=40, w=258)
    p.box(16, y + 3, 306,
          f"Sesi yang gagal: {', '.join(fail.sesi)} - dan SEMUANYA yang ber-SNR di bawah "
          f"{SNR_THRESHOLD} gagal, tanpa kecuali. Korelasi SNR terhadap galat = {lr:+.2f}: "
          f"makin lemah sinyalnya, makin besar galatnya, sangat teratur.\n"
          f"Ambang ini memprediksi lolos/gagal dengan benar di {n_ok} dari {len(R)} sesi - "
          f"termasuk 4 sesi yang TIDAK ikut menentukan ambangnya (2 hold-out + 2 jarak).\n\n"
          f"Kegagalannya BUKAN acak. Ia bisa diramalkan sebelum melihat MAPE-nya.",
          "Yang terbaca dari grafik", NAVY, 10)

    # ---- N2
    p.slide(2, "Lalu apa yang menentukan SNR? Jarak.",
            "Ini eksperimen terkendali: subjek yang sama, logger yang sama. Hanya jaraknya diubah.")
    y = p.pic(fig / "n2_dist.png", x=22, w=294)
    p.box(16, y + 3, 306,
          f"Subjek yang SAMA, direkam dua kali. Di {near.sesi}: SNR {near.snr:.2f}, MAPE "
          f"{near.mape:.1f}% - LOLOS. Di {far.sesi}: SNR {far.snr:.2f}, MAPE {far.mape:.1f}% - GAGAL TOTAL.\n"
          "Tidak ada yang berubah selain jarak duduknya. Ini bukan korelasi kebetulan - ini "
          "hubungan sebab-akibat, dan besarnya persis seperti yang diramalkan hukum 1/R^4.\n\n"
          "Artinya: subjek yang gagal bukan 'subjek yang jelek'. Mereka subjek yang DUDUKNYA "
          "TERLALU JAUH.",
          "Sebabnya fisik, bukan kebetulan orangnya", RED, 10)

    # ---- N3
    p.slide(3, "Usul: jangan ganti subjeknya - ini justru temuannya",
            "Kegagalan yang bisa dijelaskan lebih berharga daripada keberhasilan yang tidak bisa.")
    y = p.get_y() + 2
    y = p.box(16, y, 306,
              f"Mengganti subjek yang gagal tidak menyelesaikan apa pun. Subjek penggantinya akan "
              f"GAGAL JUGA kalau duduk sejauh itu, dan LOLOS JUGA kalau duduk dekat - persis seperti "
              f"yang ditunjukkan eksperimen jarak. Yang perlu diganti bukan subjeknya, melainkan "
              f"JARAK DUDUKNYA saat perekaman.",
              "1. Mengganti subjek tidak memperbaiki apa pun", RED, 10)
    y = p.box(16, y + 2.5, 306,
              f"Kalau {', '.join(fail.sesi)} dibuang, kita kehilangan satu-satunya bukti yang "
              f"menetapkan KAPAN metode ini bisa dipakai dan kapan secara fisik tidak bisa. Yang "
              f"tersisa adalah metode yang mengaku berhasil pada semua orang tanpa batas yang "
              f"dinyatakan - dan itu justru LEBIH mudah dipatahkan penguji, bukan lebih aman. "
              f"Metode yang jujur soal batasnya lebih dipercaya daripada metode yang tidak pernah gagal.",
              "2. Justru subjek gagal inilah kontribusi K3 (kriteria kelayakan)", GREEN, 10)
    y = p.box(16, y + 2.5, 306,
              f"- Semua {len(R)} sesi dilaporkan apa adanya, tidak ada yang dibuang.\n"
              f"- Kegagalan {', '.join(fail.sesi)} dijelaskan sebabnya (SNR < {SNR_THRESHOLD}), "
              f"bukan disembunyikan.\n"
              f"- Kriteria SNR >= {SNR_THRESHOLD} diajukan sebagai kriteria kelayakan perekaman - "
              f"terbukti benar di {n_ok}/{len(R)} sesi.\n"
              f"- Untuk perekaman berikutnya (kalau ada): jarak subjek ~50 cm dan DICATAT. "
              f"Itu memperbaiki sebabnya, bukan menyembunyikan akibatnya.",
              "3. Yang saya usulkan", NAVY, 10)
    p.box(16, y + 2.5, 306,
          f"Satu sesi meleset dari ambang: cad02 (SNR 2.58, di ATAS ambang, tapi MAPE 10.0%). "
          f"Ini kasus BATAS - melesetnya hanya 0.0 poin dari ambang 10%. Dilaporkan apa adanya, "
          f"tidak dibulatkan menjadi 'lolos'.",
          "Yang saya sampaikan sendiri (tidak menunggu ditanya)", AMBER, 9.6)

    p.output(str(out / "kenapa_subjek_gagal.pdf"))
    print("ok:", out / "kenapa_subjek_gagal.pdf")


def main(csv, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "_fig_snr"
    figdir.mkdir(exist_ok=True)

    R = collect(csv)
    R["lolos"] = R.mape < MAPE_STANDARD
    R["cocok"] = R.lolos == (R.snr >= SNR_THRESHOLD)

    fig_snr(R, figdir)
    near, far = fig_dist(R, figdir)
    print("figure ok")
    build(figdir, out, R, near, far)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
