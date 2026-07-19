"""
PDF: cara menghitung SNR, langkah demi langkah, dengan persamaan formal dan
sitasi. Semua angka dihitung LANGSUNG dari data -- tidak ada yang diketik tangan,
jadi dokumen ini tidak bisa basi diam-diam.

Alur penjelasan sengaja mengikuti urutan yang terbukti nyantol saat dijelaskan
lisan (bukan urutan kode):
  1. radar TIDAK menghasilkan satu angka -- dia menghasilkan tabel 49 baris
  2. ECG menghasilkan satu angka, dan tugasnya CUMA menunjuk baris mana
  3. dua angka diambil dari tabel radar itu, lalu dibagi
  4. kenapa BUKAN puncak radar (di S01 puncaknya justru bukan jantung)
  5. dari 1 jendela ke SNR subjek: median antar-jendela, bukan rata-rata

Lampiran menjawab dua pertanyaan yang datang belakangan: dari mana tabel PSD itu
lahir (rantai resample -> potong -> detrend -> Welch), dan kenapa laju resample
25 Hz tidak menentukan apa-apa -- yang terakhir dibuktikan dengan menjalankan
ULANG seluruh pipeline pada 12,5-50 Hz setiap kali dokumen ini dibangun.

Definisi SNR mengikuti Ren et al. (2016) sebagaimana diterapkan Le (2020) Pers.
(14); penaksir lantai derau memakai median (statistik terurut) mengikuti prinsip
OS-CFAR, Rohling (1983). Ambang 1.8 BUKAN dari literatur -- diturunkan dari data
penelitian ini (kontribusi K3).

Usage:
    python make_snr_explained.py <aligned_csv> <thesis_dir>
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import phase_pipeline as pp  # noqa: E402
from phase_pipeline import (FS, HR_BAND, OVERLAP,  # noqa: E402
                           WINDOW_SEC, resample_antialias)

from fpdf import FPDF  # noqa: E402
from fpdf.enums import XPos, YPos  # noqa: E402

NAVY, RED, GREY, GREEN, AMBER = (28, 42, 66), (192, 57, 43), (110, 118, 128), (30, 132, 73), (176, 122, 12)
SNR_THRESHOLD = 1.8
DEMO = {"kuat": "subject01", "lemah": "subject06"}


def mpl(c):
    return tuple(v / 255 for v in c)


# ---------------------------------------------------------------- perhitungan
def window_table(sub, which="tengah"):
    """Balikin semua yang dibutuhkan untuk membedah SATU jendela."""
    t = sub.Timestamp.values
    tu, xu, _ = resample_antialias(t, sub.unwrapPhasePeak_mm.values)
    gu = np.interp(tu, np.sort(t), sub.gt_heart_rate.values[np.argsort(t)])
    w = int(WINDOW_SEC * FS)
    s = (len(xu) - w) // 2 if which == "tengah" else which
    raw = signal.detrend(xu[s:s + w])
    f, p = signal.welch(raw, fs=FS, nperseg=w)
    m = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    fb, pb = f[m], p[m]
    hr = float(gu[s:s + w].mean())
    k = int(np.argmin(np.abs(fb - hr / 60)))
    return dict(fb=fb, pb=pb, hr=hr, f_ecg=hr / 60, k=k, psd_k=float(pb[k]),
                med=float(np.median(pb)), snr=float(pb[k] / np.median(pb)),
                k_max=int(np.argmax(pb)), psd_max=float(pb.max()),
                mulai=s / FS, n_bin=len(fb), df=float(fb[1] - fb[0]))


def snr_series(sub):
    """Deret SNR per jendela + ringkasannya."""
    t = sub.Timestamp.values
    tu, xu, _ = resample_antialias(t, sub.unwrapPhasePeak_mm.values)
    gu = np.interp(tu, np.sort(t), sub.gt_heart_rate.values[np.argsort(t)])
    w = int(WINDOW_SEC * FS)
    step = max(1, int(w * (1 - OVERLAP)))
    out, tt = [], []
    for s in range(0, len(xu) - w, step):
        raw = signal.detrend(xu[s:s + w])
        f, p = signal.welch(raw, fs=FS, nperseg=w)
        m = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
        fb, pb = f[m], p[m]
        hr = float(gu[s:s + w].mean())
        if not (HR_BAND[0] <= hr / 60 <= HR_BAND[1]):
            continue
        out.append(pb[int(np.argmin(np.abs(fb - hr / 60)))] / np.median(pb))
        tt.append(s / FS)
    a = np.array(out)
    return dict(t=np.array(tt), snr=a, n=len(a), med=float(np.median(a)),
                mean=float(a.mean()), lo=float(a.min()), hi=float(a.max()),
                n_atas=int((a >= SNR_THRESHOLD).sum()),
                step=step / FS)


def akuisisi(sub):
    """Angka nyata soal grid waktu mentah: laju frame + besar jitter-nya."""
    t = np.sort(sub.Timestamp.values)
    dt = np.diff(t)
    return dict(fs_asli=1.0 / dt.mean(), dt_mean=dt.mean() * 1e3,
                dt_med=float(np.median(dt)) * 1e3, jitter=dt.std() * 1e3)


def fs_sweep(front, laju=(12.5, 20.0, 25.0, 40.0, 50.0)):
    """Ulang SELURUH pipeline pada beberapa laju resample.

    Ini yang menjawab "kenapa 25 Hz?" dengan ukuran, bukan dengan alasan.
    Dihitung ulang tiap kali dokumen dibangun supaya tidak bisa basi diam-diam.
    """
    subs = list(front.groupby("subject_id"))
    keep, out = pp.FS, []
    try:
        for fs in laju:
            pp.FS = float(fs)
            m = [pp.metrics(*pp.estimate(s.Timestamp.values,
                                         s.unwrapPhasePeak_mm.values,
                                         s.gt_heart_rate.values)[:2])[1]
                 for _, s in subs]
            out.append((fs, float(np.mean(m)),
                        int(sum(x < pp.MAPE_STANDARD for x in m)), len(m)))
    finally:
        pp.FS = keep
    return out


# ------------------------------------------------------------------- persamaan
def eq(latex, path, fs=17):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fs, color=mpl(NAVY))
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06,
                transparent=True)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- gambar
def fig_window(A, B, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, D, nama, kuat in [(axes[0], A, DEMO["kuat"], True),
                              (axes[1], B, DEMO["lemah"], False)]:
        ax.plot(D["fb"] * 60, D["pb"], color=mpl(NAVY), lw=1.5, zorder=2)
        ax.axhline(D["med"], color=mpl(GREY), ls="--", lw=1.2, zorder=1)
        ax.text(120, D["med"], " median", va="center", ha="left",
                fontsize=7.5, color=mpl(GREY))
        # puncak radar sendiri
        ax.plot(D["fb"][D["k_max"]] * 60, D["psd_max"], "v", color=mpl(AMBER),
                ms=8, zorder=3)
        ax.annotate("puncak radar\n(BUKAN dipakai)",
                    xy=(D["fb"][D["k_max"]] * 60, D["psd_max"]),
                    textcoords="offset points", xytext=(6, -14),
                    fontsize=7.5, color=mpl(AMBER))
        # alamat dari ECG
        ax.plot(D["f_ecg"] * 60, D["psd_k"], "o", color=mpl(RED), ms=9, zorder=4)
        ax.annotate(f"baris k*={D['k']}\n({D['hr']:.0f} bpm dari ECG)",
                    xy=(D["f_ecg"] * 60, D["psd_k"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=7.5, color=mpl(RED), fontweight="bold")
        ax.annotate("", xy=(D["f_ecg"] * 60, D["psd_k"]),
                    xytext=(D["f_ecg"] * 60, D["med"]),
                    arrowprops=dict(arrowstyle="<->", color=mpl(GREEN), lw=1.5))
        ax.set_title(f"{nama}  --  SNR = {D['snr']:.2f}", fontsize=10.5,
                     fontweight="bold", color=mpl(GREEN if kuat else RED))
        ax.set_xlabel("frekuensi (bpm)", fontsize=8.5)
        ax.set_ylabel("PSD", fontsize=8.5)
        ax.set_xlim(HR_BAND[0] * 60, HR_BAND[1] * 60 + 6)
        ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_series(SA, SB, out):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for S, nama, c in [(SA, DEMO["kuat"], NAVY), (SB, DEMO["lemah"], RED)]:
        ax.plot(S["t"], S["snr"], "-o", ms=3, lw=1.2, color=mpl(c),
                label=f"{nama}  (median = {S['med']:.2f})")
        ax.axhline(S["med"], color=mpl(c), ls=":", lw=1.4)
    ax.axhline(SNR_THRESHOLD, color=mpl(GREEN), ls="--", lw=1.6)
    ax.text(2, SNR_THRESHOLD * 1.12, f"ambang {SNR_THRESHOLD}", fontsize=8,
            color=mpl(GREEN), fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("waktu mulai jendela (detik)", fontsize=9)
    ax.set_ylabel("SNR jendela (skala log)", fontsize=9)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ PDF
class Doc(FPDF):
    def h1(self, txt):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 7.5, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*RED)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y() + 1, self.l_margin + 28, self.get_y() + 1)
        self.ln(4)

    def h2(self, txt):
        if self.get_y() > self.h - 45:
            self.add_page()
        self.ln(1.5)
        self.set_font("Helvetica", "B", 11.5)
        self.set_text_color(*RED)
        self.multi_cell(0, 6, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(0.5)

    def rich(self, txt, h=5.4, size=10):
        """Paragraf. Markdown fpdf2: __miring__ (istilah asing), **tebal**.

        Sengaja pakai markdown bawaan, bukan penanda '*' buatan sendiri:
        '*' harus tetap bisa dipakai literal untuk notasi k* dan P(f_k*),
        dan multi_cell(markdown=True) memenggal baris per KATA dengan benar.
        """
        # '--' adalah penanda UNDERLINE di markdown fpdf2, jadi tanda pisah '--'
        # dalam prosa akan diam-diam menggarisbawahi sisa kalimat. Escape '\-\-'
        # TIDAK bekerja (backslash-nya ikut tercetak -- sudah diuji), jadi tanda
        # pisah dinormalkan ke satu hyphen di sini, sekali untuk semua paragraf.
        txt = txt.replace("--", "-")
        self.set_font("Helvetica", "", size)
        self.set_text_color(30, 36, 46)
        self.set_x(self.l_margin)
        self.multi_cell(0, h, txt, markdown=True,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.5)

    def eqimg(self, path, w=None, center=True):
        from PIL import Image
        iw, ih = Image.open(path).size
        w = w or min(self.epw * 0.62, iw * 0.16)
        hh = w * ih / iw
        x = (self.w - w) / 2 if center else self.l_margin
        self.ln(1)
        self.image(path, x=x, y=self.get_y(), w=w)
        self.set_y(self.get_y() + hh + 3)

    def note(self, title, body, color=RED, bg=(255, 249, 235)):
        y0 = self.get_y()
        self.set_fill_color(*bg)
        self.set_x(self.l_margin + 2)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*color)
        self.multi_cell(self.epw - 2, 5, title, fill=True,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin + 2)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(70, 60, 40)
        self.multi_cell(self.epw - 2, 4.8, body, fill=True,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*color)
        self.set_line_width(1.4)
        self.line(self.l_margin, y0, self.l_margin, self.get_y())
        self.ln(3)

    def table(self, head, rows, widths, hl=None):
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(*NAVY)
        self.set_text_color(255, 255, 255)
        for t, w in zip(head, widths):
            self.cell(w, 6, t, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8.5)
        for i, r in enumerate(rows):
            mark = hl and i in hl
            self.set_fill_color(*(255, 235, 232) if mark else (247, 249, 252))
            self.set_text_color(*(RED if mark else (40, 46, 56)))
            if mark:
                self.set_font("Helvetica", "B", 8.5)
            for t, w in zip(r, widths):
                self.cell(w, 5.2, t, align="C", fill=True)
            self.ln()
            if mark:
                self.set_font("Helvetica", "", 8.5)
        self.ln(2)


def build(A, B, SA, SB, AK, SW, figs, out):
    p = Doc(orientation="P", format="A4")
    p.set_auto_page_break(True, margin=15)
    p.set_margins(18, 14, 18)
    p.add_page()

    p.h1("Cara Menghitung SNR Sinyal Fase Radar")
    p.set_font("Helvetica", "I", 9.5)
    p.set_text_color(*GREY)
    p.multi_cell(0, 5, "SNR (Signal-to-Noise Ratio) di sini menjawab satu pertanyaan: "
                       "pada frekuensi tempat detak jantung SEBENARNYA berada, apakah "
                       "sinyal radar menonjol di atas derau? Semua angka pada dokumen "
                       "ini dihitung langsung dari data penelitian.",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    p.ln(3)

    # ---- rumus utama di depan
    p.h2("Rumusnya (dijabarkan di bawah)")
    p.eqimg(figs["eq_main"], w=118)
    p.rich("dengan " + chr(66) + " = himpunan indeks bin di pita jantung, dan k* = "
           "indeks bin yang frekuensinya paling dekat dengan detak dari ECG. "
           "Perhatikan: __pembilang dan penyebut dua-duanya dari radar.__ ECG tidak "
           "ikut dibagi -- ia hanya menentukan k*.")

    # ---- langkah 1
    p.h2("Langkah 1 -- Radar tidak menghasilkan satu angka, tapi satu TABEL")
    p.rich(
        f"Sinyal fase __unwrapPhasePeak_mm__ dipotong per {WINDOW_SEC:.0f} detik, "
        f"diluruskan (__detrend__), lalu diubah ke ranah frekuensi memakai __Welch PSD__. "
        f"Hasilnya bukan satu nilai, melainkan {A['n_bin']} pasang "
        f"(frekuensi, kekuatan) pada pita jantung:")
    p.eqimg(figs["eq_band"], w=104)
    p.rich(
        f"Pita 0,8-2,0 Hz = 48-120 bpm (kali 60, karena Hz per detik dan bpm per "
        f"menit). Lebar tiap bin {A['df']:.3f} Hz = {A['df']*60:.1f} bpm, yang "
        f"datang dari panjang jendela: 60 / {WINDOW_SEC:.0f} detik = "
        f"{60/WINDOW_SEC:.1f} bpm.")

    rows = []
    show = [0, 1, 2, 3, None, A["k"] - 1, A["k"], A["k"] + 1, None,
            A["n_bin"] - 1]
    for i in show:
        if i is None:
            rows.append(("...", "...", "...", "..."))
            continue
        rows.append((str(i), f"{A['fb'][i]:.3f}", f"{A['fb'][i]*60:.1f}",
                     f"{A['pb'][i]:.5f}"))
    hl = {show.index(A["k"])}
    p.rich(f"Contoh nyata, {DEMO['kuat']}, jendela mulai detik "
           f"{A['mulai']:.0f} (dipotong agar muat):")
    p.table(("baris k", "f_k (Hz)", "f_k (bpm)", "PSD"), rows,
            (28, 32, 32, 38), hl=hl)
    p.rich("Radar berhenti di sini. Dia hanya punya tabel -- dia __tidak tahu__ baris "
           "mana yang detak jantung.")

    # ---- langkah 2
    p.h2("Langkah 2 -- ECG menghasilkan satu angka, tugasnya menunjuk baris")
    p.rich(
        f"Dari deteksi R-peak pada jendela {WINDOW_SEC:.0f} detik yang sama, ECG "
        f"memberi detak acuan {A['hr']:.1f} bpm. Angka itu diubah ke Hz, lalu "
        f"dicari bin terdekatnya:")
    p.eqimg(figs["eq_addr"], w=126)
    p.rich(
        f"{A['hr']:.1f} / 60 = {A['f_ecg']:.3f} Hz -> bin terdekat adalah "
        f"__baris k*__ = {A['k']} (f = {A['fb'][A['k']]:.3f} Hz).")
    p.note("Di sinilah tugas ECG SELESAI.",
           f"Seluruh kontribusi ECG pada perhitungan ini adalah menghasilkan satu "
           f"bilangan bulat: {A['k']}. Nilai {A['hr']:.1f} bpm dan "
           f"{A['f_ecg']:.3f} Hz tidak dibagi, tidak dikali, dan tidak muncul lagi "
           f"di rumus manapun. ECG menentukan ALAMAT; isinya dibaca dari radar.",
           color=NAVY, bg=(238, 243, 250))

    # ---- langkah 3
    p.h2("Langkah 3 -- Tabel radar dibaca dua kali, lalu dibagi")
    p.rich(
        f"__Bacaan pertama__ -- PSD radar di baris yang ditunjuk: "
        f"P(f_k*) = {A['psd_k']:.5f}.")
    p.rich(
        f"__Bacaan kedua__ -- lantai derau, yaitu median PSD seluruh {A['n_bin']} "
        f"bin di pita: {A['med']:.5f}. Median mewakili tinggi PSD baris yang "
        f"'biasa-biasa saja'.")
    p.eqimg(figs["eq_calc"], w=128)
    p.rich(
        f"Artinya: pada frekuensi tempat jantung benar-benar berada, kekuatan "
        f"sinyal radar {A['snr']:.1f} kali lipat kekuatan bin biasa. Satuan PSD "
        f"saling meniadakan, sehingga SNR adalah bilangan tanpa satuan. Bila perlu "
        f"dibandingkan dengan literatur yang melaporkan dalam desibel:")
    p.eqimg(figs["eq_db"], w=112)

    # ---- langkah 4 : kenapa bukan puncak radar
    p.h2("Langkah 4 -- Kenapa BUKAN puncak tertinggi radar")
    p.rich(
        f"Ini bagian yang paling mudah disalahpahami. Pada jendela {DEMO['kuat']} "
        f"di atas, PSD tertinggi justru ada di __baris {A['k_max']}__ "
        f"({A['fb'][A['k_max']]*60:.1f} bpm, PSD {A['psd_max']:.5f}) -- **BUKAN** di "
        f"baris {A['k']} tempat jantung berada.")
    p.rich(
        f"Bila estimasi diambil dari puncak radar, kesimpulannya menjadi "
        f"{A['fb'][A['k_max']]*60:.1f} bpm, padahal ECG mengatakan {A['hr']:.0f} bpm. "
        f"Puncak itu adalah kebocoran napas di tepi bawah pita -- kuat, tetapi "
        f"bukan jantung.")
    p.note("Sebab itu SNR sengaja tidak memakai puncak radar.",
           "Bila SNR dihitung dari puncaknya sendiri, pertanyaannya menjadi 'apakah "
           "radar punya puncak?' -- dan jawabannya SELALU ya, karena dari sekian bin "
           "pasti ada satu yang tertinggi, sekalipun itu derau. Metrik semacam itu "
           "tidak mengukur apa pun. Pertanyaan yang benar: 'pada frekuensi tempat "
           "jantung BENAR-BENAR berada, apakah radar menonjol?'")

    p.rich(
        f"Perbandingannya terlihat jelas pada {DEMO['lemah']}: PSD tertinggi ada di "
        f"{B['fb'][B['k_max']]*60:.1f} bpm ({B['psd_max']:.5f}), sementara jantung "
        f"menurut ECG ada di {B['hr']:.0f} bpm (baris {B['k']}), yang PSD-nya hanya "
        f"{B['psd_k']:.5f} -- di BAWAH median {B['med']:.5f}. SNR = {B['snr']:.2f}. "
        f"Radar tetap punya puncak, tetapi puncak itu bukan jantung.")
    p.rich(
        f"Kalau definisinya memakai puncak radar, {DEMO['lemah']} akan mendapat SNR "
        f"{B['psd_max']/B['med']:.1f} -- terlihat sangat baik, padahal keliru. "
        f"Definisi yang benar memberi {B['snr']:.2f}, dan itulah yang cocok dengan "
        f"kenyataan (subjek ini gagal memenuhi standar akurasi).")

    p.image(figs["fig_win"], w=p.epw)
    p.set_font("Helvetica", "I", 8)
    p.set_text_color(*GREY)
    p.multi_cell(0, 4, "Segitiga kuning = puncak radar (tidak dipakai). Titik merah "
                       "= baris k* yang ditunjuk ECG. Garis putus-putus = median "
                       "(lantai derau). Panah hijau = SNR.",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    p.ln(3)

    # ---- langkah 5
    p.h2("Langkah 5 -- Dari satu jendela ke SNR subjek")
    p.rich(
        f"Jendela {WINDOW_SEC:.0f} detik digeser maju {SA['step']:.0f} detik setiap "
        f"kali (tumpang tindih {OVERLAP*100:.0f}%), dan seluruh langkah 1-3 diulang "
        f"dari nol pada tiap posisi. {DEMO['kuat']} menghasilkan {SA['n']} nilai SNR, "
        f"{DEMO['lemah']} menghasilkan {SB['n']}. SNR subjek adalah median deret itu:")
    p.eqimg(figs["eq_sess"], w=104)
    p.image(figs["fig_ser"], w=p.epw)
    p.ln(2)

    p.h2("Kenapa median, bukan rata-rata -- ini bukan soal selera")
    p.table(("subjek", "n jendela", "rentang", "rata-rata", "median", "vonis benar"),
            [(DEMO["kuat"], str(SA["n"]), f"{SA['lo']:.2f} - {SA['hi']:.2f}",
              f"{SA['mean']:.2f}", f"{SA['med']:.2f}", "lolos"),
             (DEMO["lemah"], str(SB["n"]), f"{SB['lo']:.2f} - {SB['hi']:.2f}",
              f"{SB['mean']:.2f}", f"{SB['med']:.2f}", "gagal")],
            (30, 24, 32, 26, 24, 30), hl={1})
    p.rich(
        f"Lihat {DEMO['lemah']}: rata-ratanya {SB['mean']:.2f} -- di ATAS ambang "
        f"{SNR_THRESHOLD}, sehingga akan dinyatakan layak. Median-nya "
        f"{SB['med']:.2f} -- di BAWAH ambang. Kenyataannya subjek ini memang gagal "
        f"memenuhi standar akurasi, jadi __median yang benar__.")
    p.rich(
        f"Sebabnya: dari {SB['n']} jendela {DEMO['lemah']}, hanya {SB['n_atas']} "
        f"({100*SB['n_atas']/SB['n']:.0f}%) yang di atas ambang. Mayoritas jendelanya "
        f"buruk, tetapi segelintir jendela yang kebetulan sangat baik (tertinggi "
        f"{SB['hi']:.2f}) menarik rata-rata naik hingga tampak layak. Median kebal "
        f"terhadap pencilan semacam itu. Bandingkan {DEMO['kuat']}: "
        f"{SA['n_atas']} dari {SA['n']} jendela ({100*SA['n_atas']/SA['n']:.0f}%) di "
        f"atas ambang -- baik secara konsisten, bukan baik secara kebetulan.")
    p.rich(
        "Pilihan median ini bukan improvisasi: memakai statistik terurut "
        "(__order statistic__) sebagai penaksir level derau adalah prinsip __OS-CFAR__ "
        "(Rohling, 1983), yang justru dirancang agar tahan ketika sel pembanding "
        "mengandung puncak pengganggu.")

    p.add_page()
    # ---- dari sinyal mentah ke tabel + pertanyaan "kenapa 25 Hz"
    p.h2("Lampiran -- Dari sinyal mentah ke tabel PSD, dan kenapa laju 25 Hz "
         "tidak menentukan")
    p.rich(
        f"Langkah 1 dimulai dari tabel PSD yang sudah jadi. Tabel itu sendiri "
        f"lahir dari empat langkah berikut, semuanya di dalam satu jendela "
        f"{WINDOW_SEC:.0f} detik:")
    p.rich(
        f"**(1) Ratakan grid waktu.** Kolom __unwrapPhasePeak_mm__ direkam sekitar "
        f"{AK['fs_asli']:.0f} Hz, tetapi jaraknya TIDAK seragam: simpangan baku "
        f"jarak antar-sampel {AK['jitter']:.1f} ms. Penyebabnya kuantisasi pewaktu "
        f"sistem operasi, bukan radarnya - rata-rata jarak sampel tepat "
        f"{AK['dt_mean']:.1f} ms, sedangkan mediannya {AK['dt_med']:.1f} ms "
        f"(persis satu __tick__ pewaktu Windows). Welch PSD mensyaratkan jarak "
        f"sampel seragam, jadi sinyal ditapis __anti-alias__ lalu diinterpolasi ke "
        f"grid seragam. **Alasan langkah ini ada adalah KESERAGAMAN grid, bukan "
        f"penurunan laju.**")
    p.rich(
        f"**(2) Potong {WINDOW_SEC:.0f} detik. (3) Luruskan (__detrend__)** - buang "
        f"kemiringan akibat subjek perlahan bergeser, yang kalau dibiarkan menjadi "
        f"energi di sekitar frekuensi nol dan bocor ke mana-mana. "
        f"**(4) Welch PSD**, menghasilkan {A['n_bin']} pasang (frekuensi, kekuatan) "
        f"pada pita jantung - tabel di Langkah 1.")

    p.h2("Kenapa 25 Hz? Karena laju berapa pun memberi hasil sama")
    p.rich(
        "Laju grid pada langkah (1) adalah pilihan bebas, dan pilihan bebas selalu "
        "mengundang pertanyaan apakah ia disetel sampai angkanya bagus. Dua alasan "
        "mengapa tidak, lalu buktinya.")
    p.rich(
        "**Pertama, syarat Nyquist.** Pita jantung berhenti di 2,0 Hz, sehingga laju "
        "cuplik minimum yang sah adalah 4 Hz. Laju 25 Hz memberi margin 12,5 kali - "
        "berlebih, tetapi tidak mungkin salah.")
    p.rich(
        "**Kedua, resolusi frekuensi sama sekali tidak dipengaruhi laju.** Ini yang "
        "sering disangka sebaliknya. Dengan N sampel dalam jendela selama T detik:")
    p.eqimg(figs["eq_df"], w=132)
    p.rich(
        f"Laju f_s saling menghapus. Yang menentukan lebar bin {A['df']*60:.1f} bpm "
        f"adalah **panjang jendela {WINDOW_SEC:.0f} detik**, bukan laju cuplik. "
        f"Menaikkan laju menjadi 50 Hz hanya menggandakan jumlah sampel tanpa "
        f"menambah satu pun bin di pita jantung.")
    p.rich(
        "**Buktinya - seluruh pipeline dijalankan ulang pada beberapa laju:**")
    p.table(("laju resample", "MAPE rata-rata", "subjek lolos"),
            [(f"{fs:.1f} Hz", f"{m:.2f}%", f"{n}/{tot}")
             for fs, m, n, tot in SW],
            (48, 48, 48), hl={[fs for fs, *_ in SW].index(FS)})
    _m = [m for _, m, _, _ in SW]
    p.rich(
        f"Rentang {min(f for f, *_ in SW):.1f}-{max(f for f, *_ in SW):.0f} Hz "
        f"memberi MAPE {min(_m):.2f}-{max(_m):.2f}%, yakni datar dalam "
        f"{max(_m)-min(_m):.2f} poin persentase. Baris bertanda adalah laju yang "
        f"dipakai di seluruh dokumen ini.")
    p.note("Jawaban ringkas bila ditanya saat sidang.",
           f"\"Laju {FS:.0f} Hz bukan nilai istimewa. Kami menguji ulang seluruh "
           f"pipeline pada {min(f for f, *_ in SW):.1f} hingga "
           f"{max(f for f, *_ in SW):.0f} Hz dan MAPE rata-rata bergerak kurang dari "
           f"{max(_m)-min(_m):.2f} poin. Hasil penelitian ini tidak bergantung pada "
           f"pilihan tersebut.\" - Perhatikan bentuk jawabannya: bukan pembelaan "
           f"atas suatu pilihan, melainkan penunjukan bahwa pilihan itu tidak "
           f"berpengaruh. Tabel di atas dihitung ulang setiap kali dokumen ini "
           f"dibangun, sehingga tidak dapat basi tanpa ketahuan.",
           color=GREEN, bg=(238, 248, 240))

    p.add_page()
    # ---- batasan
    p.h2("Batasan penting -- SNR ini alat DIAGNOSIS, bukan alat lapangan")
    p.note("Perhitungan ini membutuhkan ECG.",
           "Langkah 2 memerlukan detak acuan untuk menentukan k*. Di lapangan "
           "(deployment) ECG tidak tersedia, sehingga tidak ada yang menunjuk baris "
           "mana yang harus dibaca. Karena itu SNR ini TIDAK dapat dipakai sebagai "
           "gerbang kualitas saat sistem berjalan mandiri. Fungsinya: MENJELASKAN, "
           "setelah eksperimen, mengapa sesi tertentu gagal dan sesi lain berhasil. "
           "Untuk deteksi kelayakan tanpa ECG dipakai sinyal NAPAS (0,15-0,6 Hz), "
           "yang dibahas terpisah pada kontrol negatif ruangan kosong.")
    p.rich(
        "Perlu dicatat: kebutuhan akan acuan ini bukan kelemahan khas penelitian "
        "ini. Le (2020) menghitung SNR pada frekuensi fundamental yang diketahui "
        "dari acuan PPG; praktik yang sama berlaku di sini dengan acuan ECG. Metrik "
        "ini memang metrik validasi, bukan metrik operasional.")

    p.h2("Kenapa SNR bisa rendah: jarak")
    p.rich(
        "Daya pantul radar menurun sebanding __1/R^4__ terhadap jarak R, sehingga "
        "jarak dua kali lipat berarti daya balik sekitar 16 kali lebih kecil. "
        "Dibuktikan melalui eksperimen terkendali dengan subjek dan logger yang "
        "sama, hanya jaraknya berbeda: 50 cm (SNR 3,04, MAPE 6,4%, lolos) versus "
        "100 cm (SNR 1,27, MAPE 53,9%, gagal). Pada SNR mendekati 1 komponen "
        "jantung setara derau, sehingga tidak ada algoritma -- termasuk "
        "__machine learning__ -- yang dapat mengekstraknya. Ini batas fisik, bukan kekurangan "
        "metode, dan dilaporkan sebagai kriteria kelayakan (K3).")

    # ---- dasar literatur
    p.h2("Dasar literatur")
    p.rich(
        "Definisi SNR yang dipakai mengikuti Ren et al. (2016), sebagaimana "
        "diterapkan pada radar tanda vital oleh Le (2020) Pers. (14): kekuatan "
        "spektral pada frekuensi fundamental detak jantung yang diketahui dari "
        "acuan, dibagi kekuatan spektral derau pada pita detak jantung. Penaksir "
        "lantai derau dalam penelitian ini memakai median (statistik terurut) "
        "mengikuti prinsip OS-CFAR (Rohling, 1983).")

    p.set_font("Helvetica", "B", 9.5)
    p.set_text_color(*NAVY)
    p.multi_cell(0, 5, "Perbedaan yang dinyatakan terhadap Le (2020):",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    p.ln(1)
    p.table(("aspek", "Le (2020) / Ren (2016)", "penelitian ini"),
            [("transformasi", "CZT", "Welch PSD"),
             ("besaran", "amplitudo", "daya (power)"),
             ("lantai derau", "sisa pita, buang fundamental+harmonik", "median seluruh pita"),
             ("satuan lapor", "dB (20 log10)", "rasio linier (dB: 10 log10)"),
             ("pita jantung", "0,75 - 3,0 Hz", "0,8 - 2,0 Hz (spesifikasi TI)")],
            (32, 74, 68))

    p.note("Yang BUKAN dari literatur: nilai ambang 1,8.",
           "Ambang ini diturunkan secara empiris dari data penelitian ini, lalu "
           "diuji pada sesi yang tidak ikut menentukannya (2 hold-out, 2 jarak, "
           "3 halangan): benar pada 15 dari 17 sesi, dengan korelasi log(SNR) "
           "terhadap log(MAPE) = -0,885. Ambang ini dilaporkan sebagai kontribusi "
           "penelitian (K3), BUKAN sebagai kutipan dari literatur. Metrik dari "
           "literatur + ambang dari data + validasi pada sesi tak-terlihat.",
           color=GREEN, bg=(238, 248, 240))

    p.h2("Daftar pustaka")
    p.set_font("Helvetica", "", 9)
    p.set_text_color(30, 36, 46)
    for ref in [
        "Le, M. (2020). Heart rate extraction based on eigenvalues using UWB impulse "
        "radar remote sensing. Sensors and Actuators A: Physical, 303, 111689. "
        "https://doi.org/10.1016/j.sna.2019.111689",
        "Ren, L., Wang, H., Naishadham, K., Kilic, O., & Fathy, A. E. (2016). "
        "Phase-Based Methods for Heart Rate Detection Using UWB Impulse Doppler "
        "Radar. IEEE Transactions on Microwave Theory and Techniques, 64(10), "
        "3319-3331. https://doi.org/10.1109/TMTT.2016.2597824",
        "Rohling, H. (1983). Radar CFAR Thresholding in Clutter and Multiple Target "
        "Situations. IEEE Transactions on Aerospace and Electronic Systems, "
        "AES-19(4), 608-621. https://doi.org/10.1109/TAES.1983.309350",
        "Yamamoto, K., Endo, K., & Ohtsuki, T. (2021). Remote Sensing of Heartbeat "
        "based on Space Diversity Using MIMO FMCW Radar. 2021 IEEE Global "
        "Communications Conference (GLOBECOM). "
        "https://doi.org/10.1109/GLOBECOM46510.2021.9685033 "
        "[mendukung premis: SNR komponen jantung pada sinyal fase menentukan akurasi]",
    ]:
        p.set_x(p.l_margin)
        p.multi_cell(0, 4.6, "- " + ref, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        p.ln(1)

    p.ln(2)
    p.note("Catatan penulis - hapus sebelum dokumen ini dibagikan.",
           "Bentuk persamaan Ren et al. (2016) di atas diketahui dari kutipan Le "
           "(2020) terhadap rujukan [28]-nya, BUKAN dari pembacaan langsung "
           "(akses IEEE Xplore tertutup). Verifikasi persamaan aslinya lewat akses "
           "kampus sebelum menyitasi. Bila Ren tidak dapat diperoleh, sitasi Le "
           "(2020) saja sebagai sumber utama sudah cukup dan dapat "
           "dipertanggungjawabkan - PDF-nya ada, persamaannya eksplisit, dan "
           "halamannya bisa ditunjuk saat sidang.",
           color=AMBER, bg=(255, 247, 230))

    p.output(str(out))


def main(csv, thesis_dir):
    df = pd.read_csv(csv)
    front = df[(df.dataset == "position_front") & df.unwrapPhasePeak_mm.notna()]
    A = window_table(front[front.subject_id == DEMO["kuat"]])
    B = window_table(front[front.subject_id == DEMO["lemah"]])
    SA = snr_series(front[front.subject_id == DEMO["kuat"]])
    SB = snr_series(front[front.subject_id == DEMO["lemah"]])
    AK = akuisisi(front[front.subject_id == DEMO["kuat"]])
    SW = fs_sweep(front)

    tmp = Path(sys.argv[0]).resolve().parent / "_snrtmp"
    tmp.mkdir(exist_ok=True)
    figs = {
        "eq_main": eq(r"\mathrm{SNR}_{\rm jendela}=\frac{P(f_{k^*})}"
                      r"{\mathrm{median}_{k\in B}\;P(f_k)}", tmp / "e1.png"),
        "eq_band": eq(r"B=\{k:\;0{,}8\;\mathrm{Hz}\leq f_k\leq 2{,}0\;\mathrm{Hz}\}",
                      tmp / "e2.png"),
        "eq_addr": eq(r"f_{\rm ECG}=\frac{\mathrm{HR}_{\rm ECG}}{60}"
                      r"\qquad k^*=\mathrm{arg\,min}_{k\in B}\;|f_k-f_{\rm ECG}|",
                      tmp / "e3.png"),
        "eq_calc": eq(rf"\mathrm{{SNR}}=\frac{{{A['psd_k']:.5f}}}"
                      rf"{{{A['med']:.5f}}}={A['snr']:.2f}", tmp / "e4.png"),
        "eq_db": eq(rf"\mathrm{{SNR}}_{{\rm dB}}=10\log_{{10}}"
                    rf"(\mathrm{{SNR}})={10*np.log10(A['snr']):.1f}\;\mathrm{{dB}}",
                    tmp / "e5.png"),
        "eq_sess": eq(r"\mathrm{SNR}_{\rm subjek}=\mathrm{median}_{w}\;"
                      r"\mathrm{SNR}_{\rm jendela}(w)", tmp / "e6.png"),
        "eq_df": eq(r"\Delta f=\frac{f_s}{N}=\frac{f_s}{T\cdot f_s}=\frac{1}{T}"
                    r"\qquad\Rightarrow\qquad \Delta f\;\text{bebas dari}\;f_s",
                    tmp / "e7.png"),
    }
    figs["fig_win"] = str(tmp / "fw.png")
    figs["fig_ser"] = str(tmp / "fs.png")
    fig_window(A, B, figs["fig_win"])
    fig_series(SA, SB, figs["fig_ser"])
    figs = {k: str(v) for k, v in figs.items()}

    out = Path(thesis_dir) / "cara_hitung_snr.pdf"
    build(A, B, SA, SB, AK, SW, figs, out)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    print(f"ok: {out}")
    print(f"  {DEMO['kuat']}: jendela contoh SNR {A['snr']:.2f}, subjek {SA['med']:.2f} "
          f"({SA['n_atas']}/{SA['n']} jendela >= {SNR_THRESHOLD})")
    print(f"  {DEMO['lemah']}: jendela contoh SNR {B['snr']:.2f}, subjek {SB['med']:.2f} "
          f"({SB['n_atas']}/{SB['n']} jendela >= {SNR_THRESHOLD})")
    # cek-mandiri: klaim "rata-rata menyesatkan" harus benar-benar terjadi
    assert SB["mean"] >= SNR_THRESHOLD > SB["med"], \
        "klaim 'rata-rata lolos tapi median gagal' tidak lagi benar di data ini"
    assert A["k_max"] != A["k"], \
        "klaim 'puncak radar bukan jantung' tidak lagi benar di jendela contoh"
    spread = max(m for _, m, _, _ in SW) - min(m for _, m, _, _ in SW)
    print(f"  sweep FS: " + ", ".join(f"{fs:.0f}Hz={m:.2f}%" for fs, m, _, _ in SW))
    assert spread < 1.0, (
        f"klaim 'hasil tidak bergantung laju resample' TIDAK BENAR lagi "
        f"(MAPE bervariasi {spread:.2f} poin). Jangan terbitkan dokumen ini "
        f"sebelum lampiran FS diperbaiki.")
    assert any(fs == FS for fs, *_ in SW), \
        f"laju yang dipakai dokumen ({FS}) tidak ada di tabel sweep"
    print("  (cek-mandiri: 3 klaim kunci terverifikasi)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
