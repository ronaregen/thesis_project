"""
Gabungkan semua deck (yang angkanya SUDAH final) jadi satu PDF presentasi,
lengkap dengan cover + pembatas tiap bagian.

Dikecualikan dengan sengaja:
  - diagnosis_riset.pdf, rekomendasi_riset.pdf  -> masih angka lama (6/10, MAPE
    9.8%) dan menyarankan rekam ulang. Kalau digabung akan KONTRADIKSI dengan
    deck final. Regenerate dulu lewat make_report.py kalau memang mau dimasukkan.

Urutan disusun sebagai satu alur presentasi, bukan tumpukan mentah:
  spine (alur) -> mekanisme -> bukti -> kriteria -> metodologi -> lampiran.

Usage:
    python make_combined.py <thesis_dir>
"""

import subprocess
import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

NAVY, RED, GREY, GREEN, AMBER = (28, 42, 66), (192, 57, 43), (110, 118, 128), (30, 132, 73), (176, 122, 12)
W, H = 338.7, 190.5

# (file, judul bagian, satu-baris deskripsi). Urutan = urutan di PDF akhir.
SECTIONS = [
    ("fase_vs_ecg.pdf", "Bagian 1  -  Deck Utama: Sinyal Fase vs ECG",
     "Pertanyaan tunggal, hasil, bukti detak tertangkap, kontrol negatif, hold-out, kriteria kelayakan."),
    ("cara_hitung_mae_mape.pdf", "Bagian 2  -  Metodologi: Cara Menghitung MAE dan MAPE",
     "Langkah demi langkah, dari ECG dan sinyal fase mentah sampai menjadi angka galat."),
    ("kenapa_subjek_gagal.pdf", "Bagian 3  -  Kriteria Kelayakan: SNR dan Jarak (detail)",
     "Kenapa subjek yang gagal adalah temuan, bukan alasan untuk mengganti subjek."),
    ("bukti_detak_jantung.pdf", "Bagian 4  -  Bukti Detak Tertangkap (detail)",
     "Kontrol negatif (ruangan kosong) dan hold-out, versi lengkap."),
    ("antisipasi_pertanyaan.pdf", "Lampiran A  -  Antisipasi Pertanyaan Penguji",
     "Cadangan tanya-jawab: HRV, metrik kualitas tanpa ECG, dan pembelaan metodologis."),
    ("logger_vs_fase.pdf", "Lampiran B  -  Perbandingan dengan Output BPM Bawaan (CADANGAN)",
     "HANYA dibuka jika penguji spesifik meminta perbandingan dengan output alat. Bukan jalur utama."),
]

TITLE = "Estimasi Detak Jantung dari Sinyal Fase Mentah Radar FMCW TI IWR1443BOOST"
SUBTITLE = "Analisis akar kegagalan pipeline bawaan dan rancangan pipeline pengganti tervalidasi ANSI/CTA-2065"


def page():
    p = FPDF(orientation="L", format=(H, W))
    p.set_auto_page_break(False)
    p.add_page()
    return p


def cover(path, sections):
    p = page()
    p.set_fill_color(*NAVY)
    p.rect(0, 0, W, 58, style="F")
    p.set_xy(20, 17)
    p.set_font("Helvetica", "B", 20)
    p.set_text_color(255, 255, 255)
    p.multi_cell(W - 40, 9, TITLE, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    p.set_x(20)
    p.set_font("Helvetica", "", 11)
    p.set_text_color(200, 208, 220)
    p.multi_cell(W - 40, 5.5, SUBTITLE)

    p.set_draw_color(*RED)
    p.set_line_width(1.2)
    p.line(20, 66, 60, 66)

    p.set_xy(20, 72)
    p.set_font("Helvetica", "B", 12)
    p.set_text_color(*NAVY)
    p.cell(0, 7, "Isi presentasi", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    y = 82
    for i, (_, title, desc) in enumerate(sections, 1):
        p.set_xy(22, y)
        p.set_font("Helvetica", "B", 10.5)
        p.set_text_color(*(GREEN if title.startswith("Bagian") else AMBER))
        p.multi_cell(W - 44, 5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        p.set_x(22)
        p.set_font("Helvetica", "", 9)
        p.set_text_color(80, 88, 98)
        p.multi_cell(W - 44, 4.4, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        y = p.get_y() + 2.5

    p.set_xy(20, H - 14)
    p.set_font("Helvetica", "", 8.5)
    p.set_text_color(*GREY)
    p.cell(0, 5, "Semua angka dihasilkan langsung dari skrip di code/evaluation/ -- "
                 "tidak ada yang diketik tangan. Standar penilaian: ANSI/CTA-2065 (MAPE < 10%).")
    p.output(str(path))


def divider(path, n, title, desc):
    p = page()
    p.set_fill_color(245, 247, 250)
    p.rect(0, 0, W, H, style="F")
    p.set_fill_color(*NAVY)
    p.rect(0, H / 2 - 34, 6, 68, style="F")
    p.set_xy(24, H / 2 - 24)
    p.set_font("Helvetica", "B", 13)
    p.set_text_color(*RED)
    p.cell(0, 8, f"{n:02d}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    p.set_x(24)
    p.set_font("Helvetica", "B", 22)
    p.set_text_color(*NAVY)
    p.multi_cell(W - 60, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    p.set_x(24)
    p.ln(2)
    p.set_font("Helvetica", "", 12)
    p.set_text_color(90, 98, 108)
    p.multi_cell(W - 70, 6, desc)
    p.output(str(path))


def main(thesis_dir):
    thesis = Path(thesis_dir)
    tmp = thesis / "_combine_tmp"
    tmp.mkdir(exist_ok=True)

    present = [(f, t, d) for f, t, d in SECTIONS if (thesis / f).exists()]
    missing = [f for f, _, _ in SECTIONS if not (thesis / f).exists()]
    if missing:
        print("LEWAT (file tidak ada):", ", ".join(missing))

    order = []
    cov = tmp / "00_cover.pdf"
    cover(cov, present)
    order.append(cov)

    for i, (f, title, desc) in enumerate(present, 1):
        dv = tmp / f"{i:02d}_div.pdf"
        divider(dv, i, title, desc)
        order += [dv, thesis / f]

    out = thesis / "PRESENTASI_LENGKAP.pdf"
    subprocess.run(["pdfunite", *map(str, order), str(out)], check=True)

    for x in tmp.iterdir():
        x.unlink()
    tmp.rmdir()

    n = subprocess.run(["pdfinfo", str(out)], capture_output=True, text=True).stdout
    pages = next(l.split()[1] for l in n.splitlines() if l.startswith("Pages"))
    print(f"\nok: {out}  ({pages} halaman, {len(present)} bagian)")
    for i, (f, title, _) in enumerate(present, 1):
        print(f"  {i}. {title.split('  -  ')[0]:<12} <- {f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "thesis")
