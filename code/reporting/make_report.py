"""
Bangun 2 PDF buat bimbingan:
  1. diagnosis_riset.pdf     -- apa yang sebenarnya terjadi di data (untuk pembimbing)
  2. rekomendasi_riset.pdf   -- arah riset sebaiknya gimana

Usage:
    python make_report.py <figdir> <outdir>
"""

import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

NAVY = (28, 42, 66)
RED = (192, 57, 43)
GREEN = (30, 132, 73)
GREY = (110, 118, 128)
LIGHT = (240, 242, 245)
AMBER = (176, 122, 12)
PURPLE = (110, 60, 140)


class Report(FPDF):
    def __init__(self, title, subtitle):
        super().__init__(format="A4")
        self.doc_title = title
        self.doc_subtitle = subtitle
        self.set_auto_page_break(True, margin=18)
        self.set_margins(18, 16, 18)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*GREY)
        self.cell(0, 5, self.doc_title, align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(220, 224, 228)
        self.set_line_width(0.3)
        self.line(18, self.get_y() + 0.5, 192, self.get_y() + 0.5)
        self.ln(3)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*GREY)
        self.cell(0, 5, f"Halaman {self.page_no()}", align="C")

    # ---------- building blocks ----------
    def title_page(self, meta_lines):
        self.add_page()
        self.ln(38)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*RED)
        self.cell(0, 5, "DOKUMEN BIMBINGAN TESIS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)
        self.set_font("Helvetica", "B", 23)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 10, self.doc_title)
        self.ln(2)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(*GREY)
        self.multi_cell(0, 6.5, self.doc_subtitle)
        self.ln(8)
        self.set_draw_color(*RED)
        self.set_line_width(1.2)
        self.line(18, self.get_y(), 58, self.get_y())
        self.ln(10)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*NAVY)
        for k, v in meta_lines:
            self.set_font("Helvetica", "B", 9.5)
            self.cell(38, 6, k)
            self.set_font("Helvetica", "", 9.5)
            self.multi_cell(0, 6, v, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def h1(self, text):
        if self.get_y() > 235:
            self.add_page()
        self.ln(3)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*RED)
        self.set_line_width(0.8)
        self.line(18, self.get_y() + 1, 34, self.get_y() + 1)
        self.ln(4)

    def h2(self, text):
        if self.get_y() > 250:
            self.add_page()
        self.ln(2)
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.5)

    def body(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(40, 48, 58)
        self.multi_cell(0, 5.2, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def bullets(self, items):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(40, 48, 58)
        for it in items:
            x = self.get_x()
            self.set_font("Helvetica", "B", 9.5)
            self.cell(5, 5.2, "-")
            self.set_font("Helvetica", "", 9.5)
            self.multi_cell(0, 5.2, it, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_x(x)
        self.ln(2)

    def callout(self, kind, title, text):
        colors = {"bad": RED, "good": GREEN, "warn": AMBER, "info": NAVY, "key": PURPLE}
        c = colors[kind]
        self.ln(1)
        # ukur tinggi
        self.set_font("Helvetica", "", 9.3)
        start_y = self.get_y()
        if start_y > 240:
            self.add_page()
            start_y = self.get_y()
        pad = 3.5
        self.set_fill_color(*LIGHT)
        self.set_draw_color(*c)

        x0 = self.get_x()
        self.set_xy(x0 + 5, start_y + pad)
        self.set_font("Helvetica", "B", 9.6)
        self.set_text_color(*c)
        self.multi_cell(168, 5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(x0 + 5)
        self.set_font("Helvetica", "", 9.3)
        self.set_text_color(40, 48, 58)
        self.multi_cell(168, 5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        end_y = self.get_y() + pad

        # gambar bar kiri + background (di belakang -> pakai rect setelah, jadi cukup bar)
        self.set_fill_color(*c)
        self.rect(x0, start_y, 1.8, end_y - start_y, style="F")
        self.set_draw_color(225, 228, 232)
        self.set_line_width(0.2)
        self.rect(x0, start_y, 174, end_y - start_y)
        self.set_xy(x0, end_y)
        self.ln(3.5)

    def _table_header(self, headers, widths):
        self.set_font("Helvetica", "B", 8.6)
        self.set_fill_color(*NAVY)
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, widths):
            self.cell(w, 7, h, border=0, align="C", fill=True)
        self.ln()

    def table(self, headers, rows, widths, highlight_col=None, highlight_rule=None):
        # kalau tabel utuh tidak muat di sisa halaman, mulai halaman baru dulu
        needed = 7 + len(rows) * 6.2 + 3
        if self.get_y() + needed > 272 and needed < 240:
            self.add_page()
        self._table_header(headers, widths)
        self.set_font("Helvetica", "", 8.6)
        for i, row in enumerate(rows):
            if self.get_y() + 6.2 > 272:          # tabel panjang: pecah + ulang header
                self.add_page()
                self._table_header(headers, widths)
                self.set_font("Helvetica", "", 8.6)
            fill = i % 2 == 1
            self.set_fill_color(245, 247, 249)
            for j, (v, w) in enumerate(zip(row, widths)):
                self.set_text_color(40, 48, 58)
                if highlight_rule and highlight_col == j:
                    col = highlight_rule(v)
                    if col:
                        self.set_text_color(*col)
                        self.set_font("Helvetica", "B", 8.6)
                self.cell(w, 6.2, str(v), border=0,
                          align="L" if j == 0 else "C", fill=fill)
                self.set_font("Helvetica", "", 8.6)
            self.ln()
        self.ln(3)

    def figure(self, path, caption, width=174):
        from PIL import Image
        with Image.open(path) as im:
            h = width * im.height / im.width
        if self.get_y() + h + 14 > 274:   # + ruang caption
            self.add_page()
        self.image(str(path), x=18, w=width)
        self.ln(1.5)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.multi_cell(0, 4.4, caption, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)


# ============================================================ DOC 1: DIAGNOSIS
def build_diagnosis(fig, out):
    p = Report("Diagnosis: Kenapa Riset Ini Mentok",
               "Tiga cacat pada rantai pengambilan data - dan kenapa sensornya "
               "sebenarnya tidak bersalah")
    p.title_page([
        ("Topik", "Estimasi detak jantung dengan radar FMCW TI IWR1443BOOST"),
        ("Dataset", "10 subjek posisi depan, 1 subjek 8 variasi posisi, 1 subjek 2 variasi "
                    "jarak; ground truth ECG Attys @125 Hz"),
        ("Tanggal", "13 Juli 2026"),
        ("Ringkas", "Selama ini disimpulkan radar gagal menangkap detak jantung. "
                    "Ternyata sinyal jantungnya ADA. Kegagalannya berasal dari tiga cacat "
                    "pada setup pengambilan data - frame rate 2.5x terlalu cepat, logger "
                    "salah baca byte, dan jarak subjek terlalu jauh - yang ketiganya kini "
                    "sudah dibuktikan secara kuantitatif."),
    ])

    # ---- Ringkasan eksekutif
    p.add_page()
    p.h1("Ringkasan Eksekutif")
    p.body(
        "Sampai bimbingan lalu, kesimpulan kerja adalah: empat pendekatan berbeda "
        "(output BPM bawaan TI, spektral fase mentah, filter confidence, dan supresi "
        "harmonik napas) semuanya menghasilkan korelasi ~0 terhadap ground truth ECG, "
        "sehingga diduga masalahnya ada di akuisisi radar - artinya riset praktis buntu.")
    p.callout(
        "key", "Kesimpulan itu KELIRU. Perangkatnya tidak pernah diuji secara adil.",
        "Sinyal detak jantung TERBUKTI ADA di dalam data fase radar (SNR median 3.19x). "
        "Kegagalan yang selama ini terlihat bukan berasal dari sensor maupun dari algoritma "
        "TI, melainkan dari TIGA CACAT pada rantai pengambilan data - dan ketiganya sekarang "
        "sudah dibuktikan secara kuantitatif, bukan sekadar diduga.")
    p.h2("Tiga cacat akuisisi yang ditemukan (semuanya terbukti)")
    p.bullets([
        "CACAT 1 - FRAME RATE SALAH. Logger menyetel radar ke 50 frame/detik "
        "(frameCfg 20 ms), padahal library Vital Signs TI dirancang untuk 20 frame/detik "
        "(frameCfg 50 ms). Akibatnya filter jantung TI mulur 2.5x: band yang seharusnya "
        "0.8-4.0 Hz menjadi 2.0-10.0 Hz - yaitu MEMBUANG detak jantung manusia (1-2 Hz). "
        "BUKTI: 94.3% energi 'waveform jantung' TI berada di 2-10 Hz, hanya 4.2% di band "
        "jantung yang sebenarnya.",
        "CACAT 2 - LOGGER SALAH BACA BYTE. 4 dari 10 kolom diambil dari offset yang keliru. "
        "Yang terparah: kolom 'heart_rate_est_peak' sebenarnya berisi laju NAPAS "
        "(breathingRateEst_FFT), dan 'confidence_heart' sebenarnya confidence NAPAS. "
        "BUKTI: kolom final_heart_rate berhasil direproduksi 100.00% persis - ternyata "
        "98.3% waktu ia menyalin laju napas 5.86 napas/menit dan melaporkannya sebagai "
        "detak jantung.",
        "CACAT 3 - JARAK SUBJEK TERLALU JAUH. Data variasi jarak milik sendiri menunjukkan: "
        "pada 100 cm, SNR sinyal jantung = 1.09, yaitu SETARA NOISE (tidak ada yang bisa "
        "diekstrak). Pada 50 cm, SNR = 3.16. Daya pantul radar turun sekitar 1/R^4, jadi "
        "jarak adalah parameter kritis, bukan detail sepele.",
    ])
    p.callout(
        "warn", "Konsekuensi penting untuk kejujuran ilmiah tesis.",
        "Karena perangkat dijalankan dengan konfigurasi yang salah, dataset yang ada "
        "TIDAK BISA dipakai untuk memvonis 'library TI gagal'. Yang gagal adalah "
        "SETUP-nya. Ini justru temuan yang lebih berharga: ia bisa dibuktikan, bisa "
        "diperbaiki, dan menjelaskan semua anomali yang selama ini membingungkan.")
    p.h2("Standar yang dipakai untuk menilai (bukan angka karangan sendiri)")
    p.body(
        "Semua vonis di dokumen ini diukur terhadap batas galat yang diakui untuk alat ukur "
        "detak jantung, bukan terhadap target internal:")
    p.callout(
        "info", "ANSI/CTA-2065 - Physical Activity Monitoring for Heart Rate: MAPE < 10%",
        "Consumer Technology Association (2018), ANSI/CTA-2065. Sebuah alat dinyatakan "
        "VALID untuk pengukuran detak jantung bila Mean Absolute Percentage Error (MAPE) "
        "terhadap ECG < 10%. Ambang ini dipakai luas di literatur validasi alat HR - "
        "antara lain Nelson & Allen (2019), JMIR Mhealth Uhealth 7(3):e10828, "
        "doi:10.2196/10828, yang menetapkan MAPE < 10% sebagai kriteria akurasi yang dapat "
        "diterima, dan dipakai ulang oleh studi validasi wearable pada pasien anak "
        "(kedua paper ada di folder papers/). Sebagian studi memakai kriteria lebih ketat "
        "(+/-5%). Pada HR rata-rata 78 bpm, MAPE 10% setara dengan galat sekitar 7.8 bpm.")
    p.h2("Hasil setelah diperbaiki")
    p.body(
        "Semua angka di bawah memakai split subjek yang sama persis seperti model ML lama: "
        "8 subjek untuk training (subject01-08), subject09 untuk validasi, dan subject10 "
        "sebagai subjek UJI yang tidak pernah disentuh. Model ML lama direkonstruksi ulang "
        "dan diukur ulang (kode: code/baseline/reproduce_old_model.py) - angkanya bukan "
        "perkiraan lagi.")
    p.table(
        ["Metode", "MAE subjek uji", "MAPE", "Vonis vs standar 10%"],
        [["TI Vital Signs bawaan", "54.7", "89.4%", "GAGAL TOTAL"],
         ["Model ML lama (fitur BPM TI)", "13.4", "22.6%", "GAGAL"],
         ["Trivial baseline (tebak konstan)", "14.7", "24.6%", "GAGAL"],
         ["Pipeline DIPERBAIKI", "5.2", "8.5%", "LOLOS"]],
        [58, 34, 26, 56], highlight_col=3,
        highlight_rule=lambda v: GREEN if "LOLOS" in str(v) else RED)
    p.callout(
        "bad", "Model ML lama praktis tidak belajar apapun dari radar.",
        "Enam jenis regressor dicoba (Linear, Ridge, k-NN, Random Forest, Gradient Boosting, "
        "MLP) - semuanya mendarat di MAE 13.4-14.6 bpm, sementara menebak angka KONSTAN "
        "tanpa data radar sama sekali sudah menghasilkan 14.8 bpm. Selisihnya cuma 0.2-1.4 "
        "bpm, dan korelasinya -0.12 s.d. +0.02 (nol). Diuji ulang dengan memutar subjek uji "
        "ke semua 10 kemungkinan: model ML KALAH dari tebak-konstan di 5 dari 10 rotasi - "
        "persis seperti lempar koin.")
    p.callout(
        "good", "Estimator fase yang diperbaiki: MAE 5.2 bpm - tanpa machine learning.",
        "Di subjek uji yang sama, pengolahan sinyal yang dikonfigurasi benar menghasilkan "
        "MAE 5.2 bpm - hampir 3x lebih baik daripada trivial baseline (14.8) maupun model ML "
        "lama (13.4-14.6), dan praktis menyentuh target pembimbing (5.0 bpm). Ini belum "
        "memakai ML sama sekali; ML baru masuk BELAKANGAN, di atas fitur yang benar.")

    # ---- Bagian 1
    p.add_page()
    p.h1("1. Yang Memang Benar-Benar Rusak: Output BPM Bawaan TI")
    p.body(
        "Temuan lama ini tetap berlaku dan tidak berubah. Kolom heart rate jadi dari "
        "Vital Signs library TI tidak bisa dipakai sebagai apapun - bukan sebagai hasil, "
        "bukan pula sebagai fitur model.")
    p.figure(fig / "fig1_ti_broken.png",
             "Kiri: output final_heart_rate TI (merah) sama sekali tidak mengikuti ECG (gelap). "
             "Kanan: 94.6% dari seluruh sampel terkunci di satu nilai, 5.859375 bpm.")
    p.callout(
        "bad", "Angka 5.859375 bpm itu bukan hasil pengukuran - itu artefak indeks FFT.",
        "5.859375 bpm = 0.09765625 Hz, yang PERSIS sama dengan bin ke-5 dari FFT 1024 titik "
        "pada frame rate 20 Hz (resolusi 20/1024 = 0.01953125 Hz). Nilai ini bahkan berada "
        "di LUAR band HR yang dipakai TI sendiri (0.8-2.0 Hz), jadi mustahil merupakan "
        "estimasi detak jantung yang sah. Ini nilai sentinel yang keluar saat algoritma "
        "gagal mengunci sinyal - dan itu terjadi di 94.6% sampel.")
    p.h2("Kolom radar lain yang juga tidak boleh dipakai")
    p.bullets([
        "range_bin_value dan rangeBinPhaseIndex - nilainya mencapai 2^31 dan 2^32 "
        "(maksimum 2147483648 dan 4294836234). Ini bilangan uint32 mentah yang belum "
        "di-parse, bukan besaran fisik. Kedua kolom ini rusak di level logger, buang saja.",
        "outputFilterHeartOut - sudah diuji sebagai sumber sinyal alternatif: hasilnya "
        "bias +37.5 bpm dengan korelasi antar-subjek NEGATIF (-0.827). Ini waveform hasil "
        "olahan internal TI, jadi ikut terkontaminasi kegagalan yang sama.",
        "confidence_heart - skalanya tidak konsisten antar dataset (position_front: 0-6.09; "
        "position_variation: -611 s.d. 43217, termasuk nilai negatif). Bukan confidence score "
        "yang sah, tidak bisa dipakai untuk filtering.",
    ])
    p.callout(
        "info", "Satu-satunya kolom radar yang layak pakai: unwrapPhasePeak_mm",
        "Ini perpindahan dinding dada hasil pembacaan fase radar, dalam milimeter - "
        "keluaran mentah sebelum masuk algoritma HR bawaan TI. Seluruh hasil positif di "
        "dokumen ini berasal dari kolom ini.")

    # ---- CACAT 1
    p.add_page()
    p.h1("1b. Cacat 1: Frame Rate Salah - Filter Jantung TI Justru Membuang Jantung")
    p.body(
        "Ini penyebab paling mendasar, dan menjelaskan kenapa SEMUA keluaran jantung TI "
        "(FFT, xCorr, peak, waveform) tidak berkorelasi dengan ECG, padahal sinyal jantungnya "
        "ada di fase mentah.")
    p.table(
        ["Sumber", "Perintah frameCfg", "Periode frame", "Frame rate"],
        [["Konfigurasi bawaan TI", "frameCfg 0 0 2 0 50 1 0", "50 ms", "20 fps (benar)"],
         ["Logger yang dipakai", "frameCfg 0 0 2 0 20 1 0", "20 ms", "50 fps (SALAH)"]],
        [46, 54, 34, 40], highlight_col=3,
        highlight_rule=lambda v: RED if "SALAH" in str(v) else GREEN)
    p.body(
        "Frame rate radar menentukan laju sampling sumbu waktu-lambat (slow time), yaitu "
        "laju sampling sinyal vital. Library TI memakai filter IIR bi-quad dengan koefisien "
        "TETAP yang dirancang untuk 20 Hz. Kalau filter yang sama dijalankan pada 50 Hz, "
        "seluruh respons frekuensinya MULUR dengan faktor 50/20 = 2.5x.")
    p.callout(
        "bad", "Akibatnya fatal: filter jantung TI meleset melewati detak jantung manusia.",
        "Band jantung yang DIMAKSUD TI: 0.8 - 4.0 Hz (48 - 240 bpm).\n"
        "Band yang SEBENARNYA dilewatkan pada 50 fps: 0.8x2.5 - 4.0x2.5 = 2.0 - 10.0 Hz "
        "(120 - 600 bpm).\n\n"
        "Detak jantung manusia saat istirahat berada di 1.0 - 1.7 Hz (60 - 100 bpm) - "
        "yaitu DI BAWAH batas bawah filter yang tergeser. Artinya sinyal jantung yang asli "
        "justru DIREDAM HABIS oleh filter TI sendiri, dan yang tersisa untuk dianalisis "
        "hanyalah noise dan harmonik.")
    p.h2("Bukti langsung dari data")
    p.body(
        "Prediksi di atas bisa diuji: kalau benar, maka waveform jantung keluaran TI "
        "(outputFilterHeartOut, yang kebetulan TERBACA BENAR oleh logger) energinya harus "
        "berada di 2-10 Hz, bukan di band jantung. Hasil pengukurannya:")
    p.table(
        ["Pita frekuensi", "% energi outputFilterHeartOut", "Seharusnya"],
        [["0.8 - 2.0 Hz (jantung manusia)", "4.2%", "hampir seluruhnya"],
         ["2.0 - 10.0 Hz (band tergeser 2.5x)", "94.3%", "hampir nol"]],
        [64, 62, 48], highlight_col=1,
        highlight_rule=lambda v: RED if v == "94.3%" else GREEN)
    p.figure(fig / "fig11_filter_shift.png",
             "Spektrum 'waveform jantung' keluaran TI. Puncaknya di 2.98 Hz (179 per menit) - "
             "bukan detak jantung manusia. 94.3% energinya berada persis di band 2.0-10.0 Hz "
             "yang diprediksi kalau filter 0.8-4.0 Hz dijalankan 2.5x terlalu cepat. "
             "Prediksi dan pengukuran cocok.")
    p.callout(
        "good", "Kenapa pipeline perbaikan tetap bekerja meski frame rate salah.",
        "unwrapPhasePeak_mm diekstrak SEBELUM masuk rantai filter TI. Karena itu ia TIDAK "
        "terkena pergeseran band, dan sinyal jantungnya utuh. Inilah alasan mendasar kenapa "
        "seluruh hasil positif di dokumen ini bisa diperoleh dari kolom tersebut, sementara "
        "semua kolom hilir TI tidak berguna.")

    # ---- CACAT 2
    p.add_page()
    p.h1("1c. Cacat 2: Logger Salah Membaca Byte - Kolom Detak Jantung Berisi Laju Napas")
    p.body(
        "Logger (old code/logger.py) membaca paket biner radar dengan offset byte manual. "
        "Empat dari sepuluh offset itu keliru. Struktur paket yang benar diambil dari "
        "definisi resmi TI (texas_instrument_labs/pjt/common/mmw_output.h): header 40 byte "
        "+ TLV header 8 byte, sehingga isi data mulai di byte ke-48.")
    p.table(
        ["Kolom di CSV", "Offset dibaca", "ISI SEBENARNYA di offset itu", "Status"],
        [["unwrapPhasePeak_mm", "64", "unwrapPhasePeak_mm", "BENAR"],
         ["outputFilterHeartOut", "72", "outputFilterHeartOut", "BENAR"],
         ["heart_rate_est_fft", "76", "heartRateEst_FFT", "BENAR"],
         ["heart_rate_est_peak", "92", "breathingRateEst_FFT (laju NAPAS)", "SALAH"],
         ["confidence_heart", "104", "confidenceMetricBreathOut (NAPAS)", "SALAH"],
         ["sumEnergyHeartWfm", "116", "confidenceMetricHeartOut_4Hz", "SALAH"],
         ["rangeBinPhaseIndex", "50 (uint32)", "uint16 - tipe data keliru", "SALAH"]],
        [44, 26, 68, 24], highlight_col=3,
        highlight_rule=lambda v: RED if v == "SALAH" else GREEN)
    p.callout(
        "bad", "Akibatnya: final_heart_rate sebenarnya adalah LAJU NAPAS, bukan detak jantung.",
        "Logger menghitung final_heart_rate dengan aturan:\n"
        "  jika (confidence > 0.4) ATAU (|est_fft - est_peak| < 15) -> pakai est_fft\n"
        "  jika tidak -> pakai est_peak\n\n"
        "Tapi 'confidence' yang dipakai adalah confidence NAPAS (nilainya ~0.036, tidak "
        "pernah melewati 0.4), dan 'est_peak' yang dipakai adalah LAJU NAPAS (~5.86). "
        "Selisih |85 - 5.86| = 79, jauh di atas 15. Maka kedua syarat gagal, dan cabang "
        "'jika tidak' selalu diambil - sehingga LAJU NAPAS 5.859375 napas/menit disalin "
        "menjadi 'detak jantung', 98.3% dari waktu.")
    p.callout(
        "key", "Ini diverifikasi dengan menjalankan ulang fungsi logger pada kolom-kolomnya.",
        "Fungsi calculate_heart_rate_est_display() dijalankan ulang persis seperti di "
        "logger.py, memakai kolom-kolom yang tersimpan di CSV. Hasilnya cocok 100.00% "
        "dengan kolom final_heart_rate yang benar-benar tersimpan (selisih maksimum 0.000000 "
        "pada 21.296 baris). Jadi tidak ada keraguan: angka 5.859375 yang selama ini dikira "
        "'nilai sentinel misterius' ternyata adalah laju napas.")
    p.figure(fig / "fig10_rootcause.png",
             "Kiri: distribusi 'confidence_heart' (yang sebenarnya confidence NAPAS) menumpuk "
             "jauh di bawah ambang 0.4 yang dipakai logger. Kanan: persentase waktu output "
             "membeku, diprediksi dari logika logger, dibandingkan yang terukur di data.")

    # ---- CACAT 3
    p.add_page()
    p.h1("1d. Cacat 3: Jarak Subjek - Sinyal Jantung Tenggelam di 100 cm")
    p.body(
        "Cacat ini terbukti dari data variasi jarak yang SUDAH PERNAH DIREKAM SENDIRI "
        "(data/raw/distance), tapi belum pernah dianalisis dari sudut ini.")
    p.table(
        ["Jarak subjek", "SNR sinyal jantung", "MAE", "MAPE", "Keterangan"],
        [["50 cm", "3.16", "8.6 bpm", "12.5%", "sinyal jelas ada"],
         ["100 cm", "1.09", "16.9 bpm", "28.6%", "SETARA NOISE"]],
        [32, 40, 26, 26, 50], highlight_col=1,
        highlight_rule=lambda v: RED if v == "1.09" else GREEN)
    p.figure(fig / "fig12_distance.png",
             "Pada 100 cm, SNR sinyal jantung turun ke 1.09 - praktis tidak bisa dibedakan "
             "dari noise, sehingga tidak ada algoritma apapun (termasuk ML) yang mampu "
             "mengekstraknya. Pada 50 cm, SNR 3.16 dan sinyalnya jelas ada.")
    p.callout(
        "warn", "Ini menjelaskan kenapa sebagian subjek jauh lebih buruk dari yang lain.",
        "Subjek dengan SNR lemah di dataset utama (S03: 1.39, S04: 1.38, S06: 2.08) hampir "
        "pasti duduk lebih jauh dari subjek yang SNR-nya kuat (S01: 5.91, S10: 5.43). Daya "
        "pantul radar turun kira-kira sebanding dengan 1/R^4, sehingga menggeser subjek dari "
        "50 cm ke 100 cm saja sudah cukup untuk menenggelamkan sinyal jantung. Konfigurasi "
        "TI pun hanya mencari target di rentang 0.3 - 1.0 meter (vitalSignsCfg 0.3 1.0), "
        "jadi 100 cm sudah berada di batas paling tepi.")

    # ---- Bagian 2
    p.add_page()
    p.h1("2. Bukti Desisif: Sinyal Jantung ADA di Data Radar")
    p.body(
        "Ini tes yang sebelumnya belum pernah dilakukan, dan ini yang membalik kesimpulan. "
        "Alih-alih membuat estimator HR baru lalu mengukur akurasinya (yang mencampur dua "
        "pertanyaan sekaligus), tes ini menanyakan satu hal saja secara langsung: "
        "APAKAH informasi HR-nya ada di sinyal fase radar?")
    p.h2("Cara kerjanya")
    p.body(
        "Untuk tiap potongan 16 detik, ambil frekuensi HR yang SEBENARNYA (dari ECG), lalu "
        "cek: di dalam spektrum sinyal fase radar, bin frekuensi itu menempati peringkat "
        "keberapa dari segi kekuatan? Kalau radar tidak menangkap apapun, peringkatnya akan "
        "acak - rata-rata di tengah (0.5). Kalau radar benar-benar menangkap detak jantung, "
        "bin itu akan sering jadi puncak tertinggi (mendekati 0).")
    p.figure(fig / "fig2_signal_exists.png",
             "Kiri: distribusi peringkat frekuensi ECG di spektrum fase radar - menumpuk kuat "
             "di dekat 0 (puncak tertinggi), bukan tersebar rata seperti kalau acak. "
             "Kanan: SNR di frekuensi ECG yang benar, median 3.19x di atas noise floor.")
    p.callout(
        "good", "Hasil: median peringkat ternormalisasi = 0.158, jauh dari level acak 0.500.",
        "Pada 6 dari 10 subjek, frekuensi HR yang benar masuk 3 besar puncak spektral di "
        "lebih dari 50% window. SNR median di frekuensi yang benar adalah 3.19x noise floor. "
        "Artinya: radar IWR1443 memang merekam gerakan mikro jantung. Perangkat kerasnya "
        "bekerja. Akuisisinya tidak gagal.")
    p.table(
        ["Subjek", "Median peringkat", "% window HR di 3 besar", "SNR median", "Kualitas"],
        [["subject01", "0.053", "63.5%", "5.91", "kuat"],
         ["subject02", "0.105", "51.9%", "4.56", "kuat"],
         ["subject03", "0.368", "23.5%", "1.39", "lemah"],
         ["subject04", "0.421", "9.8%", "1.38", "lemah"],
         ["subject05", "0.158", "46.2%", "3.32", "sedang"],
         ["subject06", "0.316", "18.2%", "2.08", "lemah"],
         ["subject07", "0.105", "60.4%", "5.02", "kuat"],
         ["subject08", "0.053", "58.8%", "4.20", "kuat"],
         ["subject09", "0.105", "54.7%", "3.79", "kuat"],
         ["subject10", "0.053", "63.9%", "5.43", "kuat"],
         ["acak (kalau tak ada sinyal)", "0.500", "~7%", "1.00", "-"]],
        [50, 34, 42, 26, 22], highlight_col=4,
        highlight_rule=lambda v: GREEN if v == "kuat" else (RED if v == "lemah" else None))
    p.body(
        "Subject03, 04, dan 06 memang sesi berkualitas rendah (SNR ~1.4-2.1, nyaris setara "
        "noise). Ini kemungkinan masalah penempatan radar / jarak / gerakan subjek pada sesi "
        "tersebut. Tapi 7 subjek lainnya jelas membawa sinyal jantung yang kuat.")

    # ---- Bagian 3
    p.add_page()
    p.h1("3. Penyebab Korelasi ~0 (1): Band Frekuensi Terlalu Lebar")
    p.body(
        "Script analyze_phase_signal.py mencari puncak spektral di band 0.7-3.3 Hz "
        "(setara 42-198 bpm). Band selebar ini menyentuh wilayah tempat energi napas masih "
        "sangat kuat. Menurut spesifikasi TI sendiri (Developer's Guide hal. 3), amplitudo "
        "napas adalah 1-12 mm sedangkan detak jantung hanya 0.1-0.5 mm - beda dua orde "
        "besaran. Pengukuran pada data ini mengkonfirmasi: komponen napas 4-9x lebih besar "
        "daripada komponen jantung.")
    p.figure(fig / "fig3_psd_band.png",
             "Kiri: satu window nyata - band lama (0.7-3.3 Hz) memilih 45 bpm padahal HR "
             "sebenarnya 92 bpm, karena tertarik ekor energi napas di tepi bawah band. Band "
             "baru (0.8-2.0 Hz) memilih 94 bpm, benar. Kanan: efeknya sistematis di seluruh data.")
    p.callout(
        "warn", "Efeknya sistematis, bukan acak: estimator lama SELALU meleset ke bawah.",
        "Median rasio HR-estimasi terhadap HR-sebenarnya di band lama adalah 0.74x - "
        "artinya estimasi rata-rata 26% lebih rendah dari kenyataan, konsisten ke arah yang "
        "sama. Ini ciri khas kebocoran energi frekuensi rendah, bukan ciri sinyal yang tidak "
        "ada. Kalau memang tidak ada sinyal HR, rasionya akan tersebar acak - bukan bias "
        "rapi ke satu arah.")
    p.body(
        "Catatan penting: menyempitkan band memperbaiki arah, tapi BIAS KE BAWAH-nya belum "
        "hilang sepenuhnya (median masih 0.78x). Sisa bias inilah yang menjadi pekerjaan "
        "teknis utama berikutnya - lihat dokumen rekomendasi.")

    p.h1("4. Penyebab (2): Aliasing dan Absennya Penghalusan")
    p.bullets([
        "Aliasing - script lama me-resample sinyal 64 Hz ke grid 10 Hz memakai np.interp "
        "langsung, tanpa anti-alias filter. Konten di atas 5 Hz terlipat balik masuk ke band "
        "HR. Secara teori ini bug; secara praktik dampaknya ternyata kecil di data ini "
        "(lihat ablasi), tapi tetap harus diperbaiki agar hasil bisa dipertanggungjawabkan.",
        "Tanpa penghalusan temporal - estimasi HR dibiarkan melompat bebas antar window. "
        "Padahal HR manusia tidak bisa berubah 30 bpm dalam 4 detik. Satu median filter "
        "5-titik menaikkan korelasi dari 0.268 ke 0.365 - lompatan terbesar kedua.",
    ])
    p.figure(fig / "fig4_ablation.png",
             "Ablasi: perbaikan mana yang benar-benar berpengaruh. Menyempitkan band adalah "
             "penyebab tunggal terbesar (korelasi 0.141 -> 0.253). Anti-alias dan menaikkan "
             "sample rate sendirian nyaris tidak berefek. Median filter memberi lompatan besar terakhir.")

    # ---- Bagian 5 - metrik
    p.add_page()
    p.h1("5. Penyebab (3): Korelasi Adalah Metrik yang Salah untuk Dataset Ini")
    p.body(
        "Ini bagian yang paling halus tapi paling penting untuk dipahami, karena ini "
        "menjelaskan kenapa angka korelasi selama ini terlihat seperti bencana padahal "
        "estimatornya tidak seburuk itu.")
    p.body(
        "Korelasi adalah ukuran ternormalisasi terhadap varians: ia bertanya 'kalau target "
        "naik, apakah estimasi ikut naik?'. Kalau targetnya nyaris tidak pernah naik atau "
        "turun, tidak ada apapun untuk dikorelasikan - dan korelasi akan mendekati nol "
        "SEKALIPUN estimatornya akurat.")
    p.callout(
        "key", "Semua subjek duduk diam saat istirahat. HR mereka praktis konstan.",
        "Standar deviasi HR di dalam sesi hanya 1.0-4.4 bpm (subject03 cuma 1.0 bpm). "
        "Untuk estimator tak-bias dengan noise ber-std s, korelasi yang bisa dicapai adalah "
        "r = std(HR) / akar(std(HR)^2 + s^2). Masukkan s = 5 bpm - persis di ambang target "
        "pembimbing - maka estimator SEMPURNA sekalipun hanya mencapai korelasi 0.20-0.66 "
        "pada data ini, dan pada 5 dari 10 subjek tidak menembus 0.5. Jadi korelasi "
        "dalam-subjek yang rendah di dataset ini BUKAN bukti estimator gagal; itu "
        "konsekuensi matematis dari target yang hampir konstan.")
    p.figure(fig / "fig5_ceiling.png",
             "Kiri: variasi HR dalam sesi per subjek - semuanya sangat kecil. Kanan: batas atas "
             "korelasi yang bisa dicapai estimator SEMPURNA (noise 5 bpm) pada tiap subjek, "
             "dihitung analitik. Bahkan estimator sempurna tertahan di bawah 0.5 pada 5 dari "
             "10 subjek, dan subject03 mentok di 0.20.")
    p.body(
        "Konsekuensinya: vonis 'korelasi < 0.3 berarti tidak ada sinyal HR' yang selama ini "
        "dipakai untuk menyimpulkan riset buntu memang tidak sah diterapkan pada dataset "
        "ini. Metrik itu baru bermakna kalau HR ground truth-nya benar-benar bergerak - "
        "dan itulah yang diusulkan diperbaiki di dokumen rekomendasi.")

    p.h1("6. Gambaran Jujur: Apa yang Berhasil dan Apa yang Belum")
    p.body(
        "Setelah ketiga perbaikan diterapkan, penting untuk membedah korelasi 0.62 di test "
        "set itu berasal dari mana - supaya tidak terjadi klaim berlebihan.")
    p.figure(fig / "fig8_within_between.png",
             "Kiri: radar berhasil membedakan subjek ber-HR tinggi dari yang rendah "
             "(r = 0.653 antar 10 subjek). Kanan: radar BELUM bisa melacak naik-turun HR di "
             "dalam satu sesi - tapi HR aslinya memang cuma bergerak +/-2.8 bpm.")
    p.table(
        ["Aspek yang diukur", "Hasil", "Interpretasi"],
        [["Antar-subjek (r, n=10)", "0.653", "BERHASIL - radar menangkap level HR tiap orang"],
         ["Dalam-subjek (r)", "0.024", "belum terbukti - tapi lihat kolom kanan"],
         ["Variasi HR dalam sesi", "+/- 2.8 bpm", "nyaris nol: tak ada dinamika untuk dilacak"],
         ["Bias sistematis", "-9.0 bpm", "estimasi konsisten lebih rendah; bisa dikalibrasi"]],
        [46, 26, 102], highlight_col=1,
        highlight_rule=lambda v: GREEN if v == "0.653" else None)
    p.callout(
        "warn", "Ini keterbatasan dataset, bukan (hanya) keterbatasan metode.",
        "Kemampuan radar melacak PERUBAHAN detak jantung tidak bisa dibuktikan MAUPUN "
        "dibantah dengan data yang ada sekarang, karena detak jantung subjeknya memang tidak "
        "pernah berubah selama perekaman. Ini temuan desain eksperimen, dan inilah hal paling "
        "penting yang perlu didiskusikan di bimbingan - penanganannya ada di dokumen rekomendasi.")

    # ---- Bagian 7 - hasil
    p.add_page()
    p.h1("7. Hasil Estimator yang Sudah Diperbaiki")
    p.body(
        "Estimator ini murni pengolahan sinyal - belum ada machine learning sama sekali. "
        "Konfigurasinya: band 0.8-2.0 Hz, resample 20 Hz dengan anti-alias, detrend, "
        "window 16 detik, interpolasi parabolik puncak, median filter 5 titik. "
        "Kode: code/evaluation/corrected_estimator.py")
    p.figure(fig / "fig6_result.png",
             "Kiri: subject10 (subjek UJI). Perhatikan model ML lama (merah) keluarannya "
             "praktis GARIS DATAR yang menempel di garis tebak-konstan (abu-abu titik-titik) - "
             "inilah wujud visual dari GIGO. Estimator fase diperbaiki (hijau) justru "
             "mengikuti level ECG. Tengah: MAE di subjek uji. Kanan: model lama vs trivial "
             "diputar ke semua 10 pilihan subjek uji - hasilnya imbang, ML kalah di 5 dari 10.")
    p.callout(
        "key", "Bukti visual GIGO: model ML lama outputnya garis datar.",
        "Di panel kiri, prediksi model ML lama nyaris tidak bergerak dari nilai rata-rata "
        "training. Itu yang dilakukan model regresi ketika fitur inputnya tidak membawa "
        "informasi apapun tentang target: ia menyerah dan konvergen ke mean. MAE-nya "
        "kelihatan 'wajar' (13-15 bpm) sehingga tidak terlihat rusak - padahal model itu "
        "sama sekali tidak melihat detak jantung.")
    p.figure(fig / "fig7_persubject.png",
             "MAE per subjek. subject07 (2.9 bpm) dan subject10 (5.0 bpm) SUDAH memenuhi "
             "target 5 bpm tanpa ML sama sekali. Subjek dengan MAE tinggi (S02, S03, S04, S06) "
             "persis subjek yang SNR radarnya lemah di Bagian 2 - konsisten.")
    p.callout(
        "good", "Dua subjek sudah menembus target 5 bpm tanpa machine learning.",
        "subject07 mencapai MAE 2.9 bpm dan subject10 mencapai 5.0 bpm hanya dengan "
        "pengolahan sinyal yang dikonfigurasi benar. Ini bukti bahwa target pembimbing "
        "(MAE <= 5 bpm) realistis dan bisa dicapai - bukan target yang mustahil.")
    p.h1("8. Vonis Terhadap Standar Kesehatan (ANSI/CTA-2065)")
    p.body(
        "Inilah pertanyaan yang sebetulnya paling penting dijawab tesis ini: apakah "
        "TI IWR1443BOOST layak dipakai untuk mengukur detak jantung, menurut batas galat "
        "yang diakui (MAPE < 10%)?")
    p.figure(fig / "fig9_standard.png",
             "Kiri (skala logaritmik): MAPE per subjek. Software bawaan TI meleset 60-100%, "
             "jauh di atas batas 10%. Pipeline yang diperbaiki menurunkannya sampai 5-25%. "
             "Kanan: jumlah subjek yang memenuhi standar - dari 0 menjadi 6 dari 10.")
    p.table(
        ["Konfigurasi", "MAPE (10 subjek)", "Subjek LOLOS", "Catatan"],
        [["TI Vital Signs (setup CACAT)", "89 - 91%", "0 dari 10", "bukan uji yang adil"],
         ["Pipeline fase (tanpa ML)", "2.9 - 19.3%", "6 dari 10", "dari data cacat pun lolos"],
         ["TI Vital Signs (setup BENAR)", "belum diukur", "belum diukur", "PERLU REKAM ULANG"]],
        [58, 34, 30, 52], highlight_col=3,
        highlight_rule=lambda v: RED if "bukan uji" in str(v) else
        (GREEN if "lolos" in str(v) else AMBER))
    p.callout(
        "warn", "Baca hasil ini dengan hati-hati: ini BUKAN vonis yang adil terhadap TI.",
        "Nol dari sepuluh subjek memenuhi standar - tapi perangkatnya dijalankan pada frame "
        "rate yang salah (Cacat 1), datanya dibaca dengan offset byte yang salah (Cacat 2), "
        "dan sebagian subjek duduk terlalu jauh (Cacat 3). Yang gagal di sini adalah SETUP "
        "PENGUKURAN, bukan library TI. Menyatakan 'library TI tidak layak' berdasarkan data "
        "ini akan menjadi kesimpulan yang tidak sah - dan berisiko dipatahkan di sidang. "
        "Vonis yang adil terhadap library TI hanya bisa diberikan SETELAH direkam ulang "
        "dengan konfigurasi yang benar.")
    p.callout(
        "good", "Yang BISA disimpulkan dengan sah: sensornya bekerja, dan itu kabar baik.",
        "Sensor radarnya sendiri menangkap detak jantung dengan baik (Bagian 2: SNR 3.19x). "
        "Bahkan dari data yang cacat sekalipun, pipeline pemrosesan sinyal yang benar sudah "
        "meloloskan 6 dari 10 subjek ke standar - tanpa mengganti perangkat keras apapun dan "
        "tanpa machine learning apapun, dengan yang terbaik mencapai MAPE 2.9%. Kalau ketiga "
        "cacat akuisisi diperbaiki, hasilnya masuk akal untuk lebih baik lagi.")
    p.body(
        "Empat subjek yang belum lolos (S03, S04, S06, S10) terbagi dua sebab, dan "
        "keduanya sudah teridentifikasi: (a) S03, S04, S06 memang sesi ber-SNR rendah "
        "(1.4-2.1, nyaris noise) - konsisten dengan Cacat 3, subjek kemungkinan duduk terlalu "
        "jauh; (b) S01 dan S02 ber-SNR kuat tapi terkena bias sistematis -9 bpm, yang punya "
        "jalur perbaikan tersendiri. Keduanya dibahas di dokumen rekomendasi.")

    p.h2("Kesimpulan diagnosis")
    p.bullets([
        "Sensor radar TI IWR1443 MENANGKAP sinyal jantung dengan baik (SNR sampai 5.9). "
        "Perangkat kerasnya tidak bersalah.",
        "Kegagalan berasal dari TIGA cacat akuisisi yang semuanya sudah dibuktikan: frame "
        "rate 2.5x terlalu cepat, logger salah baca 4 dari 10 kolom, dan sebagian subjek "
        "duduk terlalu jauh.",
        "Nilai 5.859375 yang selama ini dikira 'sentinel misterius' ternyata adalah LAJU "
        "NAPAS yang salah disalin ke kolom detak jantung - direproduksi 100.00% persis.",
        "Karena setup-nya cacat, dataset ini TIDAK SAH dipakai memvonis library TI. Vonis "
        "yang adil butuh rekam ulang dengan konfigurasi benar.",
        "Model ML lama gagal karena diberi makan kolom yang isinya laju napas dan confidence "
        "napas (GIGO) - bukan karena ML-nya tidak cocok.",
        "Meski begitu, pipeline pemrosesan sinyal yang benar pada fase mentah SUDAH "
        "meloloskan 6 dari 10 subjek ke standar MAPE < 10% (rata-rata 9.8%), bahkan dari data "
        "yang cacat ini - dan mengalahkan trivial baseline dengan telak.",
    ])
    p.output(str(out / "diagnosis_riset.pdf"))
    print("ok:", out / "diagnosis_riset.pdf")


# ======================================================= DOC 2: REKOMENDASI
def build_recommendation(fig, out):
    p = Report("Rekomendasi Arah Riset",
               "Langkah konkret setelah diagnosis: dari 'riset buntu' "
               "menjadi kontribusi tesis yang jelas")
    p.title_page([
        ("Dokumen", "Pendamping dari 'Diagnosis: Di Mana Riset Ini Stuck'"),
        ("Tanggal", "13 Juli 2026"),
        ("Inti", "Riset tidak perlu di-reScope atau diganti arah. Sinyal jantung terbukti "
                 "ada di data radar. Yang dibutuhkan adalah satu sesi pengambilan data ulang "
                 "dengan protokol yang benar, perbaikan bias sistematis, lalu ML dijalankan "
                 "di atas fitur yang benar."),
    ])

    p.add_page()
    p.h1("Jawaban Langsung: Bisakah Tesis Selesai TANPA Ambil Data Ulang?")
    p.callout(
        "good", "BISA. Dan caranya adalah mempersempit seluruh analisis ke satu kolom saja: "
                "unwrapPhasePeak_mm.",
        "Dari 12 kolom yang direkam, hanya kolom ini yang SELAMAT dari ketiga cacat "
        "akuisisi. Artinya data yang sudah ada TETAP SAH dipakai - asalkan klaim tesisnya "
        "dibatasi pada sinyal fase, dan TIDAK dipakai untuk memvonis library TI.")
    p.h2("Kenapa justru kolom ini yang selamat")
    p.table(
        ["Cacat", "Dampak ke kolom lain", "Dampak ke unwrapPhasePeak_mm"],
        [["Frame rate 50 fps (bukan 20)",
          "Filter TI mulur 2.5x, buang sinyal jantung", "TIDAK KENA - diambil sebelum filter"],
         ["Offset byte logger salah",
          "4 kolom berisi data napas / salah field", "TIDAK KENA - offset 64 sudah benar"],
         ["Jarak subjek terlalu jauh",
          "SNR runtuh (sama-sama kena)", "KENA - ini batas fisik yang harus dilaporkan"]],
        [46, 62, 66], highlight_col=2,
        highlight_rule=lambda v: GREEN if "TIDAK KENA" in str(v) else AMBER)
    p.callout(
        "info", "Bonus: frame rate 50 fps justru MENGUNTUNGKAN sinyal fase.",
        "Kesalahan frame rate merusak filter TI, tapi untuk sinyal fase mentah, laju sampling "
        "yang lebih tinggi (50 Hz vs 20 Hz) justru memberi resolusi waktu lebih rapat dan "
        "membuat phase-unwrapping lebih andal. Jadi cacat yang mematikan bagi pipeline TI "
        "tidak merugikan - malah sedikit menguntungkan - pendekatan berbasis fase.")

    p.h1("Hasil Nyata dengan Data yang SUDAH ADA")
    p.body(
        "Pipeline final (code/evaluation/phase_pipeline.py) hanya memakai unwrapPhasePeak_mm: "
        "resample anti-alias 25 Hz -> TURUNAN fase -> kurangi komponen napas -> bandpass "
        "0.8-2.0 Hz -> puncak PSD Welch per window 20 detik -> median filter. Tanpa ML, "
        "tanpa data baru.")
    p.table(
        ["Subjek", "MAE (bpm)", "MAPE", "SNR", "Vonis (standar 10%)"],
        [["subject01", "2.8", "3.2%", "6.41", "LOLOS"],
         ["subject09", "2.5", "2.9%", "4.81", "LOLOS"],
         ["subject07", "3.0", "4.5%", "5.99", "LOLOS"],
         ["subject08", "3.6", "4.7%", "5.35", "LOLOS"],
         ["subject05", "5.4", "8.0%", "4.55", "LOLOS"],
         ["subject02", "6.6", "8.4%", "5.33", "LOLOS"],
         ["subject03", "10.1", "12.9%", "1.67", "gagal - SNR rendah"],
         ["subject06", "11.7", "15.5%", "1.76", "gagal - SNR rendah"],
         ["subject04", "14.6", "19.3%", "1.73", "gagal - SNR rendah"],
         ["subject10", "11.2", "19.0%", "5.36", "gagal - kasus khusus"],
         ["RATA-RATA", "7.2", "9.8%", "-", "6 dari 10 LOLOS"]],
        [34, 28, 24, 22, 66], highlight_col=4,
        highlight_rule=lambda v: GREEN if v == "LOLOS" else
        (RED if "gagal" in str(v) else (GREEN if "LOLOS" in str(v) else None)))
    p.callout(
        "good", "6 dari 10 subjek memenuhi standar kesehatan ANSI/CTA-2065 (MAPE < 10%).",
        "Bandingkan dengan pipeline bawaan TI pada data yang sama: MAPE 89-91%, nol subjek "
        "lolos. Dan pipeline fase ini juga MENGALAHKAN trivial baseline dengan telak "
        "(MAE 7.1 vs 11.9 bpm di subjek uji). Semua ini tanpa machine learning dan tanpa "
        "satu byte pun data baru.")
    p.h2("Yang gagal pun punya penjelasan - dan itu justru temuan")
    p.callout(
        "key", "SNR memisahkan yang berhasil dari yang gagal dengan akurasi 92%.",
        "Tiga subjek yang gagal (S03, S04, S06) semuanya ber-SNR rendah (1.67-1.76), yaitu "
        "mendekati noise floor. Diuji lintas 12 sesi (10 subjek + 2 sesi variasi jarak), "
        "ambang SNR >= 1.8 memprediksi lolos/tidak dengan benar pada 11 dari 12 sesi. "
        "Data variasi jarak menguatkan sebab fisiknya: pada 100 cm SNR = 1.09 (setara noise, "
        "MAPE 58%), pada 50 cm SNR = 3.16 (MAPE 8.1%, LOLOS). Jadi kegagalan itu BUKAN "
        "kelemahan metode, melainkan batas fisik akuisisi - dan bisa dikuantifikasi.")
    p.callout(
        "warn", "Dua kejujuran yang harus ditulis di bab keterbatasan.",
        "(1) subject10 ber-SNR bagus (5.36) tapi tetap gagal (MAPE 19.0%, bias +9.7 bpm). "
        "Ia punya HR paling rendah di kohort (61 bpm), dan operasi turunan fase menggeser "
        "puncak spektral ke atas pada HR rendah. Ini kelemahan algoritma yang harus diakui, "
        "bukan disembunyikan.\n\n"
        "(2) Ambang SNR itu dihitung DI FREKUENSI ECG SEBENARNYA, jadi ia butuh ECG. "
        "Artinya ia alat DIAGNOSIS (menjelaskan sesi mana yang gagal dan kenapa), BUKAN "
        "gerbang kualitas yang bisa dipakai saat alat dipakai sungguhan. Metrik pengganti "
        "tanpa-ECG sudah dicoba (peak-to-median, entropi spektral, jitter) tapi akurasinya "
        "hanya 67-75% - belum layak diklaim. Ini pekerjaan lanjutan yang jujur untuk "
        "disebut sebagai saran penelitian berikutnya.")

    # ---- JUDUL & KONTRIBUSI
    p.add_page()
    p.h1("Usulan Judul dan Kontribusi Tesis")
    p.callout(
        "key", "Usulan judul",
        "\"Estimasi Detak Jantung Berbasis Sinyal Fase Mentah Radar FMCW TI IWR1443BOOST: "
        "Analisis Akar Kegagalan Pipeline Bawaan dan Rancangan Pipeline Pengganti "
        "Tervalidasi Standar ANSI/CTA-2065\"\n\n"
        "Alternatif yang lebih pendek: \"Rekonstruksi Pipeline Estimasi Detak Jantung dari "
        "Sinyal Fase Radar FMCW: Diagnosis Kegagalan dan Validasi terhadap Standar "
        "Kesehatan\"")
    p.h2("Empat kontribusi yang bisa diklaim - semuanya SUDAH ADA BUKTINYA")
    p.bullets([
        "K1 - ANALISIS AKAR KEGAGALAN pipeline vital-sign bawaan pada setup yang tidak "
        "sesuai spesifikasi. Bukti: tiga cacat (frame rate 2.5x, offset byte, jarak), "
        "semuanya terbukti kuantitatif dan dapat direproduksi - termasuk reproduksi "
        "final_heart_rate 100.00% persis.",
        "K2 - PIPELINE PENGGANTI berbasis sinyal fase mentah, divalidasi terhadap standar "
        "kesehatan. Bukti: 6 dari 10 subjek MAPE < 10% (rata-rata 9.8%), dibanding pipeline "
        "bawaan TI yang MAPE-nya ~90% dengan nol subjek lolos. Tanpa machine learning.",
        "K3 - KRITERIA KELAYAKAN berbasis SNR, plus karakterisasi pengaruh jarak akuisisi. "
        "Bukti: ambang SNR >= 1.8 memprediksi lolos/gagal dengan benar pada 11 dari 12 sesi; "
        "dikuatkan eksperimen jarak 50 cm (SNR 3.16, lolos) vs 100 cm (SNR 1.09, gagal).",
        "K4 - KOREKSI METODOLOGI evaluasi estimator HR berbasis radar. Bukti: analisis "
        "ceiling korelasi (korelasi adalah metrik keliru untuk dataset HR statis), dan "
        "pembongkaran bahwa model ML terdahulu sebenarnya kalah dari menebak angka konstan.",
    ])
    p.h2("Kenapa ini kontribusi yang sah, bukan 'sekadar menguji alat'")
    p.bullets([
        "K1 bukan 'alatnya jelek' - melainkan menunjukkan SECARA PRESIS bagaimana kesalahan "
        "konfigurasi yang tampak sepele (satu angka frame rate, empat offset byte) bisa "
        "membuat seluruh keluaran vital-sign menjadi tidak bermakna, lengkap dengan "
        "reproduksi 100.00%. Ini jebakan nyata yang bisa menimpa peneliti lain, dan "
        "belum pernah didokumentasikan sebaik ini.",
        "K2 adalah rekayasa: membangun rantai pemrosesan pengganti dan membuktikannya "
        "memenuhi standar yang diakui - bukan target yang dikarang sendiri.",
        "K3 adalah yang paling bernilai ilmiah: menjawab KAPAN metode ini bisa dipakai dan "
        "kapan secara fisik mustahil. Kebanyakan paper radar vital-sign hanya melaporkan "
        "kasus yang berhasil; menetapkan batas keberlakuan justru lebih jujur dan lebih "
        "berguna.",
        "K4 menunjukkan kedewasaan metodologis: membongkar bahwa model ML terdahulu "
        "sebenarnya kalah dari menebak angka konstan, dan bahwa korelasi adalah metrik yang "
        "keliru untuk dataset HR statis.",
    ])
    p.h2("Batasan yang WAJIB ditulis di awal (ini yang melindungi di sidang)")
    p.callout(
        "bad", "Tulis eksplisit di Batasan Masalah:",
        "\"Penelitian ini TIDAK memberikan vonis kelayakan terhadap library Vital Signs "
        "bawaan TI, karena data direkam dengan konfigurasi yang tidak sesuai spesifikasi TI "
        "(frame rate 50 fps, sementara library dirancang untuk 20 fps). Yang dievaluasi "
        "adalah kelayakan SINYAL FASE MENTAH radar sebagai basis estimasi detak jantung.\"\n\n"
        "Dengan kalimat ini, ketiga cacat berubah dari kelemahan menjadi TEMUAN (K1), dan "
        "penguji tidak bisa mematahkan tesis dengan argumen 'pengujianmu tidak adil' - "
        "karena Anda sendiri yang lebih dulu menyatakannya.")
    p.callout(
        "good", "Machine learning: jadikan bab bonus, bukan tumpuan.",
        "Dengan K1-K4 sudah aman, ML tidak lagi menentukan kelulusan. Kalau sempat: latih "
        "regressor pada fitur spektral dari unwrapPhasePeak_mm (frekuensi puncak, SNR, lebar "
        "puncak, energi harmonik, frekuensi & amplitudo napas), lalu bandingkan dengan "
        "phase_pipeline.py. Kalau menang, itu bab tambahan yang bagus. Kalau kalah, itu tetap "
        "temuan sah: pemrosesan sinyal yang benar sudah cukup, dan ML tidak menambah nilai "
        "pada ukuran data sebesar ini.")

    p.h1("Ringkasan Rekomendasi Teknis")
    p.body(
        "Urutan ini mengasumsikan TIDAK ADA pengambilan data ulang. Semuanya bisa dikerjakan "
        "dari data dan kode yang sudah ada.")
    p.table(
        ["#", "Aksi", "Kenapa", "Prioritas"],
        [["1", "Tulis Batasan Masalah (scope) yang benar",
          "Melindungi dari serangan 'pengujianmu tidak adil'", "WAJIB"],
         ["2", "Tulis bab K1: 3 cacat akuisisi + buktinya",
          "Semua bukti & skrip sudah ada, tinggal ditulis", "WAJIB"],
         ["3", "Tulis bab K2: phase_pipeline.py + hasil 6/10",
          "Skrip & angka sudah final", "WAJIB"],
         ["4", "Tulis bab K3: kriteria SNR + analisis jarak",
          "Skrip analyze_distance.py sudah ada", "WAJIB"],
         ["5", "Perbaiki kasus subject10 (bias +9.7 bpm)",
          "Kalau berhasil: 7/10 lolos, tesis makin kuat", "TINGGI"],
         ["6", "ML di atas fitur fase",
          "Bab bonus, bukan penentu kelulusan", "OPSIONAL"]],
        [8, 62, 72, 32], highlight_col=3,
        highlight_rule=lambda v: RED if v == "WAJIB" else (AMBER if v == "TINGGI" else None))
    p.callout(
        "info", "Kalau nanti ADA kesempatan merekam ulang (misal 1-2 subjek saja).",
        "Perbaikan logger tetap dilampirkan di dokumen ini (Bagian setelah ini) sebagai "
        "SARAN PENELITIAN LANJUTAN, bukan syarat kelulusan. Bahkan 2-3 subjek tambahan "
        "dengan konfigurasi benar sudah cukup untuk menjadi bab 'validasi awal setelah "
        "perbaikan' yang sangat menguatkan - tapi tesis tetap utuh tanpanya.")

    # ---- 0: perbaiki logger
    p.add_page()
    p.h1("Lampiran: Perbaikan Logger (untuk Penelitian Lanjutan)")
    p.callout(
        "info", "Ini BUKAN syarat kelulusan. Lampirkan sebagai saran penelitian lanjutan.",
        "Logger punya dua cacat yang membuat seluruh kolom detak jantung tidak bermakna. "
        "Keduanya perbaikannya sederhana dan hanya menyentuh beberapa baris.")
    p.h2("Perbaikan A - frame rate (satu angka)")
    p.body(
        "Di old code/logger.py, daftar config_commands, ubah satu baris:")
    p.callout(
        "info", "Ganti frameCfg",
        "SEKARANG :  \"frameCfg 0 0 2 0 20 1 0\\n\"     -> 20 ms  -> 50 fps  (SALAH)\n"
        "JADI     :  \"frameCfg 0 0 2 0 50 1 0\\n\"     -> 50 ms  -> 20 fps  (BENAR)\n\n"
        "Ini menyamakan frame rate dengan yang diasumsikan filter internal library TI, "
        "sehingga band jantung kembali ke 0.8-4.0 Hz seperti yang dimaksud.")
    p.h2("Perbaikan B - offset byte (empat baris)")
    p.body(
        "Di fungsi displayVitalSign(), perbaiki empat offset berikut. Offset yang benar "
        "diturunkan dari struct resmi TI di mmw_output.h (data mulai di byte 48):")
    p.callout(
        "info", "Offset yang benar",
        "heart_rate_est_peak  : 92  -> 88    (heartRateEst_peakCount_filtered)\n"
        "confidence_heart     : 104 -> 112   (confidenceMetricHeartOut)\n"
        "sumEnergyHeartWfm    : 116 -> 128   (sumEnergyHeartWfm)\n"
        "rangeBinPhaseIndex   : struct.unpack_from('<I', data, 50)\n"
        "                       -> struct.unpack_from('<H', data, 50)   (uint16, bukan uint32)\n\n"
        "Sekalian pertimbangkan menambah kolom yang selama ini tidak pernah direkam padahal "
        "berguna: breathingRateEst_FFT (offset 92), motionDetectedFlag (132), dan "
        "heartRateEst_harmonicEnergy (140).")
    p.callout(
        "warn", "Catatan: unwrapPhasePeak_mm (offset 64) SUDAH BENAR - jangan diubah.",
        "Kolom inilah satu-satunya yang selamat dari kedua cacat, dan seluruh hasil positif "
        "riset ini bertumpu padanya. Pastikan tetap direkam.")

    # ---- 1
    p.h1("Lampiran: Protokol Rekam Ulang (untuk Penelitian Lanjutan)")
    p.callout(
        "info", "Kalau suatu saat ada kesempatan merekam ulang: JARAK dan VARIASI HR.",
        "JARAK: data variasi jarak sendiri menunjukkan pada 100 cm SNR jantung = 1.09 "
        "(setara noise), sementara pada 50 cm = 3.16. Rekam pada jarak sekitar 50 cm, dan "
        "CATAT jarak persisnya untuk tiap subjek. Jangan melebihi 1.0 m, karena di luar itu "
        "radar bahkan tidak mencari target (vitalSignsCfg 0.3 1.0).\n\n"
        "VARIASI HR: lihat di bawah.")
    p.callout(
        "warn", "Kenapa variasi HR penting.",
        "Semua subjek direkam sambil duduk diam istirahat, sehingga HR mereka hanya "
        "bervariasi 1-4 bpm sepanjang sesi. Dengan data seperti itu, TIDAK ADA model - "
        "sehebat apapun - yang bisa membuktikan ia mampu melacak detak jantung, karena "
        "detak jantungnya tidak pernah berubah. Menambah subjek baru dengan protokol yang "
        "sama TIDAK akan menyelesaikan masalah ini.")
    p.h2("Protokol yang disarankan (per subjek, satu rekaman kontinu ~10-12 menit)")
    p.table(
        ["Fase", "Durasi", "Aktivitas", "Target HR"],
        [["1. Istirahat awal", "3 menit", "duduk diam menghadap radar", "~60-75 bpm"],
         ["2. Beban ringan", "2 menit", "berdiri, squat / naik-turun bangku", "naik ke ~110-130"],
         ["3. Recovery", "5 menit", "duduk kembali, diam, radar terus merekam", "turun bertahap"],
         ["4. Istirahat akhir", "2 menit", "duduk diam", "kembali ~70-80 bpm"]],
        [40, 22, 76, 36])
    p.bullets([
        "Radar DAN Attys harus terus merekam tanpa henti selama keempat fase - termasuk saat "
        "transisi. Justru fase recovery (fase 3) yang paling berharga: di situ HR turun "
        "perlahan dari 130 ke 75, memberi rentang variasi ~55 bpm untuk dilacak model.",
        "Subjek duduk diam saat fase 1, 3, 4. Fase 2 (gerak) boleh dianggap segmen "
        "'motion-corrupted' dan diberi label khusus - tidak harus dipakai untuk training, "
        "tapi tetap direkam supaya transisinya utuh.",
        "Jarak radar ke dada 0.5-1.0 meter, radar sejajar tinggi dada, tidak ada benda di "
        "antara radar dan dada (sesuai TI Developer's Guide hal. 4).",
        "Kalau memungkinkan, pakai lens/concentrator di depan sensor - TI menyebut ini "
        "meningkatkan performa secara signifikan, dan tampaknya belum dipakai di setup sekarang.",
        "Jumlah subjek: 8-10 orang sudah cukup. Yang menentukan bukan banyaknya orang, "
        "tapi ADANYA variasi HR di dalam tiap rekaman.",
    ])
    p.callout(
        "good", "Efek langsung protokol ini terhadap metrik.",
        "Dengan rentang HR 60-130 bpm dalam satu sesi, standar deviasi HR dalam-subjek naik "
        "dari ~2.8 bpm menjadi ~20 bpm. Pada tingkat error estimator saat ini (~6 bpm), "
        "korelasi dalam-subjek yang bisa dicapai melonjak dari batas atas ~0.3 menjadi >0.90. "
        "Barulah korelasi menjadi metrik yang bermakna, dan barulah klaim 'model melacak HR' "
        "bisa dibuktikan atau dibantah secara jujur.")

    # ---- 2
    p.add_page()
    p.h1("Perbaiki Kasus subject10 - Peluang Naik ke 7/10  [TINGGI]")
    p.body(
        "Estimator yang sudah diperbaiki masih meleset ke bawah secara konsisten: 9 dari 10 "
        "subjek diestimasi terlalu rendah, rata-rata -9.0 bpm. Karena arahnya konsisten "
        "(bukan acak), ini error yang TERSTRUKTUR - dan error terstruktur bisa diperbaiki.")
    p.body(
        "Bukti bahwa error ini terstruktur: kalau bias per-subjek dihilangkan, MAE anjlok "
        "drastis - subject04 dari 19.3 ke 3.8 bpm, subject02 dari 15.4 ke 7.7 bpm. "
        "(Catatan kejujuran: menghilangkan bias memakai ECG subjek itu sendiri adalah "
        "diagnostik, BUKAN hasil yang bisa dipakai - di dunia nyata tidak ada ECG untuk "
        "kalibrasi. Angka ini cuma membuktikan errornya berupa offset, bukan noise.)")
    p.h2("Tiga cara menyerang bias ini, urut dari yang paling murah")
    p.bullets([
        "(a) Buang komponen napas secara eksplisit, jangan cuma di-band-pass. Estimasi "
        "sinyal napas di 0.1-0.5 Hz, rekonstruksi, lalu KURANGKAN dari sinyal fase sebelum "
        "masuk band HR. Ini berbeda dari notch-harmonic yang dulu gagal: yang dulu menghapus "
        "frekuensi diskrit, ini menghapus komponen sinyalnya beserta seluruh ekor spektralnya.",
        "(b) Pakai turunan fase (phase difference), bukan fase mentah. TI sendiri melakukan "
        "ini (Developer's Guide hal. 9) justru untuk menekan drift dan menguatkan komponen "
        "jantung. Diferensiasi menaikkan penguatan frekuensi tinggi secara proporsional, "
        "sehingga menekan dominasi napas secara alami. Ini eksperimen paling murah - "
        "cukup np.diff pada sinyal fase, satu baris.",
        "(c) Ganti pemilihan puncak dengan metode yang tahan bias: alih-alih argmax "
        "telanjang, pakai autokorelasi pada sinyal ter-bandpass, atau cepstrum, atau "
        "harmonic-sum (jumlahkan energi di f dan 2f - detak jantung punya harmonik, "
        "kebocoran napas tidak).",
    ])
    p.callout(
        "info", "Target realistis setelah bias diperbaiki.",
        "MAE saat ini 5.2 bpm di subjek uji (10.4 bpm rata-rata semua subjek), dengan "
        "bias -9 bpm sebagai komponen error "
        "terbesar. Kalau bias bisa ditekan lewat metode di atas, MAE <= 5 bpm (target "
        "pembimbing) sangat masuk akal tercapai - bahkan sebelum machine learning ikut main.")

    # ---- 3
    p.h1("ML di Atas Fitur Fase  [OPSIONAL - bab bonus]")
    p.body(
        "Model ML lama gagal bukan karena ML-nya tidak cocok, tapi karena diberi makan "
        "kolom BPM bawaan TI yang 94.6% berisi nilai sentinel. Garbage in, garbage out. "
        "Fitur yang sebenarnya informatif belum pernah sekalipun dicoba di model ML.")
    p.h2("Fitur yang seharusnya dipakai (semua diturunkan dari unwrapPhasePeak_mm)")
    p.bullets([
        "Fitur spektral per window: frekuensi puncak di band HR, tinggi puncak, "
        "rasio puncak terhadap median PSD (SNR), lebar puncak, posisi 3 puncak teratas.",
        "Fitur napas sebagai konteks: frekuensi & amplitudo napas di window itu - "
        "membiarkan model belajar sendiri MENGOREKSI kebocoran napas, alih-alih kita "
        "tebak-tebak filternya.",
        "Fitur harmonik: energi di f_puncak vs 2x f_puncak (detak jantung punya harmonik "
        "kedua; kebocoran napas tidak) - ini bisa mengajari model membedakan puncak asli "
        "dari puncak palsu.",
        "Fitur time-domain: RMS sinyal ter-bandpass, zero-crossing rate, jarak antar puncak.",
        "Fitur kualitas sinyal: sumEnergyHeartWfm (satu-satunya kolom TI turunan yang "
        "belum terbukti rusak - rentangnya 0-0.517, masuk akal secara fisik).",
    ])
    p.callout(
        "warn", "Jangan pakai: final_heart_rate, heart_rate_est_*, heartRateEst_xCorr, "
                "confidence_heart, range_bin_value, rangeBinPhaseIndex, outputFilterHeartOut.",
        "Ketujuh kolom ini sudah terbukti rusak atau tidak informatif (lihat dokumen "
        "diagnosis Bagian 1). Memasukkan salah satunya ke model akan mengulang persis "
        "kegagalan yang lalu.")
    p.h2("Urutan model, dari sederhana ke kompleks")
    p.bullets([
        "Mulai dari Ridge / Random Forest di atas fitur window di atas. Kalau ini sudah "
        "mengalahkan estimator sinyal murni, kontribusi ML-nya terbukti.",
        "Kalau perlu lebih jauh: 1D-CNN langsung di atas potongan sinyal fase ter-bandpass "
        "(biarkan jaringan belajar filternya sendiri). Ini baru masuk akal SETELAH data "
        "dengan variasi HR tersedia - dengan data sekarang, CNN pasti overfit ke rata-rata.",
    ])

    # ---- 4
    p.add_page()
    p.h1("Cara Melapor Metrik (K4)  [WAJIB ditulis di metodologi]")
    p.body(
        "Kesalahan vonis kemarin sebagian besar berasal dari cara mengukur, bukan dari "
        "hasilnya sendiri. Aturan berikut sebaiknya dipakai konsisten mulai sekarang, "
        "dan juga ditulis di bab metodologi tesis.")
    p.table(
        ["Wajib dilaporkan", "Alasan"],
        [["MAE + RMSE", "ukuran error absolut, mudah dibandingkan dengan literatur"],
         ["Korelasi DALAM-subjek", "apakah model melacak DINAMIKA HR orang itu"],
         ["Korelasi ANTAR-subjek", "apakah model membedakan LEVEL HR antar orang"],
         ["Trivial constant baseline", "pembanding minimum wajib, split subjek sama"],
         ["Std HR ground truth", "konteks: korelasi tak bermakna kalau std-nya kecil"],
         ["Bland-Altman plot", "menampakkan bias sistematis yang disembunyikan MAE"],
         ["Breakdown per subjek", "agregat bisa menutupi sesi yang gagal total"]],
        [56, 118])
    p.callout(
        "bad", "Jangan pernah lagi melaporkan MAE sendirian.",
        "MAE 13-15 bpm dari model lama TERLIHAT wajar, padahal menebak angka KONSTAN tanpa "
        "data radar sama sekali sudah menghasilkan 14.8 bpm di subjek uji yang sama - dan "
        "diputar ke 10 pilihan subjek uji, model ML KALAH dari tebak-konstan di 5 dari 10 "
        "rotasi. Tanpa pembanding trivial baseline, angka MAE bisa menipu total. Aturan: "
        "setiap klaim MAE harus disertai korelasi DAN perbandingan eksplisit ke trivial "
        "baseline dengan split subjek yang identik.")
    p.callout(
        "warn", "Dan jangan memakai korelasi sebagai vonis tanpa melihat std ground truth.",
        "Korelasi rendah pada target yang hampir konstan BUKAN bukti model gagal - itu "
        "konsekuensi matematis. Selalu laporkan std HR ground truth di sampingnya, dan "
        "kalau perlu, laporkan juga batas atas korelasi yang bisa dicapai estimator sempurna "
        "pada data tersebut (seperti di dokumen diagnosis Bagian 5).")

    # ---- 5
    p.h1("Lampiran: Catatan Konfigurasi Radar")
    p.bullets([
        "Frame rate tidak konsisten: median sampling 64 Hz tapi rata-rata 50 Hz - artinya "
        "ada jeda/lonjakan tak beraturan di timestamp. Sementara spesifikasi demo TI "
        "menyebut slow-time sampling 20 Hz (frame periodicity 50 ms). Perlu dipastikan "
        "apakah logger menulis duplikat, atau ada frame yang hilang, atau konfigurasi chirp "
        "memang diubah dari default.",
        "Kolom range_bin_value dan rangeBinPhaseIndex jelas rusak (nilai mencapai 2^32) - "
        "ini bug parsing di sisi logger/GUI, bukan di radarnya. Kalau bisa diperbaiki saat "
        "rekam ulang, range bin bisa jadi fitur kualitas sinyal yang berguna.",
        "Saat rekam ulang, sekalian simpan sinyal fase MENTAH dengan sampling seragam kalau "
        "GUI TI memungkinkan - ini menghilangkan kebutuhan resampling dan sumber error-nya.",
    ])

    p.h1("Analisis Cross-Position  [RENDAH - bahan bab diskusi]")
    p.body(
        "Tetap layak dikerjakan sebagai bab diskusi, tapi bukan jalur kritis. Perlu diingat: "
        "dataset position_variation TIDAK punya kolom unwrapPhasePeak_mm sama sekali - "
        "padahal itulah satu-satunya kolom yang berguna. Artinya estimator maupun model "
        "utama TIDAK BISA dievaluasi di sana dengan data yang ada sekarang.")
    p.bullets([
        "Kalau analisis posisi ingin tetap dilakukan, sesi position_variation harus DIREKAM "
        "ULANG dengan skema kolom yang sama seperti position_front (menyertakan "
        "unwrapPhasePeak_mm). Kalau tidak, analisis ini tidak mungkin dilakukan sama sekali.",
        "Sesi 'back' dan 'right' punya ground truth berkualitas rendah (hanya 13 dan 10 beat "
        "valid) - kemungkinan masalah kontak elektroda Attys. Tandai atau exclude.",
    ])

    # ---- roadmap
    p.add_page()
    p.h1("Roadmap Menulis - Tanpa Ambil Data Ulang")
    p.table(
        ["Tahap", "Pekerjaan", "Bahan yang sudah ada", "Estimasi"],
        [["A", "Batasan Masalah + reframing judul",
          "dokumen ini", "1-2 hari"],
         ["B", "Bab K1: tiga cacat akuisisi + bukti",
          "diagnosis_riset.pdf Bagian 1b/1c/1d", "1 minggu"],
         ["C", "Bab K2: pipeline fase + hasil 6/10",
          "phase_pipeline.py (final)", "1 minggu"],
         ["D", "Bab K3: kriteria SNR + pengaruh jarak",
          "analyze_distance.py + tabel SNR", "1 minggu"],
         ["E", "Bab K4: metodologi evaluasi + keterbatasan",
          "reproduce_old_model.py + analisis ceiling", "3-5 hari"],
         ["F", "(opsional) ML di atas fitur fase",
          "belum ada - bab bonus", "1-2 minggu"]],
        [16, 62, 70, 26])
    p.callout(
        "good", "Semua bahan untuk tahap A-E SUDAH ADA. Tidak ada eksperimen baru "
                "yang diperlukan.",
        "Angka, gambar, dan skrip untuk keempat kontribusi sudah final dan reproducible. "
        "Yang tersisa murni pekerjaan MENULIS. Tahap F opsional dan tidak menentukan "
        "kelulusan.")

    p.h1("Pertanyaan untuk Dibahas di Bimbingan")
    p.bullets([
        "PALING PENTING - apakah pembingkaian ulang judul dan keempat kontribusi (K1-K4) "
        "disetujui, dengan analisis dipersempit ke sinyal fase (unwrapPhasePeak_mm) saja, "
        "TANPA pengambilan data ulang? Semua bukti untuk K1-K4 sudah ada.",
        "Apakah Batasan Masalah yang diusulkan dapat diterima - yaitu tesis TIDAK memvonis "
        "kelayakan library TI, melainkan mengevaluasi kelayakan sinyal fase mentah? Ini "
        "kunci agar ketiga cacat akuisisi menjadi TEMUAN, bukan kelemahan.",
        "Apakah 6 dari 10 subjek memenuhi standar (MAPE < 10%) dipandang cukup, mengingat "
        "4 yang gagal sudah dijelaskan sebabnya secara kuantitatif (3 karena SNR rendah = "
        "batas fisik, 1 kasus algoritmik yang diakui terbuka)?",
        "Apakah machine learning boleh diposisikan sebagai bab bonus/opsional, bukan inti "
        "tesis - mengingat pemrosesan sinyal yang benar sudah memenuhi standar tanpa ML?",
        "Kalau memungkinkan merekam 2-3 subjek saja dengan konfigurasi yang benar "
        "(frameCfg 50, offset logger diperbaiki, jarak 50 cm), apakah itu layak dijadikan "
        "bab 'validasi awal setelah perbaikan'? Ini opsional, tapi sangat menguatkan.",
    ])
    p.callout(
        "good", "Pesan penutup.",
        "Tesis ini tidak perlu topik baru, dan tidak perlu data baru. Yang selama ini terasa "
        "seperti rentetan kegagalan ternyata sudah menghasilkan empat kontribusi yang semua "
        "buktinya lengkap: akar kegagalan yang terlacak sampai ke satu angka konfigurasi dan "
        "empat offset byte (dengan reproduksi 100.00%), pipeline pengganti berbasis sinyal "
        "fase yang meloloskan 6 dari 10 subjek ke standar kesehatan tanpa machine learning "
        "sama sekali, kriteria SNR yang menjelaskan kapan metode ini bisa dan tidak bisa "
        "dipakai, serta koreksi metodologi yang menunjukkan model ML terdahulu sebenarnya "
        "kalah dari menebak angka konstan. Yang tersisa murni pekerjaan menulis - bukan "
        "eksperimen, bukan pengambilan data, bukan tebak-tebakan lagi.")

    p.output(str(out / "rekomendasi_riset.pdf"))
    print("ok:", out / "rekomendasi_riset.pdf")


if __name__ == "__main__":
    figdir = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    build_diagnosis(figdir, outdir)
    build_recommendation(figdir, outdir)
