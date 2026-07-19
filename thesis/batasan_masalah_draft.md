# Draft — Batasan Masalah & Ruang Lingkup Penelitian

> **Cara pakai.** Ini draft untuk ditempel di Bab 1 (Pendahuluan), sub-bab
> Batasan Masalah, dan diringkas jadi 1 slide pembuka di sidang.
>
> **Kenapa ini WAJIB ada di depan, bukan di bab Kesimpulan:** kalimat-kalimat di
> bawah inilah yang mengubah konfigurasi 50 fps dari *cacat tersembunyi* menjadi
> *kondisi eksperimen yang dinyatakan*. Setelah ini tertulis, penguji tidak bisa
> mematahkan penelitian dengan argumen "pengujian Anda terhadap TI tidak adil" —
> karena penelitian ini memang **tidak menguji TI**.
>
> Sebaliknya, kalau ini TIDAK ditulis, tabel yang menunjukkan `final_heart_rate`
> gagal 0/10 (MAPE ~91%) otomatis terbaca sebagai vonis atas library TI — dan
> vonis itu memang tidak sah pada konfigurasi ini.

---

## 1.x Batasan Masalah

Penelitian ini dibatasi pada hal-hal berikut.

**(1) Objek yang dievaluasi adalah SINYAL FASE MENTAH, bukan library Vital Signs
bawaan TI.**

Yang diteliti adalah kelayakan sinyal `unwrapPhasePeak_mm` — keluaran fase mentah
radar yang diekstrak **sebelum** rantai pemrosesan detak jantung milik TI — sebagai
basis estimasi detak jantung. Kolom-kolom BPM turunan TI (`final_heart_rate`,
`heart_rate_est_fft`, `heartRateEst_xCorr`, dan sejenisnya) **disertakan semata
sebagai pembanding** pada kondisi akuisisi yang sama, agar perbandingan dilakukan
pada jendela waktu yang identik.

**(2) Konfigurasi akuisisi: frame rate 50 fps.**

Seluruh pengambilan data dilakukan pada frame rate 50 fps (periode frame 20 ms).
Laju ini memberikan resolusi waktu yang lebih halus untuk *phase unwrapping*,
sehingga **menguntungkan** pendekatan berbasis fase yang menjadi fokus penelitian
ini: jumlah sampel per jendela analisis lebih banyak dan rekonstruksi fase lebih
andal.

Konsekuensi yang perlu dinyatakan: konfigurasi referensi TI untuk demo Vital Signs
adalah 20 fps. Library tersebut menggunakan filter IIR dengan **koefisien tetap**
yang dirancang untuk laju cuplik 20 Hz. Ketika dijalankan pada 50 fps, seluruh
respons frekuensi filter tersebut bergeser dengan faktor 2,5×, sehingga pita yang
dimaksudkan untuk detak jantung tidak lagi berada pada rentang detak jantung
manusia.

**Karena itu, hasil kolom BPM turunan TI dalam penelitian ini TIDAK BOLEH dibaca
sebagai penilaian atas akurasi library Vital Signs TI.** Penelitian ini tidak
memberikan vonis kelayakan terhadap library tersebut. Pengujian pada konfigurasi
20 fps direkomendasikan sebagai penelitian lanjutan untuk menilai library TI
secara adil (lihat Bab 5, Saran).

**(3) Kondisi subjek: istirahat, detak jantung praktis tidak berubah.**

Seluruh subjek direkam dalam keadaan duduk diam dan beristirahat. Simpangan baku
detak jantung dalam satu sesi hanya 0,9–4,2 bpm. Akibatnya, penelitian ini
mengevaluasi ketepatan estimasi **detak jantung rata-rata per jendela**, dan
**tidak** dapat membuktikan maupun membantah kemampuan metode dalam **melacak
perubahan** detak jantung. Klaim penelitian ini terbatas pada kondisi istirahat.

Konsekuensi lanjutan: jendela analisis 40 detik yang dipakai memberikan resolusi
frekuensi yang lebih halus, namun membuat respons terhadap perubahan detak jantung
menjadi lambat. Pada dataset ini keterbatasan tersebut tidak dapat diuji, karena
detak jantung subjek memang tidak berubah.

**(4) Estimasi yang dievaluasi adalah detak jantung rata-rata, bukan HRV.**

Variabilitas detak jantung (*heart rate variability*) telah diuji dan **tidak
layak** diperoleh dari sinyal ini: deteksi detak-per-detak hanya mencapai 61%
*recall* terhadap R-peak ECG, dan RMSSD hasil radar mencapai 4× nilai ECG. Dua
batas yang menyebabkannya bersifat fisik, bukan algoritmik (SNR sinyal jantung dan
presisi waktu 20 ms/sampel). Hasil negatif ini dilaporkan sebagai bagian dari batas
keberlakuan, bukan disembunyikan.

**(5) Acuan kebenaran (ground truth): ECG Attys, 125 Hz.**

Detak jantung acuan diperoleh melalui deteksi R-peak metode Pan–Tompkins pada
sinyal ECG, bukan dari analisis spektral langsung.

**(6) Standar penilaian: ANSI/CTA-2065 (MAPE < 10%).**

Kriteria kelayakan yang dipakai adalah standar Consumer Technology Association
(2018) untuk *Physical Activity Monitoring for Heart Rate*, yaitu MAPE terhadap ECG
di bawah 10%. Kriteria ini dipilih karena merupakan standar yang diakui dan dipakai
luas dalam literatur validasi perangkat pemantau detak jantung — bukan ambang yang
ditetapkan sendiri oleh peneliti.

**(7) Batas fisik keberlakuan dinyatakan sebagai bagian dari hasil.**

Penelitian ini menetapkan kriteria kelayakan berbasis SNR sinyal jantung. Pada SNR
di bawah ~1,8, komponen detak jantung setara dengan derau, sehingga tidak ada
algoritma pemrosesan sinyal — termasuk *machine learning* — yang dapat
mengekstraknya. Sesi-sesi yang gagal memenuhi standar dilaporkan seluruhnya beserta
penjelasan fisik penyebabnya (jarak subjek dan keberadaan penghalang). **Tidak ada
subjek yang dibuang dari pelaporan.**

---

## Versi 1 slide (untuk sidang)

> **BATASAN MASALAH**
>
> - Yang dievaluasi: **sinyal fase mentah** (`unwrapPhasePeak_mm`), diambil
>   **sebelum** rantai filter TI.
> - Yang **TIDAK** dievaluasi: **library Vital Signs TI**. Kolom BPM turunannya
>   disertakan hanya sebagai pembanding pada kondisi akuisisi yang sama.
> - Akuisisi pada **50 fps** — menguntungkan pendekatan fase (resolusi waktu lebih
>   halus), namun menempatkan filter berkoefisien tetap milik TI (referensi 20 fps)
>   di luar rentang rancangannya. **Penilaian atas library TI berada di luar lingkup
>   penelitian ini**; pengujian 20 fps disarankan sebagai penelitian lanjutan.
> - Kondisi subjek: **istirahat** (std HR dalam sesi 0,9–4,2 bpm). Yang dievaluasi:
>   ketepatan detak rata-rata per jendela, **bukan** pelacakan perubahan detak.
> - Standar: **ANSI/CTA-2065, MAPE < 10%**.

---

## Catatan untuk diri sendiri — jangan ikut tercetak di tesis

**Kenapa jalur "sembunyikan 50 fps" itu lebih berbahaya, bukan lebih aman:**

1. **Angka 50 fps bukan rahasia.** Nama file konfigurasi bawaan TI adalah
   `profile_2d_VitalSigns_20fps.cfg` — angkanya ada di nama filenya, di repo publik
   TI. Siapa pun yang pernah memakai IWR1443 bisa menemukannya dalam hitungan menit.

2. **Menyarankan "coba 20 fps" tanpa menjelaskan sebabnya justru membocorkan.**
   Pertanyaan "kenapa 20 fps? kenapa spesifik?" pasti muncul, dan pada titik itu
   pilihan yang tersisa hanya: mengaku di tempat (terlihat seperti tertangkap), atau
   berbohong.

3. **MAPE 91% pada library komersial adalah angka yang MUSTAHIL diterima begitu
   saja.** Pertanyaan "kok bisa sejelek itu?" dijamin datang. Tanpa penjelasan 50
   fps, tidak ada jawaban. Dengan penjelasan itu, jawabannya justru menjadi
   kontribusi (analisis akar kegagalan sampai level source code).

4. **Menyatakan 50 fps TIDAK memaksa rekam ulang.** Rekam ulang hanya perlu jika
   penelitian ini mengklaim menilai library TI. Batasan Masalah di atas menyatakan
   secara eksplisit bahwa penelitian ini **tidak** mengklaim itu. Objek yang
   dievaluasi — sinyal fase mentah — **tidak terpengaruh** oleh konfigurasi tersebut,
   bahkan diuntungkan olehnya. Merekam ulang pada 20 fps justru akan **memperburuk**
   sinyal fase (cuplikan lebih jarang).

   Justru **menyembunyikannya** yang menciptakan risiko rekam ulang: bila penguji
   menemukan konfigurasi itu sementara tesis terbaca sebagai vonis atas TI, seluruh
   perbandingan menjadi tidak sah dan harus diulang.

**Kriteria pembimbing sudah terpenuhi:** "jika MAPE fase lebih kecil daripada MAPE
keluaran logger, tidak perlu ambil data ulang, itu menjadi kontribusi."
→ **6,5% vs 20,2%.** Terpenuhi. Bawa Batasan Masalah ini ke pembimbing lebih dulu,
sebelum ke siapa pun.
