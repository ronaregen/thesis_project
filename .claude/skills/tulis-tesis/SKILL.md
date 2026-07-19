---
name: tulis-tesis
description: Menulis atau menyunting naskah tesis Magister Rekayasa Elektro UII dengan bahasa Indonesia yang manusiawi (istilah asing dipertahankan bila padanan Indonesianya janggal, ditulis italic), serta menegakkan standar Template-Laporan-Tesis-MRE (penomoran & rujukan gambar/tabel/persamaan, caption, sitasi IEEE, struktur bab). Pakai skill ini setiap kali user minta menulis, menyunting, atau mereview bab/subbab/abstrak/paragraf tesis, kalimat untuk sidang, atau teks apa pun yang akan masuk ke naskah tesis.
---

# Tulis Tesis (MRE UII)

Tujuan: naskah yang **enak dibaca manusia**, bukan hasil terjemahan kaku, tapi tetap patuh
template. Dua aturan itu tidak bertabrakan — template mengatur format, skill ini mengatur rasa
bahasa.

Template acuan: `thesis/template/Template-Laporan-Tesis-MRE.pdf`

---

## 1. Bahasa: pakai istilah asing kalau padanannya janggal

Prinsip: **pilih kata yang dipakai insinyur waktu ngobrol.** Kalau padanan Indonesia bikin
pembaca berhenti sejenak untuk menerjemahkan balik ke istilah aslinya, padanan itu gagal.

**PAKAI istilah asing (jangan diterjemahkan):**

| Pakai ini | JANGAN |
|---|---|
| *noise* | derau |
| *Low Pass Filter*, *bandpass*, *cutoff* | tapis lolos bawah, tapis lolos rentang |
| *sampling rate*, *sample rate* | laju cuplik |
| *frame rate* | laju bingkai |
| *baseline* | garis dasar |
| *ground truth* | kebenaran dasar |
| *aliasing*, *anti-alias* | pelipatan, gejala alias |
| *windowing*, *window* (jendela analisis) | ~~jendela~~ (ambigu) |
| *phase unwrapping* | pembukaan lilitan fase |
| *peak*, *peak detection* | puncak (boleh, kalau konteks jelas) |
| *outlier* | pencilan |
| *overfitting* | penyuaian berlebih |
| *training*, *testing*, *hold-out* | pelatihan, pengujian (boleh, tapi konsisten) |
| *dataset* | himpunan data |
| *baud rate*, *duty cycle*, *chirp*, *sweep* | — |
| *confidence metric* | metrik keyakinan |

**JANGAN italic dan JANGAN diasingkan** — kata ini sudah masuk KBBI, tulis biasa:
radar, sensor, data, filter, frekuensi, amplitudo, spektrum, sinyal, algoritma, energi,
akurasi, presisi, kalibrasi, resolusi, korelasi, variasi, deteksi, estimasi, analisis,
elektroda, subjek, sesi, standar, konfigurasi, koefisien, parameter, interval, protokol,
optimal, valid, komponen, modul, sistem.

Menulis *"sinyal"* dengan italic itu salah — sudah bahasa Indonesia.

### Format italic
- Istilah asing yang **belum** diserap KBBI → *italic*, contoh: *noise*, *ground truth*,
  *Low Pass Filter*.
- Cukup italic **pada kemunculan pertama** di tiap bab. Setelah itu tulis biasa supaya halaman
  tidak penuh miring. Kecuali istilah itu jarang muncul (< 3 kali), italic terus.
- Variabel matematis selalu *italic* (`f`, `N`, `fs`), vektor/matriks **bold**, angka tegak.
- Singkatan (FMCW, ECG, MAPE, PSD, SNR) **tidak** italic.
- Nama produk/merek (IWR1443BOOST, Attys, MATLAB, Python) tidak italic.

### Istilah campuran
Kalau frasa asing dipakai sebagai kata benda dalam kalimat Indonesia, jangan dipaksa jadi
kalimat Inggris:
- ✅ "Sinyal fase difilter menggunakan *bandpass* 0,8–2,0 Hz."
- ❌ "Sinyal fase di-*bandpass filtering* pada rentang 0,8–2,0 Hz."
- ✅ "Estimasi dihaluskan dengan *median filter* orde 9."

Imbuhan Indonesia + kata asing pakai tanda hubung: di-*resample*, me-*resample*, ter-*filter*.

### Nada tulisan
- Kalimat aktif kalau bisa. "Penelitian ini mengukur…" lebih baik dari "Dilakukan pengukuran…"
- Satu gagasan satu kalimat. Kalimat > 25 kata dipecah.
- Hindari frasa kosong: "adapun", "perlu diketahui bahwa", "sebagaimana kita ketahui",
  "pada dasarnya", "dalam rangka untuk".
- Angka desimal pakai **koma** (6,5% bukan 6.5%). Ribuan pakai titik (1.329x).
- Jangan pakai "kami"/"penulis" berlebihan — pakai "penelitian ini" sebagai subjek.

---

## 2. Gambar, Tabel, Persamaan — WAJIB dinomori dan dirujuk lebih dulu

Ini aturan paling sering dilanggar. Tegakkan tanpa kompromi.

### Urutan wajib
1. Paragraf **menyebut** nomornya dulu ("… ditunjukkan pada Gambar 4. 2.")
2. **Baru** gambar/tabel muncul, setelah paragraf itu.

Tidak boleh ada gambar/tabel yang nongol tanpa disebut di teks. Tidak boleh disebut setelah
objeknya muncul ("Gambar di atas…" ❌ — sebut nomornya, di depan).

### Penomoran
Format `<Bab>. <Urut>` — **ada spasi setelah titik**, ikut template: `Gambar 1. 1`,
`Tabel 2. 1`, `Gambar 4. 12`. Nomor urut ulang dari 1 di tiap bab.

### Caption gambar
- Letak: **di bawah** gambar
- Rata: **tengah** (align center)
- Huruf: *Sentence case* (huruf besar hanya di awal + nama diri)
- **Diakhiri titik**
- Gambar dari sumber lain **wajib** cantumkan sitasi

```
Gambar 4. 3. Perbandingan MAPE sepuluh subjek terhadap ambang ANSI/CTA-2065.
Gambar 2. 1. Blok diagram radar FMCW [4].
```

### Caption tabel
- Letak: **di atas** tabel
- Rata: **kiri** (align left)
- Huruf: *Title Case* / Capitalize Each Word
- **Tanpa titik** di akhir

```
Tabel 4. 1 Hasil Estimasi Detak Jantung per Subjek
```

Tabel tidak boleh terpotong antar halaman. Kalau panjang, pecah jadi beberapa tabel bernomor
sendiri atau pindahkan ke Lampiran.

### Persamaan
- Nomor di **margin kanan**, dalam kurung, format `(bab.urut)`
- Setelah persamaan, jelaskan variabelnya diawali baris `dengan,`
- Variabel di dalam dan di luar persamaan **harus ditulis sama persis**

```
                        MAPE = (100/n) Σ |y_i − ŷ_i| / y_i                    (3.2)

dengan,
    n adalah jumlah jendela analisis,
    y_i adalah detak jantung acuan dari ECG pada jendela ke-i (bpm),
    ŷ_i adalah hasil estimasi radar pada jendela ke-i (bpm).
```

### Daftar di halaman depan
Setiap gambar/tabel baru harus ikut masuk **Daftar Gambar** / **Daftar Tabel**. Kalau menambah
gambar di tengah bab, semua nomor sesudahnya bergeser — periksa ulang rujukan di teks.

### Lampiran
Bernomor huruf (LAMPIRAN A, B, …), **wajib diacu di badan tesis**, dan tercantum di Daftar Isi.

---

## 3. Struktur & standar lain dari template

**Bab wajib:**
- BAB I PENDAHULUAN — 1.1 Latar Belakang, 1.2 Penelitian Sebelumnya
- BAB II TINJAUAN PUSTAKA
- BAB III METODOLOGI PENELITIAN
- BAB IV HASIL DAN PEMBAHASAN
- BAB V KESIMPULAN DAN PENELITIAN LANJUTAN — 5.1 Kesimpulan (poin-poin), 5.2 Penelitian Lanjutan
- Daftar Pustaka, Lampiran

**Paragraf:** paragraf pertama tiap bab/subbab pakai style `Paragraf_pertama` (tanpa indent),
paragraf berikutnya `Paragraf_lanjutan` (indent).

**Abstrak:** 300–800 kata, memuat latar belakang, rumusan masalah, metode, dan hasil.
Kata kunci maksimal 5, **huruf kecil semua**, dipisah koma.

**Glosarium:** singkatan/istilah khusus, urut abjad, format `SINGKATAN - Kepanjangan`.
Kalau sudah ada di Glosarium, **jangan** ditulis kepanjangannya lagi di badan tesis.

**Sitasi: gaya IEEE**, nomor dalam kurung siku `[1]`, urut kemunculan pertama.
Nama penulis boleh masuk kalimat: "Amrulloh [1] menjelaskan bahwa…"

**BAB II** jangan memuat teori dasar yang sudah umum diketahui lulusan Teknik Elektro.
Semua konsep wajib berreferensi.

---

## Checklist sebelum menyerahkan tulisan

- [ ] Setiap gambar/tabel punya nomor `Bab. Urut` dan **disebut di paragraf sebelum muncul**
- [ ] Caption gambar: bawah, center, sentence case, **pakai titik**
- [ ] Caption tabel: atas, kiri, Title Case, **tanpa titik**
- [ ] Gambar/tabel dari sumber lain sudah disitasi IEEE
- [ ] Persamaan bernomor kanan, variabel dijelaskan setelah kata `dengan,`
- [ ] Istilah asing yang belum diserap sudah *italic* (kemunculan pertama)
- [ ] Kata yang sudah masuk KBBI **tidak** di-italic
- [ ] Desimal pakai koma
- [ ] Tidak ada kalimat > 25 kata dan tidak ada frasa kosong
- [ ] Daftar Gambar / Daftar Tabel / Daftar Isi sudah diperbarui
- [ ] Angka hasil disertai pembanding *trivial baseline* dan split-nya disebut
