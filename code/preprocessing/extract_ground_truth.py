"""
Ekstraksi ground truth heart rate dari sinyal ECG (Attys) via R-peak detection,
lalu alignment ke timestamp radar (IWR1443BOOST).

Mendukung dua mode:
1. SINGLE  — proses satu pasang file attys/radar
2. BATCH   — proses semua pasang file di bawah data/raw/ secara otomatis,
             mengikuti konvensi struktur folder berikut:

    data/raw/
    ├── position_front/
    │   ├── subject01/attys.csv, radar.csv
    │   ├── subject02/attys.csv, radar.csv
    │   └── ...
    └── position_variation_subject01/
        ├── front/attys.csv, radar.csv
        ├── side/attys.csv, radar.csv
        └── side_left_back/attys.csv, radar.csv

   Mode batch otomatis mengisi kolom `dataset`, `subject_id`, `position` di
   output berdasarkan nama folder:
   - folder top-level `position_<X>`           -> dataset="position_<X>",
     subject_id diambil dari folder di bawahnya, position="<X>"
   - folder top-level `position_variation_<S>` -> dataset="position_variation",
     subject_id="<S>", position diambil dari folder di bawahnya

Pendekatan deteksi R-peak: Pan-Tompkins style
    1. Bandpass filter 5-15 Hz (isolasi energi kompleks QRS)
    2. Derivative (highlight slope tajam di QRS)
    3. Squaring (perkuat amplitudo, buang tanda negatif)
    4. Moving window integration (~150ms)
    5. Peak detection dengan batas fisiologis (40-180 bpm -> jarak antar puncak)

Kenapa bukan FFT band-pass langsung di sinyal ECG mentah:
    Energi sinyal ECG yang relevan untuk deteksi detak (kompleks QRS) tersebar
    di frekuensi lebih tinggi (5-15 Hz), bukan di frekuensi detak jantung itu
    sendiri (0.7-3 Hz / 40-180 bpm). Sudah dicoba band-pass 0.7-3 Hz + ambil
    puncak PSD, hasilnya tidak stabil (90-180 bpm naik-turun drastis dalam
    waktu singkat, tidak masuk akal secara fisiologis).

Usage:
    # Mode single (satu pasang file)
    python extract_ground_truth.py single <attys_csv> <radar_csv> <output_csv>

    # Mode batch (semua data di data/raw/, hasil digabung ke satu file)
    python extract_ground_truth.py batch <raw_root_dir> <output_csv>

Contoh:
    python extract_ground_truth.py single \\
        ../../data/raw/position_front/subject01/attys.csv \\
        ../../data/raw/position_front/subject01/radar.csv \\
        ../../data/processed/aligned_position_front_subject01.csv

    python extract_ground_truth.py batch \\
        ../../data/raw \\
        ../../data/processed/aligned_all.csv
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


def detect_r_peaks(ecg_values: np.ndarray, timestamps: np.ndarray, fs: float):
    """Pan-Tompkins style R-peak detection.

    Returns:
        peak_timestamps: timestamp (unix epoch) tiap R-peak yang terdeteksi
    """
    sos_bp = signal.butter(3, [5, 15], btype="bandpass", fs=fs, output="sos")
    x_bp = signal.sosfiltfilt(sos_bp, ecg_values)

    x_deriv = np.diff(x_bp, prepend=x_bp[0])
    x_sq = x_deriv ** 2

    win_samples = max(1, int(0.15 * fs))
    x_int = np.convolve(x_sq, np.ones(win_samples) / win_samples, mode="same")

    min_dist = max(1, int(0.27 * fs))  # setara HR maksimum ~220 bpm
    threshold = np.mean(x_int) + 0.5 * np.std(x_int)
    peaks, _ = signal.find_peaks(x_int, distance=min_dist, height=threshold)

    return timestamps[peaks]


def rr_to_instantaneous_hr(peak_timestamps: np.ndarray, hr_min=40, hr_max=180):
    """Konversi timestamp R-peak jadi instantaneous HR, filter yang implausible."""
    rr_intervals = np.diff(peak_timestamps)
    inst_hr = 60.0 / rr_intervals
    hr_timestamps = peak_timestamps[1:]

    valid = (inst_hr >= hr_min) & (inst_hr <= hr_max)
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"    Membuang {n_dropped} beat implausible (di luar {hr_min}-{hr_max} bpm)")

    return hr_timestamps[valid], inst_hr[valid]


def process_pair(attys_path: Path, radar_path: Path) -> pd.DataFrame:
    """Proses satu pasang file attys/radar, return dataframe hasil alignment
    (belum ada kolom dataset/subject_id/position -- ditambahkan oleh caller)."""
    attys = pd.read_csv(attys_path)
    radar = pd.read_csv(radar_path)

    t = attys["timestamp"].values
    x = attys["value"].values
    fs = 1.0 / np.median(np.diff(t))
    print(f"    Attys: {len(t)} sampel, fs terdeteksi = {fs:.1f} Hz")

    peak_timestamps = detect_r_peaks(x, t, fs)
    print(f"    Terdeteksi {len(peak_timestamps)} R-peak")

    gt_t, gt_hr = rr_to_instantaneous_hr(peak_timestamps)
    print(f"    HR ground truth: {gt_hr.min():.1f}-{gt_hr.max():.1f} bpm, "
          f"rata-rata {gt_hr.mean():.1f} bpm ({len(gt_hr)} beat valid)")

    radar_t = radar["Timestamp"].values
    mask_overlap = (radar_t >= gt_t[0]) & (radar_t <= gt_t[-1])
    n_overlap = mask_overlap.sum()
    print(f"    Overlap dengan radar: {n_overlap}/{len(radar_t)} sampel "
          f"({100 * n_overlap / len(radar_t):.1f}%)")

    if n_overlap == 0:
        raise ValueError(
            "Tidak ada overlap waktu antara ground truth ECG dan data radar. "
            "Cek apakah kedua file berasal dari sesi rekaman yang sama."
        )

    gt_interp = np.interp(radar_t[mask_overlap], gt_t, gt_hr)

    aligned = radar[mask_overlap].copy()
    aligned["gt_heart_rate"] = gt_interp
    return aligned


def parse_dataset_subject_position(pair_dir: Path, raw_root: Path):
    """Infer dataset/subject_id/position dari path folder, mengikuti konvensi:
    - position_<X>/subjectNN/                 -> dataset=position_<X>, subject=subjectNN, position=<X>
    - position_variation_<S>/<position_name>/ -> dataset=position_variation, subject=<S>, position=<position_name>
    """
    rel_parts = pair_dir.relative_to(raw_root).parts
    if len(rel_parts) < 2:
        # fallback: file langsung di raw_root tanpa subfolder terstruktur
        return "unknown", pair_dir.parent.name, "unknown"

    top_folder, sub_folder = rel_parts[0], rel_parts[1]

    m = re.match(r"^position_variation_(.+)$", top_folder)
    if m:
        return "position_variation", m.group(1), sub_folder

    m = re.match(r"^position_(.+)$", top_folder)
    if m:
        return top_folder, sub_folder, m.group(1)

    # fallback generik kalau nama folder tidak mengikuti konvensi di atas
    return top_folder, sub_folder, "unknown"


def find_pairs(raw_root: Path):
    """Cari semua folder yang berisi attys.csv + radar.csv di bawah raw_root."""
    pairs = []
    for attys_path in sorted(raw_root.rglob("attys.csv")):
        radar_path = attys_path.parent / "radar.csv"
        if radar_path.exists():
            pairs.append((attys_path.parent, attys_path, radar_path))
        else:
            print(f"  [SKIP] {attys_path.parent} punya attys.csv tapi tidak ada radar.csv")
    return pairs


def run_single(attys_path: str, radar_path: str, output_path: str):
    aligned = process_pair(Path(attys_path), Path(radar_path))
    aligned.to_csv(output_path, index=False)
    print(f"Tersimpan: {output_path} ({aligned.shape[0]} baris, {aligned.shape[1]} kolom)")


def run_batch(raw_root: str, output_path: str):
    raw_root = Path(raw_root)
    pairs = find_pairs(raw_root)
    print(f"Ditemukan {len(pairs)} pasang attys/radar di bawah {raw_root}\n")

    all_frames = []
    failed = []
    for pair_dir, attys_path, radar_path in pairs:
        dataset, subject_id, position = parse_dataset_subject_position(pair_dir, raw_root)
        print(f"[{dataset} / {subject_id} / {position}] {pair_dir}")
        try:
            aligned = process_pair(attys_path, radar_path)
        except Exception as e:
            print(f"    [GAGAL] {e}\n")
            failed.append((pair_dir, str(e)))
            continue

        aligned.insert(0, "dataset", dataset)
        aligned.insert(1, "subject_id", subject_id)
        aligned.insert(2, "position", position)
        all_frames.append(aligned)
        print()

    if not all_frames:
        raise RuntimeError("Tidak ada satu pun pasangan file yang berhasil diproses.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(output_path, index=False)

    print("=" * 60)
    print(f"Berhasil: {len(all_frames)}/{len(pairs)} sesi")
    if failed:
        print(f"Gagal: {len(failed)} sesi:")
        for pair_dir, err in failed:
            print(f"  - {pair_dir}: {err}")
    print(f"\nTersimpan: {output_path} ({combined.shape[0]} baris total, {combined.shape[1]} kolom)")
    print("\nRingkasan per dataset/posisi:")
    print(combined.groupby(["dataset", "subject_id", "position"]).size().rename("n_samples"))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("single", "batch"):
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "single":
        if len(sys.argv) != 5:
            print(__doc__)
            sys.exit(1)
        run_single(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        if len(sys.argv) != 4:
            print(__doc__)
            sys.exit(1)
        run_batch(sys.argv[2], sys.argv[3])
