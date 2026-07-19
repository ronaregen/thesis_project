"""
Evaluasi seberapa reliable kolom-kolom HR bawaan TI Vital Sign library
dibandingkan ground truth ECG, pada dataset yang sudah di-align.

Kalau file input hasil dari mode batch (punya kolom dataset/subject_id/position),
breakdown ditampilkan per kombinasi tersebut, plus ringkasan keseluruhan.

Opsional: filter baris berdasarkan `confidence_heart` minimum, buat cek apakah
saat algoritma TI ngaku yakin (confidence tinggi), outputnya beneran lebih
akurat dibanding rata-rata.

Usage:
    python compare_baseline.py <aligned_csv> [min_confidence]

Contoh:
    python compare_baseline.py ../../data/processed/aligned_all.csv
    python compare_baseline.py ../../data/processed/aligned_all.csv 0.05
"""

import sys
import numpy as np
import pandas as pd

BASELINE_COLUMNS = [
    "final_heart_rate",
    "heart_rate_est_peak",
    "heart_rate_est_fft",
    "heart_rate_est_fft_4hz",
    "heartRateEst_xCorr",
]

GROUP_COLUMNS = ["dataset", "subject_id", "position"]


def evaluate_group(df: pd.DataFrame, label: str):
    gt = df["gt_heart_rate"].values
    print(f"\n=== {label} (n={len(df)}) ===")
    print(f"Ground truth HR: {gt.min():.1f}-{gt.max():.1f} bpm, rata-rata {gt.mean():.1f} bpm\n")
    print(f"{'Kolom':<25s} {'MAE (bpm)':>10s} {'Korelasi':>10s} {'% stuck di modus':>18s}")
    print("-" * 66)

    for col in BASELINE_COLUMNS:
        if col not in df.columns:
            continue
        values = df[col].values
        mae = np.mean(np.abs(values - gt))
        corr = np.corrcoef(values, gt)[0, 1] if np.std(values) > 0 else float("nan")
        modes = pd.Series(values).mode()
        if modes.empty:
            pct_stuck = float("nan")
        else:
            pct_stuck = 100 * np.mean(values == modes.iloc[0])
        print(f"{col:<25s} {mae:>10.1f} {corr:>10.3f} {pct_stuck:>17.1f}%")


def main(aligned_path: str, min_confidence: float = None):
    df = pd.read_csv(aligned_path)

    if min_confidence is not None:
        n_before = len(df)
        df = df[df["confidence_heart"] >= min_confidence]
        print(f"Filter confidence_heart >= {min_confidence}: {len(df)}/{n_before} "
              f"baris tersisa ({100 * len(df) / n_before:.2f}%)\n")
        if df.empty:
            print("Tidak ada baris tersisa setelah filter. Berhenti.")
            return

    has_group_cols = all(c in df.columns for c in GROUP_COLUMNS)

    if has_group_cols and df[GROUP_COLUMNS].drop_duplicates().shape[0] > 1:
        # breakdown per dataset/subject/position
        for keys, sub_df in df.groupby(GROUP_COLUMNS):
            label = " / ".join(str(k) for k in keys)
            evaluate_group(sub_df, label)
        evaluate_group(df, "SEMUA DATA DIGABUNG")
    else:
        evaluate_group(df, "Semua data")

    print("\nCatatan: korelasi mendekati 0 + % stuck tinggi = kolom kemungkinan besar")
    print("berisi sentinel/flag value saat algoritma gagal deteksi, bukan hasil valid.")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    conf = float(sys.argv[2]) if len(sys.argv) == 3 else None
    main(sys.argv[1], conf)
