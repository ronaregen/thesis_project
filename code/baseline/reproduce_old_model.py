"""
Rekonstruksi model ML lama buat dapet angka MAE yang SEBENARNYA (bukan tebakan).

Setup yang direkonstruksi (sesuai ingatan user + CLAUDE.md poin 8):
  - Fitur : heart_rate_est_fft, heart_rate_est_fft_4hz, heartRateEst_xCorr,
            heart_rate_est_peak, confidence_heart   <- kolom BPM bawaan TI
  - Split : 8 subjek train / 1 subjek validasi / 1 subjek uji (berbasis subjek)
  - Target: gt_heart_rate (dari R-peak ECG Attys)

Karena tipe regressor persisnya gak dicatat, dicoba beberapa model yang wajar
dipakai orang (Linear/Ridge/RF/GBM/MLP/kNN) supaya ketahuan RENTANG MAE-nya,
bukan cuma satu angka.

Tiap model dibandingin ke TRIVIAL BASELINE (prediksi konstan = mean gt training,
TANPA fitur radar sama sekali) di split yang SAMA -- ini pembanding wajib
(CLAUDE.md poin 8).

Juga dijalanin ulang untuk semua 10 pilihan subjek uji (leave-one-subject-out)
supaya ketahuan angkanya sensitif atau nggak terhadap subjek mana yang kebetulan
jadi test set.

Usage:
    python reproduce_old_model.py <aligned_csv>
"""

import sys
import warnings

# Radar log ~64 Hz, tapi kolom BPM bawaan TI cuma di-update ~1 Hz (window
# processing TI 16 detik, refresh tiap 1 detik) -- jadi baris berturut-turut
# nyaris duplikat. Ambil tiap baris ke-8 (~8 Hz): statistiknya sama, jalannya
# jauh lebih cepat. Set 1 kalau mau pakai semua baris.
ROW_STRIDE = 8

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TI_FEATURES = [
    "heart_rate_est_fft",
    "heart_rate_est_fft_4hz",
    "heartRateEst_xCorr",
    "heart_rate_est_peak",
    "confidence_heart",
]
TARGET = "gt_heart_rate"


def models():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "k-NN (k=10)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=0, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=0),
        "MLP (64,32)": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=60,
                         early_stopping=True, random_state=0)),
    }


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 else float("nan")


def evaluate_split(df, train_ids, val_id, test_id, verbose=True):
    tr = df[df.subject_id.isin(train_ids)]
    va = df[df.subject_id == val_id]
    te = df[df.subject_id == test_id]

    Xtr, ytr = tr[TI_FEATURES].values, tr[TARGET].values
    Xva, yva = va[TI_FEATURES].values, va[TARGET].values
    Xte, yte = te[TI_FEATURES].values, te[TARGET].values

    # TRIVIAL BASELINE: prediksi konstan = mean gt training
    const = ytr.mean()
    mae_triv_va = mean_absolute_error(yva, np.full_like(yva, const))
    mae_triv_te = mean_absolute_error(yte, np.full_like(yte, const))

    if verbose:
        print(f"\n{'='*82}")
        print(f"Train: {len(train_ids)} subjek ({train_ids[0]}..{train_ids[-1]}, n={len(tr)})"
              f" | Val: {val_id} (n={len(va)}) | Test: {test_id} (n={len(te)})")
        print(f"{'='*82}")
        print(f"Prediksi konstan trivial = {const:.1f} bpm (mean gt training, TANPA fitur radar)")
        print(f"HR sebenarnya - val: {yva.mean():.1f} bpm | test: {yte.mean():.1f} bpm\n")
        print(f"{'Model':<20s} {'MAE val':>8s} {'MAE test':>9s} {'r test':>8s} "
              f"{'vs trivial':>11s} {'Vonis':>10s}")
        print("-" * 82)
        print(f"{'TRIVIAL BASELINE':<20s} {mae_triv_va:>8.1f} {mae_triv_te:>9.1f} "
              f"{0.0:>8.3f} {'-':>11s} {'-':>10s}")

    results = {}
    for name, m in models().items():
        m.fit(Xtr, ytr)
        pv = m.predict(Xva)
        pt = m.predict(Xte)
        mae_va = mean_absolute_error(yva, pv)
        mae_te = mean_absolute_error(yte, pt)
        r_te = corr(pt, yte)
        delta = mae_te - mae_triv_te
        verdict = "KALAH" if delta >= 0 else "menang"
        results[name] = dict(mae_val=mae_va, mae_test=mae_te, r_test=r_te,
                             delta=delta, verdict=verdict)
        if verbose:
            print(f"{name:<20s} {mae_va:>8.1f} {mae_te:>9.1f} {r_te:>8.3f} "
                  f"{delta:>+10.1f} {verdict:>10s}")

    if verbose:
        print("-" * 82)
        print("'vs trivial' = MAE test model dikurangi MAE test trivial baseline.")
        print("Positif = model LEBIH BURUK daripada nebak angka konstan tanpa data radar.")

    return results, mae_triv_te


def main(path):
    df = pd.read_csv(path)
    df = df[df.dataset == "position_front"].copy()
    df = df.dropna(subset=TI_FEATURES + [TARGET])
    n_full = len(df)
    df = df.iloc[::ROW_STRIDE].reset_index(drop=True)

    subs = sorted(df.subject_id.unique())
    print(f"Dataset: position_front, {len(subs)} subjek, {len(df)} baris "
          f"(dari {n_full}, tiap baris ke-{ROW_STRIDE})")
    print(f"Fitur (kolom BPM bawaan TI): {', '.join(TI_FEATURES)}")

    # --- split persis seperti yang diingat user: train 1-8, val 9, test 10 ---
    main_res, main_triv = evaluate_split(df, subs[:8], subs[8], subs[9])

    # --- leave-one-subject-out: apakah angkanya sensitif ke pilihan test subject? ---
    print(f"\n\n{'='*82}")
    print("SENSITIVITAS: ulangi untuk SEMUA pilihan subjek uji (10 rotasi)")
    print("Tiap rotasi: 8 train / 1 val / 1 test, semua berbasis subjek")
    print(f"{'='*82}")

    agg = {name: [] for name in models()}
    triv_all = []
    for i in range(10):
        test_id = subs[i]
        val_id = subs[(i + 1) % 10]
        train_ids = [s for s in subs if s not in (test_id, val_id)]
        res, triv = evaluate_split(df, train_ids, val_id, test_id, verbose=False)
        triv_all.append(triv)
        for name, r in res.items():
            agg[name].append(r["mae_test"])

    print(f"\n{'Model':<20s} {'MAE test: mean':>15s} {'median':>8s} {'min':>7s} {'max':>7s} "
          f"{'kalah trivial':>14s}")
    print("-" * 82)
    triv_all = np.array(triv_all)
    print(f"{'TRIVIAL BASELINE':<20s} {triv_all.mean():>15.1f} "
          f"{np.median(triv_all):>8.1f} {triv_all.min():>7.1f} {triv_all.max():>7.1f} "
          f"{'-':>14s}")
    for name, maes in agg.items():
        maes = np.array(maes)
        n_lose = int(np.sum(maes >= triv_all))
        print(f"{name:<20s} {maes.mean():>15.1f} {np.median(maes):>8.1f} "
              f"{maes.min():>7.1f} {maes.max():>7.1f} {f'{n_lose}/10 rotasi':>14s}")

    print("\n" + "=" * 82)
    print("KESIMPULAN")
    print("=" * 82)
    best = min(main_res.items(), key=lambda kv: kv[1]["mae_test"])
    worst = max(main_res.items(), key=lambda kv: kv[1]["mae_test"])
    print(f"Di split asli (train 1-8 / val 09 / test 10):")
    print(f"  MAE test trivial baseline : {main_triv:.1f} bpm")
    print(f"  MAE test model terbaik    : {best[1]['mae_test']:.1f} bpm ({best[0]})")
    print(f"  MAE test model terburuk   : {worst[1]['mae_test']:.1f} bpm ({worst[0]})")
    print(f"  Rentang MAE semua model   : "
          f"{min(r['mae_test'] for r in main_res.values()):.1f} - "
          f"{max(r['mae_test'] for r in main_res.values()):.1f} bpm")
    n_lose_main = sum(1 for r in main_res.values() if r["verdict"] == "KALAH")
    print(f"  Model yang KALAH dari trivial baseline: {n_lose_main}/{len(main_res)}")
    print(f"  Korelasi test semua model : "
          f"{min(r['r_test'] for r in main_res.values()):+.3f} s.d. "
          f"{max(r['r_test'] for r in main_res.values()):+.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
