"""
Generate semua figure buat dokumen bimbingan (diagnosis + rekomendasi).

Usage:
    python make_figures.py <aligned_csv> <outdir>
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

C_BAD, C_GOOD, C_GT, C_NEU = "#c0392b", "#1e8449", "#2c3e50", "#7f8c8d"
FS = 25.0
HR_BAND = (0.8, 2.0)
OLD_BAND = (0.7, 3.3)


def resample_aa(t, x, fs=FS):
    o = np.argsort(t)
    t, x = t[o], x[o]
    fso = 1.0 / np.median(np.diff(t))
    x = signal.sosfiltfilt(signal.butter(6, fs / 2 * 0.9, "low", fs=fso, output="sos"), x)
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    return tu, np.interp(tu, t, x), o


def estimate(t, phase, gt, band=HR_BAND, fs=FS, win_sec=20.0, aa=True, med=True):
    """Pipeline FINAL (lihat code/evaluation/phase_pipeline.py): turunan fase +
    kurangi napas + band jantung. 6/10 subjek lolos MAPE < 10%."""
    o = np.argsort(t)
    t, phase, gt = t[o], phase[o], gt[o]
    if aa:
        _, xu, _ = resample_aa(t, phase, fs)
        tu = np.arange(t[0], t[-1], 1.0 / fs)
    else:
        tu = np.arange(t[0], t[-1], 1.0 / fs)
        xu = np.interp(tu, t, phase)
    gu = np.interp(tu, t, gt)
    d = np.diff(xu, prepend=xu[0])                       # turunan fase
    sos_br = signal.butter(4, (0.15, 0.6), "bandpass", fs=fs, output="sos")
    d = d - signal.sosfiltfilt(sos_br, signal.detrend(d))  # kurangi napas
    sig = signal.sosfiltfilt(signal.butter(4, band, "bandpass", fs=fs, output="sos"),
                             signal.detrend(d))
    w = int(win_sec * fs)
    st = max(1, int(w * 0.25))
    e, r, tt = [], [], []
    for s in range(0, len(sig) - w, st):
        seg = signal.detrend(sig[s:s + w])
        f, p = signal.welch(seg, fs=fs, nperseg=w)
        m = (f >= band[0]) & (f <= band[1])
        if not m.any():
            continue
        e.append(f[m][np.argmax(p[m])] * 60)
        r.append(gu[s:s + w].mean())
        tt.append(tu[s + w // 2] - tu[0])
    e, r, tt = np.array(e), np.array(r), np.array(tt)
    if med and len(e) >= 5:
        e = signal.medfilt(e, 5)
    return tt, e, r


def fig1_ti_broken(front, out):
    """TI output vs ground truth — output TI datar/stuck."""
    sub = front[front.subject_id == "subject01"].sort_values("Timestamp")
    t = sub.Timestamp.values - sub.Timestamp.values[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.1),
                                   gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(t, sub.gt_heart_rate, color=C_GT, lw=1.4, label="Ground truth ECG (Attys)")
    ax1.plot(t, sub.final_heart_rate, color=C_BAD, lw=1.0, alpha=.85,
             label="TI final_heart_rate")
    ax1.set_xlabel("waktu (detik)")
    ax1.set_ylabel("heart rate (bpm)")
    ax1.set_title("TI Vital Signs vs ECG — subject01", fontweight="bold", fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.set_ylim(0, 120)

    v = front.final_heart_rate.dropna()
    ax2.hist(v, bins=np.arange(0, 130, 2), color=C_BAD, alpha=.85)
    ax2.axvline(5.859375, color="k", ls="--", lw=1)
    ax2.annotate("5.859375 bpm\n= 94.6% sampel\n(nilai sentinel)", xy=(5.86, 0),
                 xytext=(30, 0.55), textcoords=("data", "axes fraction"), fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", lw=.9))
    ax2.set_xlabel("final_heart_rate (bpm)")
    ax2.set_ylabel("jumlah sampel")
    ax2.set_title("Distribusi output TI\n(10 subjek)", fontweight="bold", fontsize=10)
    fig.savefig(out / "fig1_ti_broken.png")
    plt.close(fig)


def fig2_signal_exists(front, out):
    """Tes desisif: rank frekuensi GT di PSD — sinyal HR ADA."""
    ranks, snrs = [], []
    for sid, sub in front.groupby("subject_id"):
        t = sub.Timestamp.values
        o = np.argsort(t)
        t = t[o]
        x = sub.unwrapPhasePeak_mm.values[o]
        g = sub.gt_heart_rate.values[o]
        tu, xu, _ = resample_aa(t, x)
        gu = np.interp(tu, t, g)
        w = int(16 * FS)
        for s in range(0, len(xu) - w, w // 2):
            seg = signal.detrend(xu[s:s + w])
            gt = gu[s:s + w].mean() / 60
            if not (HR_BAND[0] <= gt <= HR_BAND[1]):
                continue
            f, p = signal.welch(seg, fs=FS, nperseg=w)
            m = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
            fb, pb = f[m], p[m]
            gi = int(np.argmin(np.abs(fb - gt)))
            ranks.append((np.sum(pb > pb[gi])) / (len(fb) - 1))
            snrs.append(pb[gi] / np.median(pb))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.1))
    ax1.hist(ranks, bins=20, range=(0, 1), color=C_GOOD, alpha=.85)
    ax1.axhline(len(ranks) / 20, color=C_NEU, ls="--", lw=1.2,
                label="level acak (kalau sinyal HR tak ada)")
    ax1.axvline(np.median(ranks), color=C_BAD, lw=1.6,
                label=f"median = {np.median(ranks):.3f}")
    ax1.set_xlabel("rank ternormalisasi frekuensi GT di PSD\n(0 = jadi puncak tertinggi, 0.5 = acak)")
    ax1.set_ylabel("jumlah window")
    ax1.set_title("Frekuensi ECG muncul sbg puncak\ndi fase radar", fontweight="bold", fontsize=10)
    ax1.legend(fontsize=7)

    ax2.hist(np.clip(snrs, 0, 15), bins=30, color=C_GOOD, alpha=.85)
    ax2.axvline(1.0, color=C_NEU, ls="--", lw=1.2, label="SNR = 1 (setara noise)")
    ax2.axvline(np.median(snrs), color=C_BAD, lw=1.6, label=f"median = {np.median(snrs):.2f}")
    ax2.set_xlabel("SNR di frekuensi ground truth")
    ax2.set_ylabel("jumlah window")
    ax2.set_title("Kekuatan puncak HR\ndi frekuensi yang benar", fontweight="bold", fontsize=10)
    ax2.legend(fontsize=7)
    fig.savefig(out / "fig2_signal_exists.png")
    plt.close(fig)


def fig3_psd_band(front, out):
    """PSD contoh + histogram rasio est/GT: band lama sistematis meleset ke bawah."""
    # cari window contoh di mana band lama meleset tapi band baru benar
    best = None
    for sid, sub in front.groupby("subject_id"):
        t = sub.Timestamp.values
        x = sub.unwrapPhasePeak_mm.values
        g = sub.gt_heart_rate.values
        tu, xu, o = resample_aa(t, x)
        gu = np.interp(tu, np.sort(t), g[np.argsort(t)])
        w = int(16 * FS)
        for s in range(0, len(xu) - w, w):
            seg = signal.detrend(xu[s:s + w])
            gt = gu[s:s + w].mean()
            f, p = signal.welch(seg, fs=FS, nperseg=w)
            mo = (f >= OLD_BAND[0]) & (f <= OLD_BAND[1])
            mn = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
            fo = f[mo][np.argmax(p[mo])] * 60
            fn = f[mn][np.argmax(p[mn])] * 60
            if abs(fn - gt) < 3 and abs(fo - gt) > 12:
                score = abs(fo - gt)
                if best is None or score > best[0]:
                    best = (score, f, p, gt, fo, fn)
    _, f, p, gt, fo, fn = best

    # histogram rasio est/GT, band lama vs baru
    r_old, r_new = [], []
    for sid, sub in front.groupby("subject_id"):
        t = sub.Timestamp.values
        x = sub.unwrapPhasePeak_mm.values
        g = sub.gt_heart_rate.values
        tu, xu, _ = resample_aa(t, x)
        gu = np.interp(tu, np.sort(t), g[np.argsort(t)])
        w = int(16 * FS)
        for s in range(0, len(xu) - w, w // 2):
            seg = signal.detrend(xu[s:s + w])
            gt_w = gu[s:s + w].mean()
            ff, pp = signal.welch(seg, fs=FS, nperseg=w)
            mo = (ff >= OLD_BAND[0]) & (ff <= OLD_BAND[1])
            mn = (ff >= HR_BAND[0]) & (ff <= HR_BAND[1])
            r_old.append(ff[mo][np.argmax(pp[mo])] * 60 / gt_w)
            r_new.append(ff[mn][np.argmax(pp[mn])] * 60 / gt_w)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.3),
                                  gridspec_kw={"width_ratios": [1.45, 1]})
    ax.semilogy(f, p, color=C_GT, lw=1.3)
    ax.axvspan(OLD_BAND[0], OLD_BAND[1], color=C_BAD, alpha=.10)
    ax.axvspan(HR_BAND[0], HR_BAND[1], color=C_GOOD, alpha=.18)
    ax.axvline(gt / 60, color=C_GOOD, ls="--", lw=1.6, label=f"HR sebenarnya (ECG) = {gt:.0f} bpm")
    mo = (f >= OLD_BAND[0]) & (f <= OLD_BAND[1])
    mn = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    ax.plot(fo / 60, p[mo].max(), "v", color=C_BAD, ms=10,
            label=f"puncak band LAMA (0.7–3.3 Hz) = {fo:.0f} bpm  ✗")
    ax.plot(fn / 60, p[mn].max(), "^", color=C_GOOD, ms=10,
            label=f"puncak band BARU (0.8–2.0 Hz) = {fn:.0f} bpm  ✓")
    ax.text(0.04, 0.06, "ekor energi NAPAS bocor\nke tepi bawah band lama",
            transform=ax.transAxes, fontsize=7.5, color=C_BAD, fontweight="bold")
    ax.set_xlim(0, 3.6)
    ax.set_xlabel("frekuensi (Hz)")
    ax.set_ylabel("PSD fase (mm²/Hz)")
    ax.set_title("Contoh window: band lama salah pilih puncak", fontweight="bold", fontsize=9.5)
    ax.legend(fontsize=7, loc="upper right")

    bins = np.linspace(0.3, 2.0, 45)
    ax2.hist(r_old, bins=bins, color=C_BAD, alpha=.65,
             label=f"band LAMA 0.7–3.3 Hz (median {np.median(r_old):.2f}x)")
    ax2.hist(r_new, bins=bins, color=C_GOOD, alpha=.65,
             label=f"band BARU 0.8–2.0 Hz (median {np.median(r_new):.2f}x)")
    ax2.axvline(1.0, color="k", ls="--", lw=1.4)
    ax2.text(1.03, .90, "benar\n(1.0x)", transform=ax2.get_xaxis_transform(), fontsize=7)
    ax2.set_xlabel("rasio HR estimasi / HR sebenarnya")
    ax2.set_ylabel("jumlah window")
    ax2.set_title("Band baru geser ke arah benar, tapi\nBIAS KE BAWAH masih tersisa",
                  fontweight="bold", fontsize=9.5)
    ax2.legend(fontsize=6.5)
    fig.savefig(out / "fig3_psd_band.png")
    plt.close(fig)


def fig8_within_between(front, out):
    """Yang JUJUR: estimator benar antar-subjek, tapi HR dalam sesi nyaris tak bergerak."""
    res = {}
    for sid, sub in front.groupby("subject_id"):
        _, e, r = estimate(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                           sub.gt_heart_rate.values)
        res[sid] = (e, r)
    sids = sorted(res)
    me = np.array([res[s][0].mean() for s in sids])
    mg = np.array([res[s][1].mean() for s in sids])

    wi_e = np.concatenate([res[s][0] - res[s][0].mean() for s in sids])
    wi_g = np.concatenate([res[s][1] - res[s][1].mean() for s in sids])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax1.scatter(mg, me, s=70, color=C_GOOD, zorder=3, edgecolors="w", lw=1)
    for s, a, b in zip(sids, mg, me):
        ax1.annotate(s.replace("subject", "S"), (a, b), fontsize=6.5,
                     xytext=(4, -8), textcoords="offset points")
    lim = [55, 95]
    ax1.plot(lim, lim, "k--", lw=1, label="ideal")
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_xlabel("HR rata-rata sebenarnya, per subjek (bpm)")
    ax1.set_ylabel("HR rata-rata estimasi radar (bpm)")
    ax1.set_title(f"BERHASIL antar-subjek: r = {np.corrcoef(me, mg)[0,1]:.3f}\n"
                  "radar bisa bedakan orang ber-HR tinggi vs rendah",
                  fontweight="bold", fontsize=9, color=C_GOOD, pad=8)
    ax1.legend(fontsize=7, loc="upper left")

    ax2.scatter(wi_g, wi_e, s=8, alpha=.3, color=C_BAD, edgecolors="none")
    ax2.axhline(0, color="k", lw=.8)
    ax2.axvline(0, color="k", lw=.8)
    ax2.set_xlim(-12, 12)
    ax2.set_ylim(-30, 30)
    ax2.set_xlabel("variasi HR sebenarnya dalam sesi (bpm, mean dibuang)")
    ax2.set_ylabel("variasi HR estimasi (bpm)")
    ax2.set_title(f"GAGAL dalam-subjek: r = {np.corrcoef(wi_e, wi_g)[0,1]:.3f}\n"
                  f"tapi HR asli cuma bergerak ±{np.std(wi_g):.1f} bpm —\n"
                  "TIDAK ADA dinamika yang bisa dilacak",
                  fontweight="bold", fontsize=9, color=C_BAD, pad=8)
    fig.suptitle("Temuan paling penting: radar menangkap LEVEL HR, "
                 "tapi dataset tak punya DINAMIKA HR untuk dibuktikan",
                 fontweight="bold", fontsize=10.5, y=1.10)
    fig.savefig(out / "fig8_within_between.png")
    plt.close(fig)


def fig4_ablation(out):
    labels = ["Asli\n(band 0.7–3.3 Hz)", "+ band sempit\n0.8–2.0 Hz",
              "+ anti-alias\n+ 20 Hz", "+ detrend\n+ window 16 s",
              "+ median filter\n(final)"]
    corr = [0.141, 0.253, 0.252, 0.268, 0.365]
    mae = [15.6, 11.7, 11.8, 11.6, 10.5]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.2))
    cols = [C_BAD] + [C_NEU] * 3 + [C_GOOD]
    ax1.bar(x, corr, color=cols, alpha=.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.8)
    ax1.set_ylabel("korelasi vs ECG")
    ax1.set_title("Korelasi (makin tinggi makin baik)", fontweight="bold", fontsize=10)
    for i, v in enumerate(corr):
        ax1.text(i, v + .012, f"{v:.3f}", ha="center", fontsize=7.5, fontweight="bold")
    ax1.set_ylim(0, .44)

    ax2.bar(x, mae, color=cols, alpha=.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=6.8)
    ax2.set_ylabel("MAE (bpm)")
    ax2.set_title("MAE (makin rendah makin baik)", fontweight="bold", fontsize=10)
    for i, v in enumerate(mae):
        ax2.text(i, v + .25, f"{v:.1f}", ha="center", fontsize=7.5, fontweight="bold")
    ax2.set_ylim(0, 18)
    fig.suptitle("Ablasi: perbaikan mana yang paling berpengaruh (agregat 10 subjek)",
                 fontweight="bold", fontsize=10.5, y=1.04)
    fig.savefig(out / "fig4_ablation.png")
    plt.close(fig)


def fig5_ceiling(front, out):
    """Std GT kecil -> ceiling korelasi rendah, walau estimator sempurna.

    Basis dihitung pada GT ter-window (16 s) -- persis besaran yang dibandingkan
    estimator, supaya konsisten dengan angka MAE/korelasi di dokumen.
    """
    # Ceiling analitik (deterministik, bukan simulasi):
    # est = gt + noise, noise independen dgn std s
    #   -> corr(est, gt) = std(gt) / sqrt(std(gt)^2 + s^2)
    NOISE = 5.0
    sids, stds, ceils = [], [], []
    for sid, sub in front.groupby("subject_id"):
        t = sub.Timestamp.values
        o = np.argsort(t)
        tu = np.arange(t[o][0], t[o][-1], 1.0 / FS)
        gu = np.interp(tu, t[o], sub.gt_heart_rate.values[o])
        w = int(16 * FS)
        g = np.array([gu[s:s + w].mean() for s in range(0, len(gu) - w, w // 4)])
        sd = float(np.std(g))
        sids.append(sid.replace("subject", "S"))
        stds.append(sd)
        ceils.append(sd / np.sqrt(sd ** 2 + NOISE ** 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.3))
    x = np.arange(len(sids))
    ax1.bar(x, stds, color=C_NEU, alpha=.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sids, fontsize=7.5)
    ax1.set_ylabel("std HR dalam sesi (bpm)")
    ax1.set_ylim(0, 5.4)
    ax1.set_title("HR subjek nyaris tidak bergerak\n(duduk diam, istirahat)",
                  fontweight="bold", fontsize=10)
    for i, v in enumerate(stds):
        ax1.text(i, v + .1, f"{v:.1f}", ha="center", fontsize=7)

    ax2.bar(x, ceils, color=C_BAD, alpha=.9)
    ax2.axhline(0.5, color=C_GOOD, ls="--", lw=1.3, label="korelasi 0.5")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sids, fontsize=7.5)
    ax2.set_ylim(0, 0.92)
    ax2.set_ylabel("korelasi maksimum yang bisa dicapai")
    n_below = sum(1 for c in ceils if c < 0.5)
    ax2.set_title(f"Ceiling korelasi estimator SEMPURNA\n"
                  f"(error 5 bpm) — {n_below} dari 10 subjek < 0.5",
                  fontweight="bold", fontsize=10)
    ax2.legend(fontsize=7.5, loc="upper left")
    for i, v in enumerate(ceils):
        ax2.text(i, v + .015, f"{v:.2f}", ha="center", fontsize=7)
    fig.suptitle("Kenapa korelasi per-subjek menyesatkan di dataset ini",
                 fontweight="bold", fontsize=10.5, y=1.05)
    fig.savefig(out / "fig5_ceiling.png")
    plt.close(fig)


def _old_ml_predictions(front_all, test_id="subject10"):
    """Rekonstruksi model ML lama (fitur BPM TI) -> prediksi di subjek uji.

    Dipakai buat nunjukin secara visual bahwa model itu keluarannya praktis
    GARIS DATAR (regresi ke mean) -- bukti GIGO.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    feats = ["heart_rate_est_fft", "heart_rate_est_fft_4hz", "heartRateEst_xCorr",
             "heart_rate_est_peak", "confidence_heart"]
    d = front_all.dropna(subset=feats + ["gt_heart_rate"]).iloc[::8]
    subs = sorted(d.subject_id.unique())
    train = [s for s in subs if s not in (test_id, "subject09")]
    tr = d[d.subject_id.isin(train)]
    te = d[d.subject_id == test_id].sort_values("Timestamp")

    m = GradientBoostingRegressor(n_estimators=100, random_state=0)
    m.fit(tr[feats].values, tr.gt_heart_rate.values)
    pred = m.predict(te[feats].values)
    t = te.Timestamp.values - te.Timestamp.values[0]
    return t, pred, te.gt_heart_rate.values, float(tr.gt_heart_rate.mean())


def fig6_result(front, out):
    """Hasil di SUBJEK UJI (subject10) -- split asli user: 8 train / 1 val / 1 test."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2),
                             gridspec_kw={"width_ratios": [1.75, 1, 1.15]})

    # --- panel 1: time series subject10 (test) : GT vs model lama vs estimator baru
    sub = front[front.subject_id == "subject10"]
    tt, e, r = estimate(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                        sub.gt_heart_rate.values)
    t_ml, pred_ml, gt_ml, const = _old_ml_predictions(front)

    axes[0].plot(tt, r, color=C_GT, lw=2.0, label="Ground truth ECG", zorder=3)
    axes[0].plot(t_ml, pred_ml, color=C_BAD, lw=1.2, alpha=.9,
                 label="Model ML lama (fitur BPM TI)")
    axes[0].axhline(const, color=C_NEU, ls=":", lw=1.6,
                    label=f"Trivial: tebak konstan {const:.0f} bpm")
    axes[0].plot(tt, e, color=C_GOOD, lw=1.3, alpha=.95,
                 label="Estimator fase DIPERBAIKI")
    axes[0].set_xlabel("waktu (detik)")
    axes[0].set_ylabel("heart rate (bpm)")
    axes[0].set_title("subject10 = subjek UJI (tidak pernah dilatih)",
                      fontweight="bold", fontsize=9.5)
    axes[0].legend(fontsize=6.3, loc="upper right", ncol=2)
    axes[0].set_ylim(40, 100)

    # --- panel 2: bar MAE di subjek uji
    names = ["Model ML\nlama\n(fitur TI)", "Trivial\nbaseline", "Pipeline\nDIPERBAIKI",
             "Batas standar\nkesehatan\n(MAPE 10%)"]
    mae_ml = float(np.mean(np.abs(pred_ml - gt_ml)))
    mae_triv = float(np.mean(np.abs(gt_ml - const)))
    mae_est = float(np.mean(np.abs(e - r)))
    thr_std = 0.10 * float(np.mean(gt_ml))   # MAPE 10% dikonversi ke bpm di HR subjek ini
    vals = [mae_ml, mae_triv, mae_est, thr_std]
    cols = [C_BAD, C_NEU, C_GOOD, "#8e44ad"]
    axes[1].bar(np.arange(4), vals, color=cols, alpha=.9)
    axes[1].set_xticks(np.arange(4))
    axes[1].set_xticklabels(names, fontsize=6.3)
    axes[1].set_ylabel("MAE (bpm)")
    axes[1].set_title("MAE di subjek uji\n(subject10)", fontweight="bold", fontsize=9.5)
    for i, v in enumerate(vals):
        axes[1].text(i, v + .3, f"{v:.1f}", ha="center", fontsize=7.5, fontweight="bold")
    axes[1].set_ylim(0, 18)

    # --- panel 3: sensitivitas 10 rotasi -- model lama vs trivial
    # per-rotasi leave-one-subject-out, dihitung di reproduce_old_model.py
    # (Gradient Boosting = varian model lama yang PALING bagus dari 6 yang dicoba)
    rot = np.arange(1, 11)
    mae_gbm = [16.6, 5.1, 3.8, 4.0, 6.6, 4.7, 8.0, 4.5, 5.7, 12.4]
    mae_tri = [16.5, 4.5, 4.1, 3.1, 7.7, 4.5, 10.8, 4.1, 8.7, 14.1]
    w = .38
    axes[2].bar(rot - w / 2, mae_gbm, w, color=C_BAD, alpha=.9, label="Model ML lama")
    axes[2].bar(rot + w / 2, mae_tri, w, color=C_NEU, alpha=.9, label="Trivial baseline")
    axes[2].set_xticks(rot)
    axes[2].set_xticklabels([f"S{i:02d}" for i in rot], fontsize=6)
    axes[2].set_xlabel("subjek yang jadi test set")
    axes[2].set_ylabel("MAE (bpm)")
    axes[2].set_title("Model lama vs trivial, 10 rotasi:\nML KALAH di 5 dari 10 (lempar koin)",
                      fontweight="bold", fontsize=9.5)
    axes[2].legend(fontsize=6.5)
    fig.savefig(out / "fig6_result.png")
    plt.close(fig)


def fig7_persubject(front, out):
    """MAE per subjek estimator diperbaiki, tandai sesi lemah."""
    sids, maes = [], []
    for sid, sub in front.groupby("subject_id"):
        _, e, r = estimate(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                           sub.gt_heart_rate.values)
        sids.append(sid.replace("subject", "S"))
        maes.append(float(np.mean(np.abs(e - r))))
    fig, ax = plt.subplots(figsize=(9, 2.9))
    cols = [C_GOOD if m <= 7.8 else C_BAD for m in maes]
    ax.bar(np.arange(len(sids)), maes, color=cols, alpha=.9)
    ax.axhline(7.8, color="#8e44ad", ls="--", lw=1.6,
               label="batas standar ANSI/CTA-2065 (MAPE 10% = ~7.8 bpm @ HR rata-rata 78)")
    ax.set_xticks(np.arange(len(sids)))
    ax.set_xticklabels(sids)
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("MAE per subjek — pipeline diperbaiki (belum ada ML/kalibrasi bias)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7.5)
    for i, v in enumerate(maes):
        ax.text(i, v + .3, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_ylim(0, 22)
    fig.savefig(out / "fig7_persubject.png")
    plt.close(fig)


def fig9_standard(front, front_all, out):
    """Kepatuhan terhadap standar kesehatan ANSI/CTA-2065: MAPE < 10%."""
    sids, mape_est, mape_ti = [], [], []
    for sid, sub in front.groupby("subject_id"):
        _, e, r = estimate(sub.Timestamp.values, sub.unwrapPhasePeak_mm.values,
                           sub.gt_heart_rate.values)
        sids.append(sid.replace("subject", "S"))
        mape_est.append(100 * float(np.mean(np.abs(e - r) / r)))
        ti = front_all[front_all.subject_id == sid]
        v = ti["final_heart_rate"].dropna()
        g = ti.loc[v.index, "gt_heart_rate"]
        mape_ti.append(100 * float(np.mean(np.abs(v - g) / g)))

    x = np.arange(len(sids))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4),
                                   gridspec_kw={"width_ratios": [1.55, 1]})
    w = .38
    ax1.bar(x - w / 2, mape_ti, w, color=C_BAD, alpha=.9,
            label="TI Vital Signs bawaan")
    ax1.bar(x + w / 2, mape_est, w, color=C_GOOD, alpha=.9,
            label="Pipeline diperbaiki")
    ax1.axhline(10, color="#8e44ad", ls="--", lw=1.8,
                label="Batas standar ANSI/CTA-2065: MAPE 10%")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sids, fontsize=7.5)
    ax1.set_ylabel("MAPE (%)  -- makin kecil makin baik")
    ax1.set_yscale("log")
    ax1.set_ylim(2, 200)
    ax1.set_title("Kepatuhan terhadap standar kesehatan (skala log)",
                  fontweight="bold", fontsize=10)
    ax1.legend(fontsize=7, loc="upper right")
    for i, v in enumerate(mape_est):
        ax1.text(i + w / 2, v * 1.06, f"{v:.0f}", ha="center", fontsize=6.2,
                 color=C_GOOD, fontweight="bold")

    n_pass_est = sum(1 for v in mape_est if v < 10)
    n_pass_ti = sum(1 for v in mape_ti if v < 10)
    ax2.bar([0, 1], [n_pass_ti, n_pass_est], color=[C_BAD, C_GOOD], alpha=.9, width=.55)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["TI Vital Signs\nbawaan", "Pipeline\ndiperbaiki"], fontsize=8)
    ax2.set_ylabel("jumlah subjek LOLOS standar")
    ax2.set_ylim(0, 10.6)
    ax2.set_yticks(range(0, 11, 2))
    ax2.axhline(10, color=C_NEU, ls=":", lw=1.2)
    ax2.text(1.42, 10.15, "10 subjek", fontsize=6.5, color=C_NEU)
    ax2.set_title("Subjek yang memenuhi\nMAPE < 10%", fontweight="bold", fontsize=10)
    for i, v in enumerate([n_pass_ti, n_pass_est]):
        ax2.text(i, v + .2, f"{v}/10", ha="center", fontsize=11, fontweight="bold",
                 color=[C_BAD, C_GOOD][i])
    fig.suptitle("Vonis terhadap standar kesehatan: perangkat GAGAL dengan software bawaan, "
                 "LOLOS separuh setelah diperbaiki",
                 fontweight="bold", fontsize=10.5, y=1.04)
    fig.savefig(out / "fig9_standard.png")
    plt.close(fig)


def fig10_rootcause(front_all, out):
    """Root cause di level SOURCE CODE: logika beku di GUI TI."""
    sids, frozen_pred = [], []
    for sid, sub in front_all.groupby("subject_id"):
        c = sub.sort_values("Timestamp").confidence_heart.values
        cm, upd = 0.0, 0
        for v in c:
            cm = 0.5 * v + 0.5 * cm       # persis exp. average di GUI TI
            if cm > 0.4:                   # thresh_HeartCM
                upd += 1
        sids.append(sid.replace("subject", "S"))
        frozen_pred.append(100 * (1 - upd / len(c)))

    conf = front_all["confidence_heart"].dropna().values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.3))
    ax1.hist(np.clip(conf, 0, 1.0), bins=60, color=C_BAD, alpha=.85)
    ax1.axvline(0.4, color="#8e44ad", ls="--", lw=2,
                label="thresh_HeartCM = 0.4\n(ambang di kode TI)")
    ax1.axvline(np.median(conf), color=C_GT, lw=1.6,
                label=f"median confidence = {np.median(conf):.3f}")
    ax1.set_xlabel("confidence_heart")
    ax1.set_ylabel("jumlah sampel")
    ax1.set_title("Confidence nyaris SELALU di bawah ambang\n-> output HR tak pernah di-update",
                  fontweight="bold", fontsize=9.5)
    ax1.legend(fontsize=6.8)

    x = np.arange(len(sids))
    ax2.bar(x, frozen_pred, color=C_BAD, alpha=.9,
            label="Prediksi dari SOURCE CODE TI")
    ax2.axhline(94.6, color=C_GOOD, ls="--", lw=1.8,
                label="Terukur di data: 94.6% stuck")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sids, fontsize=7)
    ax2.set_ylim(80, 102)
    ax2.set_ylabel("% waktu output HR BEKU")
    ax2.set_title("Prediksi kode vs kenyataan data:\nCOCOK (94.3% vs 94.6%)",
                  fontweight="bold", fontsize=9.5)
    ax2.legend(fontsize=6.8, loc="lower right")
    fig.suptitle("Akar masalah ditemukan di source code TI, bukan sekadar dugaan",
                 fontweight="bold", fontsize=10.5, y=1.05)
    fig.savefig(out / "fig10_rootcause.png")
    plt.close(fig)


def fig11_filter_shift(front_all, out):
    """Bukti: filter jantung TI mulur 2.5x karena frame rate salah (50 fps vs 20 fps)."""
    d = front_all[front_all.subject_id == "subject01"].sort_values("Timestamp")
    t = d.Timestamp.values
    x = d.outputFilterHeartOut.values
    tu = np.arange(t[0], t[-1], 1 / 50.0)
    xu = np.interp(tu, t, x)
    f, p = signal.welch(xu, fs=50.0, nperseg=1024)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    ax1.semilogy(f, p, color=C_GT, lw=1.3)
    ax1.axvspan(0.8, 2.0, color=C_GOOD, alpha=.22)
    ax1.axvspan(2.0, 10.0, color=C_BAD, alpha=.12)
    ax1.text(1.4, 0.40, "band\njantung\nMANUSIA", transform=ax1.get_xaxis_transform(),
             ha="center", va="top", fontsize=7, color=C_GOOD, fontweight="bold")
    ax1.text(6.5, 0.97, "band yang SEBENARNYA dilewatkan filter TI\n"
                        "(0.8-4.0 Hz x 2.5 = 2.0-10.0 Hz)",
             transform=ax1.get_xaxis_transform(), ha="center", va="top",
             fontsize=7, color=C_BAD, fontweight="bold")
    ax1.set_xlim(0, 13)
    ax1.set_xlabel("frekuensi (Hz)")
    ax1.set_ylabel("PSD outputFilterHeartOut")
    ax1.set_title("'Waveform jantung' hasil filter TI - isinya bukan jantung",
                  fontweight="bold", fontsize=9.5)

    tot = np.trapezoid(p, f)
    bands = [(0.8, 2.0, "0.8-2.0 Hz\n(jantung asli)"), (2.0, 10.0, "2.0-10.0 Hz\n(band tergeser)")]
    vals = []
    for lo, hi, _ in bands:
        m = (f >= lo) & (f < hi)
        vals.append(100 * np.trapezoid(p[m], f[m]) / tot)
    ax2.bar([0, 1], vals, color=[C_GOOD, C_BAD], alpha=.9, width=.55)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([b[2] for b in bands], fontsize=7.5)
    ax2.set_ylabel("% energi total")
    ax2.set_ylim(0, 105)
    for i, v in enumerate(vals):
        ax2.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold",
                 color=[C_GOOD, C_BAD][i])
    ax2.set_title("Energi sinyal jantung TI\nnyaris seluruhnya di band yang SALAH",
                  fontweight="bold", fontsize=9.5)
    fig.suptitle("Bukti frame rate salah: filter jantung TI mulur 2.5x dan justru "
                 "MEMBUANG sinyal jantung",
                 fontweight="bold", fontsize=10.5, y=1.05)
    fig.savefig(out / "fig11_filter_shift.png")
    plt.close(fig)


def fig12_distance(out):
    """Pengaruh jarak: SNR runtuh ke noise floor di 100 cm."""
    # dihitung di code/evaluation/analyze_distance.py
    dist = ["50 cm", "100 cm"]
    snr = [3.16, 1.09]
    mape = [12.5, 28.6]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax1.bar([0, 1], snr, color=[C_GOOD, C_BAD], alpha=.9, width=.5)
    ax1.axhline(1.0, color=C_NEU, ls="--", lw=1.6, label="SNR = 1.0 (setara noise)")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(dist)
    ax1.set_ylabel("SNR di frekuensi jantung")
    ax1.set_ylim(0, 3.3)
    ax1.legend(fontsize=7.5)
    for i, v in enumerate(snr):
        ax1.text(i, v + .07, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax1.set_title("Di 100 cm, sinyal jantung\nTENGGELAM di noise",
                  fontweight="bold", fontsize=9.5)

    ax2.bar([0, 1], mape, color=[C_GOOD, C_BAD], alpha=.9, width=.5)
    ax2.axhline(10, color="#8e44ad", ls="--", lw=1.6, label="batas standar 10%")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(dist)
    ax2.set_ylabel("MAPE (%)")
    ax2.set_ylim(0, 33)
    ax2.legend(fontsize=7.5)
    for i, v in enumerate(mape):
        ax2.text(i, v + .7, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_title("Akurasi ikut runtuh\nseiring jarak", fontweight="bold", fontsize=9.5)
    fig.suptitle("Data jarak milik sendiri: jarak subjek sangat menentukan "
                 "(daya radar turun ~1/R^4)",
                 fontweight="bold", fontsize=10.5, y=1.04)
    fig.savefig(out / "fig12_distance.png")
    plt.close(fig)


def main(csv, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    front_all = df[df.dataset == "position_front"].copy()
    front = front_all[front_all.unwrapPhasePeak_mm.notna()].copy()

    for fn in (fig1_ti_broken, fig2_signal_exists, fig3_psd_band,
               fig5_ceiling, fig6_result, fig7_persubject, fig8_within_between):
        fn(front, out)
        print("ok:", fn.__name__)
    fig4_ablation(out)
    print("ok: fig4_ablation")
    fig9_standard(front, front_all, out)
    print("ok: fig9_standard")
    fig10_rootcause(front_all, out)
    print("ok: fig10_rootcause")
    fig11_filter_shift(front_all, out)
    print("ok: fig11_filter_shift")
    fig12_distance(out)
    print("ok: fig12_distance")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
