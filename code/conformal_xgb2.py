"""Split-conformal intervals and calibration for the proposed two-stage GBM.

Runs entirely off the stored per-seed predictions, so nothing is retrained
and nothing can drift from what Section 4 reports. Quantiles are fitted on
the validation block and coverage is measured once on the test block.
"""
import os, json
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression

WORK = Path(os.environ.get("MADSCAG_WORK", "."))
PREP, EXP, EVAL = WORK / "preprocessed_v7", WORK / "experiments_v7", WORK / "eval_v7"
ALPHA, SEEDS, H = 0.10, [42, 1, 2, 7], [1, 2, 3]


def y(split, kind, h):
    return np.load(PREP / ("y_%s_%s_h%d.npy" % (kind, split, h)))


def preds(variant, seed, split, h):
    z = np.load(EXP / ("%s_s%d" % (variant, seed)) / ("preds_%s.npz" % split))
    return (np.asarray(z["h%d_pa" % h], float).ravel(),
            np.asarray(z["h%d_mu" % h], float).ravel())


def ece(prob, lab, bins=10):
    """Expected calibration error over equal-frequency bins."""
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    tot = 0.0
    for i in range(bins):
        m = (prob > edges[i]) & (prob <= edges[i + 1])
        if m.sum():
            tot += m.sum() / len(prob) * abs(prob[m].mean() - lab[m].mean())
    return tot


rows = []
for h in H:
    ysa, yuj = y("sahih", "ltsj", h), y("uji", "ltsj", h)
    dsa, duj = y("sahih", "diamati", h), y("uji", "diamati", h)
    msa, muj = ~np.isnan(ysa), ~np.isnan(yuj)
    vsa, vuj = ~np.isnan(dsa), ~np.isnan(duj)
    q_s, cov_s, ece_raw, ece_iso, wid = [], [], [], [], []
    for s in SEEDS:
        pa_s, mu_s = preds("BL_XGB2", s, "sahih", h)
        pa_u, mu_u = preds("BL_XGB2", s, "uji", h)
        # split conformal on validation residuals
        r = np.abs(ysa[msa] - mu_s[msa])
        n = len(r)
        k = int(np.ceil((n + 1) * (1 - ALPHA)))
        qh = np.sort(r)[min(k, n) - 1]
        lo = np.maximum(mu_u[muj] - qh, 0.0)
        hi = mu_u[muj] + qh
        q_s.append(qh)
        cov_s.append(float(((yuj[muj] >= lo) & (yuj[muj] <= hi)).mean()))
        wid.append(float(np.mean(hi - lo)))
        # Stage A calibration, raw and isotonic-on-validation
        ece_raw.append(ece(pa_u[vuj], duj[vuj]))
        iso = IsotonicRegression(out_of_bounds="clip").fit(pa_s[vsa], dsa[vsa])
        ece_iso.append(ece(np.clip(iso.predict(pa_u[vuj]), 0, 1), duj[vuj]))
    rows.append(dict(horizon=h, q=float(np.mean(q_s)),
                     coverage=float(np.mean(cov_s)),
                     cov_min=float(np.min(cov_s)), cov_max=float(np.max(cov_s)),
                     width=float(np.mean(wid)),
                     ece_raw=float(np.mean(ece_raw)),
                     ece_iso=float(np.mean(ece_iso)), n_test=int(muj.sum())))

EVAL.mkdir(parents=True, exist_ok=True)
with open(EVAL / "conformal_xgb2.json", "w") as f:
    json.dump(rows, f, indent=2)

print("nominal coverage %.2f | %d seeds\n" % (1 - ALPHA, len(SEEDS)))
print("%-3s %8s %9s %14s %8s %9s %9s %7s" %
      ("h", "q", "coverage", "range", "width", "ECE raw", "ECE iso", "n"))
for r in rows:
    print("%-3d %8.4f %9.4f  %.4f-%.4f %8.4f %9.4f %9.4f %7d" %
          (r["horizon"], r["q"], r["coverage"], r["cov_min"], r["cov_max"],
           r["width"], r["ece_raw"], r["ece_iso"], r["n_test"]))
