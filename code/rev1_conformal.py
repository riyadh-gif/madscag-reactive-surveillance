# ══════════════════════════════════════════════════════════════
# Revision round 1: conformal coverage and width, by stratum
# ══════════════════════════════════════════════════════════════
"""Reviewer 2 asked whether nominal marginal coverage hides undercoverage
in the villages that motivate the system. It does.

Two constructions are scored on the same rows:

  global    one split-conformal quantile per horizon, as published
  mondrian  group-conditional quantiles on predicted-severity quintiles,
            with the quintile edges fitted on the validation block only

Coverage is always reported beside mean interval width, because an
interval clipped to zero can cover a zero outcome while saying nothing.
Cells smaller than MIN_CELL are reported but marked, and every cell
carries a Clopper-Pearson interval.

Nothing is retrained: validation and test predictions are read from the
stored per-seed npz files.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

WORK = Path(os.environ["MADSCAG_WORK"])
PREP = WORK / "preprocessed_v7"
EXP = WORK / "experiments_v7"
OUT = WORK / "eval_v7" / "revisi1"
OUT.mkdir(parents=True, exist_ok=True)

VARIANT = "BL_XGB2"
SEEDS = [42, 1, 2, 7]
HORIZONS = [1, 2, 3]
ALPHA = 0.10
N_QUINT = 5
MIN_CELL = 100          # report a point estimate
MIN_DIFF_CELL = 300     # claim a difference
SEV_BANDS = [("y=0", 0.0, 0.0), ("0<y<=0.5", 0.0, 0.5),
             ("0.5<y<=1", 0.5, 1.0), ("y>1", 1.0, np.inf)]


def preds(seed, split, h):
    z = np.load(EXP / f"{VARIANT}_s{seed}" / f"preds_{split}.npz")
    return (np.asarray(z[f"h{h}_pa"], "float64").ravel(),
            np.asarray(z[f"h{h}_mu"], "float64").ravel())


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def conformal_quantile(resid, alpha=ALPHA):
    n = len(resid)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(resid)[min(k, n) - 1])


def cell_row(construction, h, dim, level, cov, wid, clipped):
    n = len(cov)
    k = int(np.sum(cov))
    lo, hi = clopper_pearson(k, n)
    return dict(construction=construction, horizon=h, dimension=dim, level=str(level),
                n=n, coverage=float(np.mean(cov)) if n else np.nan,
                cp_lo=lo, cp_hi=hi, width=float(np.mean(wid)) if n else np.nan,
                clipped_share=float(np.mean(clipped)) if n else np.nan,
                reportable=n >= MIN_CELL, comparable=n >= MIN_DIFF_CELL)


def main():
    sm = pd.read_csv(PREP / "sample_meta_uji.csv")
    tab = np.load(PREP / "TAB_uji.npy", mmap_mode="r")
    year = np.asarray(tab[:, 260], "float64")
    rows = []

    for h in HORIZONS:
        ysa = np.load(PREP / f"y_ltsj_sahih_h{h}.npy").astype("float64")
        yuj = np.load(PREP / f"y_ltsj_uji_h{h}.npy").astype("float64")
        msa, muj = ~np.isnan(ysa), ~np.isnan(yuj)

        cov_g, wid_g, clip_g = [], [], []
        cov_m, wid_m, clip_m = [], [], []
        pa_acc, mu_acc = [], []

        for s in SEEDS:
            _, mu_s = preds(s, "sahih", h)
            pa_u, mu_u = preds(s, "uji", h)
            pa_acc.append(pa_u)
            mu_acc.append(mu_u)
            r_sa = np.abs(ysa[msa] - mu_s[msa])

            # global construction, exactly as published
            q = conformal_quantile(r_sa)
            lo = np.maximum(mu_u[muj] - q, 0.0)
            hi = mu_u[muj] + q
            cov_g.append(((yuj[muj] >= lo) & (yuj[muj] <= hi)).astype(float))
            wid_g.append(hi - lo)
            clip_g.append((mu_u[muj] - q < 0).astype(float))

            # Mondrian on predicted-severity quintiles, edges fitted on validation
            edges = np.quantile(mu_s[msa], np.linspace(0, 1, N_QUINT + 1)[1:-1])
            g_sa = np.digitize(mu_s[msa], edges)
            g_uj = np.digitize(mu_u[muj], edges)
            qg = {g: conformal_quantile(r_sa[g_sa == g]) for g in np.unique(g_sa)}
            qq = np.array([qg.get(g, q) for g in g_uj])
            lo_m = np.maximum(mu_u[muj] - qq, 0.0)
            hi_m = mu_u[muj] + qq
            cov_m.append(((yuj[muj] >= lo_m) & (yuj[muj] <= hi_m)).astype(float))
            wid_m.append(hi_m - lo_m)
            clip_m.append((mu_u[muj] - qq < 0).astype(float))

        y = yuj[muj]
        pa = np.mean(pa_acc, axis=0)[muj]
        mu = np.mean(mu_acc, axis=0)[muj]
        meta = sm[muj].reset_index(drop=True)
        yr = year[muj]
        tt = meta["t"].values
        terc = np.digitize(tt, np.quantile(tt, [1 / 3, 2 / 3]))
        prop_q = np.digitize(pa, np.quantile(pa, [0.25, 0.50, 0.75]))
        mu_q = np.digitize(mu, np.quantile(mu, np.linspace(0, 1, N_QUINT + 1)[1:-1]))

        for name, cov, wid, clip in [("global", cov_g, wid_g, clip_g),
                                     ("mondrian", cov_m, wid_m, clip_m)]:
            C = np.mean(cov, axis=0)
            W = np.mean(wid, axis=0)
            K = np.mean(clip, axis=0)
            rows.append(cell_row(name, h, "overall", "all", C, W, K))
            for kab, m in meta.groupby("kab").groups.items():
                m = np.asarray(m)
                rows.append(cell_row(name, h, "regency", kab, C[m], W[m], K[m]))
            for g in np.unique(terc):
                m = terc == g
                rows.append(cell_row(name, h, "time_block", f"tercile{g + 1}", C[m], W[m], K[m]))
            for g in np.unique(yr):
                m = yr == g
                rows.append(cell_row(name, h, "year", int(g), C[m], W[m], K[m]))
            for g in np.unique(prop_q):
                m = prop_q == g
                rows.append(cell_row(name, h, "propensity_quartile", f"Q{g + 1}", C[m], W[m], K[m]))
            for g in np.unique(mu_q):
                m = mu_q == g
                rows.append(cell_row(name, h, "pred_severity_quintile", f"P{g + 1}", C[m], W[m], K[m]))
            for lab, a, b in SEV_BANDS:
                m = (y == 0) if b == 0 else ((y > a) & (y <= b))
                if m.sum():
                    rows.append(cell_row(name, h, "severity_band", lab, C[m], W[m], K[m]))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "conformal_subgroups.csv", index=False)

    pd.set_option("display.width", 200)
    g = res[(res.dimension == "overall")]
    print("\nGATE marginal coverage (published 0.9002 / 0.9039 / 0.9051, width 0.655 at h=1)")
    print(g[["construction", "horizon", "n", "coverage", "width", "clipped_share"]].round(4).to_string(index=False))

    print("\n=== coverage by realized severity band ===")
    b = res[res.dimension == "severity_band"]
    print(b[["construction", "horizon", "level", "n", "coverage", "cp_lo", "cp_hi", "width"]].round(4).to_string(index=False))

    print("\n=== h=1 coverage by regency and propensity quartile ===")
    r = res[(res.horizon == 1) & (res.dimension.isin(["regency", "propensity_quartile"]))]
    print(r[["construction", "dimension", "level", "n", "coverage", "cp_lo", "cp_hi", "width"]].round(4).to_string(index=False))

    print("\n=== h=1 Mondrian by predicted-severity quintile ===")
    q = res[(res.horizon == 1) & (res.dimension == "pred_severity_quintile")]
    print(q[["construction", "level", "n", "coverage", "width"]].round(4).to_string(index=False))

    small = res[~res.reportable]
    print(f"\ncells below {MIN_CELL} rows: {len(small)}")
    print("wrote " + str(OUT / "conformal_subgroups.csv"))


if __name__ == "__main__":
    main()
