# ══════════════════════════════════════════════════════════════
# Revision round 1: the split-boundary overlap, measured
# ══════════════════════════════════════════════════════════════
"""Window origins respect the split markers, but targets at longer horizons
do not. A validation window whose origin sits in the last periods of the
validation block has its target inside the test block, so the conformal
quantile for h=2 and h=3 is fitted on residuals whose outcomes lie in the
evaluation window.

This measures both halves of that sentence: how many validation targets
cross the boundary, and what the coverage becomes when they are removed
from the calibration set. Writing the second half to a file matters because
it is the half that establishes the defect is immaterial, and a number that
only ever existed in a console is not evidence.

Nothing is retrained; only the calibration set changes.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(os.environ["MADSCAG_WORK"])
PREP = WORK / "preprocessed_v7"
EXP = WORK / "experiments_v7"
OUT = WORK / "eval_v7" / "revisi1"
OUT.mkdir(parents=True, exist_ok=True)

VARIANT = "BL_XGB2"
SEEDS = [42, 1, 2, 7]
HORIZONS = [1, 2, 3]
ALPHA = 0.10


def conformal_quantile(resid, alpha=ALPHA):
    n = len(resid)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(resid)[min(k, n) - 1])


def main():
    sm_s = pd.read_csv(PREP / "sample_meta_sahih.csv", usecols=["t"])
    sm_u = pd.read_csv(PREP / "sample_meta_uji.csv", usecols=["t"])
    first_test = int(sm_u.t.min())
    rows = []

    for h in HORIZONS:
        ysa = np.load(PREP / f"y_ltsj_sahih_h{h}.npy").astype("float64")
        yuj = np.load(PREP / f"y_ltsj_uji_h{h}.npy").astype("float64")
        obs_s, obs_u = ~np.isnan(ysa), ~np.isnan(yuj)
        target = sm_s.t.to_numpy() + h - 1
        crosses = obs_s & (target >= first_test)
        clean = obs_s & ~crosses

        for label, mask in [("as published", obs_s), ("validation trimmed", clean)]:
            cov, wid = [], []
            for s in SEEDS:
                mu_s = np.asarray(np.load(EXP / f"{VARIANT}_s{s}" / "preds_sahih.npz")[f"h{h}_mu"], "float64")
                mu_u = np.asarray(np.load(EXP / f"{VARIANT}_s{s}" / "preds_uji.npz")[f"h{h}_mu"], "float64")
                q = conformal_quantile(np.abs(ysa[mask] - mu_s[mask]))
                lo = np.maximum(mu_u[obs_u] - q, 0.0)
                hi = mu_u[obs_u] + q
                cov.append(float(((yuj[obs_u] >= lo) & (yuj[obs_u] <= hi)).mean()))
                wid.append(float(np.mean(hi - lo)))
            rows.append(dict(horizon=h, calibration=label,
                             n_calibration=int(mask.sum()),
                             n_crossing=int(crosses.sum()),
                             crossing_share=float(crosses.sum() / obs_s.sum()),
                             first_test_period=first_test,
                             coverage=float(np.mean(cov)),
                             width=float(np.mean(wid)), n_seeds=len(SEEDS)))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "conformal_split_boundary.csv", index=False)

    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print("\nshift from trimming:")
    for h in HORIZONS:
        a = res[(res.horizon == h) & (res.calibration == "as published")].coverage.iloc[0]
        b = res[(res.horizon == h) & (res.calibration == "validation trimmed")].coverage.iloc[0]
        print(f"  h={h}: {a:.4f} -> {b:.4f}  ({(b - a) * 100:+.2f} points)")
    print("\nwrote " + str(OUT / "conformal_split_boundary.csv"))


if __name__ == "__main__":
    main()
