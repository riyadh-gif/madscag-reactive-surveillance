# ══════════════════════════════════════════════════════════════
# What the conformal construction buys over a trivial interval
# ══════════════════════════════════════════════════════════════
"""Near-nominal marginal coverage is not by itself evidence that a conformal
construction adds anything, because a target that is half zeros can be
covered by a single band applied to every village. This measures that
reference: the constant interval from zero to the calibration quantile of
the outcome, against the per-village conformal interval, at the same nominal
level. The gap between them is what the construction earns.
"""
import json
import os
from pathlib import Path

import numpy as np

WORK = Path(os.environ["MADSCAG_WORK"])
PREP = WORK / "preprocessed_v7"
EXP = WORK / "experiments_v7"
OUT = WORK / "eval_v7" / "revisi1"
SEEDS = [42, 1, 2, 7]
ALPHA = 0.10


def quantile(residuals, alpha=ALPHA):
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(residuals)[min(k, n) - 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in (1, 2, 3):
        ysa = np.load(PREP / f"y_ltsj_sahih_h{h}.npy").astype("float64")
        yuj = np.load(PREP / f"y_ltsj_uji_h{h}.npy").astype("float64")
        cal, test = ~np.isnan(ysa), ~np.isnan(yuj)

        cov, wid = [], []
        for s in SEEDS:
            mu_s = np.asarray(np.load(EXP / f"BL_XGB2_s{s}" / "preds_sahih.npz")[f"h{h}_mu"], "float64")
            mu_u = np.asarray(np.load(EXP / f"BL_XGB2_s{s}" / "preds_uji.npz")[f"h{h}_mu"], "float64")
            q = quantile(np.abs(ysa[cal] - mu_s[cal]))
            lo = np.maximum(mu_u[test] - q, 0.0)
            hi = mu_u[test] + q
            cov.append(float(((yuj[test] >= lo) & (yuj[test] <= hi)).mean()))
            wid.append(float(np.mean(hi - lo)))

        qc = quantile(np.abs(ysa[cal]))
        rows.append(dict(horizon=h,
                         conformal_coverage=float(np.mean(cov)),
                         conformal_width=float(np.mean(wid)),
                         constant_coverage=float((yuj[test] <= qc).mean()),
                         constant_width=qc, n_test=int(test.sum()),
                         n_calibration=int(cal.sum()), n_seeds=len(SEEDS)))
        r = rows[-1]
        print("h=%d  conformal %.4f (width %.3f)   constant [0,q] %.4f (width %.3f)"
              % (h, r["conformal_coverage"], r["conformal_width"],
                 r["constant_coverage"], r["constant_width"]))

    (OUT / "conformal_constant_benchmark.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")
    print("\nwrote", OUT / "conformal_constant_benchmark.json")


if __name__ == "__main__":
    main()
