# ══════════════════════════════════════════════════════════════
# Revision round 1: the ablation arms, re-tested honestly
# ══════════════════════════════════════════════════════════════
"""Re-reads the stored ablation metrics and fixes three defects in how
they were tested. Nothing is retrained.

1. Ties. One arm removes a channel that has no path to the output, so all
   of its cells are exactly zero. A sign test that counts ties as evidence
   returns a tiny p-value from nothing at all. Ties are excluded here
   (Pratt's rule), and an arm whose cells are all ties is reported as
   "no effect detected" with no p-value.

2. The unit. Four seeds times three horizons is not twelve independent
   cells: the horizons share seeds, the same test block and the same
   fitted pipeline. The 12-cell count is kept as a descriptive consistency
   statement, and the inferential unit is the seed (n=4), where the
   sign-test floor is 2^-4 = 0.125 and nothing can reach 0.05. That floor
   is printed rather than worked around.

3. Multiplicity. Nine arms times two metrics is eighteen tests, and the
   published text reads them one at a time. Holm is applied across all
   eighteen.

Output: eval_v7/revisi1/ablation_tests.csv
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

WORK = Path(os.environ["MADSCAG_WORK"])
OUT = WORK / "eval_v7" / "revisi1"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "BASE_v7"
METRICS = [("auc_a", "lower is worse", -1), ("mae_c", "higher is worse", +1)]
ARMS = ["NO_ZI_B_v7", "NO_MPE_v7", "NO_SPATIAL_v7", "NO_PRESSURE_v7",
        "NO_DIFF_v7", "BETA_OBS_v7", "NO_GATE_v7", "NO_ANOM_v7", "NO_OBSFLAG_v7"]
LABEL = {"NO_ZI_B_v7": "zero-inflated head", "NO_MPE_v7": "missingness encoder",
         "NO_SPATIAL_v7": "neighbour channel", "NO_PRESSURE_v7": "pressure channel",
         "NO_DIFF_v7": "diffusion loss term", "BETA_OBS_v7": "observation modulator (on)",
         "NO_GATE_v7": "zero-inflation gate", "NO_ANOM_v7": "anomaly channel",
         "NO_OBSFLAG_v7": "observation flag (control)"}


def holm(p):
    p = np.asarray(p, "float64")
    keep = np.isfinite(p)
    adj = np.full_like(p, np.nan)
    idx = np.where(keep)[0]
    order = idx[np.argsort(p[idx])]
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(order) - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def sign_test_pratt(d):
    """Two-sided exact sign test with ties dropped. Returns (p, n_used)."""
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return float("nan"), 0
    k = int((nz > 0).sum())
    p = 2 * min(stats.binom.cdf(k, n, 0.5), 1 - stats.binom.cdf(k - 1, n, 0.5))
    return float(min(p, 1.0)), n


def main():
    m = pd.read_csv(WORK / "eval_v7" / "metrics_full.csv")
    u = m[m.split == "uji"]
    rows = []

    for metric, direction, sign in METRICS:
        base = u[u.variant == BASE].set_index(["seed", "horizon"])[metric]
        for arm in ARMS:
            s = u[u.variant == arm].set_index(["seed", "horizon"])[metric]
            ix = s.index.intersection(base.index)
            d = (s.loc[ix] - base.loc[ix]).astype("float64")
            cells = len(d)
            ties = int((d == 0).sum())
            p_cell, n_cell = sign_test_pratt(d.to_numpy())

            # the inferential unit: average each seed over its horizons
            per_seed = d.groupby(level=0).mean()
            p_seed, n_seed = sign_test_pratt(per_seed.to_numpy())
            se = stats.sem(per_seed) if len(per_seed) > 1 else np.nan
            ci = (stats.t.interval(0.95, len(per_seed) - 1, loc=per_seed.mean(), scale=se)
                  if np.isfinite(se) and se > 0 else (np.nan, np.nan))

            rows.append(dict(
                metric=metric, arm=arm, label=LABEL[arm], direction=direction,
                cells=cells, ties=ties, all_ties=bool(ties == cells),
                mean_effect=float(d.mean()), median_effect=float(d.median()),
                seeds=len(per_seed), seed_mean=float(per_seed.mean()),
                seed_ci_lo=float(ci[0]), seed_ci_hi=float(ci[1]),
                consistency=f"{int((d > 0).sum())}+/{int((d < 0).sum())}-/{ties}=",
                hurts_when_removed=bool(np.sign(d.mean()) == sign),
                p_sign_cells_pratt=p_cell, n_cells_used=n_cell,
                p_sign_seeds_pratt=p_seed, n_seeds_used=n_seed))

    res = pd.DataFrame(rows)
    res["p_holm_18"] = holm(res.p_sign_cells_pratt.values)
    res["p_holm_18_seedunit"] = holm(res.p_sign_seeds_pratt.values)
    res = res.sort_values(["metric", "mean_effect"], key=lambda c: c.abs()
                          if c.name == "mean_effect" else c, ascending=[True, False])
    res.to_csv(OUT / "ablation_tests.csv", index=False)

    floor = 2.0 ** -4 * 2
    pd.set_option("display.width", 220)
    for metric, _, _ in METRICS:
        print(f"\n=== {metric}, arms ordered by |effect| ===")
        g = res[res.metric == metric]
        print(g[["label", "mean_effect", "seed_ci_lo", "seed_ci_hi", "consistency",
                 "p_sign_cells_pratt", "p_holm_18", "p_sign_seeds_pratt",
                 "all_ties"]].round(5).to_string(index=False))
    print(f"\nexact two-sided sign-test floor with 4 seeds: p = {floor:.3f}")
    print("so no arm can reach 0.05 on the seed as the unit; the 12-cell p-values")
    print("are reported only because the published text used them, and they")
    print("borrow degrees of freedom from horizons that share seeds.")
    noop = res[res.all_ties].label.unique()
    print(f"\narms with every cell an exact tie (no effect detected, no p-value): {list(noop)}")
    print("wrote " + str(OUT / "ablation_tests.csv"))


if __name__ == "__main__":
    main()
