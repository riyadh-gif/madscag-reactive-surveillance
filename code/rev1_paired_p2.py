# ══════════════════════════════════════════════════════════════
# Revision round 1, phase 2: the pre-registered ablation family
# ══════════════════════════════════════════════════════════════
"""Family A: five feature-group ablations of the proposed system.

The protocol is Phase 1's, unchanged and imported rather than restated:
the test period is the unit, n=23; the pairing is period against period on
the same observed set with the same k; Wilcoxon signed-rank is primary and
the paired t is printed beside it; no equivalence is claimed anywhere.

Registered before the numbers were read:
  primary    capture C(0.50) at h=1, seed-ensembled, Holm across the five
  secondary  capture C(0.25), descriptive, because a tighter budget is more
             sensitive to ordering than to magnitude
  replication h=2 and h=3, descriptive, no p-value
  mechanism  capture is rank-based and touches only the top half, so a large
             move in stage-level error with a flat capture is the expected
             signature of a block carrying magnitude but not ordering

Reads `capture_periods_p2.csv`; the Phase 1 file is never opened.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

from rev1_paired import holm, compare, series, TOST_MARGINS

WORK = Path(os.environ["MADSCAG_WORK"])
OUT = WORK / "eval_v7" / "revisi1"

ARMS = {
    "ab_noobshist": ("AB_NOOBSHIST", "observation history", 13),
    "ab_nosevhist": ("AB_NOSEVHIST", "severity history", 25),
    "ab_noclim": ("AB_NOCLIM", "climate summaries", 204),
    "ab_nonbr": ("AB_NONBR", "neighbour pressure", 2),
    "ab_last1": ("AB_LAST1", "window mean and standard deviation", 172),
}
PRIMARY_H, PRIMARY_F, SECONDARY_F = 1, 0.50, 0.25


def main():
    cp = pd.read_csv(OUT / "capture_periods_p2.csv")
    rows = []
    for arm, (run, group, ncol) in ARMS.items():
        for h in [1, 2, 3]:
            for frac in sorted(cp.frac.unique()):
                a = series(cp, h, frac, "proposed", "model_sev", "ens")
                b = series(cp, h, frac, arm, "model_sev", "ens")
                if len(a) == 0 or len(b) == 0:
                    continue
                r = dict(arm=run, removed_group=group, n_columns_removed=ncol,
                         h=h, frac=float(frac))
                r.update(compare(a, b))
                rows.append(r)
    res = pd.DataFrame(rows)

    # Holm inside the pre-registered family: five arms, h=1, f=0.50
    fam = res[(res.h == PRIMARY_H) & np.isclose(res.frac, PRIMARY_F)]
    res["p_holm5"] = np.nan
    res.loc[fam.index, "p_holm5"] = holm(fam.p_wilcoxon.values)

    # stage-level effects, from the frozen-schema metrics table
    m = pd.read_csv(WORK / "eval_v7" / "metrics_full.csv")
    u = m[m.split == "uji"]
    base = u[u.variant == "BL_XGB2"].set_index(["seed", "horizon"])
    stage = []
    for arm, (run, group, ncol) in ARMS.items():
        s = u[u.variant == run].set_index(["seed", "horizon"])
        ix = s.index.intersection(base.index)
        for met in ["auc_a", "mae_c"]:
            d = (s.loc[ix, met] - base.loc[ix, met]).astype("float64")
            per_seed = d.groupby(level=0).mean()
            from scipy import stats as st
            se = st.sem(per_seed) if len(per_seed) > 1 else np.nan
            ci = (st.t.interval(0.95, len(per_seed) - 1, loc=per_seed.mean(), scale=se)
                  if np.isfinite(se) and se > 0 else (np.nan, np.nan))
            stage.append(dict(arm=run, metric=met, cells=len(d),
                              mean_effect=float(d.mean()),
                              seed_mean=float(per_seed.mean()),
                              seed_ci_lo=float(ci[0]), seed_ci_hi=float(ci[1]),
                              consistency=f"{int((d>0).sum())}+/{int((d<0).sum())}-/{int((d==0).sum())}="))
    stage = pd.DataFrame(stage)

    # one table for the manuscript, replacing the nine-arm figure pair
    tab = []
    for arm, (run, group, ncol) in ARMS.items():
        p = res[(res.arm == run) & (res.h == PRIMARY_H) & np.isclose(res.frac, PRIMARY_F)].iloc[0]
        sec = res[(res.arm == run) & (res.h == PRIMARY_H) & np.isclose(res.frac, SECONDARY_F)]
        h2 = res[(res.arm == run) & (res.h == 2) & np.isclose(res.frac, PRIMARY_F)]
        h3 = res[(res.arm == run) & (res.h == 3) & np.isclose(res.frac, PRIMARY_F)]
        sa = stage[(stage.arm == run) & (stage.metric == "auc_a")].iloc[0]
        sm = stage[(stage.arm == run) & (stage.metric == "mae_c")].iloc[0]
        # The verdict follows the pre-registered test, not the size of the number.
        # A large delta that the signed-rank test does not resolve is reported as
        # unresolved, never as an effect.
        holm_p = float(res.loc[p.name, "p_holm5"])
        stage_ci_excludes_zero = (sa.seed_ci_lo * sa.seed_ci_hi > 0
                                  or sm.seed_ci_lo * sm.seed_ci_hi > 0)
        consistent = all(np.sign(x) == np.sign(-p.diff_mean) and abs(x) > 0.002
                         for x in [-sec.diff_mean.iloc[0] if len(sec) else 0.0,
                                   -h2.diff_mean.iloc[0] if len(h2) else 0.0,
                                   -h3.diff_mean.iloc[0] if len(h3) else 0.0])
        if holm_p < 0.05:
            verdict = "decision-layer effect"
        elif abs(p.diff_mean) >= 0.01 and consistent:
            verdict = "direction consistent across budgets and horizons, not resolved by 23 periods"
        elif abs(p.diff_mean) < 0.01 and stage_ci_excludes_zero:
            verdict = "stage-level effect with no demonstrated decision-layer effect"
        else:
            verdict = "no effect demonstrated"
        tab.append(dict(arm=run, removed_group=group, n_columns_removed=ncol,
                        capture_h1=round(float(p.mean_b), 4),
                        delta_h1=round(float(-p.diff_mean), 4),
                        ci_lo=round(float(-p.ci_hi), 4), ci_hi=round(float(-p.ci_lo), 4),
                        p_wilcoxon=round(float(p.p_wilcoxon), 4),
                        p_holm5=round(float(res.loc[p.name, "p_holm5"]), 4),
                        wins_for_arm=int(p.losses), losses_for_arm=int(p.wins),
                        delta_f025=round(float(-sec.diff_mean.iloc[0]), 4) if len(sec) else np.nan,
                        delta_h2=round(float(-h2.diff_mean.iloc[0]), 4) if len(h2) else np.nan,
                        delta_h3=round(float(-h3.diff_mean.iloc[0]), 4) if len(h3) else np.nan,
                        d_auc_a=round(float(sa.seed_mean), 5),
                        d_auc_ci=f"[{sa.seed_ci_lo:+.5f}, {sa.seed_ci_hi:+.5f}]",
                        d_mae_c=round(float(sm.seed_mean), 5),
                        d_mae_ci=f"[{sm.seed_ci_lo:+.5f}, {sm.seed_ci_hi:+.5f}]",
                        endpoint_verdict=verdict))
    T1 = pd.DataFrame(tab).sort_values("delta_h1")

    res.to_csv(OUT / "paired_tests_p2.csv", index=False)
    stage.to_csv(OUT / "ablation_proposed_stage.csv", index=False)
    T1.to_csv(OUT / "table_ablation_proposed.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== Family A: capture at f=0.50, h=1. Delta is arm minus proposed ===")
    print(T1[["arm", "n_columns_removed", "capture_h1", "delta_h1", "ci_lo", "ci_hi",
              "p_wilcoxon", "p_holm5", "wins_for_arm", "endpoint_verdict"]].to_string(index=False))
    print("\n=== secondary and replication (descriptive, no p-values) ===")
    print(T1[["arm", "delta_f025", "delta_h2", "delta_h3"]].to_string(index=False))
    print("\n=== stage-level effects, seed as the unit ===")
    print(T1[["arm", "d_auc_a", "d_auc_ci", "d_mae_c", "d_mae_ci"]].to_string(index=False))
    print("\nproposed system reference: C(0.50) h=1 = %.4f" %
          series(cp, 1, 0.50, "proposed", "model_sev", "ens").mean())
    print("wrote table_ablation_proposed.csv, paired_tests_p2.csv, ablation_proposed_stage.csv")


if __name__ == "__main__":
    main()
