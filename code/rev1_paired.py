# ══════════════════════════════════════════════════════════════
# Revision round 1: paired inference across the 23 test periods
# ══════════════════════════════════════════════════════════════
"""One protocol for every decision-layer contrast in Section 4.

Unit of analysis is the test period, n=23. Villages inside a period
compete for the same k slots, so the village is not independent. The
pairing is period against period: same observed set, same k, two ranking
rules. Wilcoxon signed-rank is primary because the per-period differences
are bounded and driven by a few large-severity villages; the paired t is
reported beside it as a sensitivity, never alone.

Multiplicity: Holm inside the pre-registered family of three horizons.

No equivalence is claimed anywhere. When the primary test fails to reject,
the reported result is the measured difference, its confidence interval and
the win count, and nothing further: failing to reject is not evidence that
the systems are the same. The TOST columns are a margin sensitivity, printed
at three margins at once precisely so that no single margin can be chosen
after seeing its p-value. On the h=1 contrast none of the three certifies
equivalence at a margin an operations office could act on, and the output
says so rather than leaving the reader to pick a column.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

WORK = Path(os.environ["MADSCAG_WORK"])
OUT = WORK / "eval_v7" / "revisi1"

PRIMARY_FRAC = 0.50
FRACS_REPORTED = [0.25, 0.50, 0.75]
TOST_MARGINS = [0.02, 0.025, 0.03]

# (label, variant_a, policy_a, variant_b, policy_b)
CONTRASTS = [
    ("proposed_vs_recurrent", "proposed", "model_sev", "recurrent_hybrid", "model_sev"),
    ("sev_vs_product", "proposed", "model_sev", "proposed", "model_agg"),
    ("proposed_vs_persistence", "proposed", "model_sev", "reference", "persistence"),
    ("proposed_vs_random", "proposed", "model_sev", "reference", "random"),
    ("proposed_vs_grud", "proposed", "model_sev", "grud", "model_sev"),
    ("proposed_vs_lstm", "proposed", "model_sev", "lstm", "model_sev"),
    ("proposed_vs_tcn", "proposed", "model_sev", "tcn", "model_sev"),
    ("persistence_vs_random", "reference", "persistence", "reference", "random"),
]


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, "float64")
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(p) - i) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, "float64")
    n = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    prev = 1.0
    for rank, idx in zip(range(n, 0, -1), order[::-1]):
        prev = min(prev, p[idx] * n / rank)
        adj[idx] = prev
    return adj


def tost(d, margin):
    """Two one-sided tests on the paired mean difference."""
    n = len(d)
    se = stats.sem(d)
    if not np.isfinite(se) or se == 0:
        return float("nan")
    df = n - 1
    p_lo = 1.0 - stats.t.cdf((d.mean() + margin) / se, df)
    p_hi = stats.t.cdf((d.mean() - margin) / se, df)
    return float(max(p_lo, p_hi))


def series(cp, h, frac, variant, policy, estimand):
    """Capture per period. 'ens4' falls back to 'ens' for variants that were
    only ever run with four seeds, so the 4-vs-4 comparison is well defined."""
    est = "none" if variant == "reference" else estimand
    m = ((cp.h == h) & np.isclose(cp.frac, frac) & (cp.variant == variant)
         & (cp.policy == policy) & (cp.estimand == est))
    s = cp[m].set_index("t").capture.sort_index()
    if len(s) == 0 and est == "ens4":
        return series(cp, h, frac, variant, policy, "ens")
    return s


def compare(a, b):
    ix = a.index.intersection(b.index)
    d = (a.loc[ix] - b.loc[ix]).astype("float64")
    n = len(d)
    se = stats.sem(d)
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se) if se > 0 else (np.nan, np.nan)
    try:
        wp = float(stats.wilcoxon(a.loc[ix], b.loc[ix]).pvalue)
    except ValueError:
        wp = float("nan")
    rec = dict(n_periods=n, mean_a=float(a.loc[ix].mean()), mean_b=float(b.loc[ix].mean()),
               diff_mean=float(d.mean()), diff_median=float(d.median()),
               ci_lo=float(ci[0]), ci_hi=float(ci[1]), se=float(se),
               p_wilcoxon=wp,
               p_ttest=float(stats.ttest_rel(a.loc[ix], b.loc[ix]).pvalue),
               wins=int((d > 0).sum()), losses=int((d < 0).sum()),
               ties=int((d == 0).sum()))
    for m in TOST_MARGINS:
        rec["tost_p_" + str(m)] = tost(d, m)
    return rec


def main():
    cp = pd.read_csv(OUT / "capture_periods.csv")
    rows = []
    for estimand in ["ens", "ens4", "perseed"]:
        for h in [1, 2, 3]:
            for frac in FRACS_REPORTED:
                for label, va, pa, vb, pb in CONTRASTS:
                    a = series(cp, h, frac, va, pa, estimand)
                    b = series(cp, h, frac, vb, pb, estimand)
                    if len(a) == 0 or len(b) == 0:
                        continue
                    r = dict(estimand=estimand, h=h, frac=frac, contrast=label)
                    r.update(compare(a, b))
                    rows.append(r)
    res = pd.DataFrame(rows)

    # Holm inside the pre-registered 3-horizon family, per contrast, at f=0.50
    res["p_holm"] = np.nan
    prim = res[np.isclose(res.frac, PRIMARY_FRAC)]
    for (est, label), g in prim.groupby(["estimand", "contrast"]):
        res.loc[g.index, "p_holm"] = holm(g.p_wilcoxon.values)
    res.to_csv(OUT / "paired_tests.csv", index=False)

    # exploratory budget grid, BH corrected
    grid = []
    for h in [1, 2, 3]:
        for frac in sorted(cp.frac.unique()):
            a = series(cp, h, frac, "proposed", "model_sev", "ens")
            b = series(cp, h, frac, "recurrent_hybrid", "model_sev", "ens")
            if len(a) and len(b):
                r = dict(h=h, frac=float(frac), contrast="proposed_vs_recurrent")
                r.update(compare(a, b))
                grid.append(r)
            a = series(cp, h, frac, "reference", "persistence", "none")
            b = series(cp, h, frac, "reference", "random", "none")
            if len(a) and len(b):
                r = dict(h=h, frac=float(frac), contrast="persistence_vs_random")
                r.update(compare(a, b))
                grid.append(r)
    gr = pd.DataFrame(grid)
    gr["p_bh"] = bh(gr.p_wilcoxon.fillna(1.0).values)
    gr.to_csv(OUT / "paired_budget_grid.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== PRIMARY: proposed vs recurrent hybrid, f=0.50, ensemble estimand ===")
    p = res[(res.contrast == "proposed_vs_recurrent") & (res.estimand == "ens")
            & np.isclose(res.frac, 0.50)]
    print(p[["h", "mean_a", "mean_b", "diff_mean", "ci_lo", "ci_hi", "p_wilcoxon",
             "p_ttest", "p_holm", "wins", "losses",
             "tost_p_0.02", "tost_p_0.025", "tost_p_0.03"]].round(4).to_string(index=False))
    print("  the tost_p_* columns are a margin sensitivity, not a verdict: no")
    print("  equivalence is claimed at any margin, and the reported result is")
    print("  the difference with its interval and win count.")

    print("\n=== same contrast: 4 seeds both sides, and per-seed estimand ===")
    q = res[(res.contrast == "proposed_vs_recurrent") & (res.estimand != "ens")
            & np.isclose(res.frac, 0.50)]
    print(q[["estimand", "h", "mean_a", "mean_b", "diff_mean", "ci_lo", "ci_hi",
             "p_wilcoxon", "p_ttest", "wins"]].round(4).to_string(index=False))

    print("\n=== other contrasts, h=1, f=0.50, ensemble ===")
    o = res[(res.estimand == "ens") & (res.h == 1) & np.isclose(res.frac, 0.50)
            & (res.contrast != "proposed_vs_recurrent")]
    print(o[["contrast", "mean_a", "mean_b", "diff_mean", "ci_lo", "ci_hi",
             "p_wilcoxon", "p_ttest", "wins", "losses"]].round(4).to_string(index=False))

    print("\n=== sev vs product across horizons (ensemble, f=0.50) ===")
    s = res[(res.contrast == "sev_vs_product") & (res.estimand == "ens")
            & np.isclose(res.frac, 0.50)]
    print(s[["h", "diff_mean", "p_wilcoxon", "p_ttest", "p_holm", "wins"]].round(4).to_string(index=False))

    print("\n=== persistence vs random across the budget grid, h=1 ===")
    pr = gr[(gr.contrast == "persistence_vs_random") & (gr.h == 1)]
    print(pr[["frac", "mean_a", "mean_b", "diff_mean", "p_wilcoxon", "p_bh", "wins"]].round(4).to_string(index=False))

    print("\nwrote " + str(OUT / "paired_tests.csv") + " and " + str(OUT / "paired_budget_grid.csv"))


if __name__ == "__main__":
    main()
