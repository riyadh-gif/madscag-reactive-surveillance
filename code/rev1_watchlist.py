# ══════════════════════════════════════════════════════════════
# Revision round 1: watchlist rules at a matched list size
# ══════════════════════════════════════════════════════════════
"""Rebuilds Table 5 and adds the composite rule the manuscript names but
never ran: filter on reporting-gap length first, then rank the survivors
by predicted severity.

Every rule draws LIST_SIZE villages per period from the same candidate
pool and is scored the same way, on the rows an officer visited anyway.
Two columns the published table lacks are added, because the comparison
is not otherwise readable:

  visited_share   the fraction of each list that was visited at all, which
                  differs by a factor of six across rules and is the
                  reason the attack rates are not directly comparable
  Wilson CI       on the attack rate, since n ranges from ~100 to ~580

The gap rule is a mechanical proxy for the reactive visit process itself,
so its high attack rate is close to tautological; that is stated in the
output rather than left for the reader to infer.
"""
import os, json
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
HORIZONS = [1, 2, 3]
LIST_SIZE = 42
SEV_Q, VIS_Q = 0.75, 0.25
GAP_MIN_GRID = [1, 2, 3, 4]


def mean_preds(variant, h):
    pas, mus = [], []
    for fp in sorted(EXP.glob(f"{variant}_s*/preds_uji.npz")):
        z = np.load(fp)
        pas.append(np.asarray(z[f"h{h}_pa"], "float64").ravel())
        mus.append(np.asarray(z[f"h{h}_mu"], "float64").ravel())
    return np.mean(pas, axis=0), np.mean(mus, axis=0), len(pas)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - half) / d), float((c + half) / d)


def score(d, picks, label):
    """picks: boolean mask over all test rows, already period-wise sized."""
    sel = d[picks]
    obs = sel[sel["yl"].notna()]
    k = int((obs["yl"] > 0).sum())
    n = len(obs)
    lo, hi = wilson(k, n)
    return dict(rule=label, listed=int(len(sel)),
                visited=n, visited_share=float(n / max(len(sel), 1)),
                attacked=k,
                attack_rate=float(k / n) if n else np.nan,
                wilson_lo=lo, wilson_hi=hi,
                mean_Y=float(obs["yl"].mean()) if n else np.nan,
                total_Y=float(obs["yl"].sum()) if n else np.nan)


def topk_mask(d, col, k=LIST_SIZE, ascending=False, pool=None):
    """Per-period top-k by one column, optionally inside a candidate pool."""
    m = pd.Series(False, index=d.index)
    for t, g in d.groupby("t"):
        gg = g if pool is None else g[pool.loc[g.index]]
        if len(gg) == 0:
            continue
        take = min(k, len(gg))
        idx = gg.nsmallest(take, col).index if ascending else gg.nlargest(take, col).index
        m.loc[idx] = True
    return m


def main():
    sm = pd.read_csv(PREP / "sample_meta_uji.csv")
    gap = pd.read_csv(WORK / "gap_lookup.csv")
    rows, summaries = [], []

    for h in HORIZONS:
        yl = np.load(PREP / f"y_ltsj_uji_h{h}.npy").astype("float64")
        pa, mu, nseed = mean_preds(VARIANT, h)
        d = pd.DataFrame({"t": sm["t"].values, "vid": sm["vid"].values,
                          "yl": yl, "pa": pa, "mu": mu})
        # the gap is a property of the origin row, known when the list is drawn
        d = d.merge(gap, on=["vid", "t"], how="left")
        assert d["gap"].notna().all(), "gap join incomplete"
        d["sq"] = d.groupby("t")["mu"].rank(pct=True)
        d["pq"] = d.groupby("t")["pa"].rank(pct=True)

        rules = {
            "gap_length": topk_mask(d, "gap"),
            "pred_severity": topk_mask(d, "mu"),
            "low_visit_prob": topk_mask(d, "pa", ascending=True),
        }
        # B_t as published: an intersection, not a fixed-size list
        rules["blind_spot_Bt"] = (d["sq"] >= SEV_Q) & (d["pq"] <= VIS_Q)
        # the composite the manuscript says is "not run here"
        for g_min in GAP_MIN_GRID:
            pool = d["gap"] >= g_min
            rules[f"gap_ge{g_min}_then_severity"] = topk_mask(d, "mu", pool=pool)

        for label, mask in rules.items():
            r = score(d, mask, label)
            r["h"] = h
            r["n_seeds"] = nseed
            rows.append(r)

        allobs = d[d["yl"].notna()]
        rows.append(dict(rule="all_observed_rows", listed=len(d), visited=len(allobs),
                         visited_share=float(len(allobs) / len(d)),
                         attacked=int((allobs["yl"] > 0).sum()),
                         attack_rate=float((allobs["yl"] > 0).mean()),
                         wilson_lo=wilson(int((allobs["yl"] > 0).sum()), len(allobs))[0],
                         wilson_hi=wilson(int((allobs["yl"] > 0).sum()), len(allobs))[1],
                         mean_Y=float(allobs["yl"].mean()),
                         total_Y=float(allobs["yl"].sum()), h=h, n_seeds=nseed))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "watchlist_rules.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nGATE h=1 (published: gap 102 visited 97.1%, s-hat 576 78.5%, "
          "B_t 214 64.0%, low p-hat 171 61.4%, all rows 6790 49.4%)")
    g = res[res.h == 1]
    print(g[["rule", "listed", "visited", "visited_share", "attack_rate",
             "wilson_lo", "wilson_hi", "mean_Y", "total_Y"]].round(4).to_string(index=False))

    print("\n=== composite rule across horizons ===")
    c = res[res.rule.str.startswith("gap_ge")]
    print(c[["h", "rule", "listed", "visited", "attack_rate", "wilson_lo",
             "wilson_hi", "mean_Y", "total_Y"]].round(4).to_string(index=False))

    print("\nwrote " + str(OUT / "watchlist_rules.csv"))


if __name__ == "__main__":
    main()
