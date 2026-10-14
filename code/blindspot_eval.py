# ══════════════════════════════════════════════════════════════
# Measuring the blind-spot set
# ══════════════════════════════════════════════════════════════
"""Counts B_t and tests whether it holds what the definition claims.

Eq. (4) of the manuscript defines the blind-spot set as the villages whose
predicted conditional severity is high while their predicted probability
of being visited is low. Nothing so far measures it. The Introduction's
second contribution rests on that set being publishable mid-season and
worth publishing, and neither half has a number behind it.

Both thresholds are within-period quantiles of the predicted quantities,
not absolute cut-offs, because the patrol force is a fixed size and a
watchlist has to be a fixed size too. The defaults put a village in B_t
when s-hat is in the period's top quartile and p-hat in its bottom
quartile.

The test is the awkward one this data allows. Villages in B_t are by
construction the ones least likely to be visited, so most of them carry no
outcome. The subset that was visited anyway is the only place the claim
can be checked, and that subset is not a random sample of B_t. The result
is reported with that stated, not hidden: it is a lower-bound style check,
not a clean validation.

A rank-sum test on severity accompanies the rates, because the observed
B_t subset is small and a difference in means alone would not carry.

Output: eval_v7/blindspot.json
"""
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

WORK_DIR = Path(os.environ.get("MADSCAG_WORK", "."))
PREP_DIR = WORK_DIR / "preprocessed_v7"
EXP_DIR = WORK_DIR / "experiments_v7"
EVAL_DIR = WORK_DIR / "eval_v7"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

VARIANT = "BL_XGB2"
HORIZON = 1
SEV_Q = 0.75         # s-hat at or above this within-period quantile
VIS_Q = 0.25         # p-hat at or below this within-period quantile
SENSITIVITY = [(0.75, 0.25), (0.80, 0.20), (0.90, 0.10), (0.70, 0.30)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def mean_preds(variant, h):
    pas, mus = [], []
    for fp in sorted(EXP_DIR.glob(f"{variant}_s*/preds_uji.npz")):
        z = np.load(fp)
        pas.append(np.asarray(z[f"h{h}_pa"], dtype="float64").ravel())
        mus.append(np.asarray(z[f"h{h}_mu"], dtype="float64").ravel())
    if not pas:
        raise SystemExit(f"no stored predictions for {variant}")
    return np.mean(pas, axis=0), np.mean(mus, axis=0), len(pas)


def evaluate(d, sev_q, vis_q):
    """Everything Section 4.3 reports, for one threshold pair."""
    inB = (d["sq"] >= sev_q) & (d["pq"] <= vis_q)
    n_periods = d["t"].nunique()
    obs = d["yl"].notna()

    o_in = obs & inB
    o_out = obs & ~inB
    y_in = d.loc[o_in, "yl"].to_numpy()
    y_out = d.loc[o_out, "yl"].to_numpy()

    res = dict(
        sev_q=sev_q, vis_q=vis_q,
        n_B=int(inB.sum()),
        share_of_panel=float(inB.mean()),
        per_period=float(inB.sum() / n_periods),
        n_B_observed=int(o_in.sum()),
        obs_rate_in_B=float(obs[inB].mean()) if inB.sum() else float("nan"),
        obs_rate_outside=float(obs[~inB].mean()),
        attack_rate_in_B=float((y_in > 0).mean()) if len(y_in) else float("nan"),
        attack_rate_outside=float((y_out > 0).mean()),
        mean_sev_in_B=float(y_in.mean()) if len(y_in) else float("nan"),
        mean_sev_outside=float(y_out.mean()),
    )
    if len(y_in) >= 20:
        u, p = mannwhitneyu(y_in, y_out, alternative="greater")
        # rank-biserial: the readable effect size for a rank-sum test
        res["rank_biserial"] = float(2 * u / (len(y_in) * len(y_out)) - 1)
        res["mannwhitney_p"] = float(p)
    return res


def main():
    sm = pd.read_csv(PREP_DIR / "sample_meta_uji.csv")
    yl = np.load(PREP_DIR / f"y_ltsj_uji_h{HORIZON}.npy")
    pa, mu, nseed = mean_preds(VARIANT, HORIZON)
    log(f"{VARIANT}: {nseed} seeds, {len(sm)} test rows, "
        f"{sm['t'].nunique()} periods, {np.isfinite(yl).sum()} observed")

    d = pd.DataFrame({"t": sm["t"], "vid": sm["vid"], "yl": yl,
                      "pa": pa, "mu": mu})
    # within-period ranks: the watchlist is drawn fresh each period
    d["sq"] = d.groupby("t")["mu"].rank(pct=True)
    d["pq"] = d.groupby("t")["pa"].rank(pct=True)

    main_res = evaluate(d, SEV_Q, VIS_Q)
    sens = [evaluate(d, s, v) for s, v in SENSITIVITY]

    # how stable is the watchlist from one period to the next? a list that
    # reshuffles completely every period is not something a patrol office
    # can act on.
    inB = (d["sq"] >= SEV_Q) & (d["pq"] <= VIS_Q)
    sets = {t: set(g.loc[inB.loc[g.index], "vid"])
            for t, g in d.groupby("t")}
    ts = sorted(sets)
    jac = [len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1)
           for a, b in zip(ts, ts[1:]) if sets[a] or sets[b]]
    main_res["period_to_period_jaccard"] = float(np.mean(jac)) if jac else float("nan")

    out = dict(variant=VARIANT, horizon=HORIZON, n_seeds=nseed,
               main=main_res, sensitivity=sens)
    with open(EVAL_DIR / "blindspot.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    m = main_res
    print(f"\nB_t at s-hat>={SEV_Q}, p-hat<={VIS_Q}")
    print(f"  size            {m['n_B']} rows, {m['share_of_panel']*100:.1f}% of panel, "
          f"{m['per_period']:.1f} villages per period")
    print(f"  observed        {m['n_B_observed']} ({m['obs_rate_in_B']*100:.1f}% of B_t) "
          f"vs {m['obs_rate_outside']*100:.1f}% outside")
    print(f"  attack rate     {m['attack_rate_in_B']*100:.1f}% inside "
          f"vs {m['attack_rate_outside']*100:.1f}% outside")
    print(f"  mean severity   {m['mean_sev_in_B']:.4f} vs {m['mean_sev_outside']:.4f}")
    if "mannwhitney_p" in m:
        print(f"  rank-sum        rb={m['rank_biserial']:.3f}, p={m['mannwhitney_p']:.2e}")
    print(f"  list stability  mean Jaccard between consecutive periods "
          f"{m['period_to_period_jaccard']:.3f}")
    print("\nsensitivity")
    print(pd.DataFrame(sens)[["sev_q", "vis_q", "n_B", "per_period",
                              "n_B_observed", "attack_rate_in_B",
                              "attack_rate_outside"]].round(4).to_string(index=False))
    log(f"✓ {EVAL_DIR / 'blindspot.json'}")


if __name__ == "__main__":
    main()
