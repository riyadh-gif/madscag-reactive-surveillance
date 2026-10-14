# ══════════════════════════════════════════════════════════════
# Revision round 1, phase 2: regional and temporal hold-out refits
# ══════════════════════════════════════════════════════════════
"""Reviewer 2 asked for regency-wise and year-block robustness. Both are
hold-out refits, so both go through `leakage_check.fit_two_stage`, whose
body is the proposed system's configuration verbatim and which already
takes the three splits as dictionaries. Nothing in that function changes.

Leave-one-regency-out runs in three variants, because a village's feature
vector can carry its neighbours' information across a regency border:

  A  as fitted            the honest upper bound on what a refit achieves
  B  seal 264, 265        the two neighbour-pressure summaries, built from
                          a five-nearest-neighbour graph fitted once over
                          every village, so neighbours cross borders
  C  seal B plus 85/171/257
                          `coverage_rate`, built at v7_pipeline.py:226 as
                          a panel-wide per-period observation mean, which
                          carries the held-out regency's surveillance
                          intensity into every row of every other regency

The gap between A and C is the measure of how much of a "held-out" result
was never held out at all.

Leave-one-year-out is deliberately not implemented. Holding out 2022 would
train on 2023 to 2026, which is the future leakage this project's own
`leakage_check.py` exists to condemn, and 2020 contributes 264 rows. The
temporal arm is instead four rolling-origin folds with an expanding
training window, the published split being the last of them, so the series
is anchored to a result we have already reported.

Seeds follow `leakage_check.py`'s own convention of two, and the full-panel
comparison is restricted to the same two seeds so that a four-seed ensemble
is never compared against a two-seed one.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from leakage_check import fit_two_stage

WORK = Path(os.environ["MADSCAG_WORK"])
PREP = WORK / "preprocessed_v7"
EXP = WORK / "experiments_v7"
OUT = WORK / "eval_v7" / "revisi1"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["latih", "sahih", "uji"]
SEEDS = [42, 1]
HORIZON = 1
FRAC = 0.50
MIN_OBS = 5
MIN_CELL = 100
SEAL = {"A": [], "B": [264, 265], "C": [264, 265, 85, 171, 257]}
N_FOLDS = 4


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_all(meta_h):
    TAB = {sp: np.load(PREP / f"TAB_{sp}.npy") for sp in SPLITS}
    yd = {sp: {h: np.load(PREP / f"y_diamati_{sp}_h{h}.npy") for h in meta_h}
          for sp in SPLITS}
    yl = {sp: {h: np.load(PREP / f"y_ltsj_{sp}_h{h}.npy") for h in meta_h}
          for sp in SPLITS}
    sm = {sp: pd.read_csv(PREP / f"sample_meta_{sp}.csv") for sp in SPLITS}
    return TAB, yd, yl, sm


def seal_cols(TAB, cols):
    if not cols:
        return TAB
    out = {}
    for sp, a in TAB.items():
        b = a.copy()
        b[:, cols] = 0.0
        out[sp] = b
    return out


def subset(TAB, yd, yl, masks, H):
    T = {sp: TAB[sp][masks[sp]] for sp in SPLITS}
    D = {sp: {h: yd[sp][h][masks[sp]] for h in H} for sp in SPLITS}
    L = {sp: {h: yl[sp][h][masks[sp]] for h in H} for sp in SPLITS}
    return T, D, L


def capture_periods(t, yl, score, frac=FRAC):
    """Per-period capture on one row set."""
    d = pd.DataFrame({"t": t, "yl": yl, "s": score})
    out = {}
    for tt, g in d.groupby("t"):
        o = g[g["yl"].notna()]
        if len(o) < MIN_OBS:
            continue
        k = max(1, int(round(frac * len(o))))
        out[int(tt)] = float(o.nlargest(k, "s")["yl"].sum() / max(o["yl"].sum(), 1e-9))
    return pd.Series(out)


def stored_full(h, seeds=SEEDS):
    mus = []
    for s in seeds:
        mus.append(np.asarray(np.load(EXP / f"BL_XGB2_s{s}" / "preds_uji.npz")[f"h{h}_mu"],
                              "float64").ravel())
    return np.mean(mus, axis=0)


def paired(a, b):
    ix = a.index.intersection(b.index)
    d = (a.loc[ix] - b.loc[ix]).astype("float64")
    se = stats.sem(d) if len(d) > 1 else np.nan
    ci = (stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=se)
          if np.isfinite(se) and se > 0 else (np.nan, np.nan))
    try:
        w = float(stats.wilcoxon(a.loc[ix], b.loc[ix]).pvalue)
    except ValueError:
        w = float("nan")
    return dict(n=len(d), mean_a=float(a.loc[ix].mean()), mean_b=float(b.loc[ix].mean()),
                diff=float(d.mean()), ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                p_wilcoxon=w, wins=int((d > 0).sum()))


def run_loro(meta, TAB, yd, yl, sm):
    H = meta["horizons"]
    full_mu = stored_full(HORIZON)
    rows = []
    for variant, cols in SEAL.items():
        T0 = seal_cols(TAB, cols)
        for kab in sorted(sm["uji"].kab.unique()):
            masks = {sp: (sm[sp].kab != kab).to_numpy() for sp in ["latih", "sahih"]}
            masks["uji"] = (sm["uji"].kab == kab).to_numpy()
            if masks["uji"].sum() < MIN_CELL:
                continue
            T, D, L = subset(T0, yd, yl, masks, H)
            mus = []
            for seed in SEEDS:
                t0 = time.time()
                _, mu = fit_two_stage(T, D, L, meta, HORIZON, seed)
                mus.append(mu)
                log(f"  LORO{variant} {kab} seed {seed} {time.time()-t0:.0f}s")
            mu_refit = np.mean(mus, axis=0)
            t_u = sm["uji"].t.to_numpy()[masks["uji"]]
            yl_u = yl["uji"][HORIZON][masks["uji"]]
            c_refit = capture_periods(t_u, yl_u, mu_refit)
            c_full = capture_periods(t_u, yl_u, full_mu[masks["uji"]])
            r = dict(experiment="loro", variant=variant, fold=kab,
                     rows=int(masks["uji"].sum()),
                     n_train=int(masks["latih"].sum()),
                     train_share_kept=float(masks["latih"].mean()),
                     observed=int(np.isfinite(yl_u).sum()),
                     capture_refit=float(c_refit.mean()),
                     capture_full=float(c_full.mean()),
                     diff=float(c_refit.mean() - c_full.mean()),
                     periods=int(len(c_refit)))
            rows.append(r)
            log(f"  LORO{variant} {kab}: refit {r['capture_refit']:.4f} "
                f"full {r['capture_full']:.4f}")
    return pd.DataFrame(rows)


def run_rolling(meta, TAB, yd, yl, sm):
    """Expanding-window folds cut on the origin period, published split last."""
    H = meta["horizons"]
    allT = np.concatenate([TAB[sp] for sp in SPLITS], axis=0)
    allt = np.concatenate([sm[sp].t.to_numpy() for sp in SPLITS])
    allD = {h: np.concatenate([yd[sp][h] for sp in SPLITS]) for h in H}
    allL = {h: np.concatenate([yl[sp][h] for sp in SPLITS]) for h in H}
    periods = np.array(sorted(np.unique(allt)))
    n_test, n_val = 23, 20
    rows = []
    for k in range(N_FOLDS):
        end = len(periods) - k * n_test
        test_p = periods[end - n_test:end]
        val_p = periods[end - n_test - n_val:end - n_test]
        train_p = periods[:end - n_test - n_val]
        if len(train_p) < 20:
            log(f"  fold {N_FOLDS-k}: training window too short, skipped")
            continue
        masks = {"latih": np.isin(allt, train_p), "sahih": np.isin(allt, val_p),
                 "uji": np.isin(allt, test_p)}
        T = {sp: allT[masks[sp]] for sp in SPLITS}
        D = {sp: {h: allD[h][masks[sp]] for h in H} for sp in SPLITS}
        L = {sp: {h: allL[h][masks[sp]] for h in H} for sp in SPLITS}
        mus = []
        for seed in SEEDS:
            t0 = time.time()
            _, mu = fit_two_stage(T, D, L, meta, HORIZON, seed)
            mus.append(mu)
            log(f"  fold {N_FOLDS-k} seed {seed} {time.time()-t0:.0f}s")
        c = capture_periods(allt[masks["uji"]], allL[HORIZON][masks["uji"]],
                            np.mean(mus, axis=0))
        rows.append(dict(experiment="rolling_origin", fold=int(N_FOLDS - k),
                         train_periods=len(train_p),
                         test_t_min=int(test_p.min()), test_t_max=int(test_p.max()),
                         rows=int(masks["uji"].sum()),
                         observed=int(np.isfinite(allL[HORIZON][masks["uji"]]).sum()),
                         capture=float(c.mean()), periods=int(len(c)),
                         is_published_split=bool(k == 0)))
        log(f"  fold {N_FOLDS-k}: capture {rows[-1]['capture']:.4f}")
    return pd.DataFrame(rows)


def main():
    import json
    with open(PREP / "meta_v7.json", encoding="utf-8") as f:
        meta = json.load(f)
    TAB, yd, yl, sm = load_all(meta["horizons"])
    log(f"loaded; seeds {SEEDS}, horizon {HORIZON}")

    loro = run_loro(meta, TAB, yd, yl, sm)
    loro.to_csv(OUT / "loro_folds.csv", index=False)
    roll = run_rolling(meta, TAB, yd, yl, sm)
    roll.to_csv(OUT / "rolling_origin.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== leave-one-regency-out, capture at f=0.50, h=1, on held-out rows ===")
    print(loro.round(4).to_string(index=False))
    print("\nper variant, fold as the unit:")
    fam = []
    for v in sorted(loro.variant.unique()):
        g = loro[loro.variant == v]
        a = pd.Series(g.capture_refit.values, index=g.fold.values)
        b = pd.Series(g.capture_full.values, index=g.fold.values)
        r = paired(a, b)
        r["variant"] = v
        fam.append(r)
    fam = pd.DataFrame(fam)[["variant", "n", "mean_a", "mean_b", "diff",
                             "ci_lo", "ci_hi", "p_wilcoxon", "wins"]]
    print(fam.round(4).to_string(index=False))
    fam.to_csv(OUT / "loro_tests.csv", index=False)

    print("\n=== spatial leakage: how much a held-out fold was not held out ===")
    piv = loro.pivot(index="fold", columns="variant", values="capture_refit")
    if {"A", "C"}.issubset(piv.columns):
        piv["A_minus_C"] = piv["A"] - piv["C"]
    print(piv.round(4).to_string())

    print("\n=== rolling-origin folds (fold 4 is the published split) ===")
    print(roll.round(4).to_string(index=False))
    print("\nwrote loro_folds.csv, loro_tests.csv, rolling_origin.csv")


if __name__ == "__main__":
    main()
