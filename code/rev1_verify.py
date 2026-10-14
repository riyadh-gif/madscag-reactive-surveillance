# ══════════════════════════════════════════════════════════════
# Revision round 1: verifications and the blind-spot set at every horizon
# ══════════════════════════════════════════════════════════════
"""Four things that have to be checked before they are written down.

1. The single-stage control. If its Stage B is the same fitted object as
   the proposed system's, then every severity-ranked number reported for
   it is identical by construction and controls nothing at the decision
   layer. Checked on the stored prediction arrays, not inferred.

2. The reactivity index. The manuscript reports 2.02 without saying which
   rows it covers. Computed here on the full register and on the test
   block separately, with intervals, so the caption can name its scope.

3. The watchlist tie-break. The published table lists 102 visited rows for
   the gap rule where a strict top-42-per-period gives 101. Both rules are
   evaluated so the revised table can state which one it uses.

4. The blind-spot set at h=2 and h=3. The published statistics exist only
   at h=1, while the deployment argument rests on the longer horizons.

Nothing is retrained.
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu

WORK = Path(os.environ["MADSCAG_WORK"])
INPUT = Path(os.environ["MADSCAG_INPUT"])
PREP = WORK / "preprocessed_v7"
EXP = WORK / "experiments_v7"
OUT = WORK / "eval_v7" / "revisi1"
OUT.mkdir(parents=True, exist_ok=True)
PANEL = INPUT / "dataset" / "datasiap" / "panel_cuaca.csv"

HORIZONS = [1, 2, 3]
LIST_SIZE = 42
SEV_Q, VIS_Q = 0.75, 0.25


def mean_preds(variant, h, key="mu"):
    a = []
    for fp in sorted(EXP.glob(f"{variant}_s*/preds_uji.npz")):
        a.append(np.asarray(np.load(fp)[f"h{h}_{key}"], "float64").ravel())
    return np.mean(a, axis=0), len(a)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - half) / d), float((c + half) / d)


def check_single_stage():
    """Is BL_SINGLE's Stage B the same fitted object as BL_XGB2's?"""
    out = []
    for h in HORIZONS:
        for key in ["mu", "pa"]:
            diffs = []
            for fp in sorted(EXP.glob(f"BL_XGB2_s*/preds_uji.npz")):
                s = fp.parent.name.rsplit("_s", 1)[1]
                g = EXP / f"BL_SINGLE_s{s}" / "preds_uji.npz"
                if not g.exists():
                    continue
                a = np.asarray(np.load(fp)[f"h{h}_{key}"], "float64")
                b = np.asarray(np.load(g)[f"h{h}_{key}"], "float64")
                diffs.append(float(np.max(np.abs(a - b))))
            out.append(dict(horizon=h, array=key, seeds=len(diffs),
                            max_abs_diff=float(np.max(diffs)) if diffs else np.nan,
                            identical=bool(diffs and np.max(diffs) == 0.0)))
    return pd.DataFrame(out)


def reactivity(p, label):
    o = p[p.diamati > 0]
    atk = (o.ltsj.fillna(0) > 0).astype(int)
    gap0 = o.gap == 0
    gap1 = o.gap >= 1
    r0, n0 = atk[gap0].mean(), int(gap0.sum())
    r1, n1 = atk[gap1].mean(), int(gap1.sum())
    # delta method on the log ratio, then back-transform
    se = np.sqrt((1 - r1) / (r1 * n1) + (1 - r0) / (r0 * n0))
    lo, hi = np.exp(np.log(r1 / r0) - 1.96 * se), np.exp(np.log(r1 / r0) + 1.96 * se)
    grad = [float(atk[o.gap == g].mean()) for g in [0, 1, 2, 3]]
    grad.append(float(atk[o.gap >= 4].mean()))
    return dict(scope=label, R=float(r1 / r0), ci_lo=float(lo), ci_hi=float(hi),
                rate_after_gap=float(r1), rate_no_gap=float(r0),
                n_after_gap=n1, n_no_gap=n0,
                gradient_gap_0_1_2_3_4plus=[round(x, 4) for x in grad])


def main():
    sm = pd.read_csv(PREP / "sample_meta_uji.csv")
    gap = pd.read_csv(WORK / "gap_lookup.csv")
    report = {}

    # ---- 1. single-stage control -------------------------------------
    ss = check_single_stage()
    ss.to_csv(OUT / "verify_single_stage.csv", index=False)
    report["single_stage_identical_stageB"] = bool(
        ss[(ss.array == "mu")].identical.all())
    report["single_stage_identical_stageA"] = bool(
        ss[(ss.array == "pa")].identical.all())

    # ---- 2. reactivity index ----------------------------------------
    p = pd.read_csv(PANEL, low_memory=False,
                    usecols=["kab", "kec", "desa", "t", "diamati", "ltsj"])
    p["vid"] = p.kab + "|" + p.kec + "|" + p.desa
    p = p.drop_duplicates(subset=["vid", "t"], keep="last").merge(gap, on=["vid", "t"], how="left")
    rows = [reactivity(p, "full panel"),
            reactivity(p[p.t >= int(sm.t.min())], "test block")]
    react = pd.DataFrame(rows)
    react.to_csv(OUT / "verify_reactivity.csv", index=False)

    # ---- 3. watchlist tie-break -------------------------------------
    tie_rows = []
    for h in HORIZONS:
        yl = np.load(PREP / f"y_ltsj_uji_h{h}.npy").astype("float64")
        mu, _ = mean_preds("BL_XGB2", h)
        d = pd.DataFrame({"t": sm["t"].values, "vid": sm["vid"].values,
                          "yl": yl, "mu": mu}).merge(gap, on=["vid", "t"], how="left")
        for rule, col, asc in [("gap_length", "gap", False), ("pred_severity", "mu", False)]:
            for mode in ["strict", "with_ties"]:
                keep = []
                for t, g in d.groupby("t"):
                    if mode == "strict":
                        keep.append(g.nlargest(LIST_SIZE, col))
                    else:
                        thr = g[col].nlargest(LIST_SIZE).iloc[-1]
                        keep.append(g[g[col] >= thr])
                sel = pd.concat(keep)
                obs = sel[sel.yl.notna()]
                k, n = int((obs.yl > 0).sum()), len(obs)
                lo, hi = wilson(k, n)
                tie_rows.append(dict(horizon=h, rule=rule, tie_rule=mode,
                                     listed=len(sel), visited=n,
                                     attack_rate=float(k / n) if n else np.nan,
                                     wilson_lo=lo, wilson_hi=hi,
                                     mean_Y=float(obs.yl.mean()) if n else np.nan))
    ties = pd.DataFrame(tie_rows)
    ties.to_csv(OUT / "verify_watchlist_tiebreak.csv", index=False)

    # ---- 4. blind-spot set at every horizon --------------------------
    bs_rows = []
    for h in HORIZONS:
        yl = np.load(PREP / f"y_ltsj_uji_h{h}.npy").astype("float64")
        pa, nseed = mean_preds("BL_XGB2", h, "pa")
        mu, _ = mean_preds("BL_XGB2", h, "mu")
        d = pd.DataFrame({"t": sm["t"].values, "vid": sm["vid"].values,
                          "yl": yl, "pa": pa, "mu": mu})
        d["sq"] = d.groupby("t")["mu"].rank(pct=True)
        d["pq"] = d.groupby("t")["pa"].rank(pct=True)
        inB = (d.sq >= SEV_Q) & (d.pq <= VIS_Q)
        obs = d.yl.notna()
        y_in = d.loc[obs & inB, "yl"].to_numpy()
        y_out = d.loc[obs & ~inB, "yl"].to_numpy()
        k_in = int((y_in > 0).sum())
        lo, hi = wilson(k_in, len(y_in))
        rec = dict(horizon=h, n_seeds=nseed, n_B=int(inB.sum()),
                   per_period=float(inB.sum() / d.t.nunique()),
                   n_B_observed=int(len(y_in)),
                   obs_rate_in_B=float(obs[inB].mean()),
                   obs_rate_outside=float(obs[~inB].mean()),
                   attack_rate_in_B=float((y_in > 0).mean()) if len(y_in) else np.nan,
                   wilson_lo=lo, wilson_hi=hi,
                   attack_rate_outside=float((y_out > 0).mean()),
                   mean_sev_in_B=float(y_in.mean()) if len(y_in) else np.nan,
                   mean_sev_outside=float(y_out.mean()))
        if len(y_in) >= 20:
            u, pv = mannwhitneyu(y_in, y_out, alternative="greater")
            rec["rank_biserial"] = float(2 * u / (len(y_in) * len(y_out)) - 1)
            rec["mannwhitney_p"] = float(pv)
        sets = {t: set(g.loc[inB.loc[g.index], "vid"]) for t, g in d.groupby("t")}
        ts = sorted(sets)
        jac = [len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1)
               for a, b in zip(ts, ts[1:]) if sets[a] or sets[b]]
        rec["period_to_period_jaccard"] = float(np.mean(jac)) if jac else np.nan
        bs_rows.append(rec)
    bs = pd.DataFrame(bs_rows)
    bs.to_csv(OUT / "blindspot_horizons.csv", index=False)

    with open(OUT / "verify_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pd.set_option("display.width", 215)
    print("\n=== 1. single-stage control against the proposed system ===")
    print(ss.to_string(index=False))
    print("\n=== 2. reactivity index (manuscript quotes 2.02 with no scope) ===")
    print(react.to_string(index=False))
    print("\n=== 3. watchlist tie-break (published table: gap rule 102 visited, 97.1%) ===")
    print(ties[ties.horizon == 1].round(4).to_string(index=False))
    print("\n=== 4. blind-spot set at every horizon ===")
    print(bs.round(4).to_string(index=False))
    print("\nwrote verify_single_stage.csv, verify_reactivity.csv, "
          "verify_watchlist_tiebreak.csv, blindspot_horizons.csv")


if __name__ == "__main__":
    main()
