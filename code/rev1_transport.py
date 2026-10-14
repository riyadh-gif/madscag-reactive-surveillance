# ══════════════════════════════════════════════════════════════
# Revision round 1, phase 2: does Stage B transport across propensity?
# ══════════════════════════════════════════════════════════════
"""Reviewer 1: Stage B is fitted only on rows with M=1, yet its score then
ranks villages with a low probability of being visited, and under the
paper's own MNAR argument the conditional law of Y need not transport.
The reviewer asks for the assumption to be stated and its sensitivity
tested where possible.

This is the test the register permits. Stage B is refitted on the
high-visit-propensity third of the observed training rows and evaluated on
the low-propensity third of the observed test rows, then the other way
round, with matched-stratum controls so the comparison is like for like.
The transport penalty is the cross-stratum error minus the matched-stratum
error on an identical set of evaluation rows.

Propensity on the training rows is cross-fitted, not fitted in sample: the
training block is cut in two by period, Stage A is fitted on one half and
predicts the other, and vice versa, so no row is scored by a model that saw
it. Validation and test propensities come from the stored BL_XGB2 runs,
which are already out of sample.

What this supports: whether the conditional severity function learned where
visits are frequent carries to villages that are rarely visited but were
visited anyway. What it cannot support: anything about villages never
visited at all. Low-propensity observed rows are overwhelmingly reactive
visits, so they are a selected high-severity subset, not a sample of the
low-propensity population. That belongs beside the result, not in a
limitations paragraph.
"""
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error

WORK = Path(os.environ["MADSCAG_WORK"])
PREP = WORK / "preprocessed_v7"
EXP = WORK / "experiments_v7"
OUT = WORK / "eval_v7" / "revisi1"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1, 2, 7]
HORIZON = 1
FRAC = 0.50
MIN_OBS = 5
XGB_KW = dict(n_estimators=600, max_depth=6, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              objective="reg:absoluteerror", eval_metric="mae",
              early_stopping_rounds=50, n_jobs=4, verbosity=0)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def stage_a_crossfit(TAB, yd, t, meta, seed=42):
    """Out-of-fold visit probability for every training row."""
    import xgboost as xgb
    cw = meta["class_weights"][str(HORIZON)]["pos"]
    mid = np.median(np.unique(t))
    half = t <= mid
    out = np.full(len(t), np.nan)
    for tr, te in [(half, ~half), (~half, half)]:
        m = tr & ~np.isnan(yd)
        clf = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                scale_pos_weight=cw, eval_metric="aucpr",
                                random_state=seed, n_jobs=4, verbosity=0)
        clf.fit(TAB[m], yd[m].astype(int), verbose=False)
        out[te] = clf.predict_proba(TAB[te])[:, 1]
    return out


def stored_pa_mu(split, h, seeds=SEEDS):
    pas, mus = [], []
    for s in seeds:
        z = np.load(EXP / f"BL_XGB2_s{s}" / f"preds_{split}.npz")
        pas.append(np.asarray(z[f"h{h}_pa"], "float64").ravel())
        mus.append(np.asarray(z[f"h{h}_mu"], "float64").ravel())
    return np.mean(pas, axis=0), np.mean(mus, axis=0)


def terciles_within_period(p, t, observed):
    """Label observed rows HIGH / MID / LOW by within-period propensity."""
    lab = np.full(len(p), "", dtype=object)
    df = pd.DataFrame({"t": t, "p": p, "obs": observed})
    for tt, g in df[df.obs].groupby("t"):
        q = g["p"].quantile([1 / 3, 2 / 3]).to_numpy()
        lo, hi = q[0], q[1]
        idx = g.index.to_numpy()
        v = g["p"].to_numpy()
        lab[idx[v <= lo]] = "LOW"
        lab[idx[(v > lo) & (v < hi)]] = "MID"
        lab[idx[v >= hi]] = "HIGH"
    return lab


def capture_periods(t, yl, score):
    d = pd.DataFrame({"t": t, "yl": yl, "s": score})
    out = {}
    for tt, g in d.groupby("t"):
        o = g[g["yl"].notna()]
        if len(o) < MIN_OBS:
            continue
        k = max(1, int(round(FRAC * len(o))))
        out[int(tt)] = float(o.nlargest(k, "s")["yl"].sum() / max(o["yl"].sum(), 1e-9))
    return pd.Series(out)


def main():
    import xgboost as xgb
    with open(PREP / "meta_v7.json", encoding="utf-8") as f:
        meta = json.load(f)
    TAB = {sp: np.load(PREP / f"TAB_{sp}.npy") for sp in ["latih", "sahih", "uji"]}
    yl = {sp: np.load(PREP / f"y_ltsj_{sp}_h{HORIZON}.npy").astype("float64")
          for sp in ["latih", "sahih", "uji"]}
    yd_l = np.load(PREP / f"y_diamati_latih_h{HORIZON}.npy").ravel()
    sm = {sp: pd.read_csv(PREP / f"sample_meta_{sp}.csv") for sp in ["latih", "sahih", "uji"]}

    log("cross-fitting Stage A on the training block for out-of-fold propensity")
    pa_l = stage_a_crossfit(TAB["latih"], yd_l, sm["latih"].t.to_numpy(), meta)
    pa_s, _ = stored_pa_mu("sahih", HORIZON)
    pa_u, mu_u_full = stored_pa_mu("uji", HORIZON)

    obs = {sp: ~np.isnan(yl[sp]) for sp in ["latih", "sahih", "uji"]}
    lab = {"latih": terciles_within_period(pa_l, sm["latih"].t.to_numpy(), obs["latih"]),
           "sahih": terciles_within_period(pa_s, sm["sahih"].t.to_numpy(), obs["sahih"]),
           "uji": terciles_within_period(pa_u, sm["uji"].t.to_numpy(), obs["uji"])}
    for sp in lab:
        c = pd.Series(lab[sp][obs[sp]]).value_counts().to_dict()
        log(f"  {sp}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))

    # fit Stage B inside each training stratum, predict once on all observed test rows
    preds = {}
    for stratum in ["HIGH", "LOW"]:
        per_seed = []
        for seed in SEEDS:
            tr = obs["latih"] & (lab["latih"] == stratum)
            va = obs["sahih"] & (lab["sahih"] == stratum)
            t0 = time.time()
            reg = xgb.XGBRegressor(random_state=seed, **XGB_KW)
            reg.fit(TAB["latih"][tr], yl["latih"][tr],
                    eval_set=[(TAB["sahih"][va], yl["sahih"][va])], verbose=False)
            per_seed.append(np.clip(reg.predict(TAB["uji"]), 0, None))
            log(f"  Stage B trained on {stratum} (n={int(tr.sum())}), seed {seed}, "
                f"{time.time()-t0:.0f}s")
        preds[stratum] = np.mean(per_seed, axis=0)

    # transport penalties on identical evaluation rows
    rows = []
    for target in ["LOW", "HIGH"]:
        m = obs["uji"] & (lab["uji"] == target)
        matched = target
        cross = "HIGH" if target == "LOW" else "LOW"
        y = yl["uji"][m]
        mae_matched = mean_absolute_error(y, preds[matched][m])
        mae_cross = mean_absolute_error(y, preds[cross][m])
        mae_full = mean_absolute_error(y, mu_u_full[m])
        rows.append(dict(evaluated_on=target, n_rows=int(m.sum()),
                         trained_matched=matched, trained_cross=cross,
                         mae_matched=float(mae_matched), mae_cross=float(mae_cross),
                         transport_penalty=float(mae_cross - mae_matched),
                         mae_full_panel_model=float(mae_full),
                         mean_y=float(np.mean(y)),
                         attack_rate=float(np.mean(y > 0))))
    pen = pd.DataFrame(rows)

    # Family C: decision layer inside the lowest-propensity quartile
    q = np.nanquantile(pa_u[obs["uji"]], 0.25)
    mq = obs["uji"] & (pa_u <= q)
    t_q = sm["uji"].t.to_numpy()[mq]
    c_full = capture_periods(t_q, yl["uji"][mq], mu_u_full[mq])
    c_high = capture_periods(t_q, yl["uji"][mq], preds["HIGH"][mq])
    ix = c_full.index.intersection(c_high.index)
    d = (c_high.loc[ix] - c_full.loc[ix]).astype("float64")
    se = stats.sem(d)
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=se)
    famC = dict(test="transportability", unit="test period", n_units=len(d),
                metric="capture C(0.50) in the lowest propensity quartile",
                value_full=float(c_full.loc[ix].mean()),
                value_high_trained=float(c_high.loc[ix].mean()),
                delta=float(d.mean()), ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                p_wilcoxon=float(stats.wilcoxon(c_high.loc[ix], c_full.loc[ix]).pvalue),
                p_ttest=float(stats.ttest_rel(c_high.loc[ix], c_full.loc[ix]).pvalue),
                wins=int((d > 0).sum()), n_rows=int(mq.sum()))

    pen.to_csv(OUT / "transport_penalty.csv", index=False)
    pd.DataFrame([famC]).to_csv(OUT / "transport_familyC.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== transport penalty: Stage B fitted in one propensity stratum, "
          "scored in another ===")
    print(pen.round(4).to_string(index=False))
    print("\n=== Family C: decision layer inside the lowest propensity quartile ===")
    for k, v in famC.items():
        print(f"  {k:28s} {v if not isinstance(v, float) else round(v, 4)}")
    print("\nwrote transport_penalty.csv, transport_familyC.csv")


if __name__ == "__main__":
    main()
