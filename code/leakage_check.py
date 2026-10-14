# ══════════════════════════════════════════════════════════════
# What a random split would have reported
# ══════════════════════════════════════════════════════════════
"""Quantifies the cost of splitting this panel at random.

The manuscript claims, in Related Work, that no forecast in this area has
been scored under a leakage-free chronological protocol. That is a claim
about other people's practice; this script measures what the practice
would have bought us, on our own data, with our own model.

Both arms fit the identical two-stage XGBoost of baselines.run_xgb2, with
identical hyperparameters, identical horizons and identical seeds. The
only difference is which rows land in which block:

  chrono   the pipeline's own chronological blocks (69,004/15,850/14,649)
  random   the same rows pooled and reshuffled, cut at the same sizes

A random split leaks in two directions here, and both are worth naming.
It puts future periods into the training block, and because every row is
a 12-period window, it also puts windows that overlap a test row's own
history into training. The second channel is the quieter one.

Metrics are recomputed inside this script for BOTH arms rather than read
from metrics_full.csv for the chronological one, so the comparison cannot
be an artefact of two different metric implementations. The chronological
numbers should reproduce metrics_full.csv; the script prints both so the
agreement is visible.

Output: eval_v7/leakage_check.json
"""
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

WORK_DIR = Path(os.environ.get("MADSCAG_WORK", "."))
PREP_DIR = WORK_DIR / "preprocessed_v7"
EVAL_DIR = WORK_DIR / "eval_v7"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["latih", "sahih", "uji"]
SEEDS = [42, 1]
SHUFFLE_SEED = 20260914


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_all(meta):
    """Returns TAB, yd, yl per split, plus the block sizes."""
    H = meta["horizons"]
    TAB = {sp: np.load(PREP_DIR / f"TAB_{sp}.npy") for sp in SPLITS}
    yd = {sp: {h: np.load(PREP_DIR / f"y_diamati_{sp}_h{h}.npy") for h in H}
          for sp in SPLITS}
    yl = {sp: {h: np.load(PREP_DIR / f"y_ltsj_{sp}_h{h}.npy") for h in H}
          for sp in SPLITS}
    return TAB, yd, yl, [len(TAB[sp]) for sp in SPLITS]


def reshuffle(TAB, yd, yl, sizes, H):
    """Pool every row, permute once, cut at the same three sizes."""
    rng = np.random.default_rng(SHUFFLE_SEED)
    allT = np.concatenate([TAB[sp] for sp in SPLITS], axis=0)
    perm = rng.permutation(len(allT))
    cuts = np.cumsum(sizes)[:-1]
    idx = np.split(perm, cuts)
    T2, yd2, yl2 = {}, {}, {}
    for sp, ix in zip(SPLITS, idx):
        T2[sp] = allT[ix]
        yd2[sp] = {h: np.concatenate([yd[s][h] for s in SPLITS])[ix] for h in H}
        yl2[sp] = {h: np.concatenate([yl[s][h] for s in SPLITS])[ix] for h in H}
    return T2, yd2, yl2


def fit_two_stage(TAB, yd, yl, meta, h, seed):
    """Verbatim configuration of baselines.run_xgb2, one horizon."""
    import xgboost as xgb
    cw = meta["class_weights"][str(h)]["pos"]
    m = ~np.isnan(yd["latih"][h])
    ms = ~np.isnan(yd["sahih"][h])
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=cw, eval_metric="aucpr",
        early_stopping_rounds=50, random_state=seed, n_jobs=4, verbosity=0)
    clf.fit(TAB["latih"][m], yd["latih"][h][m].astype(int),
            eval_set=[(TAB["sahih"][ms], yd["sahih"][h][ms].astype(int))],
            verbose=False)
    mo = ~np.isnan(yl["latih"][h])
    mso = ~np.isnan(yl["sahih"][h])
    reg = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="reg:absoluteerror", eval_metric="mae",
        early_stopping_rounds=50, random_state=seed, n_jobs=4, verbosity=0)
    reg.fit(TAB["latih"][mo], yl["latih"][h][mo],
            eval_set=[(TAB["sahih"][mso], yl["sahih"][h][mso])],
            verbose=False)
    pa = clf.predict_proba(TAB["uji"])[:, 1]
    mu = np.clip(reg.predict(TAB["uji"]), 0, None)
    return pa, mu


def score(pa, mu, yd_u, yl_u, ref_mean):
    """The columns Section 4.1 needs, nothing more."""
    out = {}
    vd = ~np.isnan(yd_u)
    if vd.sum() and len(np.unique(yd_u[vd])) == 2:
        yv, pv = yd_u[vd].astype(int), pa[vd]
        out["auc_a"] = float(roc_auc_score(yv, pv))
        out["ap_a"] = float(average_precision_score(yv, pv))
    vl = ~np.isnan(yl_u)
    yv, mv, cv = yl_u[vl], mu[vl], (pa * mu)[vl]
    out["mae_b"] = float(mean_absolute_error(yv, mv))
    yb = (yv > 0).astype(int)
    if len(np.unique(yb)) == 2:
        out["auc_b_nz"] = float(roc_auc_score(yb, mv))
    out["mae_c"] = float(mean_absolute_error(yv, cv))
    bl_mean = float(np.mean(np.abs(yv - ref_mean)))
    bl_zero = float(np.mean(np.abs(yv)))
    out["skill_c_vs_mean"] = 1.0 - out["mae_c"] / max(bl_mean, 1e-8)
    out["skill_c_vs_zero"] = 1.0 - out["mae_c"] / max(bl_zero, 1e-8)
    return out


def main():
    with open(PREP_DIR / "meta_v7.json", encoding="utf-8") as f:
        meta = json.load(f)
    H = meta["horizons"]
    TAB, yd, yl, sizes = load_all(meta)
    log(f"block sizes {sizes}, horizons {H}")

    arms = {"chrono": (TAB, yd, yl)}
    arms["random"] = reshuffle(TAB, yd, yl, sizes, H)
    log(f"reshuffled under seed {SHUFFLE_SEED}")

    rows = []
    for arm, (T, D, L) in arms.items():
        # each protocol is judged against a constant fitted on its OWN
        # training block; using the chronological mean for both would
        # hand one arm an easier reference.
        for h in H:
            ref = float(np.nanmean(L["latih"][h]))
            for seed in SEEDS:
                pa, mu = fit_two_stage(T, D, L, meta, h, seed)
                r = dict(arm=arm, horizon=h, seed=seed, ref_mean=ref)
                r.update(score(pa, mu, D["uji"][h], L["uji"][h], ref))
                rows.append(r)
                log(f"  {arm:7s} h={h} s={seed}  auc_a={r.get('auc_a', float('nan')):.4f}"
                    f"  mae_c={r['mae_c']:.4f}  skill_vs_zero={r['skill_c_vs_zero']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "leakage_check.csv", index=False)
    piv = df.groupby(["arm", "horizon"])[
        ["auc_a", "ap_a", "mae_b", "auc_b_nz", "mae_c",
         "skill_c_vs_mean", "skill_c_vs_zero"]].mean().round(4)
    print(piv.to_string())
    with open(EVAL_DIR / "leakage_check.json", "w", encoding="utf-8") as f:
        json.dump({"shuffle_seed": SHUFFLE_SEED, "seeds": SEEDS,
                   "block_sizes": sizes,
                   "summary": json.loads(piv.reset_index().to_json(orient="records"))},
                  f, indent=2)
    log(f"✓ {EVAL_DIR / 'leakage_check.json'}")


if __name__ == "__main__":
    main()
