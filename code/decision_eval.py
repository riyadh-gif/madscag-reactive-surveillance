# ══════════════════════════════════════════════════════════════
# Decision-layer evaluation for every stored model
# ══════════════════════════════════════════════════════════════
"""
Runs the patrol-budget simulation and the lead-time analysis for each
variant in experiments_v7, reading its stored preds_uji.npz rather than
retraining anything.

The decision layer is the part of this study that actually drives field
behaviour, so it has to be evaluated for the *best* implementation of the
factorisation, not only for the recurrent one it was first built on.
Ranking uses conditional severity mu (the "model-mu" rule), which the
original study found to dominate expected-severity ranking: multiplying
by attendance penalises exactly the low-attendance villages where hidden
outbreaks sit.

Seeds are averaged within a variant before ranking, matching how an
ensemble would be deployed.

Output: eval_v7/decision_compare.csv
"""
import os, sys, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

INPUT_DIR = Path(os.environ.get("MADSCAG_INPUT", "."))
WORK_DIR = Path(os.environ.get("MADSCAG_WORK", "."))
PREP_DIR = WORK_DIR / "preprocessed_v7"
EXP_DIR = WORK_DIR / "experiments_v7"
EVAL_DIR = WORK_DIR / "eval_v7"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

BUDGET_FRACS = [0.25, 0.50, 0.75]
SIM_SEED = 42


def log(m):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_meta():
    with open(PREP_DIR / "meta_v7.json", encoding="utf-8") as f:
        return json.load(f)


def variant_preds(name, h=1):
    """Mean pa and mu over the variant's seeds, on the test split."""
    pas, mus = [], []
    for fp in sorted(EXP_DIR.glob(f"{name}_s*/preds_uji.npz")):
        z = np.load(fp)
        pas.append(np.asarray(z[f"h{h}_pa"], dtype="float64").ravel())
        mus.append(np.asarray(z[f"h{h}_mu"], dtype="float64").ravel())
    if not pas:
        return None, None, 0
    return np.mean(pas, axis=0), np.mean(mus, axis=0), len(pas)


def budget_sim(df, fracs, seed=SIM_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for frac in fracs:
        for pol in ["model", "persistence", "random"]:
            det, cap = [], []
            for _, g in df.groupby("t"):
                obs = g[g["yl"].notna()]
                if len(obs) < 5:
                    continue
                k = max(1, int(round(frac * len(obs))))
                if pol == "model":
                    sel = obs.nlargest(k, "score")
                elif pol == "persistence":
                    sel = obs.nlargest(k, "pres")
                else:
                    sel = obs.sample(n=k, random_state=int(rng.integers(1e9)))
                det.append((sel["yl"] > 0).sum() / max((obs["yl"] > 0).sum(), 1))
                cap.append(sel["yl"].sum() / max(obs["yl"].sum(), 1e-9))
            if det:
                rows.append(dict(frac=frac, policy=pol,
                                 detection_rate=float(np.mean(det)),
                                 severity_capture=float(np.mean(cap))))
    return pd.DataFrame(rows)


def leadtime(sm, yl1, score, q=0.70):
    """Share of eventual outbreak villages already in the top (1-q) tier."""
    idx = {(v, int(t)): i
           for i, (v, t) in enumerate(zip(sm["vid"], sm["t"]))}
    thr = float(np.quantile(score, q))
    flags, n = [], 0
    vids, ts = sm["vid"].to_numpy(), sm["t"].to_numpy()
    for i in range(len(sm)):
        if not (yl1[i] > 0):
            continue
        j = idx.get((vids[i], int(ts[i]) - 0))
        if j is None:
            continue
        n += 1
        flags.append(score[j] > thr)
    return (float(np.mean(flags)) if flags else np.nan), n


def main():
    meta = load_meta()
    sm = pd.read_csv(PREP_DIR / "sample_meta_uji.csv")
    yl1 = np.load(PREP_DIR / "y_ltsj_uji_h1.npy")
    tab = np.load(PREP_DIR / "TAB_uji.npy")
    pres_i = 3 * meta["n_features"] + meta["tab_extras"].index("ltsj_lag1_last")
    pres = tab[:, pres_i].astype("float64")

    names = sorted({p.parent.name.rsplit("_s", 1)[0]
                    for p in EXP_DIR.glob("*/preds_uji.npz")})
    log(f"variants with stored predictions: {len(names)}")

    rows = []
    for name in names:
        pa, mu, nseed = variant_preds(name)
        if pa is None:
            continue
        df = pd.DataFrame({"t": sm["t"], "yl": yl1, "score": mu, "pres": pres})
        b = budget_sim(df, BUDGET_FRACS)
        lt, n_out = leadtime(sm, yl1, mu)
        rec = dict(variant=name, n_seeds=nseed, leadtime_flag_rate=lt,
                   n_outbreaks=n_out)
        for _, r in b[b.policy == "model"].iterrows():
            rec[f"det@{r.frac:.2f}"] = r.detection_rate
            rec[f"sev@{r.frac:.2f}"] = r.severity_capture
        rows.append(rec)
        log(f"  {name:16s} n={nseed}  sev@0.50={rec.get('sev@0.50', float('nan')):.4f}")

    # reference policies do not depend on the model
    df0 = pd.DataFrame({"t": sm["t"], "yl": yl1, "score": pres, "pres": pres})
    ref = budget_sim(df0, BUDGET_FRACS)
    for pol in ["persistence", "random"]:
        rec = dict(variant=f"REF_{pol}", n_seeds=0)
        for _, r in ref[ref.policy == pol].iterrows():
            rec[f"det@{r.frac:.2f}"] = r.detection_rate
            rec[f"sev@{r.frac:.2f}"] = r.severity_capture
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("sev@0.50", ascending=False)
    out.to_csv(EVAL_DIR / "decision_compare.csv", index=False)
    log(f"✓ {EVAL_DIR / 'decision_compare.csv'}")
    cols = ["variant", "n_seeds", "det@0.25", "sev@0.25", "det@0.50",
            "sev@0.50", "det@0.75", "sev@0.75", "leadtime_flag_rate"]
    print(out[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
