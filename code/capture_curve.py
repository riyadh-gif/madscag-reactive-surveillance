# ══════════════════════════════════════════════════════════════
# Budget capture curve for every ranking rule the paper discusses
# ══════════════════════════════════════════════════════════════
"""Computes C(f) over a full budget grid for four ways of ordering villages.

Section 3.7 states that ranking by conditional severity s-hat rather than
by the aggregate p-hat * s-hat is consequential rather than cosmetic, and
promises Section 4 will show the cost of choosing wrongly. decision_eval.py
only ever ranks by s-hat, so the aggregate curve has never been computed.
This script computes both, alongside the two reference policies, on a grid
dense enough to draw.

Nothing is retrained: the four stored BL_XGB2 seeds are read from their
preds_uji.npz and averaged, which is how an ensemble would be deployed.

The four rules:
  model_sev   order by s-hat            (the deployed rule)
  model_agg   order by p-hat * s-hat    (the rule Section 3.1 rejects)
  persistence order by last recorded severity
  random      uniform draw, averaged over repeats

Output: eval_v7/capture_curve.csv
"""
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd

WORK_DIR = Path(os.environ.get("MADSCAG_WORK", "."))
PREP_DIR = WORK_DIR / "preprocessed_v7"
EXP_DIR = WORK_DIR / "experiments_v7"
EVAL_DIR = WORK_DIR / "eval_v7"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

VARIANT = "BL_XGB2"
HORIZON = 1
FRACS = np.round(np.arange(0.05, 0.96, 0.05), 2)
RANDOM_REPEATS = 20
SIM_SEED = 42
MIN_OBS = 5          # periods with fewer observed villages are skipped


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


def capture(df, col, frac, rng=None):
    """Share of the period's recorded outbreak severity inside the top f."""
    caps, dets = [], []
    for _, g in df.groupby("t"):
        o = g[g["yl"].notna()]
        if len(o) < MIN_OBS:
            continue
        k = max(1, int(round(frac * len(o))))
        if col == "random":
            sel = o.sample(n=k, random_state=int(rng.integers(1e9)))
        else:
            sel = o.nlargest(k, col)
        caps.append(sel["yl"].sum() / max(o["yl"].sum(), 1e-9))
        dets.append((sel["yl"] > 0).sum() / max((o["yl"] > 0).sum(), 1))
    return float(np.mean(caps)), float(np.mean(dets))


def main():
    with open(PREP_DIR / "meta_v7.json", encoding="utf-8") as f:
        meta = json.load(f)
    sm = pd.read_csv(PREP_DIR / "sample_meta_uji.csv")
    yl = np.load(PREP_DIR / f"y_ltsj_uji_h{HORIZON}.npy")
    tab = np.load(PREP_DIR / "TAB_uji.npy")
    pres_i = 3 * meta["n_features"] + meta["tab_extras"].index("ltsj_lag1_last")
    pres = tab[:, pres_i].astype("float64")

    pa, mu, nseed = mean_preds(VARIANT, HORIZON)
    log(f"{VARIANT}: {nseed} seeds averaged, {len(sm)} test rows, "
        f"{sm['t'].nunique()} periods")

    df = pd.DataFrame({"t": sm["t"], "yl": yl, "model_sev": mu,
                       "model_agg": pa * mu, "persistence": pres})

    rng = np.random.default_rng(SIM_SEED)
    rows = []
    for f in FRACS:
        for col in ["model_sev", "model_agg", "persistence"]:
            c, d = capture(df, col, f)
            rows.append(dict(frac=float(f), policy=col,
                             severity_capture=c, detection_rate=d))
        cs, ds = zip(*(capture(df, "random", f, rng)
                       for _ in range(RANDOM_REPEATS)))
        rows.append(dict(frac=float(f), policy="random",
                         severity_capture=float(np.mean(cs)),
                         detection_rate=float(np.mean(ds)),
                         random_sd=float(np.std(cs))))

    out = pd.DataFrame(rows)
    out.to_csv(EVAL_DIR / "capture_curve.csv", index=False)
    piv = out.pivot(index="frac", columns="policy",
                    values="severity_capture").round(4)
    piv["sev_minus_agg"] = (piv["model_sev"] - piv["model_agg"]).round(4)
    print(piv.to_string())
    log(f"✓ {EVAL_DIR / 'capture_curve.csv'}")


if __name__ == "__main__":
    main()
