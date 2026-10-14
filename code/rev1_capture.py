# ══════════════════════════════════════════════════════════════
# Revision round 1: per-period capture for every policy and horizon
# ══════════════════════════════════════════════════════════════
"""Writes the per-period capture values the paired tests need.

capture_curve.py computes C(f) as a mean over periods and discards the
per-period vector, so no paired inference was possible. This keeps the
vector. Conventions are copied from capture_curve.py exactly so the
published means reproduce: observed set is yl.notna(), k = max(1,
round(f * n_obs)), periods with fewer than MIN_OBS observed villages are
skipped, and seed predictions are averaged before scoring.

Two estimands are produced side by side:
  ens      seed-ensembled predictions, then capture   (primary)
  perseed  capture per seed, then averaged            (secondary)

Nothing is retrained.
"""
import os, time
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path(os.environ["MADSCAG_WORK"])
PREP = WORK / "preprocessed_v7"
EXP = WORK / "experiments_v7"
OUT = WORK / "eval_v7" / "revisi1"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3]
# Phase 2 reruns this script over the ablation arms only, on the three budgets
# the pre-registered tests use, and writes to its own file so the Phase 1
# outputs quoted in the response letter are never overwritten.
FRACS = (np.array([float(x) for x in os.environ["REV1_FRACS"].split(",")])
         if os.environ.get("REV1_FRACS")
         else np.round(np.arange(0.05, 0.96, 0.05), 2))
OUT_FILE = os.environ.get("REV1_CAPTURE_FILE", "capture_periods.csv")
MEANS_FILE = os.environ.get("REV1_MEANS_FILE", "capture_means.csv")
MIN_OBS = 5
RANDOM_REPEATS = 20
SIM_SEED = 42

# variant -> label used in the paper
VARIANTS = {
    "BL_XGB2": "proposed",
    "BASE_v7": "recurrent_hybrid",
    "BL_GRUD": "grud",
    "BL_LSTM": "lstm",
    "BL_TCN": "tcn",
}
# feature-group ablations of the proposed system (phase 2). Names may never
# start with BL_XGB2, because the seed glob below would fold them into the
# proposed system's own ensemble without raising anything.
ABLATIONS = {
    "AB_NOOBSHIST": "ab_noobshist",
    "AB_NOSEVHIST": "ab_nosevhist",
    "AB_NOCLIM": "ab_noclim",
    "AB_NONBR": "ab_nonbr",
    "AB_LAST1": "ab_last1",
}
if os.environ.get("REV1_ABLATIONS"):
    VARIANTS = dict(VARIANTS, **ABLATIONS)
# 1.9: the published headline pairs a 5-seed ensemble against a 4-seed one
SEED_SUBSETS = {"BASE_v7": [42, 1, 2, 7]}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def seed_paths(variant):
    return sorted(EXP.glob(f"{variant}_s*/preds_uji.npz"))


def load_preds(variant, h, seeds=None):
    """Returns list of (seed, pa, mu)."""
    out = []
    for fp in seed_paths(variant):
        s = int(fp.parent.name.rsplit("_s", 1)[1])
        if seeds is not None and s not in seeds:
            continue
        z = np.load(fp)
        out.append((s, np.asarray(z[f"h{h}_pa"], "float64").ravel(),
                    np.asarray(z[f"h{h}_mu"], "float64").ravel()))
    return out


def per_period(df, col, frac, rng=None):
    """Capture and detection per period for one ranking column."""
    rows = []
    for t, g in df.groupby("t"):
        o = g[g["yl"].notna()]
        if len(o) < MIN_OBS:
            continue
        k = max(1, int(round(frac * len(o))))
        if col == "random":
            sel = o.sample(n=k, random_state=int(rng.integers(1e9)))
        else:
            sel = o.nlargest(k, col)
        tot = max(o["yl"].sum(), 1e-9)
        pos = max((o["yl"] > 0).sum(), 1)
        rows.append(dict(t=int(t), n_obs=len(o), k=k,
                         capture=float(sel["yl"].sum() / tot),
                         detection=float((sel["yl"] > 0).sum() / pos)))
    return pd.DataFrame(rows)


def main():
    sm = pd.read_csv(PREP / "sample_meta_uji.csv")
    tab = np.load(PREP / "TAB_uji.npy", mmap_mode="r")
    pres = np.asarray(tab[:, 262], "float64")  # ltsj_lag1_last
    rng = np.random.default_rng(SIM_SEED)
    recs = []

    for h in HORIZONS:
        yl = np.load(PREP / f"y_ltsj_uji_h{h}.npy").astype("float64")
        base = pd.DataFrame({"t": sm["t"].values, "yl": yl})

        # reference policies: no model involved
        for pol, col in [("persistence", pres)]:
            d = base.assign(**{pol: col})
            for f in FRACS:
                r = per_period(d, pol, f)
                r["h"], r["frac"], r["variant"] = h, float(f), "reference"
                r["policy"], r["estimand"], r["n_seeds"] = pol, "none", 0
                recs.append(r)
        for f in FRACS:
            reps = [per_period(base.assign(random=0.0), "random", f, rng)
                    for _ in range(RANDOM_REPEATS)]
            r = reps[0][["t", "n_obs", "k"]].copy()
            r["capture"] = np.mean([x["capture"].values for x in reps], axis=0)
            r["detection"] = np.mean([x["detection"].values for x in reps], axis=0)
            r["h"], r["frac"], r["variant"] = h, float(f), "reference"
            r["policy"], r["estimand"], r["n_seeds"] = "random", "none", RANDOM_REPEATS
            recs.append(r)

        for variant, label in VARIANTS.items():
            for tag, seeds in [("ens", None)] + (
                    [("ens4", SEED_SUBSETS[variant])] if variant in SEED_SUBSETS else []):
                P = load_preds(variant, h, seeds)
                if not P:
                    continue
                pa = np.mean([p for _, p, _ in P], axis=0)
                mu = np.mean([m for _, _, m in P], axis=0)
                d = base.assign(model_sev=mu, model_agg=pa * mu)
                for pol in ["model_sev", "model_agg"]:
                    for f in FRACS:
                        r = per_period(d, pol, f)
                        r["h"], r["frac"], r["variant"] = h, float(f), label
                        r["policy"], r["estimand"] = pol, tag
                        r["n_seeds"] = len(P)
                        recs.append(r)

            # secondary estimand: capture per seed, then average over seeds
            P = load_preds(variant, h)
            for pol in ["model_sev", "model_agg"]:
                for f in FRACS:
                    per = []
                    for _, pa_s, mu_s in P:
                        d = base.assign(model_sev=mu_s, model_agg=pa_s * mu_s)
                        per.append(per_period(d, pol, f).set_index("t"))
                    r = per[0][["n_obs", "k"]].copy()
                    r["capture"] = np.mean([x["capture"] for x in per], axis=0)
                    r["detection"] = np.mean([x["detection"] for x in per], axis=0)
                    r = r.reset_index()
                    r["h"], r["frac"], r["variant"] = h, float(f), label
                    r["policy"], r["estimand"] = pol, "perseed"
                    r["n_seeds"] = len(P)
                    recs.append(r)
        log(f"h={h} done")

    out = pd.concat(recs, ignore_index=True)
    cols = ["h", "frac", "variant", "policy", "estimand", "n_seeds",
            "t", "n_obs", "k", "capture", "detection"]
    out[cols].to_csv(OUT / OUT_FILE, index=False)
    log(f"wrote {OUT / OUT_FILE}  rows={len(out)}")

    # means, for the correctness gate against the published table
    m = (out[out.estimand.isin(["ens", "none"])]
         .groupby(["h", "frac", "variant", "policy"], as_index=False)
         .capture.mean())
    m.to_csv(OUT / MEANS_FILE, index=False)
    g = m[(m.h == 1) & (np.isclose(m.frac, 0.50))]
    print("\nGATE h=1 f=0.50 (published: proposed .6492, recurrent .6467, "
          "persistence .4825, random .4976)")
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
