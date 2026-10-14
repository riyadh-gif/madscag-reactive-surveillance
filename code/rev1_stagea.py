# ══════════════════════════════════════════════════════════════
# Revision round 1: Stage A by stratum, and the seed stability band
# ══════════════════════════════════════════════════════════════
"""Two things the decision-layer work left open.

Attendance is the one quantity identified on the complete roster, because
the visit indicator is recorded for every village-period, so it is the
right place to answer Reviewer 2's request for regency-wise performance.
AUC and average precision are computed per regency, per year and per time
block, at all three horizons, with the seed spread beside each figure.

The protocol also calls for seed variability to be reported as a stability
band rather than folded into the estimand. The band is the spread of
leave-one-seed-out ensembles: each one drops a single seed, re-averages the
remaining predictions and re-scores, which is the closest thing to asking
what the deployed system would have been had one seed not been run.

Nothing is retrained.
"""
import os
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

WORK = Path(os.environ["MADSCAG_WORK"])
PREP = WORK / "preprocessed_v7"
EXP = WORK / "experiments_v7"
OUT = WORK / "eval_v7" / "revisi1"
OUT.mkdir(parents=True, exist_ok=True)

PROPOSED, RECURRENT = "BL_XGB2", "BASE_v7"
HORIZONS = [1, 2, 3]
FRAC = 0.50
MIN_OBS = 5
MIN_CELL = 100


def load_seeds(variant, h, key):
    out = {}
    for fp in sorted(EXP.glob(f"{variant}_s*/preds_uji.npz")):
        s = int(fp.parent.name.rsplit("_s", 1)[1])
        out[s] = np.asarray(np.load(fp)[f"h{h}_{key}"], "float64").ravel()
    return out


def safe_auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def capture_mean(d, col, frac=FRAC):
    caps = []
    for _, g in d.groupby("t"):
        o = g[g["yl"].notna()]
        if len(o) < MIN_OBS:
            continue
        k = max(1, int(round(frac * len(o))))
        caps.append(o.nlargest(k, col)["yl"].sum() / max(o["yl"].sum(), 1e-9))
    return float(np.mean(caps))


def main():
    sm = pd.read_csv(PREP / "sample_meta_uji.csv")
    tab = np.load(PREP / "TAB_uji.npy", mmap_mode="r")
    year = np.asarray(tab[:, 260], "float64").astype(int)
    rows, band = [], []

    for h in HORIZONS:
        yd = np.load(PREP / f"y_diamati_uji_h{h}.npy").ravel()
        yl = np.load(PREP / f"y_ltsj_uji_h{h}.npy").astype("float64")
        for variant, label in [(PROPOSED, "proposed"), (RECURRENT, "recurrent_hybrid")]:
            pas = load_seeds(variant, h, "pa")
            mus = load_seeds(variant, h, "mu")
            seeds = sorted(pas)
            ens = np.mean([pas[s] for s in seeds], axis=0)

            d = pd.DataFrame({"t": sm["t"].values, "kab": sm["kab"].values,
                              "year": year, "y": yd, "p": ens})
            terc = np.digitize(sm["t"].values, np.quantile(sm["t"].values, [1 / 3, 2 / 3]))
            d["block"] = [f"tercile{g + 1}" for g in terc]

            def add(dim, level, m):
                if m.sum() < MIN_CELL:
                    return
                per_seed = [safe_auc(d.y[m], pas[s][m]) for s in seeds]
                rows.append(dict(horizon=h, system=label, dimension=dim, level=str(level),
                                 n=int(m.sum()), positives=int(d.y[m].sum()),
                                 auc_ensemble=safe_auc(d.y[m], d.p[m]),
                                 ap_ensemble=float(average_precision_score(d.y[m], d.p[m]))
                                 if len(np.unique(d.y[m])) > 1 else np.nan,
                                 auc_seed_min=float(np.nanmin(per_seed)),
                                 auc_seed_max=float(np.nanmax(per_seed)),
                                 n_seeds=len(seeds)))

            add("overall", "all", np.ones(len(d), bool))
            for k in sorted(d.kab.unique()):
                add("regency", k, (d.kab == k).to_numpy())
            for y_ in sorted(d.year.unique()):
                add("year", int(y_), (d.year == y_).to_numpy())
            for b in sorted(d.block.unique()):
                add("time_block", b, (d.block == b).to_numpy())

            # leave-one-seed-out stability band on the decision layer
            dd = pd.DataFrame({"t": sm["t"].values, "yl": yl})
            full = np.mean([mus[s] for s in seeds], axis=0)
            base_cap = capture_mean(dd.assign(mu=full), "mu")
            loo = []
            for drop in seeds:
                keep = [s for s in seeds if s != drop]
                mu = np.mean([mus[s] for s in keep], axis=0)
                loo.append(capture_mean(dd.assign(mu=mu), "mu"))
            per_seed_cap = [capture_mean(dd.assign(mu=mus[s]), "mu") for s in seeds]
            band.append(dict(horizon=h, system=label, n_seeds=len(seeds),
                             capture_full_ensemble=base_cap,
                             loo_min=float(np.min(loo)), loo_max=float(np.max(loo)),
                             loo_range=float(np.max(loo) - np.min(loo)),
                             per_seed_min=float(np.min(per_seed_cap)),
                             per_seed_max=float(np.max(per_seed_cap)),
                             per_seed_sd=float(np.std(per_seed_cap, ddof=1))))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "stagea_strata.csv", index=False)
    bd = pd.DataFrame(band)
    bd.to_csv(OUT / "seed_stability_band.csv", index=False)

    pd.set_option("display.width", 210)
    print("\nGATE Stage A overall AUC (published 0.7889 / 0.7185 / 0.6886 for the proposal)")
    print(res[(res.dimension == "overall")][
        ["horizon", "system", "n", "auc_ensemble", "auc_seed_min", "auc_seed_max"]]
        .round(4).to_string(index=False))

    print("\n=== Stage A by regency, h=1 ===")
    print(res[(res.horizon == 1) & (res.dimension == "regency")][
        ["system", "level", "n", "positives", "auc_ensemble", "ap_ensemble",
         "auc_seed_min", "auc_seed_max"]].round(4).to_string(index=False))

    print("\n=== Stage A by year and time block, h=1 ===")
    print(res[(res.horizon == 1) & (res.dimension.isin(["year", "time_block"]))][
        ["system", "dimension", "level", "n", "auc_ensemble", "ap_ensemble"]]
        .round(4).to_string(index=False))

    print("\n=== seed stability band on C(0.50) ===")
    print(bd.round(4).to_string(index=False))
    print("\nwrote " + str(OUT / "stagea_strata.csv") + " and " + str(OUT / "seed_stability_band.csv"))


if __name__ == "__main__":
    main()
