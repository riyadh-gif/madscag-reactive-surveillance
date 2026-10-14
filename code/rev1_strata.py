# ══════════════════════════════════════════════════════════════
# Revision round 1: capture by regency, by year, and by organism
# ══════════════════════════════════════════════════════════════
"""Reviewer 2 asked for regency-wise, year-block and organism-group
sensitivity. None of it needs retraining.

Regency and year are scored by letting each stratum run its own budget:
villages compete for slots inside their own regency (or year block), which
is how a patrol office actually allocates. That is a different quantity
from the panel-wide curve and is labelled as such.

Organisms are not a stratification but an exact decomposition. The register
splits severity into six per-pest columns that sum to the total on every
observed row, so the per-organism capture differences sum to the headline
difference by construction. They are therefore reported as signed
contribution shares with period bootstrap intervals, descriptive, plus one
omnibus sign-flip permutation test over the period-by-organism matrix, so
six more comparisons do not enter the multiplicity family.

`opt_dominan` and the per-pest severity columns sit in the pipeline's
leakage exclusion list and never enter any feature vector. Everything here
is a post-hoc descriptor of the sample.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

WORK = Path(os.environ["MADSCAG_WORK"])
INPUT = Path(os.environ["MADSCAG_INPUT"])
PREP = WORK / "preprocessed_v7"
EXP = WORK / "experiments_v7"
OUT = WORK / "eval_v7" / "revisi1"
OUT.mkdir(parents=True, exist_ok=True)

PANEL = INPUT / "dataset" / "datasiap" / "panel_cuaca.csv"
PEST_COLS = ["ltsj_pbp", "ltsj_hdb", "ltsj_wbc", "ltsj_tikus", "ltsj_blas",
             "ltsj_opt_lain"]
PROPOSED, RECURRENT = "BL_XGB2", "BASE_v7"
HORIZONS = [1, 2, 3]
FRAC = 0.50
MIN_OBS = 5
N_BOOT = 2000
N_PERM = 10000
RNG = np.random.default_rng(7)


def mean_preds(variant, h):
    mus = []
    for fp in sorted(EXP.glob(f"{variant}_s*/preds_uji.npz")):
        mus.append(np.asarray(np.load(fp)[f"h{h}_mu"], "float64").ravel())
    return np.mean(mus, axis=0), len(mus)


def pick(g, col, frac=FRAC):
    """Per-period selection: top ceil-rounded share of the observed set."""
    o = g[g["yl"].notna()]
    if len(o) < MIN_OBS:
        return None, None
    k = max(1, int(round(frac * len(o))))
    return o, o.nlargest(k, col)


def capture_by(d, col, group_col):
    """Capture inside each stratum, each stratum running its own budget."""
    rows = []
    for (grp, t), g in d.groupby([group_col, "t"]):
        o, sel = pick(g, col)
        if o is None:
            continue
        tot = max(o["yl"].sum(), 1e-9)
        rows.append(dict(stratum=grp, t=int(t), n_obs=len(o),
                         capture=float(sel["yl"].sum() / tot)))
    return pd.DataFrame(rows)


def main():
    sm = pd.read_csv(PREP / "sample_meta_uji.csv")
    tab = np.load(PREP / "TAB_uji.npy", mmap_mode="r")
    year = np.asarray(tab[:, 260], "float64").astype(int)
    panel = pd.read_csv(PANEL, low_memory=False,
                        usecols=["kab", "kec", "desa", "t", "diamati", "ltsj",
                                 "opt_dominan"] + PEST_COLS)
    panel["vid"] = panel.kab + "|" + panel.kec + "|" + panel.desa
    panel = panel.drop_duplicates(subset=["vid", "t"], keep="last")

    strat_rows, contrib_rows, omni_rows = [], [], []

    for h in HORIZONS:
        yl = np.load(PREP / f"y_ltsj_uji_h{h}.npy").astype("float64")
        mu_p, n_p = mean_preds(PROPOSED, h)
        mu_r, n_r = mean_preds(RECURRENT, h)
        d = pd.DataFrame({"t": sm["t"].values, "vid": sm["vid"].values,
                          "kab": sm["kab"].values, "year": year,
                          "yl": yl, "prop": mu_p, "rec": mu_r})
        # the target row for horizon h sits at t + h - 1
        d["t_target"] = d["t"] + h - 1
        j = panel[["vid", "t", "opt_dominan"] + PEST_COLS].rename(columns={"t": "t_target"})
        d = d.merge(j, on=["vid", "t_target"], how="left")

        # ---- regency and year strata, each running its own budget ----
        for group_col in ["kab", "year"]:
            for who, col in [("proposed", "prop"), ("recurrent", "rec")]:
                c = capture_by(d, col, group_col)
                c["h"], c["system"], c["dimension"] = h, who, group_col
                strat_rows.append(c)

        # ---- organism decomposition, h-specific ----
        obs = d[d["yl"].notna()].copy()
        miss = obs[PEST_COLS].isna().any(axis=1).mean()
        resid = float(np.nanmax(np.abs(obs[PEST_COLS].sum(axis=1) - obs["yl"])))
        per_t = []
        for t, g in obs.groupby("t"):
            if len(g) < MIN_OBS:
                continue
            k = max(1, int(round(FRAC * len(g))))
            sel_p = g.nlargest(k, "prop")
            sel_r = g.nlargest(k, "rec")
            tot = max(g["yl"].sum(), 1e-9)
            row = dict(t=int(t), total_Y=float(g["yl"].sum()))
            for c in PEST_COLS:
                row[c] = float((sel_p[c].sum() - sel_r[c].sum()) / tot)
                row[c + "_mass"] = float(g[c].sum())
                row[c + "_cap_prop"] = float(sel_p[c].sum() / max(g[c].sum(), 1e-9))
                row[c + "_cap_rec"] = float(sel_r[c].sum() / max(g[c].sum(), 1e-9))
            per_t.append(row)
        P = pd.DataFrame(per_t)
        M = P[PEST_COLS].to_numpy()          # 23 x 6 signed contributions

        for i, c in enumerate(PEST_COLS):
            v = M[:, i]
            boot = RNG.choice(v, size=(N_BOOT, len(v)), replace=True).mean(axis=1)
            contrib_rows.append(dict(
                h=h, organism=c.replace("ltsj_", ""),
                mass=float(P[c + "_mass"].sum()),
                mass_share=float(P[c + "_mass"].sum() / P["total_Y"].sum()),
                capture_proposed=float(np.average(P[c + "_cap_prop"], weights=P[c + "_mass"] + 1e-12)),
                capture_recurrent=float(np.average(P[c + "_cap_rec"], weights=P[c + "_mass"] + 1e-12)),
                contribution=float(v.mean()),
                boot_lo=float(np.quantile(boot, 0.025)),
                boot_hi=float(np.quantile(boot, 0.975))))

        # one omnibus test: sign-flip permutation, period as the unit
        obs_stat = float(np.abs(M.mean(axis=0)).sum())
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            s = RNG.choice([-1.0, 1.0], size=(M.shape[0], 1))
            null[b] = np.abs((M * s).mean(axis=0)).sum()
        omni_rows.append(dict(h=h, statistic=obs_stat,
                              p_permutation=float((np.sum(null >= obs_stat) + 1) / (N_PERM + 1)),
                              sum_of_contributions=float(M.mean(axis=0).sum()),
                              pest_cols_missing_share=float(miss),
                              max_decomposition_residual=resid,
                              n_seeds_proposed=n_p, n_seeds_recurrent=n_r))

    strat = pd.concat(strat_rows, ignore_index=True)
    strat.to_csv(OUT / "capture_strata_periods.csv", index=False)
    contrib = pd.DataFrame(contrib_rows)
    contrib.to_csv(OUT / "organism_contributions.csv", index=False)
    omni = pd.DataFrame(omni_rows)
    omni.to_csv(OUT / "organism_omnibus.csv", index=False)

    pd.set_option("display.width", 210)
    print("\n=== capture at f=0.50 by regency, each regency on its own budget ===")
    g = (strat[(strat.dimension == "kab")]
         .groupby(["h", "system", "stratum"], as_index=False)
         .agg(capture=("capture", "mean"), periods=("capture", "size"),
              n_obs=("n_obs", "mean")))
    piv = g[g.h == 1].pivot(index="stratum", columns="system", values="capture")
    piv["diff"] = piv["proposed"] - piv["recurrent"]
    print(piv.round(4).to_string())

    print("\n=== capture at f=0.50 by year ===")
    gy = (strat[(strat.dimension == "year")]
          .groupby(["h", "system", "stratum"], as_index=False)
          .agg(capture=("capture", "mean"), periods=("capture", "size")))
    py = gy[gy.h == 1].pivot(index="stratum", columns="system", values="capture")
    py["diff"] = py["proposed"] - py["recurrent"]
    print(py.round(4).to_string())

    print("\n=== organism decomposition at h=1 (contributions sum to the headline diff) ===")
    c1 = contrib[contrib.h == 1]
    print(c1[["organism", "mass", "mass_share", "capture_proposed",
              "capture_recurrent", "contribution", "boot_lo", "boot_hi"]].round(4).to_string(index=False))
    print("sum of contributions: %.4f" % c1.contribution.sum())

    print("\n=== omnibus sign-flip permutation ===")
    print(omni.round(5).to_string(index=False))
    print("\nwrote " + str(OUT / "organism_contributions.csv"))


if __name__ == "__main__":
    main()
