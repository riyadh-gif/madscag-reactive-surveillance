# ══════════════════════════════════════════════════════════════
# Revision round 1: panel, horizon and coverage accounting
# ══════════════════════════════════════════════════════════════
"""Reviewer 1 asked three counting questions the manuscript never answers.

1. Why 114,757 rows when 1,401 villages x 156 periods is 218,556.
2. Exactly how many eligible targets exist at each horizon, and what
   happens to windows that cross a split boundary.
3. Which denominator produces the quoted coverage figures.

All three are answered from the raw register and the stored tensors, with
no model involved, so the numbers in the response letter are reproducible
by a reader holding the panel.
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path(os.environ["MADSCAG_WORK"])
INPUT = Path(os.environ["MADSCAG_INPUT"])
PREP = WORK / "preprocessed_v7"
OUT = WORK / "eval_v7" / "revisi1"
OUT.mkdir(parents=True, exist_ok=True)
PANEL = INPUT / "dataset" / "datasiap" / "panel_cuaca.csv"

SPLITS = ["latih", "sahih", "uji"]
HORIZONS = [1, 2, 3]


def main():
    p = pd.read_csv(PANEL, low_memory=False,
                    usecols=["kab", "kec", "desa", "t", "diamati", "ltsj", "bagian"])
    p["vid"] = p.kab + "|" + p.kec + "|" + p.desa
    p = p.drop_duplicates(subset=["vid", "t"], keep="last")

    n_v, n_t = p.vid.nunique(), p.t.nunique()
    g = p.groupby("vid").t.agg(["min", "max", "count"])
    span = g["max"] - g["min"] + 1
    obs = int((p.diamati > 0).sum())

    panel_acc = dict(
        rows=int(len(p)), villages=int(n_v), periods=int(n_t),
        t_min=int(p.t.min()), t_max=int(p.t.max()),
        rectangular_cells=int(n_v * n_t),
        villages_present_every_period=int((g["count"] == n_t).sum()),
        villages_entering_late=int((g["min"] > p.t.min()).sum()),
        villages_leaving_early=int((g["max"] < p.t.max()).sum()),
        villages_with_contiguous_block=int((g["count"] == span).sum()),
        sum_of_spans=int(span.sum()),
        interior_holes=int((span - g["count"]).sum()),
        observed_rows=obs,
        observed_share_of_rows=float(obs / len(p)),
        ltsj_notna=int(p.ltsj.notna().sum()),
    )

    # split boundaries, read off the register's own marker
    splits = (p.groupby("bagian").t.agg(["min", "max", "count"])
              .rename(columns={"min": "t_min", "max": "t_max", "count": "rows"}))

    # horizon eligibility, from the stored tensors
    hz = []
    for sp in SPLITS:
        sm = pd.read_csv(PREP / f"sample_meta_{sp}.csv")
        for h in HORIZONS:
            yd = np.load(PREP / f"y_diamati_{sp}_h{h}.npy").ravel()
            yl = np.load(PREP / f"y_ltsj_{sp}_h{h}.npy").ravel()
            hz.append(dict(split=sp, horizon=h, windows=int(len(sm)),
                           periods=int(sm.t.nunique()),
                           origin_t_min=int(sm.t.min()), origin_t_max=int(sm.t.max()),
                           target_t_min=int(sm.t.min() + h - 1),
                           target_t_max=int(sm.t.max() + h - 1),
                           eligible_targets=int(np.isfinite(yd).sum()),
                           observed_targets=int((yd > 0).sum()),
                           severity_present=int(np.isfinite(yl).sum()),
                           attacked_targets=int(np.nansum(yl > 0))))
    hz = pd.DataFrame(hz)

    # coverage: which denominator gives the quoted 37.9 / 35.2 / 48.1
    cov = []
    for sp in SPLITS:
        sm = pd.read_csv(PREP / f"sample_meta_{sp}.csv")
        y = np.load(PREP / f"y_diamati_{sp}_h1.npy").ravel()
        s = pd.DataFrame({"t": sm.t.values, "y": y})
        per = s.groupby("t").y.mean()
        cov.append(dict(basis="modelling sample (quoted in the manuscript)",
                        split=sp, mean_of_period_rates=float(per.mean()),
                        pooled_rate=float(s.y.mean()), periods=int(len(per)),
                        rows=int(len(s))))
    allper = pd.concat([pd.read_csv(PREP / f"sample_meta_{sp}.csv").assign(
        y=np.load(PREP / f"y_diamati_{sp}_h1.npy").ravel())[["t", "y"]]
        for sp in SPLITS]).groupby("t").y.mean()
    cov.append(dict(basis="modelling sample (quoted in the manuscript)", split="all",
                    mean_of_period_rates=float(allper.mean()),
                    pooled_rate=np.nan, periods=int(len(allper)), rows=np.nan))

    # register-side denominators, for the corrected sentence
    per_reg = p.groupby("t").agg(in_panel=("vid", "nunique"),
                                 obs=("diamati", lambda s: int((s > 0).sum())))
    cov.append(dict(basis="register, villages present that period", split="all",
                    mean_of_period_rates=float((per_reg.obs / per_reg.in_panel).mean()),
                    pooled_rate=float(per_reg.obs.sum() / per_reg.in_panel.sum()),
                    periods=int(len(per_reg)), rows=int(len(p))))
    cov.append(dict(basis="register, full 1401-village roster", split="all",
                    mean_of_period_rates=float((per_reg.obs / n_v).mean()),
                    pooled_rate=float(per_reg.obs.sum() / (n_v * n_t)),
                    periods=int(len(per_reg)), rows=int(len(p))))
    # the drift claim, on both bases
    for lab, series in [("register", per_reg.obs / per_reg.in_panel)]:
        ts = sorted(series.index)
        k = len(ts) // 3
        cov.append(dict(basis=f"{lab}, first third of periods", split="drift",
                        mean_of_period_rates=float(series.loc[ts[:k]].mean()),
                        pooled_rate=np.nan, periods=k, rows=np.nan))
        cov.append(dict(basis=f"{lab}, last third of periods", split="drift",
                        mean_of_period_rates=float(series.loc[ts[-k:]].mean()),
                        pooled_rate=np.nan, periods=k, rows=np.nan))
    cov = pd.DataFrame(cov)

    with open(OUT / "panel_accounting.json", "w", encoding="utf-8") as f:
        json.dump(dict(panel=panel_acc,
                       splits=splits.reset_index().to_dict("records")), f, indent=2)
    hz.to_csv(OUT / "horizon_accounting.csv", index=False)
    cov.to_csv(OUT / "coverage_denominators.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== panel accounting ===")
    for k, v in panel_acc.items():
        print(f"  {k:34s} {v}")
    print("\n=== split boundaries, from the register's own marker ===")
    print(splits.to_string())
    print("\n=== horizon eligibility ===")
    print(hz.to_string(index=False))
    print("\n=== coverage denominators (manuscript quotes 37.9 / 35.2 / 48.1) ===")
    print(cov.round(4).to_string(index=False))
    print("\nwrote " + str(OUT / "panel_accounting.json"))


if __name__ == "__main__":
    main()
