"""Baselines for Stage A, so its AUC means something.

Attendance predictability is the paper's one surviving positive claim, and it
is reported without anything to compare against. Two model-free rankers are
the obvious challengers: a village's own historical visit rate, and the
panel-wide coverage rate of the period. If either reaches the model's AUC,
Stage A is reading the base rate rather than forecasting dispatch.

Both are computed strictly from periods before the test block, so neither
sees the window it is scored on.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

WORK = Path(os.environ["MADSCAG_WORK"])
PREP = WORK / "preprocessed_v7"
EXP = WORK / "experiments_v7"
OUT = WORK / "eval_v7" / "revisi1"
SEEDS = [42, 1, 2, 7]
HORIZONS = [1, 2, 3]


def main():
    meta_u = pd.read_csv(PREP / "sample_meta_uji.csv")
    first_test = int(meta_u.t.min())

    # history from every period strictly before the test block, built from
    # the training and validation blocks: a row is observed when its h=1
    # target exists, which is the visit indicator for that period
    frames = []
    for split in ("latih", "sahih"):
        meta = pd.read_csv(PREP / f"sample_meta_{split}.csv", usecols=["vid", "t"])
        y = np.load(PREP / f"y_ltsj_{split}_h1.npy")
        meta["m"] = (~np.isnan(y)).astype(int)
        frames.append(meta)
    hist = pd.concat(frames, ignore_index=True)
    hist = hist[hist.t < first_test]
    by_village = hist.groupby("vid").agg(visits=("m", "sum"), periods=("m", "size"))
    by_village["rate"] = by_village.visits / by_village.periods
    village_rate = by_village.rate.to_dict()
    prior_overall = float(hist.m.mean())
    last_known = float(hist.groupby("t").m.mean().iloc[-1])

    rows = []
    for h in HORIZONS:
        y = np.load(PREP / f"y_ltsj_uji_h{h}.npy")
        m = (~np.isnan(y)).astype(int)
        v_rate = meta_u.vid.map(village_rate).fillna(prior_overall).to_numpy()
        cover = np.full(len(meta_u), last_known)

        model = []
        for s in SEEDS:
            p = np.load(EXP / f"BL_XGB2_s{s}" / "preds_uji.npz")[f"h{h}_pa"]
            model.append(np.asarray(p, "float64"))
        ens = np.mean(model, axis=0)

        for name, score in (("village historical visit rate", v_rate),
                            ("panel coverage rate (constant)", cover),
                            ("Stage A, seed ensemble", ens)):
            if np.allclose(score, score[0]):
                auc, ap = float("nan"), float(m.mean())
            else:
                auc = roc_auc_score(m, score)
                ap = average_precision_score(m, score)
            rows.append(dict(horizon=h, ranker=name, n=len(m), positives=int(m.sum()),
                             auc=auc, ap=ap))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "stagea_baselines.csv", index=False)
    pd.set_option("display.width", 160)
    print(res.round(4).to_string(index=False))
    print("\nwrote", OUT / "stagea_baselines.csv")

    for h in HORIZONS:
        g = res[res.horizon == h].set_index("ranker")
        lift = g.loc["Stage A, seed ensemble", "auc"] - g.loc["village historical visit rate", "auc"]
        print("h=%d  model %.4f  village-rate %.4f  lift %+.4f"
              % (h, g.loc["Stage A, seed ensemble", "auc"],
                 g.loc["village historical visit rate", "auc"], lift))


if __name__ == "__main__":
    main()
