"""Persist the fitted models of the proposed system.

The training runs scored predictions and threw the boosters away, so the
release had weights only for the recurrent family. The editor asked for
pretrained weights, and the weights that matter are the ones behind the
paper's own numbers.

Retraining is safe to do after the fact: the reproduction gate showed this
path reproduces byte-exactly at a fixed seed, so the boosters written here
are the ones that produced the published predictions. The script checks that
claim rather than assuming it, by scoring the reloaded models against the
stored predictions.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import baselines as B

OUT = Path(os.environ["MADSCAG_WORK"]) / "release_weights" / "proposed_xgb2"
SEEDS = [int(s) for s in os.environ.get("REV1_WEIGHT_SEEDS", "42,1,2,7").split(",")]
FMT = os.environ.get("REV1_WEIGHT_FMT", "ubj")


def dump(meta, seed):
    import xgboost as xgb

    H = meta["horizons"]
    TAB = {sp: B.load_tab(sp) for sp in B.SPLITS}
    yd, yl = {}, {}
    for sp in B.SPLITS:
        yd[sp], yl[sp] = B.load_y(sp, H)

    seed_dir = OUT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checks = []

    for h in H:
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

        a = seed_dir / f"stageA_attendance_h{h}.{FMT}"
        b = seed_dir / f"stageB_severity_h{h}.{FMT}"
        clf.save_model(str(a))
        reg.save_model(str(b))

        # the published predictions, for comparison
        stored = np.load(Path(os.environ["MADSCAG_WORK"]) / "experiments_v7"
                         / f"BL_XGB2_s{seed}" / "preds_uji.npz")
        pa = clf.predict_proba(TAB["uji"])[:, 1]
        mu = np.clip(reg.predict(TAB["uji"]), 0, None)
        checks.append(dict(
            horizon=h,
            max_abs_diff_p=float(np.max(np.abs(pa - stored[f"h{h}_pa"]))),
            max_abs_diff_mu=float(np.max(np.abs(mu - stored[f"h{h}_mu"]))),
            best_iteration_A=int(clf.best_iteration),
            best_iteration_B=int(reg.best_iteration),
            bytes_A=a.stat().st_size, bytes_B=b.stat().st_size))
    return checks


def main():
    meta = json.loads((Path(os.environ["MADSCAG_WORK"]) / "preprocessed_v7"
                       / "meta_v7.json").read_text())
    report = {}
    for seed in SEEDS:
        t0 = time.time()
        report[seed] = dump(meta, seed)
        print("seed %d  %.0f s" % (seed, time.time() - t0))
        for c in report[seed]:
            print("   h=%d  dp=%.2e  dmu=%.2e  %.1f + %.1f MB"
                  % (c["horizon"], c["max_abs_diff_p"], c["max_abs_diff_mu"],
                     c["bytes_A"] / 1e6, c["bytes_B"] / 1e6))
    (OUT / "reproduction_check.json").write_text(json.dumps(report, indent=2))
    worst = max(max(c["max_abs_diff_p"], c["max_abs_diff_mu"])
                for cs in report.values() for c in cs)
    total = sum(f.stat().st_size for f in OUT.rglob("*." + FMT))
    print("\nworst deviation from the published predictions: %.3e" % worst)
    print("total weight bytes: %.1f MB" % (total / 1e6))
    print("GATE:", "PASS" if worst == 0.0 else "MISMATCH")


if __name__ == "__main__":
    main()
