# ══════════════════════════════════════════════════════════════
# MADSCAG — post-hoc rescoring from stored predictions
# ══════════════════════════════════════════════════════════════
"""
Recomputes the full metric set for every run in experiments_v7 from its
stored preds_{split}.npz, without retraining anything.

Why this exists: mae_pos and the naive-constant references were added
after the campaign had started. Every run already stores its per-horizon
pa and mu on both 'sahih' and 'uji', so the added metrics are recoverable
exactly. Nothing here refits a model; mu recalibration is re-derived from
'sahih' exactly as the campaign does it.

Writes one tidy table: eval_v7/metrics_full.csv
"""
import os, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from baselines import (PREP_DIR, EXP_DIR, WORK_DIR, load_meta, load_y,
                       fit_mu_rc, positive_part_metrics, _safe, log)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, mean_absolute_error)

EVAL_DIR = WORK_DIR / "eval_v7"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def naive_references(meta, H):
    """MAE of the naive constants, per horizon, on the observed uji rows."""
    refs = {}
    for sp in ["sahih", "uji"]:
        _, yl = load_y(sp, H)
        for h in H:
            y = yl[h][~np.isnan(yl[h])]
            _, yl_tr = load_y("latih", [h])
            ytr = yl_tr[h][~np.isnan(yl_tr[h])]
            refs[(sp, h)] = dict(
                mae_const_mean=float(np.mean(np.abs(y - meta["mean_ltsj_latih"]))),
                mae_const_median=float(np.mean(np.abs(y - float(np.median(ytr))))),
                mae_const_zero=float(np.mean(np.abs(y))),
                frac_zero=float((y == 0).mean()))
    return refs


def rescore_run(run_dir, meta, refs):
    H = meta["horizons"]
    name = run_dir.name.rsplit("_s", 1)[0]
    seed = run_dir.name.rsplit("_s", 1)[1]
    rows, rc = [], {}
    for sp in ["sahih", "uji"]:
        fp = run_dir / f"preds_{sp}.npz"
        if not fp.exists(): return []
        z = np.load(fp)
        yd, yl = load_y(sp, H)
        for h in H:
            pa = np.asarray(z[f"h{h}_pa"], dtype="float64").ravel()
            mu = np.asarray(z[f"h{h}_mu"], dtype="float64").ravel()
            if sp == "sahih": rc[h] = fit_mu_rc(mu, yl[h])
            a, b = rc.get(h, (0.0, 1.0))
            mu_rc = np.clip(a + b * mu, 0, None)
            vd, vl = ~np.isnan(yd[h]), ~np.isnan(yl[h])
            r = dict(variant=name, seed=seed, split=sp, horizon=h)
            if vd.sum() and len(np.unique(yd[h][vd])) == 2:
                yv, pv = yd[h][vd].astype(int), pa[vd]
                r["auc_a"] = _safe(roc_auc_score, yv, pv)
                r["ap_a"]  = _safe(average_precision_score, yv, pv)
                r["f1_a"]  = _safe(f1_score, yv, (pv >= .5).astype(int),
                                   zero_division=0)
            if vl.sum():
                yv, mv = yl[h][vl], mu[vl]
                muc, muc_rc = (pa * mu)[vl], (pa * mu_rc)[vl]
                r["mae_b"]  = _safe(mean_absolute_error, yv, mv)
                r["rmse_b"] = _safe(
                    lambda a_, b_: float(np.sqrt(np.mean((a_ - b_) ** 2))), yv, mv)
                yb = (yv > 0).astype(int)
                if len(np.unique(yb)) == 2:
                    r["auc_b_nz"] = _safe(roc_auc_score, yb, mv)
                r["mae_b_rc"] = _safe(mean_absolute_error, yv, mu_rc[vl])
                r["mae_c"]    = _safe(mean_absolute_error, yv, muc)
                r["mae_c_rc"] = _safe(mean_absolute_error, yv, muc_rc)
                ref = refs[(sp, h)]
                r.update({f"ref_{k}": v for k, v in ref.items()})
                # skill against each naive reference, so the reader can see
                # how much of the headline depends on which constant is used
                for tag, base in [("mean", ref["mae_const_mean"]),
                                  ("median", ref["mae_const_median"]),
                                  ("zero", ref["mae_const_zero"])]:
                    if np.isfinite(r["mae_c"]) and base > 0:
                        r[f"skill_c_vs_{tag}"] = 1.0 - r["mae_c"] / base
                        r[f"skill_c_rc_vs_{tag}"] = 1.0 - r["mae_c_rc"] / base
                r.update(positive_part_metrics(yv, mv, muc, muc_rc))
            rows.append(r)
    return rows


def main():
    meta = load_meta()
    H = meta["horizons"]
    refs = naive_references(meta, H)
    log("naive references (uji):")
    for h in H:
        r = refs[("uji", h)]
        log(f"  h={h} frac_zero={r['frac_zero']:.3f} "
            f"MAE(mean)={r['mae_const_mean']:.4f} "
            f"MAE(median)={r['mae_const_median']:.4f} "
            f"MAE(zero)={r['mae_const_zero']:.4f}")
    rows = []
    for rd in sorted(EXP_DIR.iterdir()):
        if not rd.is_dir(): continue
        got = rescore_run(rd, meta, refs)
        if got: rows.extend(got); log(f"  ✓ rescored {rd.name}")
        else:   log(f"  ↷ no preds: {rd.name}")
    if not rows:
        log("nothing to rescore"); return
    df = pd.DataFrame(rows)
    out = EVAL_DIR / "metrics_full.csv"
    df.to_csv(out, index=False)
    log(f"✓ {out} ({len(df)} rows, {df.variant.nunique()} variants)")


if __name__ == "__main__":
    main()
