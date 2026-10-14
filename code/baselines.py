# ══════════════════════════════════════════════════════════════
# MADSCAG — external comparison baselines
# ══════════════════════════════════════════════════════════════
"""
Comparison baselines trained on the *identical* preprocessed tensors,
chronological splits and scoring code as the MADSCAG campaign, so that
every cell of the comparison table is like-for-like.

Baselines
  BL_CLIM     climatological constant (training-period observed mean)
  BL_PERSIST  fixed-origin persistence (last observed severity)
  BL_XGB2     two independent XGBoost stages (attendance | severity),
              no shared trunk, no factorised objective
  BL_LSTM     plain sequence LSTM, direct-MAE severity head
  BL_TCN      dilated causal temporal conv net
  BL_GRUD     GRU-D-style decayed recurrence over a single-channel
              observation mask (adaptation of Che et al. 2018 to a
              register where the *observation*, not the covariate, is
              what goes missing)

Every baseline emits pa (attendance probability) and mu (severity) per
horizon and is scored by score_run(), which is a verbatim port of the
metric block in v7_pipeline.train_one_v7. Results land in
EXP_DIR/<NAME>_s<seed>/hasil.csv with the same schema as the MADSCAG
runs, so the existing aggregation picks them up unchanged.

All tuning on 'sahih'; 'uji' scored once.
"""
import os, sys, json, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, mean_absolute_error)

INPUT_DIR = Path(os.environ.get("MADSCAG_INPUT", "."))
WORK_DIR  = Path(os.environ.get("MADSCAG_WORK", "."))
PREP_DIR  = WORK_DIR / "preprocessed_v7"
EXP_DIR   = WORK_DIR / "experiments_v7"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1, 2, 7]
SPLITS = ["latih", "sahih", "uji"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def _safe(fn, *a, **k):
    try:
        v = float(fn(*a, **k)); return v if np.isfinite(v) else float("nan")
    except Exception: return float("nan")

def fit_mu_rc(mu, yl):
    m = ~np.isnan(yl)
    if m.sum() < 10: return 0.0, 1.0
    try:
        b, a = np.polyfit(mu[m], yl[m], 1)
        if np.isfinite([a, b]).all() and b > 0: return float(a), float(b)
    except Exception: pass
    return 0.0, 1.0


# ── data ──────────────────────────────────────────────────────
def load_meta():
    with open(PREP_DIR / "meta_v7.json", encoding="utf-8") as f:
        return json.load(f)

def load_X(sp):
    names = ["climate", "spatial", "anomaly", "mask", "obs_flag", "press"]
    return {n: np.load(PREP_DIR / f"X_{n}_{sp}.npy") for n in names}

def load_tab(sp):
    return np.load(PREP_DIR / f"TAB_{sp}.npy")

def load_y(sp, H):
    yd = {h: np.load(PREP_DIR / f"y_diamati_{sp}_h{h}.npy") for h in H}
    yl = {h: np.load(PREP_DIR / f"y_ltsj_{sp}_h{h}.npy") for h in H}
    return yd, yl


# ── positive-part metrics ─────────────────────────────────────
def positive_part_metrics(yv, mu, muc, muc_rc):
    """Severity accuracy on rows that actually carry an attack.

    MAE over all observed rows is minimised by the conditional median,
    which is zero here (roughly half of observed village-periods record
    no attack), so a whole-sample MAE rewards shrinkage toward zero and
    says little about how well magnitude is predicted. These columns
    separate the two questions: mae_pos* measure magnitude accuracy where
    a magnitude exists, and frac_pos records the base rate that makes the
    whole-sample figure so easy to game.
    """
    pos = yv > 0
    out = {"frac_pos": float(pos.mean())}
    if pos.sum() >= 10:
        yp = yv[pos]
        out["mae_pos_b"]    = _safe(mean_absolute_error, yp, mu[pos])
        out["mae_pos_c"]    = _safe(mean_absolute_error, yp, muc[pos])
        out["mae_pos_c_rc"] = _safe(mean_absolute_error, yp, muc_rc[pos])
        # skill of the positive-part forecast against the best constant
        # fitted on positives (their median), i.e. the strongest naive
        # competitor on this subset.
        med = float(np.median(yp))
        bl_pos = float(np.mean(np.abs(yp - med)))
        if np.isfinite(out["mae_pos_c"]) and bl_pos > 0:
            out["skill_pos_c"] = 1.0 - out["mae_pos_c"] / bl_pos
    return out


# ── scoring: verbatim port of the campaign metric block ───────
def score_run(name, seed, meta, preds):
    """preds[split][h] = (pa, mu).  Writes hasil.csv in campaign schema."""
    H = meta["horizons"]
    rd = EXP_DIR / f"{name}_s{seed}"; rd.mkdir(parents=True, exist_ok=True)
    rows, rc = [], {}
    for sp in ["sahih", "uji"]:
        yd, yl = load_y(sp, H)
        npz = {}
        for h in H:
            pa, mu = preds[sp][h]
            pa = np.asarray(pa, dtype="float64").ravel()
            mu = np.asarray(mu, dtype="float64").ravel()
            npz[f"h{h}_pa"], npz[f"h{h}_mu"] = pa, mu
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
                r["mae_b"]  = _safe(mean_absolute_error, yv, mv)
                r["rmse_b"] = _safe(
                    lambda a_, b_: float(np.sqrt(np.mean((a_ - b_) ** 2))), yv, mv)
                yb = (yv > 0).astype(int)
                if len(np.unique(yb)) == 2:
                    r["auc_b_nz"] = _safe(roc_auc_score, yb, mv)
                r["mae_b_rc"] = _safe(mean_absolute_error, yv, mu_rc[vl])
                r["mae_c"]    = _safe(mean_absolute_error, yv, (pa * mu)[vl])
                r["mae_c_rc"] = _safe(mean_absolute_error, yv, (pa * mu_rc)[vl])
                bl = mean_absolute_error(
                    yv, np.full_like(yv, meta["mean_ltsj_latih"]))
                if np.isfinite(r["mae_c"]):
                    r["skill_c"]    = 1.0 - r["mae_c"] / max(bl, 1e-8)
                    r["skill_c_rc"] = 1.0 - r["mae_c_rc"] / max(bl, 1e-8)
                r.update(positive_part_metrics(yv, mv, (pa * mu)[vl],
                                               (pa * mu_rc)[vl]))
            rows.append(r)
        np.savez(str(rd / f"preds_{sp}.npz"), **npz)
    pd.DataFrame(rows).to_csv(rd / "hasil.csv", index=False)
    with open(rd / "cfg.json", "w") as f:
        json.dump(dict(variant=dict(name=name, kind="baseline"), seed=seed,
                       tf=tf.__version__), f, indent=2)
    log(f"  ✓ {name}_s{seed}")


# ── trivial baselines (deterministic, seed-independent) ───────
def run_clim(meta):
    """Constant severity = training-period observed mean; pa = base rate."""
    H = meta["horizons"]
    c = meta["mean_ltsj_latih"]
    yd_tr, _ = load_y("latih", H)
    preds = {}
    for sp in ["sahih", "uji"]:
        n = len(np.load(PREP_DIR / f"X_anomaly_{sp}.npy", mmap_mode="r"))
        preds[sp] = {}
        for h in H:
            base = float(np.nanmean(yd_tr[h]))
            preds[sp][h] = (np.full(n, base), np.full(n, c))
    score_run("BL_CLIM", 0, meta, preds)


def run_zero(meta):
    """All-zero severity forecast: the MAE-optimal constant on this target.

    Roughly half of observed village-periods record no attack, so the
    conditional median is zero and a constant zero is the strongest naive
    competitor under MAE. Reporting it makes explicit how much of any
    MAE-based headline is available without learning anything. pa is the
    training-period attendance base rate, so the composite pa*mu is also
    zero and mae_c equals mae_b here by construction.
    """
    H = meta["horizons"]
    yd_tr, _ = load_y("latih", H)
    preds = {}
    for sp in ["sahih", "uji"]:
        n = len(np.load(PREP_DIR / f"X_anomaly_{sp}.npy", mmap_mode="r"))
        preds[sp] = {h: (np.full(n, float(np.nanmean(yd_tr[h]))), np.zeros(n))
                     for h in H}
    score_run("BL_ZERO", 0, meta, preds)


def run_persist(meta):
    """Fixed-origin persistence: last observed severity carried forward.

    ltsj_lag1_last and diamati_lag1_last are the final-timestep values of
    the window, i.e. information available at the forecast origin.
    """
    H = meta["horizons"]
    extras = meta["tab_extras"]
    i_sev = extras.index("ltsj_lag1_last")
    i_obs = extras.index("diamati_lag1_last")
    off = meta["tab_dim"] - len(extras)
    preds = {}
    for sp in ["sahih", "uji"]:
        TAB = load_tab(sp)
        sev = TAB[:, off + i_sev].astype("float64")
        obs = TAB[:, off + i_obs].astype("float64")
        preds[sp] = {h: (obs, np.clip(sev, 0, None)) for h in H}
    score_run("BL_PERSIST", 0, meta, preds)


def ablation_cols(meta, drop_base=(), drop_extras=(), blocks=(0, 1, 2)):
    """Column indices of TAB that survive an ablation.

    TAB is [last | window-mean | window-std | extras], each of the three
    blocks holding the same 86 features in the same order, so a feature
    group is removed from every block at once. Columns are dropped rather
    than zeroed: a zeroed column still consumes colsample_bytree draws,
    which would penalise an arm for a reason unrelated to the information
    taken away.
    """
    F = meta["n_features"]
    E = meta["tab_extras"]
    off = 3 * F
    db, de = set(drop_base), set(drop_extras)
    keep = [b * F + i for b in blocks for i in range(F) if i not in db]
    keep += [off + j for j, nm in enumerate(E) if nm not in de]
    return np.asarray(keep, dtype=int)


# feature-group ablations of the PROPOSED system, for reviewer 1's request
ABL = {
    "AB_NOOBSHIST": dict(drop_base=[0, 1, 2, 85], drop_extras=["diamati_lag1_last"]),
    "AB_NOSEVHIST": dict(drop_base=list(range(3, 11)), drop_extras=["ltsj_lag1_last"]),
    "AB_NOCLIM":    dict(drop_base=list(range(17, 85))),
    "AB_NONBR":     dict(drop_extras=["press_n_mean", "press_sev_mean"]),
    "AB_LAST1":     dict(blocks=(0,)),
}


# ── XGBoost two-stage (independent stages, no shared trunk) ───
def run_xgb2(meta, seed, cols=None, name="BL_XGB2"):
    import xgboost as xgb
    H = meta["horizons"]
    TAB = {sp: (load_tab(sp) if cols is None else load_tab(sp)[:, cols])
           for sp in SPLITS}
    yd, yl = {}, {}
    for sp in SPLITS:
        yd[sp], yl[sp] = load_y(sp, H)
    preds = {sp: {} for sp in ["sahih", "uji"]}
    for h in H:
        cw = meta["class_weights"][str(h)]["pos"]
        # stage 1: attendance
        m = ~np.isnan(yd["latih"][h])
        clf = xgb.XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=cw, eval_metric="aucpr",
            early_stopping_rounds=50, random_state=seed, n_jobs=4, verbosity=0)
        ms = ~np.isnan(yd["sahih"][h])
        clf.fit(TAB["latih"][m], yd["latih"][h][m].astype(int),
                eval_set=[(TAB["sahih"][ms], yd["sahih"][h][ms].astype(int))],
                verbose=False)
        # stage 2: severity on observed rows only
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
        for sp in ["sahih", "uji"]:
            pa = clf.predict_proba(TAB[sp])[:, 1]
            mu = np.clip(reg.predict(TAB[sp]), 0, None)
            preds[sp][h] = (pa, mu)
    score_run(name, seed, meta, preds)


def run_single(meta, seed):
    """One-stage XGBoost: severity predicted directly, no attendance stage.

    This is the control for the paper's central claim. It holds the model
    family, the feature set, the split, and the tuning protocol fixed
    against BL_XGB2 and removes exactly one thing: the factorisation of
    expected severity into attendance times conditional severity. pa is
    fixed at 1, so the composite forecast equals the severity forecast and
    mae_c equals mae_b by construction. If factorising is what earns the
    result, this arm has to lose.
    """
    import xgboost as xgb
    H = meta["horizons"]
    TAB = {sp: load_tab(sp) for sp in SPLITS}
    yl = {sp: load_y(sp, H)[1] for sp in SPLITS}
    preds = {sp: {} for sp in ["sahih", "uji"]}
    for h in H:
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
        for sp in ["sahih", "uji"]:
            mu = np.clip(reg.predict(TAB[sp]), 0, None)
            preds[sp][h] = (np.ones(len(mu)), mu)
    score_run("BL_SINGLE", seed, meta, preds)


# ── neural baselines ──────────────────────────────────────────
def _masked_bce(pos_w):
    def loss(y_true, y_pred):
        v = tf.cast(~tf.math.is_nan(y_true), tf.float32)
        y = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
        p = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        w = y * pos_w + (1.0 - y)
        ce = -(y * tf.math.log(p) + (1 - y) * tf.math.log(1 - p)) * w
        return tf.reduce_sum(ce * v) / (tf.reduce_sum(v) + 1e-8)
    loss.__name__ = "masked_bce"; return loss


def _masked_mae():
    def loss(y_true, y_pred):
        v = tf.cast(~tf.math.is_nan(y_true), tf.float32)
        y = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
        mu = tf.nn.softplus(y_pred)
        return tf.reduce_sum(tf.abs(y - mu) * v) / (tf.reduce_sum(v) + 1e-8)
    loss.__name__ = "masked_mae"; return loss


class GRUDCell(layers.Layer):
    """GRU-D-style decay on a single-channel observation mask.

    Che et al. (2018) decay the hidden state toward the empirical mean by
    exp(-max(0, W*delta)) where delta is time since the last observation.
    Here the missing quantity is the *visit*, so delta is derived from the
    observation flag rather than from per-feature masks.
    """
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units_ = int(units)
        self.gru = layers.GRU(self.units_, return_sequences=False)
        self.w_decay = layers.Dense(1, activation=None, name="decay_w")

    def call(self, x, delta):
        gamma = tf.exp(-tf.nn.relu(self.w_decay(delta)))   # (B,T,1) in (0,1]
        return self.gru(tf.concat([x * gamma, gamma, delta], axis=-1))

    def get_config(self):
        c = super().get_config(); c.update({"units": self.units_}); return c


def _delta_from_obs(obs):
    """Periods since the last observation, per timestep. obs: (N,T,1)."""
    o = obs[..., 0]
    n, T = o.shape
    d = np.zeros_like(o, dtype="float32")
    last = np.full(n, -1.0, dtype="float32")
    for t in range(T):
        d[:, t] = np.where(last < 0, t, t - last)
        last = np.where(o[:, t] > 0, float(t), last)
    return d[..., None]


def _heads(shared, H, dropout=0.2):
    outs = []
    for h in H:
        a = layers.Dense(1, activation="sigmoid", name=f"pa_{h}")(
            layers.Dropout(dropout)(layers.Dense(32, activation="relu")(shared)))
        b = layers.Dense(1, name=f"mu_{h}")(
            layers.Dropout(dropout)(layers.Dense(32, activation="relu")(shared)))
        outs += [a, b]
    return outs


def _fit_neural(model, meta, Xtr, Xsa, H, ytr, ysa, seed, epochs=40):
    losses, lw = {}, {}
    for h in H:
        losses[f"pa_{h}"] = _masked_bce(meta["class_weights"][str(h)]["pos"])
        losses[f"mu_{h}"] = _masked_mae()
        lw[f"pa_{h}"] = 1.0; lw[f"mu_{h}"] = 1.0
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses,
                  loss_weights=lw)
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                         restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=4, min_lr=1e-6),
           keras.callbacks.TerminateOnNaN()]
    model.fit(Xtr, ytr, validation_data=(Xsa, ysa), epochs=epochs,
              batch_size=256, callbacks=cbs, verbose=2)
    return model


def _neural_targets(H, yd, yl):
    t = {}
    for h in H:
        t[f"pa_{h}"] = yd[h].reshape(-1, 1).astype("float32")
        t[f"mu_{h}"] = yl[h].reshape(-1, 1).astype("float32")
    return t


def _neural_predict(model, X, H, n):
    p = model.predict(X, batch_size=512, verbose=0)
    out = {}
    for i, h in enumerate(H):
        pa = np.asarray(p[i * 2]).ravel()
        mu = np.logaddexp(0.0, np.asarray(p[i * 2 + 1]).ravel())  # softplus
        out[h] = (pa, np.clip(mu, 0, None))
    return out


def run_neural(kind, meta, seed):
    H = meta["horizons"]
    J, F = meta["jendela"], meta["n_features"]
    tf.keras.utils.set_random_seed(seed)
    Xs = {sp: load_X(sp) for sp in SPLITS}
    ys = {sp: load_y(sp, H) for sp in SPLITS}

    if kind == "BL_GRUD":
        inp = keras.Input((J, F), name="x")
        ind = keras.Input((J, 1), name="delta")
        shared = layers.LayerNormalization()(GRUDCell(64)(inp, ind))
        model = keras.Model([inp, ind], _heads(shared, H))
        def prep(sp):
            return [Xs[sp]["climate"], _delta_from_obs(Xs[sp]["obs_flag"])]
    elif kind == "BL_LSTM":
        inp = keras.Input((J, F), name="x")
        shared = layers.LayerNormalization()(layers.LSTM(64)(inp))
        model = keras.Model(inp, _heads(shared, H))
        def prep(sp): return Xs[sp]["climate"]
    elif kind == "BL_TCN":
        inp = keras.Input((J, F), name="x")
        z = inp
        for d in [1, 2, 4]:
            z = layers.Conv1D(64, 3, padding="causal", dilation_rate=d,
                              activation="relu")(z)
            z = layers.LayerNormalization()(z)
        shared = layers.GlobalAveragePooling1D()(z)
        model = keras.Model(inp, _heads(shared, H))
        def prep(sp): return Xs[sp]["climate"]
    else:
        raise ValueError(kind)

    ttr = _neural_targets(H, *ys["latih"])
    tsa = _neural_targets(H, *ys["sahih"])
    model = _fit_neural(model, meta, prep("latih"), prep("sahih"), H,
                        ttr, tsa, seed)
    preds = {sp: _neural_predict(model, prep(sp), H, len(Xs[sp]["climate"]))
             for sp in ["sahih", "uji"]}
    score_run(kind, seed, meta, preds)
    del model; gc.collect(); keras.backend.clear_session()


# ── driver ────────────────────────────────────────────────────
def main():
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    seeds = ([int(s) for s in sys.argv[2].split(",")]
             if len(sys.argv) > 2 else SEEDS)
    meta = load_meta()
    log(f"baselines | horizons={meta['horizons']} | seeds={seeds}")

    def want(n): return (only is None) or (n in only)

    if want("BL_CLIM"):    run_clim(meta)
    if want("BL_ZERO"):    run_zero(meta)
    if want("BL_PERSIST"): run_persist(meta)
    for seed in seeds:
        if want("BL_XGB2"):
            t = time.time(); run_xgb2(meta, seed)
            log(f"    BL_XGB2_s{seed} {time.time()-t:.0f}s")
        if want("BL_SINGLE"):
            t = time.time(); run_single(meta, seed)
            log(f"    BL_SINGLE_s{seed} {time.time()-t:.0f}s")
        for nm, kw in ABL.items():
            if want(nm):
                t = time.time()
                run_xgb2(meta, seed, ablation_cols(meta, **kw), nm)
                log(f"    {nm}_s{seed} {time.time()-t:.0f}s")
        if want("XG2_CHK"):
            t = time.time(); run_xgb2(meta, seed, None, "XG2_CHK")
            log(f"    XG2_CHK_s{seed} {time.time()-t:.0f}s")
        if want("XG2_CHKSLICE"):
            t = time.time()
            run_xgb2(meta, seed, np.arange(meta["tab_dim"]), "XG2_CHKSLICE")
            log(f"    XG2_CHKSLICE_s{seed} {time.time()-t:.0f}s")
        for kind in ["BL_GRUD", "BL_LSTM", "BL_TCN"]:
            if want(kind):
                t = time.time(); run_neural(kind, meta, seed)
                log(f"    {kind}_s{seed} {time.time()-t:.0f}s")
    log("baselines complete")


if __name__ == "__main__":
    main()
