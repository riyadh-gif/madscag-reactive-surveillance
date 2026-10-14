# ══════════════════════════════════════════════════════════════
# MADSCAG-LSTM-ZI v7.0 — Journal-Grade Enhancement Pipeline
# ══════════════════════════════════════════════════════════════
"""
v7.0 on top of v6.1:
  L1a Stage-A hybrid (GBM + blend + isotonic calibration, fit on sahih)
  L1b Stage-B MAE polish phase + per-run / ensemble mu recalibration
  L1c Neighbor outbreak-pressure channel (X_press) + NO_PRESSURE ablation
  L2  Paired-seed ablation stats, env manifest, data MD5, resolution log
  L3  Decision simulations: budget reallocation, prospective watchlist
      validation, lead-time analysis, tuned watchlist thresholds
  R   Resilient execution: error classification + bounded auto-remediation
      (OOM -> halve batch; divergence -> shrink LR; import -> fallback)
Run modes: single notebook cell | CLI (python file --filter ...)
All tuning on sahih; uji scored once. Artifacts under /kaggle/working.
"""
import os, sys, json, time, gc, copy, hashlib, platform, traceback, zipfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, mean_absolute_error)
import joblib

# ── Paths (auto-detected) ─────────────────────────────────────
KAGGLE_MODE = os.path.exists("/kaggle")
if KAGGLE_MODE:
    _hits = sorted(Path("/kaggle/input").rglob("panel_cuaca.csv"))
    if not _hits:
        raise FileNotFoundError("panel_cuaca.csv not found under /kaggle/input")
    DATASIAP_DIR = _hits[0].parent
    INPUT_DIR, WORK_DIR = DATASIAP_DIR.parent, Path("/kaggle/working")
else:
    INPUT_DIR    = Path(os.environ.get("MADSCAG_INPUT", "input"))
    DATASIAP_DIR = INPUT_DIR / "dataset" / "datasiap"
    WORK_DIR     = Path(os.environ.get("MADSCAG_WORK", "work"))

PREP_DIR, EXP_DIR  = WORK_DIR / "preprocessed_v7", WORK_DIR / "experiments_v7"
EVAL_DIR, FIG_DIR  = WORK_DIR / "eval_v7",         WORK_DIR / "figures_v7"
ARCHIVE_DIR        = WORK_DIR / "archive"
ERRORS_PATH, RES_PATH = WORK_DIR / "errors_v7.json", WORK_DIR / "resolutions.json"
MANIFEST_PATH      = WORK_DIR / "manifest_v7.json"
for d in [PREP_DIR, EXP_DIR, EVAL_DIR, FIG_DIR, ARCHIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Small utilities ───────────────────────────────────────────
def log(m):  print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def _reset_keras():
    gc.collect()
    try: keras.backend.clear_session()
    except AttributeError:
        try: tf.keras.backend.clear_session()
        except Exception: pass
def _safe(fn, *a, **k):
    try:
        v = float(fn(*a, **k)); return v if np.isfinite(v) else float("nan")
    except Exception: return float("nan")
def _fnum(v):
    try: v = float(v)
    except Exception: return 0.0
    return 0.0 if v != v else v

# ══════════════════════════════════════════════════════════════
# 1. SAFE CONFIG
# ══════════════════════════════════════════════════════════════
def make_config(**overrides):
    cfg = {
        "pipeline_version": "v7.0",
        "run_tag": time.strftime("v7_%Y%m%d_%H%M"),
        # data
        "jendela": 12, "min_panjang": 24, "horizons": [1, 2, 3], "k_neighbors": 5,
        # training
        "batch": 256, "min_batch": 32,
        "lr":  {"warmup": 5e-4, "joint": 1e-3, "finetune": 3e-4, "polish": 1e-4},
        "epochs": {"warmup": 10, "joint": 50, "finetune": 20, "polish": 8},
        "patience": 10, "dropout": 0.2, "lambda_sc": 0.05,
        "p_tweedie": 1.3, "alpha_zi": 2.0, "beta_obs_on": 1.0,
        # experiment matrix
        "seeds_base": [42, 1, 2, 3, 7], "ablation_seeds": [42, 1, 2, 7],
        # hybrid stage-A
        "hybrid": True, "blend_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        # decision simulations
        "budget_fracs": [0.25, 0.5, 0.75], "sim_seed": 42, "hit_k": 25,
        "watch_pa_grid": [0.2, 0.3, 0.4], "watch_sev_grid": [0.3, 0.5, 0.7],
        # resilience
        "max_retries": 2, "lr_shrink": 0.3,
        # reproducibility
        "deterministic_ops": False,
        # runner flags
        "skip_prep": False, "filter": "", "force": False,
        "eval_only": False, "seeds_only": "", "no_eval": False,
    }
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg.update(overrides)
    validate_config(cfg)
    return cfg

def validate_config(c):
    if c["batch"] < c["min_batch"]: raise ValueError("batch < min_batch")
    if c["patience"] < 1: raise ValueError("patience must be >= 1")
    if not (0 < c["lr_shrink"] < 1): raise ValueError("lr_shrink must be in (0,1)")
    if not c["horizons"] or any(h < 1 for h in c["horizons"]): raise ValueError("bad horizons")
    for k in ["warmup", "joint", "finetune", "polish"]:
        if c["epochs"][k] < 1 or c["lr"][k] <= 0: raise ValueError(f"bad epochs/lr for {k}")
    for f in c["budget_fracs"]:
        if not 0 < f < 1: raise ValueError("budget_fracs must be in (0,1)")
    for w in c["blend_grid"]:
        if not 0 <= w <= 1: raise ValueError("blend_grid must be in [0,1]")
    if not c["seeds_base"] or not c["ablation_seeds"]: raise ValueError("empty seed lists")

# ══════════════════════════════════════════════════════════════
# 2. DYNAMIC ERROR SCAN & RESOLVE
# ══════════════════════════════════════════════════════════════
RESOLUTIONS = []
def _record(kind, action, detail):
    RESOLUTIONS.append(dict(time=time.strftime("%H:%M:%S"), kind=kind,
                            action=action, detail=str(detail)[:300]))
    log(f"⟳ RESOLVE [{kind}] {action} :: {detail}")

def classify_error(exc):
    s = f"{type(exc).__name__}: {exc}".lower()
    if any(k in s for k in ("out of memory", "resourceexhausted", "oom",
                            "cuda error", "memory")):            return "OOM"
    if any(k in s for k in ("non-finite", "nan", "diverg")):     return "DIVERGENCE"
    if isinstance(exc, FileNotFoundError):                       return "MISSING_ARTIFACT"
    if isinstance(exc, ImportError) or "no module named" in s:   return "IMPORT"
    return "UNKNOWN"

def run_resilient(label, fn, cfg):
    """Bounded self-healing: OOM->halve batch, DIVERGENCE->shrink LR, else re-raise."""
    for attempt in range(cfg["max_retries"] + 1):
        try:
            return fn()
        except Exception as exc:
            kind = classify_error(exc)
            if attempt == cfg["max_retries"]:
                _record(kind, "give-up", label); raise
            if kind == "OOM":
                old = cfg["batch"]
                cfg["batch"] = max(cfg["min_batch"], old // 2)
                _reset_keras()
                _record("OOM", f"batch {old}→{cfg['batch']}", label); continue
            if kind == "DIVERGENCE":
                for k in cfg["lr"]: cfg["lr"][k] *= cfg["lr_shrink"]
                _reset_keras()
                _record("DIVERGENCE", f"all LR ×{cfg['lr_shrink']}", label); continue
            _record(kind, "no-remedy", label); raise

# ══════════════════════════════════════════════════════════════
# 3. PREFLIGHT & MANIFEST
# ══════════════════════════════════════════════════════════════
def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def preflight_scan(cfg):
    rep = {"kaggle": KAGGLE_MODE,
           "gpus": [g.name for g in tf.config.list_physical_devices("GPU")]}
    try:
        import xgboost; rep["xgboost"] = xgboost.__version__
    except Exception:
        rep["xgboost"] = "unavailable -> HistGradientBoosting fallback"
    du = shutil.disk_usage(str(WORK_DIR)); rep["disk_free_gb"] = round(du.free / 1e9, 1)
    if du.free / 1e9 < 5: log("⚠ Low disk space on WORK_DIR")
    for f in ["panel_cuaca.csv", "desa_sel_final.csv"]:
        fp = DATASIAP_DIR / f
        rep[f] = "OK" if fp.exists() else "MISSING"
        assert fp.exists(), f"Preflight fatal: {fp} missing"
    log(f"PREFLIGHT: {json.dumps(rep, default=str)}")
    return rep

def write_manifest(cfg, pf):
    env = dict(python=platform.python_version(), platform=platform.platform(),
               tf=tf.__version__, numpy=np.__version__, pandas=pd.__version__,
               sklearn=sklearn.__version__)
    man = dict(pipeline=cfg["pipeline_version"], run_tag=cfg["run_tag"],
               started=time.strftime("%Y-%m-%d %H:%M:%S"),
               config=cfg, preflight=pf, env=env,
               data_md5={f: md5_of(DATASIAP_DIR / f)
                         for f in ["panel_cuaca.csv", "desa_sel_final.csv"]})
    with open(MANIFEST_PATH, "w") as f:
        json.dump(man, f, indent=2, default=str)
    log(f"Manifest → {MANIFEST_PATH}")

# ══════════════════════════════════════════════════════════════
# 4. PREPROCESSING v7
# ══════════════════════════════════════════════════════════════
def run_preprocessing_v7(cfg):
    log("═"*60); log("PREPROCESSING v7"); log("═"*60)
    J, H, K, MINP = cfg["jendela"], cfg["horizons"], cfg["k_neighbors"], cfg["min_panjang"]

    data_path = DATASIAP_DIR / "panel_cuaca.csv"
    assert data_path.exists(), f"Missing {data_path}"
    p = pd.read_csv(data_path, low_memory=False)
    assert "diamati" in p.columns and "ltsj" in p.columns
    log(f"Shape: {p.shape}")
    n_unobs_ltsj = int(((p["diamati"] == 0) & p["ltsj"].notna()).sum())
    n_obs_noltsj = int(((p["diamati"] == 1) & p["ltsj"].isna()).sum())
    log(f"Missingness semantics check: unobs-with-ltsj={n_unobs_ltsj}, "
        f"obs-without-ltsj={n_obs_noltsj} (both ~0 confirms ltsj NaN ⟺ diamati=0)")

    p = p.sort_values(["kab", "kec", "desa", "t"]).reset_index(drop=True)
    p["vid"] = p["kab"].astype(str) + "|" + p["kec"].astype(str) + "|" + p["desa"].astype(str)
    n0 = len(p); p = p.drop_duplicates(subset=["vid", "t"], keep="last")
    log(f"Dedup: {n0} → {len(p)}")

    if "ltsj_lag1" in p.columns:
        p["ltsj_lag1_raw"] = pd.to_numeric(p["ltsj_lag1"], errors="coerce").clip(lower=0)
    else:
        p["ltsj_lag1_raw"] = pd.Series(np.nan, index=p.index)
    p["diamati_lag1_raw"] = pd.to_numeric(
        p.get("diamati_lag1", pd.Series(np.nan, index=p.index)), errors="coerce")

    p = p.merge(p.groupby("t")["diamati"].mean().rename("coverage_rate")
                 .reset_index(), on="t", how="left")

    KUNCI     = ["kab", "kec", "desa", "t", "Tahun", "bl", "pe", "vid"]
    TARGET    = ["ltsj", "diamati"]
    BOCOR     = ["terkendali", "kimia", "ltss", "luas_tanam", "n_baris", "n_opt",
                 "ltsj_pbp", "ltsj_hdb", "ltsj_wbc", "ltsj_tikus", "ltsj_blas",
                 "ltsj_opt_lain", "opt_dominan"]
    NON_FITUR = ["bagian", "cuaca_sumber", "cuaca_sebar_suhu", "urutan", "basi",
                 "ltsj_terakhir", "jarak_amat", "ltsj_lag1_raw", "diamati_lag1_raw"]
    fitur = [c for c in p.columns if c not in set(KUNCI + TARGET + BOCOR + NON_FITUR)]
    assert "coverage_rate" in fitur
    obj = [c for c in fitur if not pd.api.types.is_numeric_dtype(p[c])]
    if obj:
        log(f"⚠ coercing non-numeric features: {obj}")
        for c in obj: p[c] = pd.to_numeric(p[c], errors="coerce")
    MASK_FEATS  = [c for c in ["diamati_lag1", "laju_amat_6", "laju_amat_24"]
                   if c in p.columns]
    n_mask_feat = max(1, len(MASK_FEATS))
    TAB_EXTRAS  = ["bl", "pe", "Tahun", "t", "ltsj_lag1_last",
                   "diamati_lag1_last", "press_n_mean", "press_sev_mean"]
    log(f"Features: {len(fitur)} | mask: {len(MASK_FEATS)}")

    splits = ["latih", "sahih", "uji"]
    for s in splits:
        m = p["bagian"] == s
        log(f"  {s}: {int(m.sum()):,}  t=[{p.loc[m,'t'].min()}, {p.loc[m,'t'].max()}]")
    latih_mask = p["bagian"] == "latih"

    p[fitur] = p[fitur].replace([np.inf, -np.inf], np.nan)
    isi_median = p.loc[latih_mask, fitur].median()
    allnan = isi_median[isi_median.isna()].index.tolist()
    if allnan:
        log(f"⚠ all-NaN in latih — dropped: {allnan}")
        fitur = [c for c in fitur if c not in allnan]
        isi_median = isi_median.drop(index=allnan)
    p[fitur] = p[fitur].fillna(isi_median)
    sk = StandardScaler().fit(p.loc[latih_mask, fitur])
    joblib.dump(sk, PREP_DIR / "scaler_X_v7.pkl")
    p[fitur] = sk.transform(p[fitur])

    obs = p.loc[latih_mask & (p["diamati"] == 1), "ltsj"].dropna()
    mean_ltsj_latih = float(obs.mean()) if len(obs) else 0.0

    coord_path = DATASIAP_DIR / "desa_sel_final.csv"
    assert coord_path.exists(), f"Missing {coord_path}"
    coords = pd.read_csv(coord_path)[["kab", "kec", "desa", "lat", "lon"]]
    coords["vid"] = (coords["kab"].astype(str) + "|" + coords["kec"].astype(str)
                     + "|" + coords["desa"].astype(str))
    coords = coords.dropna(subset=["lat", "lon"]).drop_duplicates("vid")
    matched = coords[coords["vid"].isin(set(p["vid"]))].reset_index(drop=True)
    assert len(matched) > K, "Too few geocoded villages for spatial graph"
    coords_rad = np.radians(matched[["lat", "lon"]].to_numpy("float64"))
    nbrs = NearestNeighbors(n_neighbors=min(K + 1, len(matched)),
                            algorithm="ball_tree", metric="haversine").fit(coords_rad)
    _, idx = nbrs.kneighbors(coords_rad)
    neighbor_map = {matched["vid"].iloc[i]: [matched["vid"].iloc[j] for j in idx[i][1:]]
                    for i in range(len(matched))}

    panjang = p.groupby("vid")["t"].nunique()
    desa_ok = set(panjang[panjang >= MINP].index)
    q = p[p["vid"].isin(desa_ok)].copy()
    grouped = {vid: g.sort_values("t").reset_index(drop=True)
               for vid, g in q.groupby("vid")}
    neigh_ltsj = {vid: dict(zip(g["t"], g["ltsj"])) for vid, g in grouped.items()}
    log(f"Eligible desa: {len(desa_ok)}")

    accum = {s: {k: [] for k in ["Xc","Xs","Xa","Xm","Xo","Xp","TAB","meta"] +
                 [f"yd_{h}" for h in H] + [f"yl_{h}" for h in H] + [f"ydf_{h}" for h in H]}
             for s in splits}
    def _lp(v): return np.log1p(v) if (not pd.isna(v)) and v >= 0 else np.nan

    for ii, (vid, df) in enumerate(grouped.items()):
        if ii % 100 == 0: log(f"  Sequences [{ii+1}/{len(grouped)}]")
        n, slot = len(df), neighbor_map.get(vid, [])[:K]
        for i in range(J, n - max(H) + 1):
            row_i = df.iloc[i]; sp = row_i.get("bagian", None)
            if sp not in splits: continue
            hist, ac = df.iloc[i - J: i], accum[sp]
            xa = hist[fitur].to_numpy("float32"); ac["Xc"].append(xa)
            t_vals = hist["t"].to_numpy()
            ns = [[float(np.log1p(max(0.0, _fnum(neigh_ltsj.get(nv, {}).get(tv, 0.0)))))
                   for tv in t_vals] for nv in slot]
            while len(ns) < K: ns.append([0.0] * J)
            ac["Xs"].append(np.asarray(ns[:K], dtype="float32").T)
            ac["Xa"].append(np.log1p(hist["ltsj_lag1_raw"].fillna(0))
                            .to_numpy("float32").reshape(-1, 1))
            ac["Xm"].append(hist[MASK_FEATS].fillna(0).to_numpy("float32")
                            if MASK_FEATS else np.zeros((J, n_mask_feat), "float32"))
            ac["Xo"].append(hist[["diamati_lag1_raw"]].fillna(0).to_numpy("float32"))
            # L1c: neighbor outbreak-pressure channel (lag-only, no leakage)
            press = np.zeros((J, 2), "float32")
            for j in range(J):
                n_att, sev = 0, 0.0
                for nv in slot:
                    lv = neigh_ltsj.get(nv, {}).get(t_vals[j], np.nan)
                    if (not pd.isna(lv)) and lv > 0: n_att += 1; sev += float(lv)
                press[j, 0] = n_att / max(len(slot), 1)
                press[j, 1] = np.log1p(sev)
            ac["Xp"].append(press)
            # L1a: tabular features for the GBM hybrid
            extras = ([_fnum(row_i.get(c)) for c in ("bl", "pe", "Tahun", "t")] +
                      [float(hist["ltsj_lag1_raw"].fillna(0).iloc[-1]),
                       float(hist["diamati_lag1_raw"].fillna(0).iloc[-1]),
                       float(press[:, 0].mean()), float(press[:, 1].mean())])
            ac["TAB"].append(np.concatenate(
                [xa[-1], xa.mean(0), xa.std(0), np.asarray(extras, "float32")])
                .astype("float32"))
            ac["meta"].append((row_i["kab"], row_i["kec"], row_i["desa"],
                               row_i["t"], vid))
            for h in H:
                ti = i + h - 1
                if ti >= n:
                    ac[f"yd_{h}"].append(np.nan); ac[f"yl_{h}"].append(np.nan)
                    ac[f"ydf_{h}"].append(np.full(K + 1, np.nan, "float32")); continue
                row = df.iloc[ti]; yl = row["ltsj"]
                ac[f"yd_{h}"].append(float(row["diamati"]))
                ac[f"yl_{h}"].append(float(yl) if not pd.isna(yl) else np.nan)
                t_t = float(row["t"]); vals = [_lp(yl)]
                for nv in slot:
                    vals.append(_lp(neigh_ltsj.get(nv, {}).get(t_t, np.nan)))
                while len(vals) < K + 1: vals.append(np.nan)
                ac[f"ydf_{h}"].append(np.asarray(vals[:K + 1], "float32"))

    log("Saving arrays …")
    class_weights = {}
    for sp in splits:
        ac = accum[sp]
        if not ac["Xc"]: raise RuntimeError(f"Split '{sp}' empty")
        for nm, key in [("X_climate","Xc"),("X_spatial","Xs"),("X_anomaly","Xa"),
                        ("X_mask","Xm"),("X_obs_flag","Xo"),("X_press","Xp"),
                        ("TAB","TAB")]:
            np.save(PREP_DIR / f"{nm}_{sp}.npy", np.array(ac[key], dtype="float32"))
        pd.DataFrame(ac["meta"], columns=["kab","kec","desa","t","vid"]).to_csv(
            PREP_DIR / f"sample_meta_{sp}.csv", index=False)
        for h in H:
            yd, yl, ydf = (np.array(ac[f"yd_{h}"], "float32"),
                           np.array(ac[f"yl_{h}"], "float32"),
                           np.array(ac[f"ydf_{h}"], "float32"))
            np.save(PREP_DIR / f"y_diamati_{sp}_h{h}.npy", yd)
            np.save(PREP_DIR / f"y_ltsj_{sp}_h{h}.npy",    yl)
            np.save(PREP_DIR / f"y_diff_{sp}_h{h}.npy",    ydf)
            if sp == "latih":
                m = ~np.isnan(yd)
                pos, neg = float((yd[m] == 1).sum()), float((yd[m] == 0).sum())
                class_weights[h] = {"pos": min(neg / max(pos, 1.0), 10.0), "neg": 1.0}
        log(f"  {sp}: {len(ac['Xc']):,} samples")
    p.loc[p["bagian"].isin(["sahih", "uji"]),
          ["vid", "t", "bagian", "diamati", "ltsj"]].to_csv(
          PREP_DIR / "panel_eval.csv", index=False)

    meta = {"jendela": J, "min_panjang": MINP, "n_features": len(fitur),
            "n_mask_features": n_mask_feat, "mask_features": MASK_FEATS,
            "n_press": 2, "k_neighbors": K, "horizons": H, "fitur": fitur,
            "tab_extras": TAB_EXTRAS, "tab_dim": 3 * len(fitur) + len(TAB_EXTRAS),
            "class_weights": {str(k): v for k, v in class_weights.items()},
            "mean_ltsj_latih": mean_ltsj_latih, "version": "v7.0",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(PREP_DIR / "meta_v7.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    for nm in ["X_climate", "X_spatial", "X_anomaly", "X_mask", "X_obs_flag",
               "X_press", "TAB"]:
        a = np.load(PREP_DIR / f"{nm}_latih.npy")
        assert np.isfinite(a).all(), f"{nm}_latih contains NaN/inf"
    log("✓ Finite-input audit passed")
    return meta

def _verify_prep_files(meta):
    need = []
    for sp in ["latih", "sahih", "uji"]:
        need += [PREP_DIR / f"X_{n}_{sp}.npy" for n in
                 ["climate","spatial","anomaly","mask","obs_flag","press"]]
        need += [PREP_DIR / f"TAB_{sp}.npy"]
        need += [PREP_DIR / f"y_{k}_{sp}_h{h}.npy" for k in
                 ["diamati","ltsj","diff"] for h in meta["horizons"]]
        need.append(PREP_DIR / f"sample_meta_{sp}.csv")
    need.append(PREP_DIR / "panel_eval.csv")
    miss = [str(f) for f in need if not f.exists()]
    assert not miss, f"Missing preprocessed files: {miss}"

# ══════════════════════════════════════════════════════════════
# 5. MODEL v7
# ══════════════════════════════════════════════════════════════
def propensity_weighted_bce(cw_pos=1.0):
    cw = tf.constant(cw_pos, tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        valid  = tf.cast(~tf.math.is_nan(y_true), tf.float32)
        y_true = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
        w   = y_true * cw + (1.0 - y_true)
        pr  = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -(y_true * tf.math.log(pr) + (1 - y_true) * tf.math.log(1 - pr))
        return tf.reduce_sum(bce * w * valid) / (tf.reduce_sum(valid) + 1e-8)
    loss.__name__ = "prop_bce"; return loss

def zi_tweedie_loss(p_tweedie=1.3):
    p = tf.constant(p_tweedie, tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true, -1), tf.float32)
        valid  = tf.cast(~tf.math.is_nan(y_true), tf.float32)
        y_true = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
        pi, mu = tf.sigmoid(y_pred[:, 0]), tf.nn.softplus(y_pred[:, 1]) + 1e-6
        zero, nonzero = tf.cast(y_true == 0, tf.float32), 1.0 - tf.cast(y_true == 0, tf.float32)
        bce = -(zero * tf.math.log(1 - pi + 1e-7) + nonzero * tf.math.log(pi + 1e-7))
        lm = tf.math.log(mu)
        t1, t2 = y_true * tf.exp((1 - p) * lm) / (1 - p), tf.exp((2 - p) * lm) / (2 - p)
        return tf.reduce_sum((bce + (-t1 + t2) * nonzero) * valid) / \
               (tf.reduce_sum(valid) + 1e-8)
    loss.__name__ = "zi_tweedie"; return loss

def masked_mse():
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        valid  = tf.cast(~tf.math.is_nan(y_true), tf.float32)
        y_true = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
        return tf.reduce_sum(tf.square(y_pred - y_true) * valid) / \
               (tf.reduce_sum(valid) + 1e-8)
    loss.__name__ = "masked_mse"; return loss

def diffusion_loss_v6(lambda_sc=0.05):
    lsc = tf.constant(lambda_sc, tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        valid  = tf.cast(~tf.math.is_nan(y_true), tf.float32)
        y_true = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
        mse = tf.reduce_sum(tf.square(y_pred - y_true) * valid) / \
              (tf.reduce_sum(valid) + 1e-8)
        v0  = valid[:, 0:1]
        sc  = tf.reduce_sum(tf.square(y_pred[:, 0:1] -
               tf.reduce_mean(y_pred[:, 1:], axis=1, keepdims=True)) * v0) / \
              (tf.reduce_sum(v0) + 1e-8)
        return mse + lsc * sc
    loss.__name__ = "diffusion_sc"; return loss

def tweedie_loss_v5(p_tweedie=1.3):
    p = tf.constant(p_tweedie, tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        valid  = tf.cast(~tf.math.is_nan(y_true), tf.float32)
        y_true = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
        mu = tf.nn.softplus(y_pred) + 1e-6; lm = tf.math.log(mu)
        t1 = y_true * tf.exp((1 - p) * lm) / (1 - p)
        t2 = tf.exp((2 - p) * lm) / (2 - p)
        return tf.reduce_sum((-t1 + t2) * valid) / (tf.reduce_sum(valid) + 1e-8)
    loss.__name__ = "tweedie_v5"; return loss

class ZIAttentionGate(layers.Layer):
    def __init__(self, alpha=2.0, beta_obs=0.0, plain=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha, self.beta_obs = float(alpha), float(beta_obs)
        self.plain = bool(plain)
        self.score_d = layers.Dense(1, activation="tanh")
    def call(self, cross, anomaly, obs_flag):
        score   = self.score_d(cross)
        if self.plain:                       # NO_GATE: plain temporal attention
            attn = tf.nn.softmax(score, axis=1)
            return tf.reduce_sum(attn * cross, axis=1), attn
        nz_w    = tf.cast(anomaly > 0, tf.float32) * self.alpha + 1.0
        gate    = 1.0 + tf.nn.relu(anomaly)
        obs_b   = 1.0 + self.beta_obs * tf.cast(obs_flag, tf.float32)
        attn    = tf.nn.softmax(score * nz_w * gate * obs_b, axis=1)
        return tf.reduce_sum(attn * cross, axis=1), attn
    def get_config(self):
        c = super().get_config(); c.update({"alpha": self.alpha,
                                            "beta_obs": self.beta_obs,
                                            "plain": self.plain}); return c

class MissingnessPatternEncoder(layers.Layer):
    def __init__(self, units=16, **kwargs):
        super().__init__(**kwargs)
        self.units_ = int(units)
        self.lstm, self.norm = layers.LSTM(self.units_), layers.LayerNormalization()
    def call(self, mask): return self.norm(self.lstm(mask))
    def get_config(self):
        c = super().get_config(); c.update({"units": self.units_}); return c

CUSTOM_OBJECTS = {"ZIAttentionGate": ZIAttentionGate,
                  "MissingnessPatternEncoder": MissingnessPatternEncoder}

def build_madscag_v7(jendela, n_features, n_mask_feat, k_neighbors, horizons,
                     n_press=2, lstm_units_c=64, lstm_units_s=32, mpe_units=16,
                     alpha_zi=2.0, beta_obs=0.0, dropout=0.2, use_zi_b=True,
                     gate_plain=False):
    i_cl = keras.Input((jendela, n_features),  name="X_climate")
    i_sp = keras.Input((jendela, k_neighbors), name="X_spatial")
    i_an = keras.Input((jendela, 1),           name="X_anomaly")
    i_mk = keras.Input((jendela, n_mask_feat), name="X_mask")
    i_ob = keras.Input((jendela, 1),           name="X_obs_flag")
    i_pr = keras.Input((jendela, n_press),     name="X_press")        # L1c
    x_cl = layers.LayerNormalization(name="ln_climate")(
        layers.Bidirectional(layers.LSTM(lstm_units_c, return_sequences=True),
                             name="bilstm_climate")(i_cl))
    sp_in = layers.Concatenate(axis=-1, name="sp_press_concat")([i_sp, i_pr])
    x_sp  = layers.LayerNormalization(name="ln_spatial")(
        layers.LSTM(lstm_units_s, return_sequences=True, name="lstm_spatial")(sp_in))
    mpe = MissingnessPatternEncoder(units=mpe_units, name="mpe")(i_mk)
    mpe_exp = layers.LayerNormalization(name="ln_mpe")(
        layers.RepeatVector(jendela)(mpe))
    cross = layers.Concatenate(axis=-1, name="cross_concat")([x_cl, x_sp, mpe_exp])
    ctx, _ = ZIAttentionGate(alpha=alpha_zi, beta_obs=beta_obs,
                             plain=gate_plain,
                             name="zi_attention_gate")(cross, i_an, i_ob)
    shared = layers.LayerNormalization(name="shared_ln")(
        layers.Dropout(dropout, name="shared_drop")(
            layers.Dense(128, activation="relu", name="shared_dense")(ctx)))
    outs = []
    for h in horizons:
        a = layers.Dense(1, activation="sigmoid", name=f"stage_a_{h}")(
            layers.Dropout(dropout, name=f"drop_a_{h}")(
              layers.Dense(32, activation="relu", name=f"dense_a_{h}")(shared)))
        b = layers.Dense(2 if use_zi_b else 1, name=f"stage_b_{h}")(
            layers.Dropout(dropout, name=f"drop_b_{h}")(
              layers.Dense(32, activation="relu", name=f"dense_b_{h}")(shared)))
        d = layers.Dense(k_neighbors + 1, name=f"diffusion_{h}")(
            layers.Dropout(dropout, name=f"drop_d_{h}")(
              layers.Dense(32, activation="relu", name=f"dense_d_{h}")(shared)))
        outs += [a, b, d]
    return keras.Model([i_cl, i_sp, i_an, i_mk, i_ob, i_pr], outs, name="MADSCAG_v7")

def compile_model_v7(model, meta, lr, p_tweedie, use_zi_b, lambda_sc, mode="std",
                     w_diff=0.4):
    """mode='polish': Stage-B only, masked-MAE objective (L1b)."""
    losses, lw = {}, {}
    for h in meta["horizons"]:
        cw = meta["class_weights"][str(h)]["pos"]
        losses[f"stage_a_{h}"]   = propensity_weighted_bce(cw)
        lw[f"stage_a_{h}"]       = 0.0 if mode == "polish" else 1.0
        losses[f"stage_b_{h}"]   = masked_mse() if mode == "polish" else (
            zi_tweedie_loss(p_tweedie) if use_zi_b else tweedie_loss_v5(p_tweedie))
        lw[f"stage_b_{h}"]       = 1.0
        losses[f"diffusion_{h}"] = diffusion_loss_v6(lambda_sc)
        lw[f"diffusion_{h}"]     = 0.0 if mode == "polish" else w_diff
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=losses, loss_weights=lw)
    return model

def decode_predictions(preds, horizons, use_zi_b=True):
    out = {}
    for i, h in enumerate(horizons):
        pa, pb = np.asarray(preds[i * 3]), np.asarray(preds[i * 3 + 1])
        pa = pa.ravel()
        if use_zi_b:
            pi = tf.sigmoid(pb[:, 0]).numpy(); mu = tf.nn.softplus(pb[:, 1]).numpy()
            mu = pi * mu
        else:
            pi, mu = None, np.maximum(np.logaddexp(0.0, pb.ravel()), 0.0)
        out[h] = {"pa": pa, "mu": mu, "harapan": pa * mu, "pi_b": pi}
    return out

def ensemble_predict(model_paths, X, horizons, use_zi_b=True, batch=512):
    assert model_paths, "no models"
    acc = {h: {"pa": [], "mu": [], "harapan": []} for h in horizons}
    for mp in model_paths:
        try:
            m = keras.models.load_model(str(mp), compile=False,
                                        custom_objects=CUSTOM_OBJECTS)
        except Exception as e:
            log(f"⚠ load failed {mp.name}: {e} — skipped"); continue
        dec = decode_predictions(m.predict(X, verbose=0, batch_size=batch),
                                 horizons, use_zi_b)
        for h in horizons:
            for k in acc[h]: acc[h][k].append(dec[h][k])
        del m; _reset_keras()
    return {h: {k: np.mean(v, 0) for k, v in acc[h].items()} for h in horizons}

class ConformalPrediction:
    def __init__(self, alpha=0.1): self.alpha, self.q_b, self.thr_a = alpha, {}, {}
    @staticmethod
    def _q(s, a):
        n = len(s); return float(np.quantile(s, min(np.ceil((n+1)*(1-a))/n, 1.0)))
    def calibrate(self, dec, yd, yl, H):
        for h in H:
            pa, mu = dec[h]["pa"], dec[h]["mu"]
            vd, vl = ~np.isnan(yd[h]), ~np.isnan(yl[h])
            if vd.sum(): self.thr_a[h] = self._q(
                np.where(yd[h][vd] == 1, 1 - pa[vd], pa[vd]), self.alpha)
            if vl.sum(): self.q_b[h] = self._q(np.abs(yl[h][vl] - mu[vl]), self.alpha)
            log(f"  H={h}: q_B={self.q_b.get(h, float('nan')):.4f}")
    def intervals(self, dec, H):
        return {h: {"ci_lo": np.maximum(dec[h]["mu"] - self.q_b.get(h, 0), 0),
                    "ci_hi": dec[h]["mu"] + self.q_b.get(h, 0)} for h in H}

# ══════════════════════════════════════════════════════════════
# 6. TRAINING v7
# ══════════════════════════════════════════════════════════════
def load_split(sp, meta, zero=None):
    zero = zero or []
    names = ["climate", "spatial", "anomaly", "mask", "obs_flag", "press"]
    X = [np.load(PREP_DIR / f"X_{n}_{sp}.npy") for n in names]
    if "spatial"  in zero: X[1] = np.zeros_like(X[1])
    if "anomaly"  in zero: X[2] = np.zeros_like(X[2])
    if "mask"     in zero: X[3] = np.zeros_like(X[3])
    if "obs_flag" in zero: X[4] = np.zeros_like(X[4])
    if "press"    in zero: X[5] = np.zeros_like(X[5])
    H = meta["horizons"]
    yd = {h: np.load(PREP_DIR / f"y_diamati_{sp}_h{h}.npy") for h in H}
    yl = {h: np.load(PREP_DIR / f"y_ltsj_{sp}_h{h}.npy")    for h in H}
    ydf = {h: np.load(PREP_DIR / f"y_diff_{sp}_h{h}.npy")   for h in H}
    return X, yd, yl, ydf

def build_targets(yd, yl, ydf, meta):
    H, K = meta["horizons"], meta["k_neighbors"]; t = {}
    for h in H:
        t[f"stage_a_{h}"]   = yd[h].reshape(-1, 1).astype("float32")
        t[f"stage_b_{h}"]   = yl[h].reshape(-1, 1).astype("float32")
        t[f"diffusion_{h}"] = ydf[h].astype("float32")
    return t

def set_head_trainable(model, stage_a=True, stage_b=True, diff=True):
    for l in model.layers:
        n = l.name
        if   n.startswith(("stage_a_","dense_a_","drop_a_")):   l.trainable = stage_a
        elif n.startswith(("stage_b_","dense_b_","drop_b_")):   l.trainable = stage_b
        elif n.startswith(("diffusion_","dense_d_","drop_d_")): l.trainable = diff

def _cbs(cfg):
    return [keras.callbacks.TerminateOnNaN(),
            keras.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=cfg["patience"],
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=5, min_lr=1e-6)]

def build_variants(cfg):
    return [dict(name="BASE_v7",        zero=[],           zi_b=True,
                 seeds=cfg["seeds_base"]),
            dict(name="NO_ZI_B_v7",     zero=[],           zi_b=False,
                 seeds=cfg["ablation_seeds"]),
            dict(name="NO_PRESSURE_v7", zero=["press"],    zi_b=True,
                 seeds=cfg["ablation_seeds"]),
            dict(name="NO_MPE_v7",      zero=["mask"],     zi_b=True,
                 seeds=cfg["ablation_seeds"]),
            # ── new ablation arms ───────────────────────────────
            dict(name="NO_SPATIAL_v7",  zero=["spatial"],  zi_b=True,
                 seeds=cfg["ablation_seeds"]),
            dict(name="NO_ANOM_v7",     zero=["anomaly"],  zi_b=True,
                 seeds=cfg["ablation_seeds"]),
            dict(name="NO_OBSFLAG_v7",  zero=["obs_flag"], zi_b=True,
                 seeds=cfg["ablation_seeds"]),
            dict(name="NO_DIFF_v7",     zero=[],           zi_b=True,
                 w_diff=0.0, seeds=cfg["ablation_seeds"]),
            dict(name="NO_GATE_v7",     zero=[],           zi_b=True,
                 gate_plain=True, seeds=cfg["ablation_seeds"]),
            # ── activating the dormant observation modulator ────
            dict(name="BETA_OBS_v7",    zero=[],           zi_b=True,
                 beta_obs=cfg["beta_obs_on"], seeds=cfg["ablation_seeds"])]

def fit_mu_rc(mu, yl):
    m = ~np.isnan(yl)
    if m.sum() < 10: return 0.0, 1.0
    try:
        b, a = np.polyfit(mu[m], yl[m], 1)
        if np.isfinite([a, b]).all() and b > 0: return float(a), float(b)
    except Exception: pass
    return 0.0, 1.0

def train_one_v7(cfg, variant, seed, meta):
    H = meta["horizons"]
    run = f"{variant['name']}_s{seed}"
    rd = EXP_DIR / run; rd.mkdir(parents=True, exist_ok=True)
    if (rd / "hasil.csv").exists() and not cfg["force"]:
        log(f"  ↷ skip {run}"); return
    t0 = time.time(); tf.keras.utils.set_random_seed(seed)

    Xtr, yd_tr, yl_tr, ydf_tr = load_split("latih", meta, variant.get("zero", []))
    Xsa, yd_sa, yl_sa, ydf_sa = load_split("sahih", meta, variant.get("zero", []))
    Xu,  yd_u,  yl_u,  ydf_u  = load_split("uji",   meta, variant.get("zero", []))
    for i, x in enumerate(Xtr):
        assert np.isfinite(x).all(), f"NaN/inf in Xtr[{i}]"
    ttr, tsa = build_targets(yd_tr, yl_tr, ydf_tr, meta), \
               build_targets(yd_sa, yl_sa, ydf_sa, meta)

    model = build_madscag_v7(meta["jendela"], meta["n_features"],
                             meta["n_mask_features"], meta["k_neighbors"], H,
                             n_press=meta["n_press"],
                             alpha_zi=variant.get("alpha", cfg["alpha_zi"]),
                             beta_obs=variant.get("beta_obs", 0.0),
                             gate_plain=variant.get("gate_plain", False),
                             dropout=cfg["dropout"], use_zi_b=variant["zi_b"])
    phases = [("warmup",  dict(stage_a=True,  stage_b=False, diff=False),
               cfg["lr"]["warmup"],   cfg["epochs"]["warmup"],   "std"),
              ("joint",   dict(stage_a=True,  stage_b=True,  diff=True),
               cfg["lr"]["joint"],    cfg["epochs"]["joint"],    "std"),
              ("finetune",dict(stage_a=False, stage_b=True,  diff=True),
               cfg["lr"]["finetune"], cfg["epochs"]["finetune"], "std"),
              ("polish",  dict(stage_a=False, stage_b=True,  diff=False),
               cfg["lr"]["polish"],   cfg["epochs"]["polish"],   "polish")]
    hist = {}
    for ph, tr, lr, ep, mode in phases:
        set_head_trainable(model, **tr)
        model = compile_model_v7(model, meta, lr, cfg["p_tweedie"],
                                 variant["zi_b"], cfg["lambda_sc"], mode,
                                 w_diff=variant.get("w_diff", 0.4))
        log(f"  {run} phase={ph} lr={lr:.0e} ep={ep} batch={cfg['batch']}")
        h = model.fit(Xtr, ttr, validation_data=(Xsa, tsa), epochs=ep,
                      batch_size=cfg["batch"], callbacks=_cbs(cfg), verbose=2)
        hist[ph] = [float(v) for v in h.history.get("val_loss", [])]

    Xp = [x[:256] for x in Xtr]
    if not np.isfinite(model.predict(Xp, batch_size=256, verbose=0)[0]).all():
        raise RuntimeError(f"{run}: non-finite predictions — NOT marked done")
    model.save(str(rd / "final.keras"))
    with open(rd / "history.json", "w") as f: json.dump(hist, f)
    with open(rd / "cfg.json", "w") as f:
        json.dump(dict(variant=variant, seed=seed, batch=cfg["batch"],
                       lr=cfg["lr"], epochs=cfg["epochs"], tf=tf.__version__,
                       finished=time.strftime("%H:%M:%S")), f, indent=2, default=str)

    rc, rows = {}, []
    for sp, X, yd, yl in [("sahih", Xsa, yd_sa, yl_sa), ("uji", Xu, yd_u, yl_u)]:
        dec = decode_predictions(model.predict(X, verbose=0, batch_size=512),
                                 H, variant["zi_b"])
        npz = {f"h{h}_pa": dec[h]["pa"] for h in H}
        npz.update({f"h{h}_mu": dec[h]["mu"] for h in H})
        np.savez(str(rd / f"preds_{sp}.npz"), **npz)
        for h in H:
            pa, mu = dec[h]["pa"], dec[h]["mu"]
            if sp == "sahih": rc[h] = fit_mu_rc(mu, yl[h])
            a, b = rc.get(h, (0.0, 1.0))
            mu_rc = np.clip(a + b * mu, 0, None)
            vd, vl = ~np.isnan(yd[h]), ~np.isnan(yl[h])
            r = dict(variant=variant["name"], seed=seed, split=sp, horizon=h)
            if vd.sum() and len(np.unique(yd[h][vd])) == 2:
                yv, pv = yd[h][vd].astype(int), pa[vd]
                r["auc_a"], r["ap_a"] = _safe(roc_auc_score, yv, pv), \
                                        _safe(average_precision_score, yv, pv)
                r["f1_a"] = _safe(f1_score, yv, (pv >= .5).astype(int),
                                  zero_division=0)
            if vl.sum():
                yv, mv = yl[h][vl], mu[vl]
                r["mae_b"]  = _safe(mean_absolute_error, yv, mv)
                r["rmse_b"] = _safe(lambda a_, b_: float(np.sqrt(np.mean((a_-b_)**2))), yv, mv)
                yb = (yv > 0).astype(int)
                if len(np.unique(yb)) == 2:
                    r["auc_b_nz"] = _safe(roc_auc_score, yb, mv)
                    if variant["zi_b"]:
                        r["auc_pi_b"] = _safe(roc_auc_score, yb, dec[h]["pi_b"][vl])
                r["mae_b_rc"]  = _safe(mean_absolute_error, yv, mu_rc[vl])
                r["mae_c"]     = _safe(mean_absolute_error, yv, (pa*mu)[vl])
                r["mae_c_rc"]  = _safe(mean_absolute_error, yv, (pa*mu_rc)[vl])
                bl = mean_absolute_error(yv, np.full_like(yv, meta["mean_ltsj_latih"]))
                if np.isfinite(r["mae_c"]):
                    r["skill_c"]    = 1.0 - r["mae_c"] / max(bl, 1e-8)
                    r["skill_c_rc"] = 1.0 - r["mae_c_rc"] / max(bl, 1e-8)
            rows.append(r)
    pd.DataFrame(rows).to_csv(rd / "hasil.csv", index=False)
    with open(rd / "mu_rc.json", "w") as f: json.dump({str(k): v for k, v in rc.items()}, f)
    del model; _reset_keras()
    log(f"  ✓ {run}: {time.time()-t0:.0f}s")

# ══════════════════════════════════════════════════════════════
# 7. PAIRED-SEED ABLATION STATISTICS (L2)
# ══════════════════════════════════════════════════════════════
def _load_variant(name):
    dfs = [pd.read_csv(fp) for fp in sorted(EXP_DIR.glob(f"{name}_s*/hasil.csv"))]
    return pd.concat(dfs, ignore_index=True) if dfs else None

def _aggregate():
    dfs = [pd.read_csv(fp) for fp in sorted(EXP_DIR.glob("*/hasil.csv"))]
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(
            EXP_DIR / "experiments_v7_summary.csv", index=False)
        log("✓ experiments_v7_summary.csv")

def paired_ablation_stats():
    base = _load_variant("BASE_v7")
    if base is None: return
    metrics = ["auc_a", "ap_a", "mae_b", "mae_b_rc", "auc_b_nz",
               "auc_pi_b", "mae_c_rc", "skill_c_rc"]
    rows = []
    for name in ["NO_ZI_B_v7", "NO_PRESSURE_v7", "NO_MPE_v7",
                 "NO_SPATIAL_v7", "NO_ANOM_v7", "NO_OBSFLAG_v7",
                 "NO_DIFF_v7", "NO_GATE_v7", "BETA_OBS_v7"]:
        ab = _load_variant(name)
        if ab is None: continue
        b = base[base.split == "uji"]; a = ab[ab.split == "uji"]
        # NO_ZI_B has no hurdle gate, so auc_pi_b does not exist for it;
        # compare only on metrics both arms actually produced.
        have = [m for m in metrics if m in b.columns and m in a.columns]
        for h in sorted(b.horizon.unique()):
            for m in have:
                mg = b[b.horizon == h][["seed", m]].merge(
                     a[a.horizon == h][["seed", m]], on="seed",
                     suffixes=("_b", "_a")).dropna()
                if len(mg) < 2: continue
                d = mg[m + "_b"] - mg[m + "_a"]
                sd = d.std(ddof=1)
                t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else np.nan
                rows.append(dict(variant=name, horizon=h, metric=m, n=len(d),
                                 mean_diff=float(d.mean()), std=float(sd),
                                 t_stat=float(t) if np.isfinite(t) else np.nan,
                                 sign_consistent=int((d > 0).all() or (d < 0).all())))
    if rows:
        pd.DataFrame(rows).to_csv(EXP_DIR / "paired_ablation_v7.csv", index=False)
        log("✓ paired_ablation_v7.csv")

# ══════════════════════════════════════════════════════════════
# 8. STAGE-A HYBRID (L1a)
# ══════════════════════════════════════════════════════════════
def _fit_gbm(Xtr, ytr, Xsa, ysa):
    try:
        import xgboost as xgb
        common = dict(n_estimators=800, max_depth=6, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.9, random_state=42,
                      n_jobs=2, verbosity=0)
        try:
            clf = xgb.XGBClassifier(**common, early_stopping_rounds=50,
                                    eval_metric="aucpr")
            clf.fit(Xtr, ytr, eval_set=[(Xsa, ysa)], verbose=False)
        except TypeError:
            clf = xgb.XGBClassifier(**common)
            clf.fit(Xtr, ytr, eval_set=[(Xsa, ysa)],
                    early_stopping_rounds=50, verbose=False)
        return clf, "xgboost"
    except Exception as e:
        _record("IMPORT", "xgboost->HistGradientBoosting", e)
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                             random_state=42)
        clf.fit(Xtr, ytr); return clf, "histgb"

def train_stage_a_hybrid(meta, cfg, pa_sa, pa_u, yd_sa, yd_u):
    H = meta["horizons"]
    TAB = {sp: np.load(PREP_DIR / f"TAB_{sp}.npy")
           for sp in ["latih", "sahih", "uji"]}
    out = {}
    for h in H:
        ytr = np.load(PREP_DIR / f"y_diamati_latih_h{h}.npy")
        msa, mu_ = ~np.isnan(yd_sa[h]), ~np.isnan(yd_u[h])
        clf, backend = _fit_gbm(TAB["latih"], ytr, TAB["sahih"][msa],
                                yd_sa[h][msa])
        s_sa = clf.predict_proba(TAB["sahih"][msa])[:, 1]
        s_u  = clf.predict_proba(TAB["uji"][mu_])[:, 1]
        best_w, best_ap = 0.5, -1.0
        for w in cfg["blend_grid"]:
            ap = _safe(average_precision_score, yd_sa[h][msa].astype(int),
                       w * pa_sa[h][msa] + (1 - w) * s_sa)
            if ap > best_ap: best_ap, best_w = ap, w
        bl_sa = best_w * pa_sa[h][msa] + (1 - best_w) * s_sa
        bl_u  = best_w * pa_u[h][mu_]  + (1 - best_w) * s_u
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(out_of_bounds="clip").fit(
                bl_sa, yd_sa[h][msa].astype(float))
            c_sa, c_u = np.clip(ir.predict(bl_sa), 0, 1), \
                        np.clip(ir.predict(bl_u), 0, 1)
        except Exception:
            c_sa, c_u = bl_sa, bl_u
        f_sa, f_u = np.full(len(pa_sa[h]), np.nan), np.full(len(pa_u[h]), np.nan)
        f_sa[msa], f_u[mu_] = c_sa, c_u
        out[h] = dict(w=best_w, backend=backend, ap_sahih=best_ap,
                      cal_sa=f_sa, cal_u=f_u)
        log(f"  Hybrid H={h}: {backend}  w={best_w}  AP(sahih)={best_ap:.4f}")
    return out

# ══════════════════════════════════════════════════════════════
# 9. DECISION SIMULATIONS (L3)
# ══════════════════════════════════════════════════════════════
def _reliability(p, y, n_bins=10):
    e = np.linspace(0, 1, n_bins + 1); rows, ece, N = [], 0.0, len(p)
    for i in range(n_bins):
        m = (p >= e[i]) & (p <= e[i+1] if i == n_bins-1 else p < e[i+1])
        if m.sum():
            rows.append(dict(bin_lo=e[i], bin_hi=e[i+1],
                             pred=float(p[m].mean()), emp=float(y[m].mean()),
                             n=int(m.sum())))
            ece += m.sum() / N * abs(p[m].mean() - y[m].mean())
    return rows, float(ece)

def simulate_budget(meta, cfg, sm_u, yl1, har1, tab_u):
    df = pd.DataFrame({"vid": sm_u["vid"], "t": sm_u["t"], "yl": yl1, "har": har1})
    pres_i = 3 * meta["n_features"] + meta["tab_extras"].index("ltsj_lag1_last")
    df["pres"] = tab_u[:, pres_i]
    rng, rows = np.random.default_rng(cfg["sim_seed"]), []
    for frac in cfg["budget_fracs"]:
        for pol in ["model", "persistence", "random"]:
            det, cap, ks = [], [], []
            for _, g in df.groupby("t"):
                obs = g[g["yl"].notna()]
                if len(obs) < 5: continue
                k = max(1, int(round(frac * len(obs))))
                if   pol == "model":      sel = obs.nlargest(k, "har")
                elif pol == "persistence": sel = obs.nlargest(k, "pres")
                else: sel = obs.sample(n=k, random_state=int(rng.integers(1e9)))
                det.append((sel["yl"] > 0).sum() / max((obs["yl"] > 0).sum(), 1))
                cap.append(sel["yl"].sum() / max(obs["yl"].sum(), 1e-9))
                ks.append(k)
            if det:
                rows.append(dict(frac=frac, policy=pol, n_periods=len(det),
                                 k_mean=float(np.mean(ks)),
                                 detection_rate=float(np.mean(det)),
                                 severity_capture=float(np.mean(cap))))
    out = pd.DataFrame(rows)
    out.to_csv(EVAL_DIR / "sim_budget.csv", index=False)
    log("✓ sim_budget.csv"); return out

def _labeled_unobserved(sm, yd1, pa1, mu1, lookup):
    P, M, Y = [], [], []
    vids, ts = sm["vid"].to_numpy(), sm["t"].to_numpy()
    for i in range(len(sm)):
        if yd1[i] != 0: continue
        nxt = lookup.get((vids[i], int(ts[i]) + 1))
        if nxt is None: continue
        d, l = nxt
        if d == 1 and not pd.isna(l):
            P.append(pa1[i]); M.append(mu1[i]); Y.append(1.0 if l > 0 else 0.0)
    return np.array(P), np.array(M), np.array(Y)

def tune_watchlist_thresholds(P, M, Y, cfg):
    best = dict(pa_thr=0.30, sev_thr=0.50, f1=-1.0, precision=np.nan, recall=np.nan)
    if len(Y) < 30 or Y.sum() == 0: return best
    for pt in cfg["watch_pa_grid"]:
        for st in cfg["watch_sev_grid"]:
            fl = (P < pt) & (M > st)
            if fl.sum() == 0: continue
            pr, rc = Y[fl].mean(), Y[fl].sum() / Y.sum()
            f1 = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0
            if f1 > best["f1"]:
                best = dict(pa_thr=pt, sev_thr=st, f1=f1,
                            precision=float(pr), recall=float(rc))
    return best

def simulate_watchlist(sm, yd1, pa1, mu1, lookup, k):
    P, M, Y = _labeled_unobserved(sm, yd1, pa1, mu1, lookup)
    if len(Y) < 30:
        log("⚠ watchlist sim: too few later-confirmed labels"); return None
    har = P * M
    auc = _safe(roc_auc_score, Y.astype(int), har)
    order, base = np.argsort(-har), Y.mean()
    hit = Y[order[:k]].mean() if len(order) >= k else np.nan
    res = dict(n_labeled=int(len(Y)), base_rate=float(base),
               auc_harapan=auc, hit_at_k=float(hit),
               lift=float(hit / base) if base > 0 else np.nan, k=k)
    pd.DataFrame([res]).to_csv(EVAL_DIR / "sim_watchlist.csv", index=False)
    log(f"✓ sim_watchlist.csv: AUC={auc:.3f} hit@{k}={hit:.3f} "
        f"(base={base:.3f}, lift={res['lift']:.2f})")
    return res

def simulate_leadtime(sm_u, yl1, har_h, H):
    idx = {(v, int(t)): i
           for i, (v, t) in enumerate(zip(sm_u["vid"], sm_u["t"]))}
    rows = []
    for h in H:
        thr = float(np.quantile(har_h[h], 0.70))
        fl, n = [], 0
        for i in range(len(sm_u)):
            if not (yl1[i] > 0): continue
            j = idx.get((sm_u["vid"].iloc[i], int(sm_u["t"].iloc[i]) - (h - 1)))
            if j is None: continue
            n += 1; fl.append(har_h[h][j] > thr)
        rows.append(dict(horizon=h, lead_periods=h - 1, threshold=thr,
                         n_outbreaks=n,
                         flag_rate=float(np.mean(fl)) if fl else np.nan,
                         random_baseline=0.30))
    out = pd.DataFrame(rows)
    out.to_csv(EVAL_DIR / "sim_leadtime.csv", index=False)
    log("✓ sim_leadtime.csv"); return out

# ══════════════════════════════════════════════════════════════
# 10. FULL EVALUATION v7
# ══════════════════════════════════════════════════════════════
def run_full_evaluation_v7(meta, base_paths, cfg):
    log("═"*60); log("FULL EVALUATION v7 (Ensemble + Hybrid + Decisions)"); log("═"*60)
    H = meta["horizons"]
    Xsa, yd_sa, yl_sa, _ = load_split("sahih", meta)
    Xu,  yd_u,  yl_u,  _ = load_split("uji",   meta)
    ens_sa, ens_u = ensemble_predict(base_paths, Xsa, H, True), \
                    ensemble_predict(base_paths, Xu,  H, True)

    rc = {h: fit_mu_rc(ens_sa[h]["mu"], yl_sa[h]) for h in H}
    mu_rc = {}
    for split, ens in [("sa", ens_sa), ("u", ens_u)]:
        mu_rc[split] = {h: np.clip(rc[h][0] + rc[h][1] * ens[h]["mu"], 0, None)
                        for h in H}
    log(f"μ-recalibration (sahih-fit): { {h: (round(rc[h][0],3), round(rc[h][1],3)) for h in H} }")

    hyb = train_stage_a_hybrid(meta, cfg,
                               {h: ens_sa[h]["pa"] for h in H},
                               {h: ens_u[h]["pa"]  for h in H}, yd_sa, yd_u) \
          if cfg["hybrid"] else None

    dec_rc_sa = {h: {"pa": ens_sa[h]["pa"], "mu": mu_rc["sa"][h]} for h in H}
    cp = ConformalPrediction(alpha=0.10)
    cp.calibrate(dec_rc_sa, yd_sa, yl_sa, H)
    ivl = cp.intervals({h: {"mu": mu_rc["u"][h]} for h in H}, H)

    rows, cal_rows = [], []
    for h in H:
        pa, mu = ens_u[h]["pa"], mu_rc["u"][h]
        vd, vl = ~np.isnan(yd_u[h]), ~np.isnan(yl_u[h])
        r = dict(horizon=h, mode="ensemble_v7+hybrid")
        if vd.sum() and len(np.unique(yd_u[h][vd])) == 2:
            yv, pv = yd_u[h][vd].astype(int), pa[vd]
            r["auc_a"], r["ap_a"] = _safe(roc_auc_score, yv, pv), \
                                    _safe(average_precision_score, yv, pv)
            r["f1_a"] = _safe(f1_score, yv, (pv >= .5).astype(int), zero_division=0)
            if hyb:
                hv = hyb[h]["cal_u"][vd]
                r["auc_a_hyb"] = _safe(roc_auc_score, yv, hv)
                r["ap_a_hyb"]  = _safe(average_precision_score, yv, hv)
                r["f1_a_hyb"]  = _safe(f1_score, yv, (hv >= .5).astype(int),
                                       zero_division=0)
                bins, ece = _reliability(hv, yv)
                for b in bins: cal_rows.append(dict(horizon=h, ece=ece, **b))
        if vl.sum():
            yv, mv = yl_u[h][vl], mu[vl]
            r["mae_b"], r["rmse_b"] = _safe(mean_absolute_error, yv, mv), \
                _safe(lambda a_, b_: float(np.sqrt(np.mean((a_-b_)**2))), yv, mv)
            yb = (yv > 0).astype(int)
            if len(np.unique(yb)) == 2: r["auc_b_nz"] = _safe(roc_auc_score, yb, mv)
            r["mae_c"] = _safe(mean_absolute_error, yv, (pa * mu)[vl])
            bl = mean_absolute_error(yv, np.full_like(yv, meta["mean_ltsj_latih"]))
            if np.isfinite(r["mae_c"]): r["skill_c"] = 1 - r["mae_c"] / max(bl, 1e-8)
            if hyb:
                hb = (hyb[h]["cal_u"] * mu)[vl]
                r["mae_c_hyb"] = _safe(mean_absolute_error, yv, hb)
                if np.isfinite(r["mae_c_hyb"]):
                    r["skill_c_hyb"] = 1 - r["mae_c_hyb"] / max(bl, 1e-8)
            lo, hi = ivl[h]["ci_lo"][vl], ivl[h]["ci_hi"][vl]
            r["coverage"] = _safe(lambda: float(np.mean((yv >= lo) & (yv <= hi))))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "metrik_ensemble_v7.csv", index=False)
    if cal_rows:
        pd.DataFrame(cal_rows).to_csv(EVAL_DIR / "cal_report.csv", index=False)
        log("✓ cal_report.csv (reliability + ECE)")

    def _f(v):
        try: return f"{float(v):.4f}"
        except Exception: return "  n/a"
    log("─"*60)
    for h in H:
        r = df[df.horizon == h].iloc[0]
        print(f"H={h}  AUC-A {_f(r.get('auc_a'))}→hyb {_f(r.get('auc_a_hyb'))} | "
              f"AP {_f(r.get('ap_a'))}→{_f(r.get('ap_a_hyb'))} | "
              f"MAE-B {_f(r.get('mae_b'))} | MAE-C {_f(r.get('mae_c'))}"
              f"→hyb {_f(r.get('mae_c_hyb'))} | Cov {_f(r.get('coverage'))}")
    log("─"*60)

    # L3 simulations (H=1 operational view)
    sm_u = pd.read_csv(PREP_DIR / "sample_meta_uji.csv")
    panel = pd.read_csv(PREP_DIR / "panel_eval.csv")
    lookup = {(r.vid, int(r.t)): (r.diamati, r.ltsj) for r in panel.itertuples()}
    pa1  = hyb[1]["cal_u"] if hyb else ens_u[1]["pa"]
    mu1  = mu_rc["u"][1]; har1 = pa1 * mu1
    tab_u = np.load(PREP_DIR / "TAB_uji.npy")
    try: simulate_budget(meta, cfg, sm_u, yl_u[1], har1, tab_u)
    except Exception as e: log(f"✗ budget sim failed: {e}")
    try:
        sm_s = pd.read_csv(PREP_DIR / "sample_meta_sahih.csv")
        Ps, Ms, Ys = _labeled_unobserved(sm_s, yd_sa[1],
                                         hyb[1]["cal_sa"] if hyb else ens_sa[1]["pa"],
                                         mu_rc["sa"][1], lookup)
        tuned = tune_watchlist_thresholds(Ps, Ms, Ys, cfg)
        log(f"Watchlist thresholds (sahih-tuned): {tuned}")
        simulate_watchlist(sm_u, yd_u[1], pa1, mu1, lookup, cfg["hit_k"])
        wl = sm_u[(pa1 < tuned["pa_thr"]) & (mu1 > tuned["sev_thr"])].copy()
        wl["prob_diamati"], wl["pred_ltsj"], wl["harapan"] = \
            pa1[(pa1 < tuned["pa_thr"]) & (mu1 > tuned["sev_thr"])], \
            mu1[(pa1 < tuned["pa_thr"]) & (mu1 > tuned["sev_thr"])], har1[(pa1 < tuned["pa_thr"]) & (mu1 > tuned["sev_thr"])]
        wl.sort_values("harapan", ascending=False).to_csv(
            EVAL_DIR / "watchlist_v7_tuned.csv", index=False)
        with open(EVAL_DIR / "watchlist_tuned.json", "w") as f:
            json.dump(tuned, f, indent=2)
        log(f"✓ watchlist_v7_tuned.csv ({len(wl)} villages)")
    except Exception as e: log(f"✗ watchlist sim failed: {e}")
    try:
        har_h = {h: (hyb[h]["cal_u"] if hyb else ens_u[h]["pa"]) * mu_rc["u"][h]
                 for h in H}
        simulate_leadtime(sm_u, yl_u[1], har_h, H)
    except Exception as e: log(f"✗ leadtime sim failed: {e}")
    return df

# ══════════════════════════════════════════════════════════════
# 11. ARCHIVE & MAIN
# ══════════════════════════════════════════════════════════════
def archive_results(tag="v7_final"):
    arc = ARCHIVE_DIR / f"{tag}.zip"
    with zipfile.ZipFile(str(arc), "w", zipfile.ZIP_DEFLATED) as zf:
        for f in EVAL_DIR.glob("*"):
            if f.is_file(): zf.write(str(f), f"eval/{f.name}")
        for f in EXP_DIR.glob("*.csv"):
            zf.write(str(f), f"experiments/{f.name}")
        for exp in EXP_DIR.iterdir():
            if exp.is_dir():
                for fn in ["hasil.csv", "history.json", "cfg.json", "mu_rc.json"]:
                    fp = exp / fn
                    if fp.exists(): zf.write(str(fp), f"experiments/{exp.name}/{fn}")
        for fn in [PREP_DIR / "meta_v7.json", MANIFEST_PATH, RES_PATH, ERRORS_PATH]:
            if fn.exists(): zf.write(str(fn), f"meta/{fn.name}")
    log(f"✓ {arc} ({arc.stat().st_size/1e6:.1f} MB)")
    return arc

def archive_models(tag="v7_models"):
    arc = ARCHIVE_DIR / f"{tag}.zip"
    with zipfile.ZipFile(str(arc), "w", zipfile.ZIP_DEFLATED) as zf:
        for exp in EXP_DIR.iterdir():
            fp = exp / "final.keras" if exp.is_dir() else None
            if fp and fp.exists():
                zf.write(str(fp), f"models/{exp.name}/final.keras")
    log(f"✓ {arc} ({arc.stat().st_size/1e6:.1f} MB)")
    return arc

def main(cfg):
    t0 = time.time()
    log("═"*60); log(f"MADSCAG {cfg['pipeline_version']} PIPELINE"); log("═"*60)
    if cfg["deterministic_ops"]:
        try: tf.config.experimental.enable_op_determinism(); log("deterministic ops ON")
        except Exception as e: log(f"⚠ determinism unavailable: {e}")
    errors = []
    pf = preflight_scan(cfg); write_manifest(cfg, pf)

    if cfg["skip_prep"]:
        assert (PREP_DIR / "meta_v7.json").exists(), "meta_v7.json missing"
        with open(PREP_DIR / "meta_v7.json", encoding="utf-8") as f:
            meta = json.load(f)
        _verify_prep_files(meta); log("✓ reused preprocessed_v7")
    else:
        meta = run_resilient("prep", lambda: run_preprocessing_v7(cfg), cfg)

    if not cfg["eval_only"]:
        flt   = [x.strip() for x in cfg["filter"].split(",") if x.strip()]
        seeds = [int(s) for s in cfg["seeds_only"].split(",") if s.strip()]
        for var in build_variants(cfg):
            if flt and var["name"] not in flt: continue
            for s in (seeds or var["seeds"]):
                log(f"▶ {var['name']} seed={s}")
                run_cfg = copy.deepcopy(cfg)
                try:
                    run_resilient(f"train:{var['name']}_s{s}",
                                  lambda: train_one_v7(run_cfg, var, s, meta), run_cfg)
                except Exception as e:
                    msg = f"{var['name']}_s{s}: {type(e).__name__}: {e}"
                    log(f"  ✗ FAILED — {msg}"); traceback.print_exc()
                    errors.append(msg)
                _reset_keras()
        try: _aggregate(); paired_ablation_stats()
        except Exception as e: errors.append(f"stats: {e}"); log(f"✗ stats: {e}")
    else:
        log("Skipping training (eval-only)")

    base = sorted(fp for fp in EXP_DIR.glob("BASE_v7_s*/final.keras")
                  if (fp.parent / "hasil.csv").exists())
    if cfg.get("no_eval"):
        log("Skipping full evaluation (--no-eval)"); base = []
    if base:
        try: run_resilient("eval",
                           lambda: run_full_evaluation_v7(meta, base, copy.deepcopy(cfg)),
                           cfg)
        except Exception as e:
            errors.append(f"eval: {e}"); log(f"✗ eval failed: {e}")
            traceback.print_exc()
    else:
        log("⚠ no completed BASE_v7 runs — skipping eval")

    try: archive_results("v7_final"); archive_models("v7_models")
    except Exception as e: errors.append(f"archive: {e}")
    with open(RES_PATH, "w") as f: json.dump(RESOLUTIONS, f, indent=2)
    if errors:
        with open(ERRORS_PATH, "w") as f: json.dump(errors, f, indent=2)
        log(f"⚠ {len(errors)} error(s) → {ERRORS_PATH}")
    log("═"*60)
    log(f"✓ v7 COMPLETE in {(time.time()-t0)/60:.1f} min | "
        f"resolutions={len(RESOLUTIONS)} | errors={len(errors)}")
    log(f"  Key artifacts: {EVAL_DIR}/ | {ARCHIVE_DIR}/ | {MANIFEST_PATH}")
    log("═"*60)

# ── ENTRY POINT ───────────────────────────────────────────────
CONFIG = make_config()
# CONFIG = make_config(filter="BASE_v7", seeds_only="42")   # smoke test
# CONFIG = make_config(skip_prep=True)

if __name__ == "__main__":
    if "ipykernel" not in sys.modules and len(sys.argv) > 1:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--skip-prep", action="store_true")
        ap.add_argument("--filter", default=""); ap.add_argument("--force", action="store_true")
        ap.add_argument("--eval-only", action="store_true")
        ap.add_argument("--no-eval", action="store_true")
        ap.add_argument("--seeds-only", default="")
        ns, _ = ap.parse_known_args()
        CONFIG = make_config(skip_prep=ns.skip_prep, filter=ns.filter,
                             force=ns.force, eval_only=ns.eval_only, no_eval=ns.no_eval,
                             seeds_only=ns.seeds_only)
    main(CONFIG)