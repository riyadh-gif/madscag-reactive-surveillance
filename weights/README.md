# Pretrained weights

Two families of fitted models, one for the system the paper proposes and one
for the recurrent alternative it is compared against.

## `proposed_xgb2/` — the proposed two-stage system

Four seeds (42, 1, 2, 7), three horizons, two models per horizon:

```
seed_<s>/stageA_attendance_h<h>.ubj    XGBClassifier, visit indicator
seed_<s>/stageB_severity_h<h>.ubj      XGBRegressor, severity on visited rows
```

UBJSON, loadable without the training code:

```python
import xgboost as xgb
clf = xgb.XGBClassifier()
clf.load_model("weights/proposed_xgb2/seed_42/stageA_attendance_h1.ubj")
p_visit = clf.predict_proba(X)[:, 1]

reg = xgb.XGBRegressor()
reg.load_model("weights/proposed_xgb2/seed_42/stageB_severity_h1.ubj")
severity = reg.predict(X).clip(0)
```

`X` is the tabular window matrix built by `code/v7_pipeline.py`, scaled with
`data/scaler_X_v7.pkl`. Column order matters and is fixed by the pipeline.

These boosters were refitted after the fact, because the original runs scored
their predictions and discarded the models. `reproduction_check.json` records
what that refit reproduced: for every seed and horizon, the maximum absolute
difference between these models' test-block predictions and the stored
`preds_uji.npz` is exactly 0. The script is `code/rev1_dump_weights.py`.

## `recurrent_keras/` — the recurrent alternative and its ablation arms

41 Keras 3 models, one per run directory, named as in `artifacts/predictions/`:

- `BASE_v7_s*` is the recurrent hybrid reported in the paper.
- The other prefixes are the ablation arms of Section 4.6: `NO_ZI_B` (zero-inflated
  head), `NO_MPE` (missingness encoder), `NO_SPATIAL` (neighbour channel),
  `NO_PRESSURE` (pressure channel), `NO_DIFF` (diffusion loss), `BETA_OBS`
  (observation modulator on), `NO_GATE` (gate replaced by attention), `NO_ANOM`
  (anomaly channel), `NO_OBSFLAG` (control arm).

```python
import keras
model = keras.models.load_model("weights/recurrent_keras/BASE_v7_s42/final.keras")
```

Loading needs the custom layers in `code/v7_pipeline.py` on the import path.

This family is the comparison, not the proposal. Section 4.2 of the paper
reports that it does not demonstrate an accuracy advantage at the operative
horizon and costs roughly ten times as much to train.

## What is not here

No surveillance data. The register was obtained through a formal data request
whose terms govern redistribution and is supplied separately on request. The
weights carry learned parameters only: no village identifiers, coordinates, or
rows of the register.
