# MADSCAG: forecasting on a reactive crop-pest surveillance register

Code and result files behind the manuscript submitted to the *International
Journal of Intelligent Engineering and Systems*, paper 20266646.

The system forecasts village-level rice pest severity on an East Java patrol
register. Expected severity is factorised into a stage that predicts whether
a village will be visited and a stage that predicts severity given a visit,
both gradient-boosted, and a decision layer ranks villages under a patrol
budget.

The manuscript itself is not in this repository.

## Contents

```
code/                  preprocessing, models, baselines, evaluation
code/figures/          figure scripts
artifacts/eval/        44 result files, one per analysis
artifacts/predictions/ per-seed predictions and metrics for 86 runs
data/                  split period membership, feature metadata, the scaler
docs/                  PETA_ANGKA.md, REPRODUCE.md
figures/               rendered figures
```

**Start with `docs/PETA_ANGKA.md`**, which maps every number in the
manuscript to the script that computed it and the file that holds it, so a
reader can rerun one calculation without reading the rest of the code.

`docs/REPRODUCE.md` gives the run order, the environment variables and the
expected values.

## Data

The analysis panel is about 92 MB and the register it derives from was
obtained under a formal data request whose terms govern redistribution. Data
is supplied separately on request rather than committed here. The bundle
contains the panel, the village coordinate file, the split indices, the
reporting-gap table and the preprocessed tensors.

Everything in `artifacts/` is an aggregate over periods, seeds or strata, or
a model output, so the published tables can be read and checked without it.

## Environment

Python 3.10.8, CPU only. Exact versions in `requirements.txt`.

The gradient-boosted stages reproduce exactly at a fixed thread count on one
machine; the scripts set `n_jobs=4`. On other hardware expect agreement to
about 1e-6. TensorFlow and Keras are needed only to retrain the recurrent
comparison models; the analysis scripts run without them.

## Licence

Code is MIT, see `LICENSE`. The data bundle is governed by the terms of the
original data request.

## Citation

See `CITATION.cff`.
