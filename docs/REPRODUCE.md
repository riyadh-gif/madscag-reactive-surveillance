# Reproducing the results

## Environment

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Windows
```

Python 3.10.8, CPU only. TensorFlow and Keras are listed but are needed only
to retrain the recurrent comparison models.

## Variables

Every script reads its locations from the environment.

```
MADSCAG_INPUT   folder holding dataset/datasiap/ (data bundle, on request)
MADSCAG_WORK    working folder for preprocessed arrays, runs and results
FIGDIR          destination for rendered figures
```

Phase 2 analyses take four more so that they never overwrite a published
file:

```
REV1_ABLATIONS=1                       include the feature-group ablations
REV1_FRACS=0.25,0.5,0.75               budgets the pre-registered tests use
REV1_CAPTURE_FILE=capture_periods_p2.csv
REV1_MEANS_FILE=capture_means_p2.csv
```

## Without the data bundle

The published artifacts are enough to inspect every reported number and the
code that produced it. Use `docs/PETA_ANGKA.md` to go from a figure in the
manuscript to the file that holds it, then read the script named beside it.

Per-seed predictions are in `artifacts/predictions/<run>_s<seed>/`, with
`preds_uji.npz` and `preds_sahih.npz` holding `h{1,2,3}_pa` and
`h{1,2,3}_mu`, and `hasil.csv` holding that run's metrics. Row position is
the join key to the split indices in the data bundle.

## With the data bundle

Place `panel_cuaca.csv` and the village coordinate file under
`$MADSCAG_INPUT/dataset/datasiap/`, then:

```
# 1. preprocessing: windows, tensors, split indices
python code/v7_pipeline.py --prep-only

# 2. the proposed system and the comparison baselines, four seeds
bash code/run_baselines.sh

# 3. the recurrent campaign, optional, hours rather than minutes
bash code/run_campaign.sh

# 4. metrics table over every run
python code/rescore.py

# 5. phase 1 analyses
python code/rev1_capture.py
python code/rev1_paired.py
python code/rev1_conformal.py
python code/rev1_watchlist.py
python code/rev1_strata.py
python code/rev1_accounting.py
python code/rev1_ablation.py
python code/rev1_stagea.py
python code/rev1_verify.py

# 6. phase 2: ablations of the proposed system, then the fold experiments
python code/baselines.py AB_NOOBSHIST,AB_NOSEVHIST,AB_NOCLIM,AB_NONBR,AB_LAST1 42,1,2,7
REV1_ABLATIONS=1 REV1_FRACS=0.25,0.5,0.75 \
  REV1_CAPTURE_FILE=capture_periods_p2.csv \
  REV1_MEANS_FILE=capture_means_p2.csv python code/rev1_capture.py
python code/rev1_paired_p2.py
python code/rev1_loro.py
python code/rev1_transport.py

# 7. figures
python code/figures/make_fig1.py   # and make_fig2, make_fig3, make_fig4
```

## Checking that a rebuild is faithful

`baselines.py` takes a column mask so that a feature group can be removed
from the design matrix. Running it with no mask must reproduce the stored
predictions of the deployed system exactly:

```
python code/baselines.py XG2_CHK,XG2_CHKSLICE 42
```

then compare every array in `XG2_CHK_s42` and `XG2_CHKSLICE_s42` against
`BL_XGB2_s42`. On the machine that produced the published artifacts all 24
arrays match byte for byte. Elsewhere expect agreement to about 1e-6,
because the histogram builder sums per-thread partials in thread order and
the scripts fix `n_jobs=4`.

## Expected values

The first numbers to check, all at horizon 1:

| Quantity | Value |
|---|---|
| C(0.50), proposed | 0.6492 |
| C(0.50), recurrent hybrid | 0.6467 |
| C(0.50), persistence and random | 0.4825 and 0.4976 |
| Conformal coverage | 0.9002, mean width 0.6549 |
| Gap watchlist, strict top 42 | 101 visited, 97.03% |
| Blind-spot intersection | 214 visited, 64.02% |
| Organism contributions | sum to +0.00251 |
| Stage A AUC, per-seed mean | 0.7889 |

If these match, everything downstream of them will.
