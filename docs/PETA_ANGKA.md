# Number map

Every figure quoted in the manuscript, the script that computes it and the
file that holds it. Paths are relative to the repository root.

Conventions used throughout the decision layer: the observed set for a
period is the rows with a recorded severity; the retained set is
`k = max(1, round(f * n_observed))` villages ranked by the score; periods
with fewer than five observed villages are skipped; seed predictions are
averaged before scoring, which is the primary estimand. The per-seed
estimand, where capture is computed per seed and then averaged, is a
different quantity and is labelled as such wherever it appears.

## Register and panel

| Number | Value | Script | File |
|---|---|---|---|
| Panel rows | 114,757 | `code/rev1_accounting.py` | `artifacts/eval/revisi1/panel_accounting.json` |
| Villages, periods | 1,401 and 156 | same | same |
| Rectangular cells | 218,556 | same | same |
| Villages with contiguous row blocks | 1,401 of 1,401 | same | same |
| Interior holes | 0 | same | same |
| Observed rows | 47,097 (41.04%) | same | same |
| Split boundaries | 48483-48593, 48594-48613, 48614-48638 | same | same |
| Windows per split | 69,004 / 15,850 / 14,649 | same | `artifacts/eval/revisi1/horizon_accounting.csv` |
| Observed targets, h=1/2/3 | 6,790 / 7,170 / 7,627 | same | same |
| Coverage, modelling sample | 0.3524 / 0.3932 / 0.4815, pooled 0.3790 | same | `artifacts/eval/revisi1/coverage_denominators.csv` |
| Coverage, register basis | 0.4146 against villages present, 0.2155 against the full roster | same | same |
| Reactivity index, full panel | 2.024, 95% CI [1.996, 2.052] | `code/rev1_verify.py` | `artifacts/eval/revisi1/verify_reactivity.csv` |
| Reactivity index, test block | 2.671, 95% CI [2.583, 2.762] | same | same |

## Forecast accuracy

| Number | Value | Script | File |
|---|---|---|---|
| Stage A AUC, per-seed mean, h=1/2/3 | 0.7889 / 0.7185 / 0.6886 | `code/rescore.py` | `artifacts/eval/metrics_full.csv` |
| Stage A AUC, seed ensemble, h=1 | 0.7898 proposed, 0.7755 recurrent | `code/rev1_stagea.py` | `artifacts/eval/revisi1/stagea_strata.csv` |
| Stage A AUC by regency, h=1 | 0.720 bondowoso to 0.852 lumajang | same | same |
| Seed stability, C(0.50) h=1 | leave-one-seed-out range 0.0057 proposed, 0.0148 recurrent | same | `artifacts/eval/revisi1/seed_stability_band.csv` |

## Decision layer

| Number | Value | Script | File |
|---|---|---|---|
| C(0.50), proposed / recurrent / product / persistence / random | 0.6492 / 0.6467 / 0.6255 / 0.4825 / 0.4976 | `code/rev1_capture.py` | `artifacts/eval/revisi1/capture_means.csv` |
| Paired difference, proposed minus recurrent, h=1 | +0.0025, 95% CI [-0.0212, +0.0262], Wilcoxon 0.463, 15 of 23 periods | `code/rev1_paired.py` | `artifacts/eval/revisi1/paired_tests.csv` |
| Same, h=2 and h=3 | +0.0104 (Wilcoxon 0.030), +0.0134 (Wilcoxon 0.065) | same | same |
| Holm across the three horizons | nothing below 0.05 | same | same |
| Four seeds on both sides, h=1 | +0.0069, Wilcoxon 0.622 | same | same |
| Per-seed estimand, h=1 | +0.0255, 95% CI [+0.0077, +0.0434], Wilcoxon 0.006 | same | same |
| Severity ranking against the product | Wilcoxon 0.011 / 0.009 / 0.033, Holm 0.027 / 0.027 / 0.033 | same | same |
| Against GRU-D, LSTM, TCN, h=1 | 0.235, 0.004, 0.0003 | same | same |
| Persistence against random | significant only at f ≤ 0.20 after Benjamini-Hochberg | same | `artifacts/eval/revisi1/paired_budget_grid.csv` |

## Uncertainty

| Number | Value | Script | File |
|---|---|---|---|
| Conformal coverage, h=1/2/3 | 0.9002 / 0.9039 / 0.9051, mean width 0.6549 at h=1 | `code/rev1_conformal.py` | `artifacts/eval/revisi1/conformal_subgroups.csv` |
| Coverage by realised severity band, h=1 | 0.998 / 1.000 / 0.461 / 0.021 | same | same |
| Intervals clipped at zero | 99.0% | same | same |
| Mondrian on predicted-severity quintiles | group coverage 0.905 to 0.875, width 0.505 to 0.858, marginal 0.9001 | same | same |
| Coverage by regency, h=1 | 0.846 lumajang to 0.929 jember | same | same |

## Watchlists and the blind-spot set

| Number | Value | Script | File |
|---|---|---|---|
| Gap rule, strict top 42 per period | 101 visited, 97.03%, Wilson [91.6, 99.0] | `code/rev1_watchlist.py` | `artifacts/eval/revisi1/watchlist_rules.csv` |
| Severity ranking | 576 visited, 78.47% | same | same |
| Blind-spot intersection | 214 visited, 64.02% | same | same |
| Low visit probability alone | 171 visited, 61.40% | same | same |
| All observed test rows | 6,790, 49.44% | same | same |
| Gap filter then severity ranking | 211 visited, 95.73%, Wilson [92.1, 97.7] | same | same |
| Tie-rule comparison | strict 101 at 97.03%, ties-inclusive 105 at 97.14% | `code/rev1_verify.py` | `artifacts/eval/revisi1/verify_watchlist_tiebreak.csv` |
| Blind-spot set across horizons | 64.0% / 63.0% / 48.8%, rank-sum 0.379 at h=3 | same | `artifacts/eval/revisi1/blindspot_horizons.csv` |

## Organisms

| Number | Value | Script | File |
|---|---|---|---|
| Decomposition residual | 0.0 on every observed row | `code/rev1_strata.py` | `artifacts/eval/revisi1/organism_omnibus.csv` |
| Contributions summing to the headline | +0.00251 | same | `artifacts/eval/revisi1/organism_contributions.csv` |
| Per organism, h=1 | hdb +0.0107, wbc +0.0098, tikus -0.0087, other -0.0054 | same | same |
| Omnibus sign-flip permutation | 0.0072 / 0.0036 / 0.0001 | same | `artifacts/eval/revisi1/organism_omnibus.csv` |

## Ablations

| Number | Value | Script | File |
|---|---|---|---|
| Recurrent arms, Holm across eighteen tests | zero-inflated head and missingness encoder survive at 0.0078 | `code/rev1_ablation.py` | `artifacts/eval/revisi1/ablation_tests.csv` |
| Neighbour channel | -0.0085 AUC, 11 of 12 cells, Holm 0.0889 | same | same |
| Sign-test floor on four seeds | 0.125 | same | same |
| Observation-flag control | all cells exact ties, no effect detected | same | same |
| Proposed system, five feature groups, h=1 | all Holm 0.8476 | `code/rev1_paired_p2.py` | `artifacts/eval/revisi1/table_ablation_proposed.csv` |
| Severity history removed | -0.0264 capture, -0.0405 at f=0.25 | same | same |
| Climate removed, 204 columns | +0.0058 capture, composite MAE -0.0019 | same | same |

## External validity

| Number | Value | Script | File |
|---|---|---|---|
| Leave-one-regency-out, three sealing variants | -0.0167, -0.0150, -0.0230, five folds of five below the pooled model | `code/rev1_loro.py` | `artifacts/eval/revisi1/loro_tests.csv` |
| Signed-rank floor on five folds | 0.0625 | same | same |
| Cross-border channel bracket | -0.0070 to +0.0196 | same | `artifacts/eval/revisi1/loro_folds.csv` |
| Rolling origin, 99 / 76 / 53 / 30 training periods | 0.6488 / 0.6253 / 0.6477 / 0.6458 | same | `artifacts/eval/revisi1/rolling_origin.csv` |
| Transport penalty, low-propensity rows | +0.0145 | `code/rev1_transport.py` | `artifacts/eval/revisi1/transport_penalty.csv` |
| Transport penalty, high-propensity rows | +0.0010 | same | same |
| Capture in the lowest propensity quartile | 0.5900 full panel against 0.5227, -0.0673, Wilcoxon 0.0179 | same | `artifacts/eval/revisi1/transport_familyC.csv` |
