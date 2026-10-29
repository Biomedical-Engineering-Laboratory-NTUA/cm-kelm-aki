# All GPU searches of the article

One row per configuration (run name without the seed). Minutes: median wall time per search on one NVIDIA RTX 6000 Ada (FP64).
Names follow `run_fullcohort_cmkelm_v4.py` (`_nocult`, `_nomem`: component ablation; `_mm<controller>`: multimeme; `_ad<flags>`: adaptive mechanisms; `_pen<x>`: complexity penalty; `_ard`: ARD weights; `randsearch`: random search, `random_search.py`; `__esrd...`: cohort and feature set; no tag: earlier development extraction; `gpumix`: mixed-precision test).

| configuration | searches | median min | total min |
|---|---|---|---|
| `cmkelm_m1000_adKAH_gpu_p20_g15__esrd_eos2` | 10 | 2.96 | 29.6 |
| `cmkelm_m1000_adR_gpu_p20_g15__esrd_eos2` | 10 | 4.99 | 49.8 |
| `cmkelm_m1000_adnone_gpu_p20_g15` | 1 | 2.95 | 3.0 |
| `cmkelm_m1000_adnone_gpu_p20_g15__esrd_eos2` | 10 | 2.99 | 29.9 |
| `cmkelm_m1000_adnone_gpumix_p20_g15` | 1 | 1.84 | 1.8 |
| `cmkelm_m1000_gpu_nocult_nomem_p20_g23__esrd_eos2` | 10 | 2.83 | 28.4 |
| `cmkelm_m1000_gpu_nocult_p20_g15__esrd_eos2` | 10 | 2.96 | 29.6 |
| `cmkelm_m1000_gpu_nomem_p20_g23__esrd_eos2` | 10 | 2.87 | 28.6 |
| `cmkelm_m1000_gpu_p20_g15__esrd` | 10 | 2.88 | 28.9 |
| `cmkelm_m1000_gpu_p20_g15__esrd-cr2_z49_eos` | 10 | 2.92 | 29.2 |
| `cmkelm_m1000_gpu_p20_g15__esrd-cr2_z49_eos2` | 5 | 2.94 | 14.8 |
| `cmkelm_m1000_gpu_p20_g15__esrd-cr4_z49_n18_eos2` | 5 | 2.94 | 14.7 |
| `cmkelm_m1000_gpu_p20_g15__esrd_eos` | 50 | 2.94 | 147.2 |
| `cmkelm_m1000_gpu_p20_g15__esrd_eos2` | 10 | 2.95 | 29.5 |
| `cmkelm_m1000_gpu_p20_g15__esrd_eos2_mi` | 10 | 2.98 | 29.8 |
| `cmkelm_m1000_gpu_p20_g15__esrd_preop2` | 10 | 2.89 | 28.9 |
| `cmkelm_m1000_gpu_p30_g25__esrd_eos` | 10 | 7.19 | 71.9 |
| `cmkelm_m1000_gpu_pen0.0005_p20_g15__esrd_eos2` | 10 | 2.96 | 29.6 |
| `cmkelm_m1000_gpu_pen0.002_p20_g15__esrd_eos2` | 10 | 2.97 | 29.7 |
| `cmkelm_m1000_gpu_pen0_p20_g15__esrd_eos2` | 10 | 2.99 | 29.9 |
| `cmkelm_m1000_mmcredit_gpu_p20_g15__esrd_eos` | 10 | 2.94 | 29.4 |
| `cmkelm_m1000_mmcredit_gpu_p20_g15__esrd_eos2` | 10 | 2.96 | 29.6 |
| `cmkelm_m1000_mmfc_gpu_p20_g15__esrd_eos` | 10 | 2.94 | 29.4 |
| `cmkelm_m1000_mmfc_gpu_p20_g15__esrd_eos2` | 10 | 2.96 | 29.6 |
| `cmkelm_m1000_mmfuzzy_gpu_p20_g15__esrd_eos` | 10 | 2.94 | 29.4 |
| `cmkelm_m1000_mmfuzzy_gpu_p20_g15__esrd_eos2` | 10 | 2.99 | 29.8 |
| `cmkelm_m1000_mmhc_gpu_p20_g15__esrd_eos` | 10 | 2.94 | 29.5 |
| `cmkelm_m1000_mmhc_gpu_p20_g15__esrd_eos2` | 10 | 2.96 | 29.6 |
| `cmkelm_m1000_mmsa_gpu_p20_g15__esrd_eos` | 10 | 2.94 | 29.4 |
| `cmkelm_m1000_mmsa_gpu_p20_g15__esrd_eos2` | 10 | 2.96 | 29.6 |
| `cmkelm_m1000_mmtabu_gpu_p20_g15__esrd_eos` | 50 | 2.95 | 147.3 |
| `cmkelm_m1000_mmtabu_gpu_p20_g15__esrd_eos2` | 10 | 2.99 | 29.8 |
| `cmkelm_m1000_mmuniform_gpu_p20_g15__esrd_eos` | 10 | 2.94 | 29.4 |
| `cmkelm_m1000_mmuniform_gpu_p20_g15__esrd_eos2` | 10 | 3.00 | 29.9 |
| `cmkelm_m1000_gpu_randsearch_n500__esrd_eos2` | 10 | 3.56 (job time / runs) | 35.6 |
| `cmkelm_m1000_gpu_ard_pen0_p20_g15__esrd_eos2` | 10 | 3.63 (job time / runs) | 36.3 |

Total: 412 searches, 21.5 GPU hours.

## Confirmatory cross-validation (October 2026)

All runs use the 5 × 2 outer folds of `src/outer_cv.py` (93 end-of-surgery variables,
pre-existing kidney failure excluded).

| Wave | Content | Runs |
|---|---|---|
| 1 | M1 and M2 searches (10 seeds × 10 folds each), M4/M6/M8/LightGBM | 210 |
| 2 | M3 (5 seeds), random search ARD and masks (10 seeds, repeat 1), GP (M5), FT-Transformer (M7) | 170 |
| 3 | Stage ≥ 2 and stage 3: M1, M2 searches, baselines, GP | 440 |
| 4 | Baseline windows 90 and 7 days: M1, M2 searches, baselines | 420 |
| Rerun | M1 primary searches, storing the ARD weights (identical predictions) | 100 |
| Interpretability | Permutation importance (10 folds), SHAP (fold 11), GP refit (fold 11) | 11 |

The searches ran on two NVIDIA RTX 6000 Ada GPUs in double precision. Each M1/M2 search
of 480 evaluations took about 3–4 min.
