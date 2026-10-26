# DE-KELM: evolutionary ensembles of Nyström kernel ELMs for postoperative AKI

Code for the article

> J. Digalakis, M. Cafaro, T. Chatzistamatiou, I. Vezakis, G. K. Matsopoulos, D. D. Koutsouris,
> "Evolutionary Ensembles of Nyström Kernel Extreme Learning Machines for Postoperative Acute
> Kidney Injury Prediction: An Equal-Budget Study of Search Complexity," submitted to
> *IEEE Journal of Biomedical and Health Informatics*.

Differential evolution with population ensembles (DE-KELM) jointly selects the input variables and
tunes a class-weighted Nyström kernel extreme learning machine (KELM). A variant specified first,
CM-KELM, adds an island model, a cultural belief space (normative, situational, topographic and
domain knowledge) and a Gaussian memetic step; its component ablation (`--no-cultural`,
`--no-memetic`) shows that plain DE suffices. The controls at equal budget reproduce the main
findings: with a binary variable mask, ensembles of random variable subsets do as well as evolved
ones (`random_search.py`); with continuous per-variable kernel weights (`--ard`), random sampling
fails (`random_search.py --ard`) and the evolutionary search matches automatic relevance
determination learned by marginal likelihood in a sparse Gaussian process (`run_gp_ard.py`).
Fitness (3-fold CV AUPRC) runs on the GPU in double precision with CuPy and reproduces the CPU
implementation to numerical precision.

The confirmatory analyses (repeated outer cross-validation, KDIGO stage >= 2, subgroups,
interpretability, clinical thresholds) follow the pre-specified protocol in
[`docs/PROTOCOL_confirmatory.md`](docs/PROTOCOL_confirmatory.md), committed before they were run;
`src/protocol_counts.py` gives the aggregate counts computed before the commit.

## Data

The study uses **INSPIRE v1.4.2** (PhysioNet, credentialed access):
https://doi.org/10.13026/1eay-yc85. The data are **not** included and must not be committed:
obtain access on PhysioNet, sign the data use agreement, download the files and set

```bash
export INSPIRE_DIR=/path/to/physionet.org/files/inspire/1.4.2
```

Only aggregate results (metrics, tables, figures) should leave your secure environment.
`.gitignore` excludes data, prediction files and logs.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install cupy-cuda12x        # optional, GPU fitness (match your CUDA version)
```

Tested with Python 3.14 and CuPy on an NVIDIA RTX 6000 Ada; the CPU pipeline and smoke test also run on Python 3.11.

## Quick check without data

```bash
bash tests/smoke_test.sh
```

runs baselines, CM-KELM, the component ablation, the feature-selection control and the statistics
on synthetic dummy data in a few minutes on a CPU.

## Layout

| Path | Content |
|---|---|
| `src/inspire_preprocessing.py`, `src/intraop_features.py`, `src/cohort_split.py` | cohort, outcome (KDIGO relative criterion), preoperative and end-of-surgery variables, the fixed 70/30 split |
| `src/nystrom_kelm.py` | class-weighted Nyström KELM |
| `src/kelm_variant.py`, `src/cmeelm.py` | genome encoding, cultural belief space |
| `src/nystrom_cmkelm.py`, `src/gpu_fitness.py` | CM-KELM search, CPU and GPU (CuPy) fitness |
| `src/population_ensemble.py` | island-champion, top-k and diverse ensembles |
| `src/adaptive_cultural.py`, `src/hybrid_memes.py`, `src/fuzzy_control.py` | ablations: adaptive knowledge sources, noise-robust fitness, multimeme local search with fuzzy/credit controllers |
| `src/run_fullcohort_cmkelm_v4.py` | main CM-KELM runner (all options) |
| `src/run_baselines_preds.py`, `src/tune_trees.py`, `src/run_fs_grid_kelm.py` | baselines, tuned boosted trees, grid KELM on standard feature subsets |
| `src/stats_eval.py`, `src/calibration_dca.py`, `src/subgroup_eval.py` | DeLong, bootstrap CIs, Holm; Platt recalibration and decision curves; subgroups |
| `src/summarize_seeds.py`, `src/seed_ensemble.py`, `src/compare_seeds.py` | seed ensembles, paired seed comparisons (Wilcoxon) |
| `src/table1.py`, `src/feature_selection_freq.py`, `src/collect_timings.py`, `src/compare_gpu_cpu.py`, `src/figs_paper.py` | tables, selection frequencies, timings, GPU-CPU equivalence, figures |
| `src/random_search.py`, `src/grid_ensemble.py`, `src/cv_fitness_ablation.py` | controls: random search over masks at the CM-KELM budget, grid-tuned KELM averaged over landmark draws, ablations on the training-set objective |
| `src/calib_oof.py`, `src/mask_stability.py`, `src/masks_overlap.py` | calibration sets outside the test set, stability and size of the selected variable sets |
| `src/run_tabdl.py` | MLP and FT-Transformer baselines (`rtdl_revisiting_models`) |
| `slurm/` | Slurm job scripts (`jobs_cpu.sh`, `jobs_gpu.sh`, `reproduce_main.sh`) |
| `docs/REPRODUCE.md` | which command produces which table and figure |

All commands are run from `src/`.

## Deep-learning baselines and C++ backends

| Path | Content |
|---|---|
| `cpp/` | C++ implementations of the Nyström-KELM fit: CBLAS/LAPACK, Eigen, CUDA, MPI + CUDA (M. Cafaro); see `cpp/README.md` |
| `dl/` | TabTransformer, tabular Mamba and their stacked ensemble (M. Cafaro), JIT classifier (M. Caforio); see `dl/README.md` |
| `src/run_dl_baselines.py`, `src/run_jit_split.py` | wrappers that train the networks of `dl/` on the article's split and save test predictions for `stats_eval.py` |
| `examples/get_diabetes.py` | downloads the public example data (UCI CDC Diabetes Health Indicators) used to test `cpp/` and `dl/` without INSPIRE |

## License

MIT (see `LICENSE`). The files in `cpp/` and `dl/` keep their authors' MIT headers, and `dl/jit/`
carries its own MIT license (M. Caforio). If you use the code, please cite the article and the software release
(`CITATION.cff`).
