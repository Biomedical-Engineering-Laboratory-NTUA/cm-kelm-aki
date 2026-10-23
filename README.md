# CM-KELM: evolutionary kernel extreme learning machine for postoperative AKI

Code for the article

> J. Digalakis, M. Cafaro, I. Vezakis, G. K. Matsopoulos, D. D. Koutsouris,
> "A GPU-Accelerated Evolutionary Kernel Extreme Learning Machine with Population Ensembles for
> Postoperative Acute Kidney Injury Prediction," submitted to *IEEE Journal of Biomedical and Health Informatics*.

An island-model differential evolution with population ensembles jointly selects the input
variables and tunes a class-weighted Nyström kernel extreme learning machine (KELM). The full
algorithm, CM-KELM, adds a cultural belief space (normative, situational, topographic and domain
knowledge) and a Gaussian memetic step; the component ablation (`--no-cultural`, `--no-memetic`)
reproduces the finding that evolutionary selection and population ensembles account for the gain.
Fitness (3-fold CV AUPRC) runs on the GPU in double precision with CuPy and reproduces the CPU
implementation to numerical precision.

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
