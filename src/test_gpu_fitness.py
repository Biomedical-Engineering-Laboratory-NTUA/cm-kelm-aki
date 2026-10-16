"""
GPU fitness: correctness and speed check (run this first on the GPU node).
=========================================================================

For a set of random genomes it evaluates the fitness with
  (a) the existing CPU function (nystrom_cmkelm.nystrom_kelm_fitness)
  (b) gpu_fitness.NystromFitness on the GPU, precision='fp64'
  (c) gpu_fitness.NystromFitness on the GPU, precision='mixed'
with both the fixed split (cv_seed=0) and a re-drawn one (cv_seed=1003), and
reports the largest absolute fitness difference and the time per evaluation.

Pass criteria:
  fp64  : max |diff| < 1e-6   (AUPRC is rank-based: round-off can swap two
          near-equal scores and move the value by ~1e-7, as seen on 27 Sep)
  mixed : max |diff| < 1e-6   (residual-checked solver with FP64 fallback);
          the fallback count shows how often the FP32 preconditioner was not enough

Usage (inside a GPU job, see jobs_gpu.sh):
    python -u test_gpu_fitness.py --m 1000 --n-genomes 8
    python -u test_gpu_fitness.py --surrogate inspire_synthetic_surrogate.csv   # no real data needed
"""

import argparse
import os
import time

import numpy as np

from kelm_variant import KELMEncoding
from nystrom_cmkelm import nystrom_kelm_fitness
from gpu_fitness import NystromFitness, cp


def load(a):
    if a.surrogate:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        df = pd.read_csv(a.surrogate)
        y = df.pop("target").values.astype(int)
        Xtr, _, ytr, _ = train_test_split(df.values, y, test_size=0.3, random_state=42, stratify=y)
        return StandardScaler().fit_transform(Xtr), ytr
    from cohort_split import load_split
    Xtr, _, ytr, _, _, _ = load_split(dummy=a.dummy, verbose=False)
    return Xtr, ytr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--n-genomes", type=int, default=8)
    ap.add_argument("--surrogate", default=None)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true", help="GPU-only timing (no reference)")
    a = ap.parse_args()

    if cp is None:
        raise SystemExit("CuPy is not installed in this environment: pip install 'cupy-cuda12x[ctk]'")
    dev = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {dev['name'].decode()}  |  CuPy {cp.__version__}  |  CUDA runtime "
          f"{cp.cuda.runtime.runtimeGetVersion()}  |  CPU threads {os.environ.get('OMP_NUM_THREADS', '?')}",
          flush=True)

    X, y = load(a)
    m = min(a.m, len(y) // 4)
    enc = KELMEncoding(X.shape[1])
    rng = np.random.default_rng(0)
    genomes = [enc.random_individual(rng) for _ in range(a.n_genomes)]
    print(f"train {X.shape}, pos {y.mean():.4f}, m={m}, {a.n_genomes} genomes x 2 splits\n", flush=True)

    f64 = NystromFitness(X, y, enc, m, backend="gpu", precision="fp64")
    fmx = NystromFitness(X, y, enc, m, backend="gpu", precision="mixed")
    f64(genomes[0]); fmx(genomes[0])                       # warm-up (cuBLAS/cuSOLVER init)
    f64.n_evals = fmx.n_evals = 0
    f64.seconds = fmx.seconds = 0.0
    fmx.n_fits = fmx.n_fallback = fmx.cg_iters = 0

    t_cpu, d64, dmx = 0.0, [], []
    print(f"{'genome':>6} {'cv':>5} {'CPU':>12} {'GPU fp64':>12} {'GPU mixed':>12} {'|d64|':>9} {'|dmix|':>9}")
    for i, g in enumerate(genomes):
        for cs in (0, 1003):
            if a.skip_cpu:
                ref = float("nan")
            else:
                t0 = time.perf_counter()
                ref = nystrom_kelm_fitness(g, X, y, enc, m, cv_seed=cs, lm_seed=cs)
                t_cpu += time.perf_counter() - t0
            v64 = f64(g, cv_seed=cs, lm_seed=cs)
            vmx = fmx(g, cv_seed=cs, lm_seed=cs)
            d64.append(abs(v64 - ref)); dmx.append(abs(vmx - ref))
            print(f"{i:6d} {cs:5d} {ref:12.8f} {v64:12.8f} {vmx:12.8f} {d64[-1]:9.1e} {dmx[-1]:9.1e}", flush=True)

    n = 2 * a.n_genomes
    print()
    if not a.skip_cpu:
        print(f"CPU (existing)  : {t_cpu / n:.3f} s per evaluation")
    print(f"GPU fp64        : {f64.seconds / n:.3f} s per evaluation"
          + ("" if a.skip_cpu else f"  -> speed-up x{t_cpu / f64.seconds:.1f}"))
    print(f"GPU mixed       : {fmx.seconds / n:.3f} s per evaluation"
          + ("" if a.skip_cpu else f"  -> speed-up x{t_cpu / fmx.seconds:.1f}"))
    if not a.skip_cpu:
        for lbl, d in (("fp64 ", d64), ("mixed", dmx)):
            ok = max(d) < 1e-6
            print(f"{lbl} max |diff| = {max(d):.2e} (median {np.median(d):.2e})  -> {'PASS' if ok else 'FAIL'}")
        print(fmx.summary())


if __name__ == "__main__":
    main()
