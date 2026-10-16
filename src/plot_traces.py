"""
Figures of the adaptation dynamics (for the paper).
===================================================

Reads the per-generation traces saved in preds/<name>.json by
AdaptiveCulturalCMKELM and produces:

  fig_ks_probs_<variant>.png  knowledge-source probabilities per island over
                              generations (mean over seeds, shaded = min/max)
  fig_diversity.png           feature-mask diversity ratio and best fitness
                              per generation, one line per variant

Usage:
    python plot_traces.py cmkelm_m1000_adKAH_p20_g15 cmkelm_m1000_adnone_p20_g15 ...
"""
import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCES = ["normative", "situational", "topographic", "free"]
PROFILES = ["normative-leaning", "situational-leaning", "topographic-leaning", "exploration-leaning"]


def traces(prefix):
    out = []
    for f in sorted(glob.glob(f"preds/{prefix}_s*.json")):
        t = json.load(open(f)).get("trace")
        if t:
            out.append(t)
    return out


def plot_probs(prefix, runs):
    probs = np.array([[g["probs"] for g in t] for t in runs])   # seeds x gens x islands x sources
    gens = np.arange(1, probs.shape[1] + 1)
    fig, axes = plt.subplots(1, probs.shape[2], figsize=(4 * probs.shape[2], 3.2), sharey=True)
    for c, ax in enumerate(np.atleast_1d(axes)):
        for s, name in enumerate(SOURCES):
            y = probs[:, :, c, s]
            ax.plot(gens, y.mean(0), label=name)
            ax.fill_between(gens, y.min(0), y.max(0), alpha=0.15)
        ax.set_title(f"island {c + 1}" + (f" ({PROFILES[c]})" if "H" in prefix.split("_ad")[-1].split("_")[0] else ""),
                     fontsize=9)
        ax.set_xlabel("generation")
    np.atleast_1d(axes)[0].set_ylabel("selection probability")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"fig_ks_probs_{prefix}.png", dpi=200)
    print(f"written fig_ks_probs_{prefix}.png ({len(runs)} seeds)")


def main(prefixes):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    for p in prefixes:
        runs = traces(p)
        if not runs:
            print(f"no traces for {p} (static FullCohortCMKELM runs have none; use --adaptive none)")
            continue
        div = np.array([[g["diversity_ratio"] for g in t] for t in runs])
        best = np.array([[g["best"] for g in t] for t in runs])
        gens = np.arange(1, div.shape[1] + 1)
        label = p.split("_ad")[-1].split("_")[0] if "_ad" in p else p
        a1.plot(gens, div.mean(0), label=label)
        a2.plot(gens, best.mean(0), label=label)
        if any(v > 0.3 for g in runs[0] for row in g["probs"] for v in row) and "none" not in p:
            plot_probs(p, runs)
    a1.set(xlabel="generation", ylabel="mask diversity / initial", title="population diversity")
    a2.set(xlabel="generation", ylabel="best CV fitness", title="search progress")
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_diversity.png", dpi=200)
    print("written fig_diversity.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
