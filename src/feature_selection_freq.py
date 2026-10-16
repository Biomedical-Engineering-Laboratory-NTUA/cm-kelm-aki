"""
How often each feature is selected by CM-KELM across search seeds.
===================================================================

Reads the `active_features` list stored in the metadata of every single-seed
prediction file matching the pattern(s) and reports, per feature, the fraction
of seeds whose best individual keeps it. Also reports gamma, lambda and the
mask size per seed. Aggregate information only (safe to share).

Usage:
    python feature_selection_freq.py "preds/cmkelm_m1000_gpu_p20_g15_s*__esrd_eos.npz"
    python feature_selection_freq.py "preds/cmkelm_m1000_gpu_p20_g15_s*__esrd.npz" --out fsel_esrd
"""

import argparse
import glob
import json
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--out", default="feature_selection_freq")
    a = ap.parse_args()

    paths = sorted({p for pat in a.patterns for p in glob.glob(pat)})
    paths = [p for p in paths if "__popens_" not in p]
    if not paths:
        raise SystemExit("no files match")
    cnt, rows = Counter(), []
    for p in paths:
        meta = json.loads(str(np.load(p, allow_pickle=False)["meta"]))
        act = meta.get("active_features") or []
        cnt.update(act)
        rows.append((meta["name"], meta.get("gamma"), meta.get("lam"), len(act)))
    n = len(paths)

    lines = [f"# Feature selection frequency over {n} seeds\n", "| seed run | gamma | lambda | features kept |",
             "|---|---|---|---|"]
    for name, g, lam, k in rows:
        lines.append(f"| {name} | {g:.4g} | {lam:.4g} | {k} |")
    ks = np.array([r[3] for r in rows])
    lines.append(f"\nMask size: mean {ks.mean():.1f}, min {ks.min()}, max {ks.max()}\n")
    lines += ["| feature | selected in | fraction |", "|---|---|---|"]
    for f, c in sorted(cnt.items(), key=lambda t: (-t[1], t[0])):
        lines.append(f"| {f} | {c}/{n} | {c / n:.1f} |")
    txt = "\n".join(lines)
    open(f"{a.out}.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
