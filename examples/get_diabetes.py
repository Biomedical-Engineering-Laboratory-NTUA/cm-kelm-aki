"""Download the public example data for the C++ and deep-learning code.

CDC Diabetes Health Indicators (BRFSS 2015), UCI Machine Learning Repository,
https://doi.org/10.24432/C53919 (253,680 rows, 21 features, binary target).

Writes data/diabetes.csv at the repository root, with the target renamed to 'target'
(the C++ and Python readers find it by name), and links data/ into every folder of
cpp/ and dl/, where the programs expect ./data/diabetes.csv.

    pip install ucimlrepo
    python examples/get_diabetes.py
"""
import os

from ucimlrepo import fetch_ucirepo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ds = fetch_ucirepo(id=891)
df = ds.data.features.copy()
df.insert(0, "target", ds.data.targets.iloc[:, 0].astype(int))
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
out = os.path.join(ROOT, "data", "diabetes.csv")
df.to_csv(out, index=False)
print(f"{out}: {df.shape[0]} rows, {df.shape[1] - 1} features, {df.target.mean():.1%} positive")

for top in ("cpp", "dl"):
    for sub in sorted(os.listdir(os.path.join(ROOT, top))):
        d = os.path.join(ROOT, top, sub)
        link = os.path.join(d, "data")
        if os.path.isdir(d) and not os.path.lexists(link):
            os.symlink(os.path.relpath(os.path.join(ROOT, "data"), d), link)
            print("linked", os.path.relpath(link, ROOT))
