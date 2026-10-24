import re
NAMES = {"logistic_regression": "Logistic regression", "dl_mamba": "Tabular Mamba", "nystrom_kelm_grid": "Grid-tuned KELM",
         "dl_transformer": "TabTransformer", "dl_stacked": "Stacked DL", "random_forest": "Random forest",
         "cmkelm": "CM-KELM", "dl_jit": "JIT Transformer", "dl_mlp": "MLP", "dl_ftt": "FT-Transformer", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
num = r"(-?[0-9.]+)"
def parse(path):
    rows = {}
    for line in open(path):
        if not line.startswith("| ") or line.startswith("| Model"): continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        key = next(v for k, v in NAMES.items() if c[0].startswith(k))
        f = lambda s: [float(x) for x in re.findall(r"-?\d+\.\d+(?:e-?\d+)?", s)]
        r = dict(auroc=f(c[1]), auprc=f(c[2]))
        if c[4] != "(reference)":
            r["dauroc"] = f(c[4]); r["p_auroc"] = float(re.search(r"Holm ([0-9.e-]+)", c[5]).group(1))
            r["dauprc"] = f(c[6]); r["p_auprc"] = float(re.search(r"Holm ([0-9.e-]+)", c[7]).group(1))
        rows[key] = r
    return rows
if __name__ == "__main__":
    import sys, json; print(json.dumps(parse(sys.argv[1]), indent=1))
