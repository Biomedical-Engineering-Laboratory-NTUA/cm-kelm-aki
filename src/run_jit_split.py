"""Train the JIT classifier (../dl/jit, M. Caforio; main.py unmodified) on the article's split.

Input: the CSV written by export_split_csv.py (features, target, split, row_id). Only the two
train_test_split calls of tabular_dataset.py are replaced by the fixed train | val | test cut, so
the test set is the article's test set. Captures the test probabilities (save_binary_curves) and
the validation probabilities of the best checkpoint, and saves them for stats_eval.py.
main.py seeds torch/numpy with 42 at import; --seed re-seeds them (42 = original behaviour).

The article used main.py's defaults and five seeds (42-46), averaged with seed_ensemble.py:
  python -u run_jit_split.py --split-csv <private_dir>/inspire_esrd_eos2_split.csv \
      --seed 42 --name dl_jit_s42__esrd_eos2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jit-dir", default=os.path.join(os.path.dirname(HERE), "dl", "jit"))
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--name", default="dl_jit__esrd_eos2")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "preds"))
    ap.add_argument("--save-dir", default=None, help="--model_save_dir of main.py (default: next to the CSV)")
    ap.add_argument("--seed", type=int, default=42, help="torch/numpy seed (main.py hard-codes 42 at import)")
    ap.add_argument("--val-out-dir", default=os.path.join(HERE, "preds_val"),
                    help="validation probabilities of the best checkpoint (for stacking)")
    a, extra = ap.parse_known_args()
    extra = [x for x in extra if x != "--"]

    df = pd.read_csv(a.split_csv)
    n = df["split"].value_counts()
    ntr, nva, nte = int(n["train"]), int(n["val"]), int(n["test"])
    assert (df["split"].iloc[:ntr] == "train").all() and (df["split"].iloc[ntr:ntr + nva] == "val").all()
    y_test = df.loc[df["split"] == "test", "target"].to_numpy().astype(int)
    y_val = df.loc[df["split"] == "val", "target"].to_numpy().astype(int)
    tmp_csv = os.path.splitext(a.split_csv)[0] + "_jit_" + a.name + ".csv"
    df.drop(columns=["split", "row_id"]).to_csv(tmp_csv, index=False)
    os.chmod(tmp_csv, 0o600)

    sys.path.insert(0, os.path.abspath(os.path.expanduser(a.jit_dir)))
    os.chdir(os.path.abspath(os.path.expanduser(a.jit_dir)))
    import tabular_dataset as td

    calls = {"k": 0}

    def fixed_split(X, y, test_size=None, stratify=None, random_state=None, **kw):
        calls["k"] += 1
        cut = ntr if calls["k"] == 1 else nva
        return X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]

    td.train_test_split = fixed_split

    import main as jm
    captured = {}
    orig = jm.save_binary_curves

    def capture(y_true, y_prob, path):
        captured["y"], captured["p"] = np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float)
        return orig(y_true, y_prob, path)

    jm.save_binary_curves = capture

    import random
    import torch
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)

    orig_eval = jm.evaluate
    seen = {}

    def evaluate(model, loader, strategy, device, args, desc="Val", detailed=False, return_outputs=False):
        if str(desc).startswith("Val"):
            seen["val_loader"] = loader
        elif desc == "Test" and "val_loader" in seen:
            _, lg, tg = orig_eval(model, seen["val_loader"], strategy, device, args, desc="Val(best)",
                                  detailed=False, return_outputs=True)
            captured["yv"] = tg.numpy().astype(int)
            captured["pv"] = torch.softmax(lg, dim=1).numpy()[:, 1].astype(float)
        return orig_eval(model, loader, strategy, device, args, desc=desc, detailed=detailed,
                         return_outputs=return_outputs)

    jm.evaluate = evaluate
    save_dir = a.save_dir or os.path.join(os.path.dirname(os.path.abspath(a.split_csv)), "runs_jit", a.name)
    sys.argv = ["main.py", "--data_path", tmp_csv, "--target_col", "target",
                "--model_save_dir", save_dir] + extra
    jm.main()

    if "p" not in captured:
        sys.exit("test probabilities were not captured (save_binary_curves not called)")
    if len(captured["y"]) != len(y_test) or not np.array_equal(captured["y"], y_test):
        sys.exit("test labels/order differ from the paper's test split -- predictions NOT saved")
    sys.path.insert(0, HERE)
    from stats_eval import save_predictions
    from sklearn.metrics import average_precision_score, roc_auc_score
    p = captured["p"]
    print(f"{a.name}: test AUROC {roc_auc_score(y_test, p):.4f}  AUPRC {average_precision_score(y_test, p):.4f}")
    save_predictions(os.path.join(a.out_dir, a.name + ".npz"), a.name, y_test, p, model="JIT classifier",
                     args=" ".join(extra), seed=a.seed)
    if "pv" in captured:
        if not np.array_equal(captured["yv"], y_val):
            sys.exit("validation labels/order differ from the paper's split -- val predictions NOT saved")
        save_predictions(os.path.join(a.val_out_dir, a.name + ".npz"), a.name, y_val, captured["pv"],
                         model="JIT classifier", split="val", seed=a.seed)
        print(f"{a.name}: val AUROC {roc_auc_score(y_val, captured['pv']):.4f} (best checkpoint)")
    os.remove(tmp_csv)


if __name__ == "__main__":
    main()
