# JITTabularClassifier

Tabular classification with the JITrans (Just In Time Transformers) encoder-decoder.

Every feature of a row becomes one token (embedding table for categorical columns, per-feature
affine map for continuous ones); the encoder turns them into a memory and the decoder refines
the class prediction autoregressively over `--feedback_steps` steps, feeding each predicted
class distribution back as its next input token.

## Contents

| File | Purpose |
|---|---|
| `main.py` | Training / validation / test, checkpoints `last.pt` / `best.pt`, resume, ROC / PR curves |
| `model.py` | Tabular JITrans encoder-decoder (`define_model`) |
| `tabular_dataset.py` | CSV loading, categorical / continuous detection, stratified 70/15/15 split, preprocessing |
| `loss.py` | Loss strategies and metric set |

## Installation

    pip install -r requirements.txt

## Training

    python main.py --data_path ./data/diabetes.csv --model_save_dir ./output/jit_diabetes

Any CSV with a target column works (`--target_col`, default `target`). Numeric columns with at
most `--cat_threshold` distinct values are treated as categorical. For binary targets the positive
class is weighted by the train neg/pos ratio (`--pos_weight` overrides it) and the best checkpoint
is selected on validation AUROC (`--select_metric`).

The output directory holds `train.log`, `last.pt`, `best.pt`, `test_metrics.json` and, for binary
tasks, `test_roc_pr_curves.png`.
