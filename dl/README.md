# Deep-learning baselines

| Folder | Model | Author |
|---|---|---|
| `tabtransformer/` | TabTransformer | M. Cafaro |
| `mamba/` | tabular Mamba (two selective state-space blocks over per-feature tokens) | M. Cafaro |
| `stacked/` | stacked TabTransformer + Mamba with a logistic-regression meta-learner (the SVM variant is not used in the article) | M. Cafaro |
| `jit/` | JIT classifier: the Just-in-Time Transformer adapted to tabular classification | M. Caforio (own MIT license in `jit/LICENSE`) |

## Stand-alone use (public example data)

```bash
pip install ucimlrepo && python examples/get_diabetes.py      # from the repository root
cd dl/mamba && python mamba.py --precision float32            # float32 (default) or float64
cd dl/tabtransformer && python transformer.py --precision float32
cd dl/stacked && python stacked-model-logistic-regression.py --precision float32
cd dl/jit && pip install -r requirements.txt && python main.py --data_path ./data/diabetes.csv --model_save_dir ./output/jit
```

## Settings used in the article

The article's results were obtained in float32 with the wrappers in `src/`, on the INSPIRE
split (85% of the training set for training, 15% for validation, the common test set), with
AdamW (learning rate 1e-3) and batches of 256, keeping the epoch with the best validation AUROC:

- `src/run_dl_baselines.py --model all --esrd cr4_z49 --features eos2`:
  TabTransformer (width 32, three layers, 8 heads, MLP 128-64, positive-class weight 20,
  plateau scheduler with factor 0.5 and patience 3), tabular Mamba (width 64, two blocks,
  state size 16) and the stacked model (`stacked_lr_h8`: its own TabTransformer with 8 heads,
  the same Mamba, logistic-regression meta-learner fitted on the validation predictions);
  weight decay 1e-4 and 100 epochs for all three;
- `src/run_jit_split.py` on the CSV of `src/export_split_csv.py`: `jit/main.py` with its defaults
  (width 32, 8 heads, 3 encoder and 2 decoder layers, 4 feedback steps, 30 epochs, weight decay
  1e-4, plateau scheduler), seeds 42-46, averaged with `src/seed_ensemble.py`.

The stand-alone scripts above use their own example settings (e.g. the stand-alone TabTransformer
uses 16 heads); the article's settings are those listed here. With the same seed, the model files
in `dl/` give the same predictions as the versions used for the article (checked in float32).
