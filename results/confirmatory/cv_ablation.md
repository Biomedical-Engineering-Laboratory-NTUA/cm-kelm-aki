# Exploratory 2 x 2 ablation of DE-KELM-ARD (confirmatory folds; not prespecified)

| Model | folds | AUROC | AUPRC |
|---|---|---|---|
| DE + ARD (M1) | 10 | 0.8896 | 0.3610 |
| DE + mask (M2) | 10 | 0.8855 | 0.3484 |
| CM + ARD | 10 | 0.8895 | 0.3551 |
| CM + mask | 10 | 0.8848 | 0.3401 |
| random search + ARD | 5 | 0.8837 | 0.3190 |
| random search + mask | 5 | 0.8839 | 0.3342 |
| random search + ARD, island ensembles | 5 | 0.8840 | 0.3204 |
| random search + mask, island ensembles | 5 | 0.8836 | 0.3317 |

| Comparison | metric | difference [corrected 95% CI] | p (unadjusted) | folds |
|---|---|---|---|---|
| DE + ARD (M1) - DE + mask (M2) (encoding, plain DE) | AUPRC | +0.0126 [+0.0030, +0.0222] | 0.0156 | 10 |
| DE + ARD (M1) - DE + mask (M2) (encoding, plain DE) | AUROC | +0.0041 [+0.0013, +0.0069] | 0.00877 | 10 |
| CM + ARD - CM + mask (encoding, cultural-memetic) | AUPRC | +0.0150 [+0.0023, +0.0276] | 0.0254 | 10 |
| CM + ARD - CM + mask (encoding, cultural-memetic) | AUROC | +0.0047 [+0.0017, +0.0076] | 0.0061 | 10 |
| DE + ARD (M1) - CM + ARD (search, ARD) | AUPRC | +0.0059 [-0.0050, +0.0167] | 0.252 | 10 |
| DE + ARD (M1) - CM + ARD (search, ARD) | AUROC | +0.0002 [-0.0024, +0.0028] | 0.894 | 10 |
| DE + mask (M2) - CM + mask (search, mask) | AUPRC | +0.0082 [-0.0034, +0.0199] | 0.143 | 10 |
| DE + mask (M2) - CM + mask (search, mask) | AUROC | +0.0007 [-0.0018, +0.0033] | 0.55 | 10 |
| DE + ARD (M1) - random search + ARD (evolution vs random, ARD) | AUPRC | +0.0436 [+0.0157, +0.0715] | 0.0123 | 5 |
| DE + ARD (M1) - random search + ARD (evolution vs random, ARD) | AUROC | +0.0055 [+0.0018, +0.0091] | 0.0141 | 5 |
| DE + mask (M2) - random search + mask (evolution vs random, mask) | AUPRC | +0.0156 [+0.0046, +0.0265] | 0.0168 | 5 |
| DE + mask (M2) - random search + mask (evolution vs random, mask) | AUROC | +0.0019 [-0.0005, +0.0043] | 0.0931 | 5 |
| DE + ARD (M1) - random search + ARD, island ensembles (evolution vs random, ARD, island ensembles) | AUPRC | +0.0423 [+0.0127, +0.0719] | 0.0166 | 5 |
| DE + ARD (M1) - random search + ARD, island ensembles (evolution vs random, ARD, island ensembles) | AUROC | +0.0051 [+0.0014, +0.0089] | 0.0182 | 5 |
| DE + mask (M2) - random search + mask, island ensembles (evolution vs random, mask, island ensembles) | AUPRC | +0.0181 [+0.0038, +0.0323] | 0.0245 | 5 |
| DE + mask (M2) - random search + mask, island ensembles (evolution vs random, mask, island ensembles) | AUROC | +0.0022 [-0.0005, +0.0048] | 0.088 | 5 |
