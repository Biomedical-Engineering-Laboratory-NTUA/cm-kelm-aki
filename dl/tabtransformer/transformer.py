# MIT License

# Copyright (c) 2026 Massimo Cafaro, Ph.D., University of Salento, Lecce, Italy, ORCID: 0000-0003-1118-7109

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

#
#
# Transformer model
#
# October 18, 2026
# Version 0.3
#
# Massimo cafaro, Ph.D.
# Associate Professor
# Department of Engineering for Innovation
# University of Salento, Lecce, Italy
#
# How to run the code to process the diabetes dataset
#
#Default single precision (float32)
#python transformer.py --precision float32

# Double precision (float64)
#python transformer.py --precision float64

# Half precision (float16 / bfloat16)
#python transformer.py --precision float16

import argparse
import copy
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    classification_report,
)


# reproducibility through same seed
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)

# device Configuration
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Computational backend in use: {device}")


# dataset and data processing
class TabularDataset(Dataset):
    """PyTorch Dataset for Tabular Data containing both categorical and continuous features."""

    def __init__(self, x_cat: np.ndarray, x_cont: np.ndarray, y: np.ndarray, dtype: torch.dtype = torch.float32):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_cont = torch.tensor(x_cont, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_cat[idx], self.x_cont[idx], self.y[idx]


def prepare_data(
    csv_path: str,
    target_col: str = "target",
    cat_cardinality_threshold: int = 10,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, DataLoader], List[int], int]:
    """Loads CSV, applies 70/15/15 stratified split, preprocesses categorical and continuous features.

    Returns DataLoaders, categorical cardinalities, and number of continuous
    features.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File '{csv_path}' not found."
        )

    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # categorize features based on uniqueness/cardinality
    cat_cols = []
    cont_cols = []
    for col in X.columns:
        # Check if integer/discrete or low cardinality
        if (
            pd.api.types.is_integer_dtype(X[col])
            or X[col].nunique() <= cat_cardinality_threshold
        ):
            cat_cols.append(col)
        else:
            cont_cols.append(col)

    # if dataset has no discrete categoricals, discretize top low-nunique features for demonstration
    if len(cat_cols) == 0:
        print(
            "No strictly categorical columns detected. Quantizing 4 features into discrete bins for Transformer processing..."
        )
        sample_cols = list(X.columns[:4])
        for c in sample_cols:
            X[c] = pd.qcut(X[c], q=5, labels=False, duplicates="drop")
            cat_cols.append(c)
            cont_cols.remove(c)

    print(
        f"Detected {len(cat_cols)} categorical feature(s) and {len(cont_cols)} continuous feature(s)."
    )

    # --- Stratified Split (70% Train, 15% Val, 15% Test) ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(
        f"Data split shapes -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}"
    )

    # Encoding Categorical Features
    cat_cardinalities = []
    X_train_cat, X_val_cat, X_test_cat = None, None, None

    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        X_train_cat = encoder.fit_transform(X_train[cat_cols]).astype(np.int64)
        X_val_cat = encoder.transform(X_val[cat_cols]).astype(np.int64)
        X_test_cat = encoder.transform(X_test[cat_cols]).astype(np.int64)

        # shift values by +1 to reserve 0 for unknown/out-of-bound tokens
        X_train_cat += 1
        X_val_cat += 1
        X_test_cat += 1

        for c in cat_cols:
            cardinality = int(X[c].nunique()) + 2  # +1 shift, +1 unknown
            cat_cardinalities.append(cardinality)
    else:
        X_train_cat = np.zeros((len(X_train), 0), dtype=np.int64)
        X_val_cat = np.zeros((len(X_val), 0), dtype=np.int64)
        X_test_cat = np.zeros((len(X_test), 0), dtype=np.int64)

    # scaling Continuous Features (fit strictly on train)
    X_train_cont, X_val_cont, X_test_cont = None, None, None
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    if len(cont_cols) > 0:
        scaler = StandardScaler()
        X_train_cont = scaler.fit_transform(X_train[cont_cols]).astype(np_dtype)
        X_val_cont = scaler.transform(X_val[cont_cols]).astype(np_dtype)
        X_test_cont = scaler.transform(X_test[cont_cols]).astype(np_dtype)
    else:
        X_train_cont = np.zeros((len(X_train), 0), dtype=np_dtype)
        X_val_cont = np.zeros((len(X_val), 0), dtype=np_dtype)
        X_test_cont = np.zeros((len(X_test), 0), dtype=np_dtype)

    # PyTorch DataLoaders
    train_dataset = TabularDataset(X_train_cat, X_train_cont, y_train, dtype=dtype)
    val_dataset = TabularDataset(X_val_cat, X_val_cont, y_val, dtype=dtype)
    test_dataset = TabularDataset(X_test_cat, X_test_cont, y_test, dtype=dtype)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=256, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=512, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=512, shuffle=False),
    }

    return dataloaders, cat_cardinalities, len(cont_cols)


# TabTransformer Model Architecture
class TabTransformer(nn.Module):
    """TabTransformer Model (Huang et al., 2020).

    Passes categorical feature embeddings through multi-head Transformer layers,
    concatenates the contextualized embeddings with normalized continuous features,
    and classifies via an MLP head.
    """

    def __init__(
        self,
        categories: List[int],
        num_continuous: int,
        dim: int = 32,
        depth: int = 4,
        heads: int = 16,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        mlp_hidden_dims: Tuple[int, ...] = (128, 64),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.dtype = dtype

        # categorical Column Embeddings
        if self.num_categories > 0:
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(cardinality, dim)
                    for cardinality in categories
                ]
            )

            # Transformer Encoder Blocks
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=attn_dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=depth
            )
        else:
            self.embeddings = None
            self.transformer = None

        # layer normalisation for continuous features
        if self.num_continuous > 0:
            self.cont_layer_norm = nn.LayerNorm(num_continuous)

        # Multi-Layer Perceptron (MLP) Classifier Head
        in_dim = (self.num_categories * dim) + self.num_continuous
        mlp_layers = []

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(ff_dropout),
                ]
            )
            in_dim = hidden_dim

        mlp_layers.append(nn.Linear(in_dim, 1))  # Output logit
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor
    ) -> torch.Tensor:
        embedded = []

        # embed categorical features
        if self.num_categories > 0:
            for i, emb_layer in enumerate(self.embeddings):
                embedded.append(emb_layer(x_cat[:, i]).unsqueeze(1))

            # shape: (batch_size, num_categories, dim)
            x_cat_emb = torch.cat(embedded, dim=1)

            # contextualize via Transformer
            x_cat_trans = self.transformer(x_cat_emb)

            # flatten categorical representations: (batch_size, num_categories * dim)
            x_cat_flat = x_cat_trans.flatten(start_dim=1)
        else:
            x_cat_flat = torch.empty((x_cat.size(0), 0), device=x_cat.device, dtype=self.dtype)

        # normalize continuous features
        if self.num_continuous > 0:
            x_cont_norm = self.cont_layer_norm(x_cont)
            # concatenate continuous and transformed categorical features
            x = torch.cat([x_cat_flat, x_cont_norm], dim=1)
        else:
            x = x_cat_flat

        # final MLP Head
        return self.mlp(x)


# model Training and validation 
def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight_val: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:

    # weight positive samples to fight class imbalance
    pos_weight = torch.tensor([pos_weight_val], device=device, dtype=dtype)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_auroc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for x_cat, x_cont, y in dataloaders["train"]:
            x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x_cat, x_cont)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)

        train_loss /= len(dataloaders["train"].dataset)

        # validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_cat, x_cont, y in dataloaders["val"]:
                x_cat, x_cont, y = (
                    x_cat.to(device),
                    x_cont.to(device),
                    y.to(device),
                )
                logits = model(x_cat, x_cont)
                probs = torch.sigmoid(logits)

                val_preds.extend(probs.cpu().float().numpy())
                val_targets.extend(y.cpu().float().numpy())

        val_auroc = roc_auc_score(val_targets, val_preds)
        scheduler.step(val_auroc)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_weights = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | AUROC: {val_auroc:.4f}"
        )

    print(f"Training Complete. Best Validation AUROC: {best_val_auroc:.4f}")
    model.load_state_dict(best_model_weights)
    return model


# evaluation and metrics Plotting
def evaluate_and_plot(model: nn.Module, test_loader: DataLoader) -> None:
    model.eval()
    y_probs, y_true = [], []

    with torch.no_grad():
        for x_cat, x_cont, y in test_loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            logits = model(x_cat, x_cont)
            probs = torch.sigmoid(logits)

            y_probs.extend(probs.cpu().float().numpy())
            y_true.extend(y.cpu().float().numpy())

    y_probs = np.array(y_probs).ravel()
    y_true = np.array(y_true).ravel()
    y_pred = (y_probs >= 0.5).astype(int)

    # compute metrics 
    acc = accuracy_score(y_true, y_pred)
    f1_binary = f1_score(y_true, y_pred, average="binary", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    auroc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)

    print("\n" + "=" * 45)
    print("           TEST SET PERFORMANCE            ")
    print("=" * 45)
    print(f" Accuracy           : {acc:.4f}")
    print(f" Binary F1-Score    : {f1_binary:.4f}")
    print(f" Macro F1-Score     : {f1_macro:.4f}")
    print(f" AUROC Score        : {auroc:.4f}")
    print(f" AUPRC Score        : {auprc:.4f}")
    print("=" * 45)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    # ROC and PR Curve computation
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    baseline_pr = np.mean(y_true)  # Positive prevalence baseline

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # subplot: ROC curve
    axes[0].plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"Tabular Transformer (AUROC = {auroc:.4f})",
    )
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(alpha=0.3)

    # Subplot 2: Precision-Recall Curve (AUPRC)
    axes[1].plot(
        recall,
        precision,
        color="forestgreen",
        lw=2,
        label=f"Tabular Transformer (AUPRC = {auprc:.4f})",
    )
    axes[1].axhline(
        y=baseline_pr,
        color="navy",
        lw=1.5,
        linestyle="--",
        label=f"Baseline (Prevalence = {baseline_pr:.4f})",
    )
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("tabtransformer_performance_curves.png", dpi=300)
    print("Saved performance plots to 'tabtransformer_performance_curves.png'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="TabTransformer Training and Evaluation")
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Precision format to use everywhere (default: float32)",
    )
    args = parser.parse_args()

    precision_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = precision_map[args.precision]

    csv_file = "./data/diabetes.csv"

    # prepare Data and splits
    dataloaders, cat_cardinalities, num_continuous = prepare_data(
        csv_file, target_col="target", dtype=dtype
    )

    # instantiate Model and move to device + dtype
    model = TabTransformer(
        categories=cat_cardinalities,
        num_continuous=num_continuous,
        dim=32,
        depth=3,
        heads=16,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_dims=(128, 64),
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    # train model
    trained_model = train_model(
        model=model,
        dataloaders=dataloaders,
        epochs=100,
        lr=1e-3,
        pos_weight_val=20.0,
        dtype=dtype,
    )

    # evaluation and plot
    evaluate_and_plot(trained_model, dataloaders["test"])


if __name__ == "__main__":
    main()