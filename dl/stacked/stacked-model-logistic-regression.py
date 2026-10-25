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
# Stacking model
# 
# Base learners: Transformer and Mamba models
# Meta-learner (hierarchical stacked model): Logistic Regression model
#
# October 18, 2026
# Version 0.8
#
# Massimo Cafaro, Ph.D.
# Associate Professor
# Department of Engineering for Innovation
# University of Salento, Lecce, Italy
#
#

import argparse
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# automatic device selection (CUDA/MPS/CPU)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Computational backend in use: {device}")


# base learner: Tabular Transformer 
class TabularDataset(Dataset):
    def __init__(self, x_cat: np.ndarray, x_cont: np.ndarray, y: np.ndarray, dtype: torch.dtype = torch.float32):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_cont = torch.tensor(x_cont, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x_cat[idx], self.x_cont[idx], self.y[idx]


class TabTransformer(nn.Module):
    def __init__(
        self,
        categories: list[int],
        num_continuous: int,
        dim: int = 32,
        depth: int = 3,
        heads: int = 16,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        mlp_hidden_dims: tuple[int, ...] = (128, 64),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.dtype = dtype

        if self.num_categories > 0:
            self.embeddings = nn.ModuleList([nn.Embedding(c, dim) for c in categories])
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=attn_dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        else:
            self.embeddings = None
            self.transformer = None

        if self.num_continuous > 0:
            self.cont_layer_norm = nn.LayerNorm(num_continuous)

        in_dim = (self.num_categories * dim) + self.num_continuous
        mlp_layers = []
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
            ])
            in_dim = hidden_dim

        mlp_layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        if self.num_categories > 0:
            embedded = [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.embeddings)]
            x_cat_emb = torch.cat(embedded, dim=1)
            x_cat_trans = self.transformer(x_cat_emb)
            x_cat_flat = x_cat_trans.flatten(start_dim=1)
        else:
            x_cat_flat = torch.empty((x_cat.size(0), 0), device=x_cat.device, dtype=self.dtype)

        if self.num_continuous > 0:
            x_cont_norm = self.cont_layer_norm(x_cont)
            x = torch.cat([x_cat_flat, x_cont_norm], dim=1)
        else:
            x = x_cat_flat

        return self.mlp(x)


# base learner: Tabular Mamba (Selective State Space Model)
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.biases = nn.Parameter(torch.zeros(num_features, d_model))
        self.feature_embed = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1)
        tokens = x_expanded * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)
        return tokens + self.feature_embed


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # S4D Real initialization for state transition matrix A
        A = torch.repeat_interleave(torch.arange(1, self.d_state + 1, dtype=dtype), self.d_inner).reshape(self.d_inner, self.d_state)
        
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=dtype))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_act = self.act(x_conv)

        x_ssm = self._selective_ssm(x_act)
        y = x_ssm * self.act(z)
        return self.out_proj(y)

    def _selective_ssm(self, u: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_inner = u.shape
        A = -torch.exp(self.A_log)

        x_dbl = self.x_proj(u)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        ys = []
        h = torch.zeros(batch_size, d_inner, self.d_state, device=u.device, dtype=u.dtype)

        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)
            u_t = u[:, t, :].unsqueeze(-1)
            B_t = B[:, t, :].unsqueeze(1)
            C_t = C[:, t, :].unsqueeze(-1)

            dA = torch.exp(dt_t * A)
            dB = dt_t * B_t
            h = dA * h + dB * u_t
            y_t = torch.matmul(h, C_t).squeeze(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        return y + u * self.D


class TabularMambaClassifier(nn.Module):
    def __init__(self, num_features: int, d_model: int = 64, d_state: int = 16, depth: int = 2, dropout: float = 0.1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(d_model),
                SelectiveSSM(d_model=d_model, d_state=d_state, dtype=dtype),
                nn.Dropout(dropout)
            ])
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        for norm, mamba_block, drop in self.layers:
            residual = tokens
            tokens = mamba_block(norm(tokens))
            tokens = drop(tokens) + residual
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)


# Dataset ingestion/preprocessing
# inputs are tailored for both Tabular Transformer (categorical + continuous split) and Mamba (flat feature vectors)
def load_and_preprocess_data(csv_path: str, batch_size: int = 256, dtype: torch.dtype = torch.float32):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File '{csv_path}' not found."
        )

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["target"])
    y = df["target"].values

    cat_cols = []
    cont_cols = []
    for col in X.columns:
        if pd.api.types.is_integer_dtype(X[col]) or X[col].nunique() <= 10:
            cat_cols.append(col)
        else:
            cont_cols.append(col)

    if len(cat_cols) == 0:
        print("Discretizing 4 features into categorical bins for Tabular Transformer processing...")
        sample_cols = list(X.columns[:4])
        for c in sample_cols:
            X[c] = pd.qcut(X[c], q=5, labels=False, duplicates="drop")
            cat_cols.append(c)
            cont_cols.remove(c)
            
    print(f"Detected {len(cat_cols)} categorical feature(s) and {len(cont_cols)} continuous feature(s).")

    # Stratified Split: 70% Train, 15% Validation, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # categorical preprocessing
    cat_cardinalities = []
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = encoder.fit_transform(X_train[cat_cols]).astype(np.int64) + 1
        X_val_cat = encoder.transform(X_val[cat_cols]).astype(np.int64) + 1
        X_test_cat = encoder.transform(X_test[cat_cols]).astype(np.int64) + 1

        for c in cat_cols:
            cat_cardinalities.append(int(X[c].nunique()) + 2)
    else:
        X_train_cat = np.zeros((len(X_train), 0), dtype=np.int64)
        X_val_cat = np.zeros((len(X_val), 0), dtype=np.int64)
        X_test_cat = np.zeros((len(X_test), 0), dtype=np.int64)

    # flat standardised representation
    scaler = StandardScaler()
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    X_train_cont = scaler.fit_transform(X_train[cont_cols]).astype(np_dtype)
    X_val_cont = scaler.transform(X_val[cont_cols]).astype(np_dtype)
    X_test_cont = scaler.transform(X_test[cont_cols]).astype(np_dtype)

    dataloaders_transformer = {
        "train": DataLoader(TabularDataset(X_train_cat, X_train_cont, y_train, dtype=dtype), batch_size=batch_size, shuffle=True),
        "val": DataLoader(TabularDataset(X_val_cat, X_val_cont, y_val, dtype=dtype), batch_size=batch_size, shuffle=False),
        "test": DataLoader(TabularDataset(X_test_cat, X_test_cont, y_test, dtype=dtype), batch_size=batch_size, shuffle=False),
    }

    scaler_full = StandardScaler()
    X_train_full = scaler_full.fit_transform(X_train).astype(np_dtype)
    X_val_full = scaler_full.transform(X_val).astype(np_dtype)
    X_test_full = scaler_full.transform(X_test).astype(np_dtype)

    dataloaders_mamba = {
        "train": DataLoader(TensorDataset(torch.tensor(X_train_full, dtype=dtype), torch.tensor(y_train, dtype=dtype)), batch_size=batch_size, shuffle=True),
        "val": DataLoader(TensorDataset(torch.tensor(X_val_full, dtype=dtype), torch.tensor(y_val, dtype=dtype)), batch_size=batch_size, shuffle=False),
        "test": DataLoader(TensorDataset(torch.tensor(X_test_full, dtype=dtype), torch.tensor(y_test, dtype=dtype)), batch_size=batch_size, shuffle=False),
    }

    split_targets = {"train": y_train, "val": y_val, "test": y_test}
    meta_info = {
        "cat_cardinalities": cat_cardinalities,
        "num_continuous": len(cont_cols),
        "num_full_features": X_train_full.shape[1]
    }

    return dataloaders_transformer, dataloaders_mamba, split_targets, meta_info


# training Tabular Tranformer
def train_tabtransformer(model: nn.Module, dataloaders: dict, pos_weight: float, epochs: int = 100, dtype: torch.dtype = torch.float32):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=dtype, device=device))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val_auc = 0.0
    best_weights = model.state_dict()

    for epoch in range(1, epochs + 1):
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
        val_probs, val_targets = [], []
        with torch.no_grad():
            for x_cat, x_cont, y in dataloaders["val"]:
                x_cat, x_cont = x_cat.to(device), x_cont.to(device)
                probs = torch.sigmoid(model(x_cat, x_cont)).cpu().float().numpy()
                val_probs.extend(probs)
                val_targets.extend(y.cpu().float().numpy())
        
        val_auc = roc_auc_score(val_targets, val_probs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = model.state_dict()
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | AUROC: {val_auc:.4f}")
    
    print(f"Optimal AUROC for Tabular Transformer: {best_val_auc:.4f}")
    model.load_state_dict(best_weights)
    return model

# training Tabular Mamba
def train_mamba(model: nn.Module, dataloaders: dict, pos_weight: float, epochs: int = 100, dtype: torch.dtype = torch.float32):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=dtype, device=device))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val_auc = 0.0
    best_weights = model.state_dict()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for x_b, y_b in dataloaders["train"]:
            x_b, y_b = x_b.to(device), y_b.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
            
        train_loss /= len(dataloaders["train"].dataset)
        
        # validation
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for x_b, y_b in dataloaders["val"]:
                x_b = x_b.to(device)
                probs = torch.sigmoid(model(x_b)).cpu().float().numpy()
                val_probs.extend(probs)
                val_targets.extend(y_b.cpu().float().numpy())
                
        val_auc = roc_auc_score(val_targets, val_probs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = model.state_dict()
            
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | AUROC: {val_auc:.4f}")
            
    print(f"Optimal AUROC for Tabular Mamba: {best_val_auc:.4f}")
    model.load_state_dict(best_weights)
    return model

# Out-of-Fold predictions
def get_predictions(model: nn.Module, dataloader: DataLoader, is_transformer: bool = True) -> np.ndarray:
    model.eval()
    probs_list = []
    with torch.no_grad():
        if is_transformer:
            for x_cat, x_cont, _ in dataloader:
                x_cat, x_cont = x_cat.to(device), x_cont.to(device)
                probs = torch.sigmoid(model(x_cat, x_cont)).cpu().float().numpy()
                probs_list.extend(probs)
        else:
            for x_b, _ in dataloader:
                x_b = x_b.to(device)
                probs = torch.sigmoid(model(x_b)).cpu().float().numpy()
                probs_list.extend(probs)
    return np.array(probs_list).ravel()


# plotting and saving AUROC/AUPRC Curves
def plot_evaluation_curves(y_true: np.ndarray, model_probs: dict[str, np.ndarray], save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AUROC
    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        score = roc_auc_score(y_true, probs)
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC = {score:.4f})")

    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # AUPRC
    for name, probs in model_probs.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        score = average_precision_score(y_true, probs)
        axes[1].plot(recall, precision, lw=2, label=f"{name} (AUPRC = {score:.4f})")

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].set_title("Precision-Recall Curve (PR)", fontsize=13, fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nROC and PR curves saved to: '{save_path}'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Stacked Ensemble with Tabular Transformer, Tabular Mamba, and Logistic Regression")
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
    print(f"Loading and preprocessing dataset with precision {args.precision}...")
    dataloaders_trans, dataloaders_mamba, split_targets, meta_info = load_and_preprocess_data(csv_file, dtype=dtype)

    y_train = split_targets["train"]
    pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

    # train Tabular Transformer
    print("\nTraining base learner: Tabular Transformer...")
    trans_model = TabTransformer(
        categories=meta_info["cat_cardinalities"],
        num_continuous=meta_info["num_continuous"],
        dim=32, depth=3, heads=4, dtype=dtype
    ).to(device=device, dtype=dtype)
    trans_model = train_tabtransformer(trans_model, dataloaders_trans, pos_weight=pos_weight, epochs=100, dtype=dtype)

    # Train Tabular Mamba
    print("\nTraining base learner: Tabular Mamba...")
    mamba_model = TabularMambaClassifier(
        num_features=meta_info["num_full_features"],
        d_model=64, d_state=16, depth=2, dtype=dtype
    ).to(device=device, dtype=dtype)
    mamba_model = train_mamba(mamba_model, dataloaders_mamba, pos_weight=pos_weight, epochs=100, dtype=dtype)

    # generate meta-features (probabilities) on validation and test Sets
    val_p_trans = get_predictions(trans_model, dataloaders_trans["val"], is_transformer=True)
    val_p_mamba = get_predictions(mamba_model, dataloaders_mamba["val"], is_transformer=False)
    X_val_meta = np.column_stack((val_p_trans, val_p_mamba))

    test_p_trans = get_predictions(trans_model, dataloaders_trans["test"], is_transformer=True)
    test_p_mamba = get_predictions(mamba_model, dataloaders_mamba["test"], is_transformer=False)
    X_test_meta = np.column_stack((test_p_trans, test_p_mamba))

    # Train meta-learner: Logistic Regression
    print("\nTraining meta-learner: Logistic Regression...")
    lr_meta_learner = LogisticRegression(max_iter=1000)
    lr_meta_learner.fit(X_val_meta, split_targets["val"])

    # ensemble probabilities and predictions
    test_meta_probs = lr_meta_learner.predict_proba(X_test_meta)[:, 1]
    test_meta_preds = lr_meta_learner.predict(X_test_meta)
    y_test = split_targets["test"]

    # evaluation of standard metrics
    acc = accuracy_score(y_test, test_meta_preds)
    prec = precision_score(y_test, test_meta_preds, zero_division=0)
    rec = recall_score(y_test, test_meta_preds, zero_division=0)
    f1 = f1_score(y_test, test_meta_preds, zero_division=0)
    auroc = roc_auc_score(y_test, test_meta_probs)
    auprc = average_precision_score(y_test, test_meta_probs)

    print("\n" + "=" * 55)
    print("      STACKED ENSEMBLE EVALUATION METRICS (TEST SET) ")
    print("=" * 55)
    print(f" Accuracy           : {acc:.4f}")
    print(f" Precision          : {prec:.4f}")
    print(f" Recall             : {rec:.4f}")
    print(f" F1-Score           : {f1:.4f}")
    print(f" AUROC Score        : {auroc:.4f}")
    print(f" AUPRC Score        : {auprc:.4f}")
    print("=" * 55)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, test_meta_preds, digits=4, zero_division=0))

    # plot of AUROC and AUPRC curves
    plot_evaluation_curves(
        y_test,
        {
            "Transformer": test_p_trans,
            "Mamba": test_p_mamba,
            "Stacked meta-learner: Logistic Regression": test_meta_probs
        },
        save_path="stacked_ensemble_performance_curves.png"
    )


if __name__ == "__main__":
    main()