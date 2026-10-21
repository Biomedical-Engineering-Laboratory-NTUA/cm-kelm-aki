#
#
# MIT License

# Copyright (c) 2026 Massimo Cafaro, Ph.D.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
#
# Mamba model
#
# October 18, 2026
# Version 0.4
#
# Massimo cafaro, Ph.D.
# Associate Professor
# Department of Engineering for Innovation
# University of Salento, Lecce, Italy
#
#
# How to run the code to process the diabetes dataset
#
#Default single precision (float32)
#python mamba.py --precision float32

# Double precision (float64)
#python mamba.py --precision float64

# Half precision (float16 / bfloat16)
#python mamba.py --precision float16

import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    classification_report
)

# set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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


# Mamba architecture (Selective State Space Model)
class FeatureTokenizer(nn.Module):
    """
    Transforms tabular scalar features (B, D) into token embeddings (B, D, d_model).
    Each feature gets its own scalar linear projection + learnable identity embedding.
    """
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        
        self.weights = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.biases = nn.Parameter(torch.zeros(num_features, d_model))
        self.feature_embed = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1)  # (B, D, 1)
        tokens = x_expanded * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)
        tokens = tokens + self.feature_embed  # (B, D, d_model)
        return tokens


class SelectiveSSM(nn.Module):
    """
    PyTorch implementation of the Selective State Space Block (Mamba).
    Processes feature sequences along the sequence dimension (D).
    """
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

        # input-dependent projections for Selective SSM
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D Real initialization for state transition matrix A matching model precision
        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=dtype), self.d_inner
        ).reshape(self.d_inner, self.d_state)
        
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=dtype))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # linear projection to expanded dimension and split branches
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)

        # 1D causal convolution
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_act = self.act(x_conv)

        # Selective SSM step
        x_ssm = self._selective_ssm(x_act)

        # gated modulation and projection
        y = x_ssm * self.act(z)
        return self.out_proj(y)

    def _selective_ssm(self, u: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_inner = u.shape
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # dynamic projections (selective mechanism)
        x_dbl = self.x_proj(u)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # sequential scan over token sequence (D features)
        ys = []
        h = torch.zeros(batch_size, d_inner, self.d_state, device=u.device, dtype=u.dtype)

        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            u_t = u[:, t, :].unsqueeze(-1)    # (B, d_inner, 1)
            B_t = B[:, t, :].unsqueeze(1)     # (B, 1, d_state)
            C_t = C[:, t, :].unsqueeze(-1)    # (B, d_state, 1)

            # continuous -> discrete discretization
            dA = torch.exp(dt_t * A)         # (B, d_inner, d_state)
            dB = dt_t * B_t                  # (B, d_inner, d_state)

            # recurrence state update: h_t = dA * h_{t-1} + dB * u_t
            h = dA * h + dB * u_t

            # output computation: y_t = sum_n(C_t * h_t)
            y_t = torch.matmul(h, C_t).squeeze(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + u * self.D
        return y


class TabularMambaClassifier(nn.Module):
    """
    Tabular Mamba model composed of Feature Tokenizer, stacked Selective SSM blocks,
    mean pooling, and binary classification head.
    """
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
        tokens = self.tokenizer(x)  # (B, D, d_model)

        for norm, mamba_block, drop in self.layers:
            residual = tokens
            tokens = mamba_block(norm(tokens))
            tokens = drop(tokens) + residual

        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)  # Aggregation over feature dimension
        logits = self.head(pooled)
        return logits.squeeze(-1)


# data loading and stratified splitting
def prepare_data(data_path: str, batch_size: int = 256, dtype: torch.dtype = torch.float32):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    # stratified split: 70% Train, 15% Validation, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_temp_labels = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    # standardize continuous feature distributions
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # PyTorch DataLoaders with chosen precision dtype
    train_ds = TensorDataset(torch.tensor(X_train, dtype=dtype), torch.tensor(y_train, dtype=dtype))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=dtype), torch.tensor(y_val, dtype=dtype))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=dtype), torch.tensor(y_temp_labels, dtype=dtype))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_train.shape[1], y_train


# model training and validation
def train_and_evaluate(
    data_path: str = "./data/diabetes.csv",
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    precision: str = "float32"
):
    precision_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = precision_map[precision]

    train_loader, val_loader, test_loader, num_features, y_train = prepare_data(data_path, batch_size, dtype=dtype)

    # instantiate model and convert all parameters to the chosen precision dtype
    model = TabularMambaClassifier(num_features=num_features, d_model=64, d_state=16, depth=2, dtype=dtype).to(device=device, dtype=dtype)

    # compute loss positive weighting to counteract class imbalance (~4.8% positive rate)
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    
    pos_weight = torch.tensor([neg_count / pos_count], dtype=dtype, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_auc = 0.0

    print(f"Starting training on device: {device} | Precision: {precision}")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_b.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device)
                logits = model(X_b)
                probs = torch.sigmoid(logits)

                val_preds.extend(probs.cpu().float().numpy())
                val_targets.extend(y_b.cpu().float().numpy())

        val_auc = roc_auc_score(val_targets, val_preds)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | AUROC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_tabular_mamba.pt")

    # final testing and computation of performance metrics
    print("\nEvaluating best model checkpoint on Test Set...")
    model.load_state_dict(torch.load("best_tabular_mamba.pt"))
    model.eval()

    test_probs, test_targets = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            logits = model(X_b)
            probs = torch.sigmoid(logits)

            test_probs.extend(probs.cpu().float().numpy())
            test_targets.extend(y_b.cpu().float().numpy())

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)

    # convert probabilities to binary predictions (default threshold = 0.5)
    test_preds = (test_probs >= 0.5).astype(int)

    # compute classification metrics
    accuracy = accuracy_score(test_targets, test_preds)
    f1 = f1_score(test_targets, test_preds, average="binary")
    auroc = roc_auc_score(test_targets, test_probs)
    auprc = average_precision_score(test_targets, test_probs)

    print("\n" + "=" * 45)
    print("           TEST PERFORMANCE METRICS          ")
    print("=" * 45)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"AUROC     : {auroc:.4f}")
    print(f"AUPRC     : {auprc:.4f}")
    print("=" * 45)
    print("\nClassification Report:\n", classification_report(test_targets, test_preds, digits=4))

    # plot AUROC and AUPRC curves
    fpr, tpr, _ = roc_curve(test_targets, test_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(test_targets, test_probs)
    baseline_pr = np.mean(test_targets)  # positive prevalence baseline for PR curve

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot: AUROC curve
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"Mamba (AUROC = {auroc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Subplot: Precision-Recall curve (AUPRC)
    axes[1].plot(recall_curve, precision_curve, color="forestgreen", lw=2, label=f"Mamba (AUPRC = {auprc:.4f})")
    axes[1].axhline(y=baseline_pr, color="navy", lw=2, linestyle="--", label=f"Baseline (Prevalence = {baseline_pr:.4f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mamba_performance_curves.png", dpi=300)
    print("Performance plots saved as 'mamba_performance_curves.png'.")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular Mamba Classifier")
    parser.add_argument("--data_path", type=str, default="./data/diabetes.csv", help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Floating point precision to use everywhere (default: float32)"
    )
    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.data_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        precision=args.precision
    )