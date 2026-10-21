#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tabular dataset builder for the JITrans classifier.

Reads any CSV containing a target column, detects categorical and continuous features,
applies a stratified 70/15/15 train/val/test split and preprocesses the features with
statistics fitted on the training split only. The detection rule, the split and the seed
mirror `transformer_standard.py`, so the two models are evaluated on identical splits.

Each sample is returned as a single float tensor (categorical indices first, then scaled
continuous values) plus an integer class label, so the image training/evaluation loops in
main.py can be reused unchanged. Category indices are small integers, exact in float32;
`model.TabularTokenizer` casts them back to long.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

MISSING_TOKEN = "__missing__"


class TabularDataset(Dataset):
    """Packs categorical indices and continuous features into one float tensor per row."""

    def __init__(self, x_cat, x_cont, y, classes):
        self.x = torch.from_numpy(
            np.concatenate([x_cat.astype(np.float32), x_cont.astype(np.float32)], axis=1)
        )
        self.y = torch.from_numpy(y.astype(np.int64))
        self.classes = list(classes)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _split_columns(X, cat_cardinality_threshold):
    """Same rule as transformer_standard.py, plus non-numeric/bool columns forced categorical."""
    cat_cols, cont_cols = [], []
    for col in X.columns:
        s = X[col]
        if (
            not pd.api.types.is_numeric_dtype(s)
            or pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_integer_dtype(s)
            or s.nunique() <= cat_cardinality_threshold
        ):
            cat_cols.append(col)
        else:
            cont_cols.append(col)
    return cat_cols, cont_cols


def _class_name(value):
    """'1.0' -> '1' for integral float labels, str() otherwise."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def build_tabular_datasets(csv_path, target_col="target", cat_cardinality_threshold=10, seed=42):
    """
    Build train/val/test datasets from a CSV file.
    Args:
        csv_path (str): Path to the CSV file.
        target_col (str): Name of the ground-truth column.
        cat_cardinality_threshold (int): Numeric columns with at most this many distinct
            values are treated as categorical.
        seed (int): Random state of the stratified split.
    Returns:
        (train_dataset, val_dataset, test_dataset, meta): meta holds everything needed to
        rebuild the model and reproduce the preprocessing (stored in the checkpoint).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File '{csv_path}' not found.")

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}.")
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_col].values)
    class_names = [_class_name(c) for c in label_encoder.classes_]
    if len(class_names) < 2:
        raise ValueError(f"Target column '{target_col}' has a single class.")

    cat_cols, cont_cols = _split_columns(X, cat_cardinality_threshold)

    # As in transformer_standard.py: without categorical columns, quantize the first
    # 4 features into 5 bins so the categorical embedding path is still exercised.
    quantile_bins = {}
    if len(cat_cols) == 0:
        for c in list(X.columns[:4]):
            X[c], bins = pd.qcut(X[c], q=5, labels=False, duplicates="drop", retbins=True)
            quantile_bins[c] = bins.tolist()
            cat_cols.append(c)
            cont_cols.remove(c)

    # Missing categorical values become their own category
    for c in cat_cols:
        if X[c].isna().any():
            X[c] = X[c].astype(object).where(X[c].notna(), MISSING_TOKEN).astype(str)

    # --- Stratified split (70% train, 15% val, 15% test) ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    # --- Categorical features: ordinal codes shifted by +1, 0 = unknown category ---
    cat_categories, cat_cardinalities = [], []
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        x_cat = [
            encoder.fit_transform(X_train[cat_cols]).astype(np.int64) + 1,
            encoder.transform(X_val[cat_cols]).astype(np.int64) + 1,
            encoder.transform(X_test[cat_cols]).astype(np.int64) + 1,
        ]
        cat_categories = [cats.tolist() for cats in encoder.categories_]
        cat_cardinalities = [len(cats) + 1 for cats in encoder.categories_]
    else:
        x_cat = [np.zeros((len(part), 0), dtype=np.int64) for part in (X_train, X_val, X_test)]

    # --- Continuous features: train-median imputation, then standardisation ---
    cont_fill, cont_mean, cont_scale = [], [], []
    if cont_cols:
        medians = X_train[cont_cols].median()
        scaler = StandardScaler()
        x_cont = [
            scaler.fit_transform(X_train[cont_cols].fillna(medians)),
            scaler.transform(X_val[cont_cols].fillna(medians)),
            scaler.transform(X_test[cont_cols].fillna(medians)),
        ]
        cont_fill = medians.tolist()
        cont_mean = scaler.mean_.tolist()
        cont_scale = scaler.scale_.tolist()
    else:
        x_cont = [np.zeros((len(part), 0), dtype=np.float32) for part in (X_train, X_val, X_test)]

    train_dataset = TabularDataset(x_cat[0], x_cont[0], y_train, class_names)
    val_dataset   = TabularDataset(x_cat[1], x_cont[1], y_val,   class_names)
    test_dataset  = TabularDataset(x_cat[2], x_cont[2], y_test,  class_names)

    counts = np.bincount(y_train, minlength=len(class_names))
    meta = {
        "csv_path": csv_path,
        "target_col": target_col,
        "class_names": class_names,
        "cat_cardinality_threshold": cat_cardinality_threshold,
        "split_seed": seed,
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "cat_categories": cat_categories,
        "cat_cardinalities": cat_cardinalities,
        "quantile_bins": quantile_bins,
        "cont_fill": cont_fill,
        "cont_mean": cont_mean,
        "cont_scale": cont_scale,
        "train_class_counts": counts.tolist(),
        # neg/pos ratio of the training split (binary only), used as default positive weight
        "neg_pos_ratio": float(counts[0] / counts[1]) if len(counts) == 2 and counts[1] > 0 else None,
    }
    return train_dataset, val_dataset, test_dataset, meta
