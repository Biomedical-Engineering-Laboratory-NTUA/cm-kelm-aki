#!/usr/bin/env python3

"""
Tabular Classification using the JITrans (Just In Time Transformers) Model

This script adapts the original autoregressive encoder-decoder JITrans architecture
(originally designed for time-series forecasting) to tabular classification, preserving
its autoregressive "feedback" (retroazione) form: the decoder refines its class
prediction over several steps, feeding each prediction back as input to the next step.

This source code is related to the paper "Just In Time Transformers", by
AHMED ALA EDDINE BENALI, MASSIMO CAFARO, ITALO EPICOCO, MARCO PULIMENO and ENRICO JUNIOR SCHIOPPA

The script parses command-line arguments, builds the train/val/test splits of a CSV file
(see tabular_dataset.py), defines the model, trains it with progress bars, saves the
last/best checkpoints, supports resuming, and reports detailed classification metrics on
validation and test sets (plus ROC / PR curves for binary tasks).

Example usage:
    python main.py --data_path ./diabetes.csv --model_save_dir ./output/jit_diabetes
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import CrossEntropyLossStrategy, LossStrategyFactory
from model import define_model
from tabular_dataset import build_tabular_datasets

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Logging setup  — writes to both stdout and <model_save_dir>/train.log
# ---------------------------------------------------------------------------


def setup_logger(log_path: str) -> logging.Logger:
    """
    Returns a logger that mirrors every message to stdout AND to log_path.
    Call this after model_save_dir is created.
    """
    logger = logging.getLogger("jitrans")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SELECT_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "auroc", "auprc"]


def get_args_parser():
    parser = argparse.ArgumentParser(
        "JITrans tabular classifier", add_help=True
    )

    # Data
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="CSV file with one row per sample and a target column",
    )
    parser.add_argument("--target_col", default="target", type=str)
    parser.add_argument(
        "--cat_threshold",
        default=10,
        type=int,
        help="Numeric columns with at most this many distinct values are categorical",
    )

    # Model
    parser.add_argument("--d_model", default=32, type=int)
    parser.add_argument("--n_heads", default=8, type=int)
    parser.add_argument("--enc_layers", default=3, type=int)
    parser.add_argument("--dec_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument(
        "--pos_weight",
        default=None,
        type=float,
        help="Loss weight of the positive class (binary tasks only). "
        "Default: train neg/pos ratio; 1 disables weighting.",
    )

    # Optimisation
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--num_workers", default=4, type=int)

    # Model selection / early stopping
    parser.add_argument(
        "--select_metric",
        default="auroc",
        choices=SELECT_METRICS,
        help="Validation metric for best checkpoint and early stopping "
        "(auroc / auprc need a binary target)",
    )
    parser.add_argument(
        "--early_stopping",
        default=20,
        type=int,
        help="Stop training after this many epochs with no improvement of the "
        "selection metric. Set to 0 to disable.",
    )

    # Model / training behaviour
    parser.add_argument("--feedback_steps", default=4, type=int)
    parser.add_argument("--loss_type", default="crossentropy", type=str)
    parser.add_argument("--label_smoothing", default=0.0, type=float)
    parser.add_argument("--deep_supervision", action="store_true")
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp", action="store_true")

    # Checkpointing / output
    parser.add_argument("--model_save_dir", default="./output", type=str)
    parser.add_argument("--resume", default="", type=str)

    return parser


# ---------------------------------------------------------------------------
# Loss strategy
# ---------------------------------------------------------------------------


def build_loss_strategy(args, num_classes, class_names, class_weights=None):
    if args.loss_type.lower() == "crossentropy":
        return CrossEntropyLossStrategy(
            num_classes=num_classes,
            class_names=class_names,
            top_k=(1, 5) if num_classes >= 5 else (1,),
            label_smoothing=args.label_smoothing,
            class_weights=class_weights,
        )
    strategy = LossStrategyFactory.create_strategy(args.loss_type)
    strategy.num_classes = num_classes
    strategy.class_names = class_names
    return strategy


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model, loader, strategy, optimizer, scaler, device, epoch, num_epochs, args, logger
):
    model.train()
    running_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=f"Train {epoch + 1}/{num_epochs}", leave=False)

    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            if args.deep_supervision:
                _, all_logits = model(inputs, return_all_steps=True)
                step_losses = [
                    strategy.compute_loss(all_logits[:, t], targets)
                    for t in range(all_logits.shape[1])
                ]
                loss = torch.stack(step_losses).mean()
            else:
                logits = model(inputs)
                loss = strategy.compute_loss(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(
            loss=f"{running_loss / n_batches:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return running_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model,
    loader,
    strategy,
    device,
    args,
    desc="Val",
    detailed=False,
    return_outputs=False,
):
    model.eval()
    all_logits = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets_dev = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            logits = model(inputs)
        loss = strategy.compute_loss(logits, targets_dev)
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.float().cpu())
        all_targets.append(targets.cpu())
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_loss = total_loss / max(n_batches, 1)
    metrics = strategy.compute_metrics(all_logits, all_targets, detailed=detailed)
    metrics["loss"] = avg_loss  # inject loss into the metrics dict
    if return_outputs:
        return metrics, all_logits, all_targets
    return metrics


# ---------------------------------------------------------------------------
# Print / log helpers
# ---------------------------------------------------------------------------


def _log_block(lines, logger):
    """Send a list of strings to the logger line by line."""
    for line in lines:
        logger.info(line)


def _scalar_lines(metrics, W):
    lines = []
    for k in ["accuracy", "balanced_accuracy", "top_1_accuracy", "top_5_accuracy"]:
        if k in metrics:
            lines.append(f"  {k:<34}: {metrics[k]:.4f}")
    lines.append(f"  {'-' * (W - 2)}")
    for k in ["auroc", "auprc"]:
        if k in metrics:
            lines.append(f"  {k:<34}: {metrics[k]:.4f}")
    for k in ["f1_binary", "f1_macro", "f1_weighted", "f1_micro"]:
        if k in metrics:
            lines.append(f"  {k:<34}: {metrics[k]:.4f}")
    lines.append(f"  {'-' * (W - 2)}")
    for k in [
        "precision_macro",
        "precision_weighted",
        "recall_macro",
        "recall_weighted",
    ]:
        if k in metrics:
            lines.append(f"  {k:<34}: {metrics[k]:.4f}")
    lines.append(f"  {'-' * (W - 2)}")
    for k in ["cohen_kappa", "matthews_corrcoef"]:
        if k in metrics:
            lines.append(f"  {k:<34}: {metrics[k]:.4f}")
    return lines


def _confidence_lines(metrics, W):
    keys = [
        "mean_confidence",
        "median_confidence",
        "min_confidence",
        "mean_correct_confidence",
        "mean_incorrect_confidence",
    ]
    if not any(k in metrics for k in keys):
        return []
    lines = [f"  {'-' * (W - 2)}"]
    for k in keys:
        if k in metrics:
            lines.append(f"  {k:<34}: {metrics[k]:.4f}")
    return lines


def _per_class_lines(metrics, W):
    if "per_class" not in metrics:
        return []
    lines = [
        "",
        "  Per-class breakdown",
        f"  {'Class':<30}  Prec   Rec    F1    Support",
        f"  {'-' * (W - 2)}",
    ]
    for cls_name, m in metrics["per_class"].items():
        conf = f"  conf={m['mean_confidence']:.3f}" if "mean_confidence" in m else ""
        lines.append(
            f"  {cls_name:<30}"
            f"  {m['precision']:.3f}"
            f"  {m['recall']:.3f}"
            f"  {m['f1']:.3f}"
            f"  {m['support']:>7}"
            f"{conf}"
        )
    return lines


def _confusion_lines(metrics, class_names, W):
    if "confusion_matrix" not in metrics:
        return []
    cm = metrics["confusion_matrix"]
    lines = ["", "  Confusion matrix (rows=true, cols=pred):"]
    header = "    " + "".join(f"{i:>8d}" for i in range(cm.shape[1]))
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append(f"    {i:<4d}" + "".join(f"{int(v):>8d}" for v in row))
    if class_names:
        lines.append("")
        lines.append("  Class index legend:")
        for i, name in enumerate(class_names):
            lines.append(f"    {i}: {name}")
    return lines


def print_best_report(metrics, epoch, class_names, logger):
    """Detailed report printed (and logged) whenever a new best is reached."""
    W = 62
    lines = [
        "",
        "★" * W,
        f"  NEW BEST  —  Epoch {epoch + 1}",
        "★" * W,
        f"  {'Val Loss':<34}: {metrics.get('loss', float('nan')):.4f}",
        f"  {'-' * (W - 2)}",
    ]
    lines += _scalar_lines(metrics, W)
    lines += _confidence_lines(metrics, W)
    lines += _per_class_lines(metrics, W)
    lines += _confusion_lines(metrics, class_names, W)
    lines += ["★" * W, ""]
    _log_block(lines, logger)


def print_final_report(metrics, class_names, logger):
    """Full report used only at end-of-training test evaluation."""
    W = 62
    lines = [
        "",
        "=" * W,
        "  TEST — FINAL RESULTS",
        "=" * W,
        f"  {'Test Loss':<34}: {metrics.get('loss', float('nan')):.4f}",
        f"  {'-' * (W - 2)}",
    ]
    lines += _scalar_lines(metrics, W)
    lines += _confidence_lines(metrics, W)
    lines += _per_class_lines(metrics, W)
    lines += _confusion_lines(metrics, class_names, W)
    lines += ["=" * W, ""]
    _log_block(lines, logger)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    path, model, optimizer, scaler, epoch, best_acc, args, class_names, extra=None
):
    """`best_acc` holds the best value of args.select_metric."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_acc": best_acc,
            "args": vars(args),
            "class_names": class_names,
            **(extra or {}),
        },
        path,
    )


def save_binary_curves(y_true, y_prob, path):
    """Save ROC and Precision-Recall curves of a binary classifier (class 1 = positive)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    prevalence = float(np.mean(y_true))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(
        fpr, tpr, color="darkorange", lw=2, label=f"JITrans (AUROC = {auroc:.4f})"
    )
    axes[0].plot(
        [0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier"
    )
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(
        "Receiver Operating Characteristic (ROC) Curve", fontweight="bold"
    )
    axes[0].legend(loc="lower right")

    axes[1].plot(
        recall,
        precision,
        color="forestgreen",
        lw=2,
        label=f"JITrans (AUPRC = {auprc:.4f})",
    )
    axes[1].axhline(
        y=prevalence,
        color="navy",
        lw=1.5,
        linestyle="--",
        label=f"Baseline (Prevalence = {prevalence:.4f})",
    )
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].set_title("Precision-Recall (PR) Curve", fontweight="bold")
    axes[1].legend(loc="lower left")

    for ax in axes:
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def to_serializable(metrics):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = to_serializable(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = get_args_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.model_save_dir, exist_ok=True)

    # Logger — mirrors to stdout and to train.log in the output directory
    log_path = os.path.join(args.model_save_dir, "train.log")
    logger = setup_logger(log_path)

    logger.info(f"Using device: {device}")
    logger.info(f"Log file    : {log_path}")

    # --- Datasets / dataloaders ---
    train_dataset, val_dataset, test_dataset, tab_meta = build_tabular_datasets(
        args.data_path,
        target_col=args.target_col,
        cat_cardinality_threshold=args.cat_threshold,
    )
    nb_classes = len(tab_meta["class_names"])
    logger.info(f"Tabular CSV : {args.data_path}  (target column: {args.target_col})")
    logger.info(
        f"Categorical features ({len(tab_meta['cat_cols'])}): {tab_meta['cat_cols']}"
    )
    logger.info(
        f"Continuous features  ({len(tab_meta['cont_cols'])}): {tab_meta['cont_cols']}"
    )
    if tab_meta["quantile_bins"]:
        logger.info(f"Quantized into 5 bins: {list(tab_meta['quantile_bins'])}")
    logger.info(f"Train class counts: {tab_meta['train_class_counts']}")
    class_names = list(
        getattr(train_dataset, "classes", [str(i) for i in range(nb_classes)])
    )

    logger.info(f"Classes ({nb_classes}): {class_names}")
    logger.info(
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )

    if args.select_metric in ("auroc", "auprc") and nb_classes != 2:
        raise ValueError(
            f"--select_metric {args.select_metric} needs a binary task "
            f"({nb_classes} classes found)."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Model / loss / optimiser ---
    model = define_model(
        device,
        cat_cardinalities=tab_meta["cat_cardinalities"],
        num_continuous=len(tab_meta["cont_cols"]),
        num_classes=nb_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        N_enc=args.enc_layers,
        N_dec=args.dec_layers,
        dropout=args.dropout,
        feedback_steps=args.feedback_steps,
        use_checkpoint=args.use_checkpoint,
    )
    class_weights = None
    if nb_classes == 2:
        if args.pos_weight is None:
            args.pos_weight = tab_meta["neg_pos_ratio"]
        class_weights = torch.tensor([1.0, args.pos_weight], device=device)
        logger.info(f"Positive class weight: {args.pos_weight:.4f}")
    logger.info(
        f"Model: d_model={args.d_model} heads={args.n_heads} "
        f"enc={args.enc_layers} dec={args.dec_layers} dropout={args.dropout} | "
        f"{sum(p.numel() for p in model.parameters()):,} parameters"
    )
    strategy = build_loss_strategy(args, nb_classes, class_names, class_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    # Halve the LR when the selection metric plateaus (as transformer_standard.py)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    logger.info(
        f"Epochs: {args.num_epochs} | Batch: {args.batch_size} | LR: {args.lr} | "
        f"Selection metric: {args.select_metric}"
    )

    # --- Resume ---
    start_epoch = 0
    best_acc = 0.0  # best value of args.select_metric
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_acc = ckpt.get("best_acc", 0.0)
        start_epoch = ckpt.get("epoch", -1) + 1
        logger.info(f"Resumed at epoch {start_epoch} (best_acc={best_acc:.4f}).")

    last_path = os.path.join(args.model_save_dir, "last.pt")
    best_path = os.path.join(args.model_save_dir, "best.pt")
    metric_label = "acc" if args.select_metric == "accuracy" else args.select_metric

    def checkpoint_extra():
        return {
            "select_metric": args.select_metric,
            "tabular_meta": tab_meta,
            "scheduler_state_dict": scheduler.state_dict(),
        }

    # Early stopping state
    es_patience = args.early_stopping  # 0 = disabled
    epochs_no_imp = 0

    # --- Training loop ---
    for epoch in range(start_epoch, args.num_epochs):
        # Training pass
        train_loss = train_one_epoch(
            model,
            train_loader,
            strategy,
            optimizer,
            scaler,
            device,
            epoch,
            args.num_epochs,
            args,
            logger,
        )

        # Validation pass (fast — no per-class detail)
        val_metrics = evaluate(
            model,
            val_loader,
            strategy,
            device,
            args,
            desc=f"Val {epoch + 1}/{args.num_epochs}",
            detailed=False,
        )
        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        val_score = val_metrics[args.select_metric]
        scheduler.step(val_score)

        # One-line epoch summary
        is_best = val_score > best_acc
        best_marker = "  ★ new best" if is_best else ""
        extra_score = (
            ""
            if args.select_metric == "accuracy"
            else f"  {metric_label}: {val_score:.4f}"
        )
        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f}  "
            f"f1: {val_metrics.get('f1_macro', 0):.4f}"
            f"{extra_score}"
            f"{best_marker}"
        )

        # Update the best value before saving last.pt, so a resume restarts from
        # the true best and cannot overwrite best.pt with a worse epoch
        if is_best:
            best_acc = val_score
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1

        # Always save last checkpoint
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scaler,
            epoch,
            best_acc,
            args,
            class_names,
            checkpoint_extra(),
        )

        # New best → save best.pt + detailed validation report
        if is_best:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                epoch,
                best_acc,
                args,
                class_names,
                checkpoint_extra(),
            )
            logger.info(
                f"  Best model saved to {best_path}  (val {metric_label}={best_acc:.4f})"
            )

            # Re-evaluate with detailed=True for the full breakdown
            val_metrics_det = evaluate(
                model,
                val_loader,
                strategy,
                device,
                args,
                desc=f"  Val (detailed) {epoch + 1}",
                detailed=True,
            )
            val_metrics_det["loss"] = val_loss  # carry the already-computed loss
            print_best_report(val_metrics_det, epoch, class_names, logger)

        # Early stopping check
        if es_patience > 0 and epochs_no_imp >= es_patience:
            logger.info(
                f"Early stopping triggered: no improvement for {es_patience} consecutive epochs."
            )
            break

    logger.info(f"\nTraining complete.")
    logger.info(f"  Last checkpoint : {last_path}")
    logger.info(f"  Best model      : {best_path}  (val {metric_label}={best_acc:.4f})")

    # --- Final evaluation on the test set using the best checkpoint ---
    if os.path.isfile(best_path):
        logger.info(f"\nLoading best checkpoint for final test: {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"  (best was epoch {ckpt['epoch'] + 1}, val {metric_label}={ckpt['best_acc']:.4f})"
        )

    test_metrics, test_logits, test_targets = evaluate(
        model,
        test_loader,
        strategy,
        device,
        args,
        desc="Test",
        detailed=True,
        return_outputs=True,
    )
    print_final_report(test_metrics, class_names, logger)

    y_true = test_targets.numpy()
    y_prob = torch.softmax(test_logits, dim=1).numpy()
    report = classification_report(
        y_true,
        y_prob.argmax(axis=1),
        labels=list(range(nb_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    logger.info("Detailed classification report (test):\n" + report)
    if nb_classes == 2 and len(np.unique(y_true)) == 2:
        curves_path = os.path.join(args.model_save_dir, "test_roc_pr_curves.png")
        save_binary_curves(y_true, y_prob[:, 1], curves_path)
        logger.info(f"ROC / PR curves saved to {curves_path}")

    # Persist test metrics as JSON
    metrics_path = os.path.join(args.model_save_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(to_serializable(test_metrics), f, indent=2)
    logger.info(f"Test metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
