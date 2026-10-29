"""Metrics and figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def score(y_true, proba) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def plot_curves(y_true, probas: dict[str, np.ndarray], path: Path) -> None:
    """ROC, precision-recall and calibration curves for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, p in probas.items():
        if np.ptp(p) == 0:  # constant predictor (prior baseline): curves are degenerate
            continue
        fpr, tpr, _ = roc_curve(y_true, p)
        axes[0].plot(fpr, tpr, label=name)
        prec, rec, _ = precision_recall_curve(y_true, p)
        axes[1].plot(rec, prec, label=name)
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        axes[2].plot(mean_pred, frac_pos, marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="ROC", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(np.mean(y_true), color="k", ls="--", lw=0.8)
    axes[1].set(title="Precision-recall", xlabel="Recall", ylabel="Precision")
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[2].set(title="Calibration", xlabel="Mean predicted probability",
                ylabel="Observed fraction late")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(model, X_test, y_test, path: Path, n_sample: int = 20000,
                    seed: int = 42) -> dict[str, float]:
    """Permutation importance (drop in ROC-AUC) on a test subsample."""
    idx = np.random.default_rng(seed).choice(
        len(X_test), size=min(n_sample, len(X_test)), replace=False
    )
    Xs, ys = X_test.iloc[idx], y_test.iloc[idx]
    result = permutation_importance(
        model, Xs, ys, scoring="roc_auc", n_repeats=3, random_state=seed, n_jobs=-1
    )
    order = np.argsort(result.importances_mean)
    top = order[-12:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(Xs.columns[top], result.importances_mean[top],
            xerr=result.importances_std[top])
    ax.set(title="Permutation importance (HistGradientBoosting)",
           xlabel="Drop in ROC-AUC when feature is shuffled")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {c: float(v) for c, v in zip(Xs.columns, result.importances_mean)}
