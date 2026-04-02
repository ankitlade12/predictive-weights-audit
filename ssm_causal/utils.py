"""
Visualization and evaluation utilities for SSM causal discovery.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)

# Use non-interactive backend for saving figures without display
matplotlib.use("Agg")


def compute_metrics(
    S: np.ndarray,
    G_true: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a causal score matrix.

    Args:
        S: Predicted causal score matrix (K, K), continuous values.
        G_true: Ground truth binary adjacency matrix (K, K).
        threshold: If provided, also compute binary metrics at this threshold.
                   If None, finds the best-F1 threshold.

    Returns:
        Dictionary with keys: auroc, auprc, f1, precision, recall,
        best_threshold, spearman_corr.
    """
    K = S.shape[0]
    mask = ~np.eye(K, dtype=bool)

    # Flatten off-diagonal entries
    scores = np.abs(S[mask]).flatten()
    labels = G_true[mask].flatten()

    results = {}

    # AUROC
    if len(np.unique(labels)) > 1:
        results["auroc"] = roc_auc_score(labels, scores)
    else:
        results["auroc"] = float("nan")

    # AUPRC
    if len(np.unique(labels)) > 1:
        prec_curve, rec_curve, _ = precision_recall_curve(labels, scores)
        results["auprc"] = auc(rec_curve, prec_curve)
    else:
        results["auprc"] = float("nan")

    # Find best threshold for F1
    from .threshold import best_f1_threshold
    _, best_thresh, best_f1 = best_f1_threshold(S, G_true)

    results["best_threshold"] = best_thresh
    results["best_f1"] = best_f1

    # Binary metrics at the chosen threshold
    t = threshold if threshold is not None else best_thresh
    pred_binary = (np.abs(S) > t).astype(float)
    np.fill_diagonal(pred_binary, 0)

    pred_flat = pred_binary[mask].flatten()
    true_flat = labels

    results["precision"] = precision_score(true_flat, pred_flat, zero_division=0)
    results["recall"] = recall_score(true_flat, pred_flat, zero_division=0)
    results["f1"] = f1_score(true_flat, pred_flat, zero_division=0)

    # False Discovery Rate = FP / (FP + TP)
    tp = np.sum((pred_flat == 1) & (true_flat == 1))
    fp = np.sum((pred_flat == 1) & (true_flat == 0))
    results["fdr"] = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    # Spearman correlation between scores and true causal strengths
    from scipy.stats import spearmanr
    corr, p_val = spearmanr(scores, labels)
    results["spearman_corr"] = corr
    results["spearman_pval"] = p_val

    return results


def plot_causal_matrix(
    S: np.ndarray,
    title: str = "Causal Score Matrix",
    var_names: Optional[List[str]] = None,
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 5),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Visualize a causal score matrix as a heatmap.

    Args:
        S: Score matrix (K, K).
        title: Figure title.
        var_names: Names for each variable.
        cmap: Colormap (default: RdBu_r, colorblind-friendly).
        save_path: If provided, save figure to this path.
        figsize: Figure size.
        vmin, vmax: Color scale limits.

    Returns:
        matplotlib Figure object.
    """
    K = S.shape[0]
    if var_names is None:
        var_names = [f"X{i}" for i in range(K)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(S, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(var_names, rotation=45, ha="right")
    ax.set_yticklabels(var_names)

    ax.set_xlabel("Source variable (j)")
    ax.set_ylabel("Target variable (i)")
    ax.set_title(title)

    # Annotate cells with values
    for i in range(K):
        for j in range(K):
            text_color = "white" if abs(S[i, j]) > 0.5 * (S.max() - S.min()) else "black"
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                    color=text_color, fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_comparison(
    matrices: Dict[str, np.ndarray],
    G_true: Optional[np.ndarray] = None,
    var_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize_per_plot: Tuple[int, int] = (5, 4),
) -> plt.Figure:
    """
    Side-by-side comparison of multiple causal score matrices.

    Args:
        matrices: Dict mapping method names to score matrices.
        G_true: If provided, shown as the first plot.
        var_names: Variable names.
        save_path: Save path.
        figsize_per_plot: Size for each subplot.
    """
    n_plots = len(matrices) + (1 if G_true is not None else 0)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(figsize_per_plot[0] * n_plots, figsize_per_plot[1])
    )

    if n_plots == 1:
        axes = [axes]

    K = list(matrices.values())[0].shape[0]
    if var_names is None:
        var_names = [f"X{i}" for i in range(K)]

    idx = 0

    if G_true is not None:
        ax = axes[idx]
        im = ax.imshow(G_true, cmap="RdBu_r", aspect="equal", vmin=0, vmax=1)
        ax.set_title("Ground Truth", fontweight="bold")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(var_names, rotation=45, ha="right")
        ax.set_yticklabels(var_names)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        idx += 1

    for name, S in matrices.items():
        ax = axes[idx]
        im = ax.imshow(np.abs(S), cmap="RdBu_r", aspect="equal", vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(var_names, rotation=45, ha="right")
        ax.set_yticklabels(var_names)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add AUROC if ground truth available
        if G_true is not None:
            metrics = compute_metrics(S, G_true)
            ax.set_xlabel(f"AUROC: {metrics['auroc']:.3f}")

        idx += 1

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curves(
    scores_dict: Dict[str, np.ndarray],
    G_true: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot ROC curves for multiple methods.

    Args:
        scores_dict: Dict mapping method names to score matrices (K, K).
        G_true: Ground truth binary adjacency (K, K).
        save_path: Save path.
    """
    from sklearn.metrics import roc_curve

    K = G_true.shape[0]
    mask = ~np.eye(K, dtype=bool)
    true_flat = G_true[mask].flatten()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, len(scores_dict)))

    for (name, S), color in zip(scores_dict.items(), colors):
        pred_flat = np.abs(S[mask]).flatten()

        if len(np.unique(true_flat)) > 1:
            fpr, tpr, _ = roc_curve(true_flat, pred_flat)
            auroc = roc_auc_score(true_flat, pred_flat)
            ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Causal Discovery Methods")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def print_metrics_table(
    results: Dict[str, Dict[str, float]],
    dataset_name: str = "",
) -> str:
    """
    Format metrics as a clean table for terminal output.

    Args:
        results: Dict mapping method names to metric dicts.
        dataset_name: Name of the dataset.

    Returns:
        Formatted table string.
    """
    header = f"\n{'=' * 80}\n"
    if dataset_name:
        header += f"Dataset: {dataset_name}\n"
    header += f"{'Method':<20} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}\n"
    header += f"{'-' * 80}\n"

    rows = []
    for method, metrics in results.items():
        row = (
            f"{method:<20} "
            f"{metrics.get('auroc', float('nan')):>8.3f} "
            f"{metrics.get('auprc', float('nan')):>8.3f} "
            f"{metrics.get('best_f1', float('nan')):>8.3f} "
            f"{metrics.get('precision', float('nan')):>8.3f} "
            f"{metrics.get('recall', float('nan')):>8.3f}"
        )
        rows.append(row)

    table = header + "\n".join(rows) + f"\n{'=' * 80}\n"
    print(table)
    return table
