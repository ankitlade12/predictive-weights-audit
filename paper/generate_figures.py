"""
Generate publication-quality figures from experiment results.

Creates NeurIPS-ready figures at 300 DPI with colorblind-friendly palettes.

Usage:
    python paper/generate_figures.py
"""

import json
import os
import glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

# NeurIPS-friendly style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    "SSM-Causal": "#0072B2",   # Blue
    "Granger": "#009E73",       # Green
    "PCMCI": "#E69F00",         # Orange
    "cLSTM": "#CC79A7",         # Pink
    "Correlation": "#999999",   # Gray
}
MARKERS = {
    "SSM-Causal": "o",
    "Granger": "s",
    "PCMCI": "^",
    "cLSTM": "D",
    "Correlation": "v",
}


def load_json(subdir, filename="results.json"):
    path = os.path.join(RESULTS_DIR, subdir, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def figure_main_comparison():
    """Figure 1: Main AUROC comparison bar chart across all datasets."""
    results = load_json("exp1_synthetic") or load_json("week3_benchmark")
    if not results:
        print("  [SKIP] Figure 1 — no synthetic results")
        return

    datasets = list(results.keys())
    methods = ["SSM-Causal", "Granger", "PCMCI", "cLSTM"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(datasets))
    width = 0.2

    for i, method in enumerate(methods):
        aurocs = []
        for ds in datasets:
            aurocs.append(results[ds].get(method, {}).get("auroc", 0))
        bars = ax.bar(x + i * width, aurocs, width,
                      label=method, color=COLORS.get(method, "#333"),
                      edgecolor="white", linewidth=0.5)

    ax.set_ylabel("AUROC")
    ax.set_title("Causal Discovery Accuracy Across Benchmarks")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([d.replace("-", "\n") for d in datasets],
                       fontsize=8)
    ax.legend(loc="lower left")
    ax.set_ylim([0, 1.15])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_main_comparison.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_main_comparison.png"))
    plt.close(fig)
    print("  Figure 1 -> fig1_main_comparison.pdf")


def figure_scaling():
    """Figure 4: Runtime scaling plots."""
    results = load_json("exp5_efficiency", "efficiency_results.json")
    if not results:
        print("  [SKIP] Figure 4 — no efficiency results")
        return

    methods = ["SSM-Causal", "Granger", "PCMCI", "cLSTM"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # K scaling
    if "scaling_K" in results:
        for method in methods:
            ks, times = [], []
            for label, row in results["scaling_K"].items():
                if row.get(method) is not None:
                    ks.append(int(label.split("=")[1]))
                    times.append(row[method])
            if times:
                ax1.plot(ks, times, f"-{MARKERS.get(method, 'o')}",
                         color=COLORS.get(method, "#333"),
                         label=method, markersize=7, linewidth=2)
        ax1.set_xlabel("Number of Variables (K)")
        ax1.set_ylabel("Wall-clock Time (s)")
        ax1.set_title("Runtime vs. K (T=2000)")
        ax1.legend()
        ax1.set_yscale("log")

    # T scaling
    if "scaling_T" in results:
        for method in methods:
            ts, times = [], []
            for label, row in results["scaling_T"].items():
                if row.get(method) is not None:
                    ts.append(int(label.split("=")[1]))
                    times.append(row[method])
            if times:
                ax2.plot(ts, times, f"-{MARKERS.get(method, 'o')}",
                         color=COLORS.get(method, "#333"),
                         label=method, markersize=7, linewidth=2)
        ax2.set_xlabel("Sequence Length (T)")
        ax2.set_ylabel("Wall-clock Time (s)")
        ax2.set_title("Runtime vs. T (K=5)")
        ax2.legend()
        ax2.set_yscale("log")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_scaling.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_scaling.png"))
    plt.close(fig)
    print("  Figure 4 -> fig4_scaling.pdf")


def figure_ablation_summary():
    """Figure 5: Ablation summary (hyperparameter sensitivity)."""
    results = load_json("ablation", "ablation_results.json")
    if not results:
        print("  [SKIP] Figure 5 — no ablation results")
        return

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    param_configs = [
        ("Model Dim (d)", "model_size"),
        ("State Dim (N)", "state_dim"),
        ("Layers (L)", "n_layers"),
        ("Data Length (T)", "data_length"),
    ]

    for ax, (title, key) in zip(axes, param_configs):
        if key not in results:
            ax.set_visible(False)
            continue
        data = results[key]
        labels = list(data.keys())
        short_labels = [l.split("=")[1] for l in labels]
        jac = [data[l]["auroc_jac"] for l in labels]
        weight = [data[l]["auroc_weight"] for l in labels]

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, jac, w, label="Jacobian", color=COLORS["SSM-Causal"])
        ax.bar(x + w / 2, weight, w, label="Weight", color="#56B4E9")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels)
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("AUROC")
        if ax == axes[0]:
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_ablation.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_ablation.png"))
    plt.close(fig)
    print("  Figure 5 -> fig5_ablation.pdf")


def figure_method_overview():
    """Figure 0: Method overview diagram (simple schematic)."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # Boxes
    boxes = [
        (0.5, 1.5, "X \\in R^{T×K}\n(Time Series)", "#E8F4FD"),
        (2.5, 1.5, "Train Mamba\n(MSE loss)", "#FFF3E0"),
        (4.8, 1.5, "Extract\nW_in, W_out", "#E8F5E9"),
        (7.0, 1.5, "S = |W_out · W_in|\n(Causal Score)", "#FCE4EC"),
        (9.2, 1.5, "Causal\nGraph G", "#F3E5F5"),
    ]

    for x, y, text, color in boxes:
        ax.add_patch(plt.Rectangle((x - 0.7, y - 0.6), 1.4, 1.2,
                                    facecolor=color, edgecolor="#333",
                                    linewidth=1.5, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=8,
                fontweight="bold", zorder=3)

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.7
        x2 = boxes[i + 1][0] - 0.7
        ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                     arrowprops=dict(arrowstyle="->", color="#333", lw=2))

    ax.set_title("SSM-Causal: Method Overview", fontsize=12, fontweight="bold",
                 pad=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig0_method_overview.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig0_method_overview.png"))
    plt.close(fig)
    print("  Figure 0 -> fig0_method_overview.pdf")


def generate_all():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating publication figures...")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Figures dir: {FIGURES_DIR}")
    print()

    figure_method_overview()
    figure_main_comparison()
    figure_scaling()
    figure_ablation_summary()

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    generate_all()
