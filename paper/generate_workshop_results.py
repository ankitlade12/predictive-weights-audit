"""Generate workshop-paper values directly from corrected result artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "corrected_workshop"
REAL_DATA_PATH = RESULTS / "real_data" / "real_data_benchmark_results.json"
INTERVENTION_PATH = RESULTS / "intervention_methods" / "causal_baselines_results.json"
SEMANTICS_PATH = RESULTS / "intervention_semantics" / "intervention_sweep_results.json"
NONLINEARITY_PATH = RESULTS / "nonlinearity_confirmation.json"
TEX_OUTPUT = ROOT / "paper" / "draft" / "generated_workshop_results.tex"
JSON_OUTPUT = RESULTS / "paper_values.json"
FIGURE_OUTPUT = ROOT / "paper" / "draft" / "workshop_evidence_summary.pdf"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return numeric


def _format_decimal(value: float, digits: int = 3) -> str:
    numeric = _finite(value, "value")
    if abs(numeric) < 0.5 * 10**-digits:
        numeric = 0.0
    return f"{numeric:.{digits}f}"


def _format_p(value: float) -> str:
    numeric = _finite(value, "p-value")
    return f"{numeric:.4f}".rstrip("0").rstrip(".")


def _real_metric(payload: dict[str, Any], dataset: str, method: str) -> dict[str, float]:
    metrics = payload["datasets"][dataset]["methods"][method]["metrics"]
    if "auroc_mean" in metrics:
        return {
            "mean": _finite(metrics["auroc_mean"], f"{dataset}/{method}/auroc_mean"),
            "std": _finite(metrics["auroc_std"], f"{dataset}/{method}/auroc_std"),
        }
    return {"mean": _finite(metrics["auroc"], f"{dataset}/{method}/auroc")}


def _render_real_cell(metric: dict[str, float]) -> str:
    mean = _format_decimal(metric["mean"])
    if "std" not in metric:
        return mean
    return f"${mean}\\!\\pm\\!{_format_decimal(metric['std'])}$"


def _intervention_cell(payload: dict[str, Any], config: str, method: str) -> dict[str, float]:
    result = payload["results"][config][method]
    inference = result["primary_inference"]
    interval = inference["bootstrap_ci"]
    return {
        "mean": _finite(result["size_matched_gap_mean"], f"{config}/{method}/mean"),
        "low": _finite(interval["low"], f"{config}/{method}/low"),
        "high": _finite(interval["high"], f"{config}/{method}/high"),
        "holm_p": _finite(inference["holm_adjusted_p"], f"{config}/{method}/holm_p"),
    }


def _render_intervention_cell(metric: dict[str, float]) -> str:
    return (
        f"${_format_decimal(metric['mean'])}\\ "
        f"[{_format_decimal(metric['low'])},{_format_decimal(metric['high'])}]$"
    )


def _nonlinearity_cell(
    payload: dict[str, Any],
    *,
    K: int,
    T: int,
    mix: float,
) -> dict[str, Any]:
    for aggregate in payload["aggregates"]:
        if (
            aggregate["K"] == K
            and aggregate["T"] == T
            and float(aggregate["nonlinearity_mix"]) == mix
        ):
            return aggregate
    raise KeyError(f"Missing nonlinearity aggregate K={K}, T={T}, mix={mix}")


def build_values() -> dict[str, Any]:
    real = _load(REAL_DATA_PATH)
    intervention = _load(INTERVENTION_PATH)
    semantics = _load(SEMANTICS_PATH)
    nonlinearity = _load(NONLINEARITY_PATH)

    methods = ("Lasso", "Ridge", "Lagged-Bottleneck", "PCMCI", "PCMCI+", "Granger")
    datasets = ("climate", "finance", "lorenz96")
    real_values = {
        method: {dataset: _real_metric(real, dataset, method) for dataset in datasets}
        for method in methods
    }

    intervention_methods = ("Bottleneck", "Lasso", "PCMCI", "Granger")
    intervention_values = {
        method: {
            "K10": _intervention_cell(intervention, "K=10_T=300", method),
            "K20": _intervention_cell(intervention, "K=20_T=300", method),
        }
        for method in intervention_methods
    }
    within_method_holm_values = {
        cell["holm_p"]
        for method_values in intervention_values.values()
        for cell in method_values.values()
    }
    if len(within_method_holm_values) != 1:
        raise ValueError(
            "The manuscript's shared within-method Holm p-value claim is no "
            f"longer valid: {sorted(within_method_holm_values)}"
        )

    significant_positive_semantics = [
        _finite(result["gap_int_specific_mean"], "intervention-semantics effect")
        for result in semantics["results"]
        if result["primary_inference"]["reject_holm"]
        and float(result["gap_int_specific_mean"]) > 0
    ]
    constant_effects = [
        _finite(result["gap_int_specific_mean"], "constant-clamp effect")
        for result in semantics["results"]
        if result["int_type"] == "constant"
    ]

    contrast = intervention["method_contrasts"]["Granger-minus-Bottleneck"]
    nonlinear_reference = {
        "K10_T300_mix0": _nonlinearity_cell(nonlinearity, K=10, T=300, mix=0.0),
        "K20_T300_mix0": _nonlinearity_cell(nonlinearity, K=20, T=300, mix=0.0),
        "K10_T600_mix0": _nonlinearity_cell(nonlinearity, K=10, T=600, mix=0.0),
        "K20_T600_mix0": _nonlinearity_cell(nonlinearity, K=20, T=600, mix=0.0),
    }
    t600_aurocs = {
        _finite(value, "T=600 AUROC")
        for aggregate in nonlinearity["aggregates"]
        if aggregate["T"] == 600 and aggregate["K"] in (10, 20)
        for value in aggregate["means"]["flat_lag_auroc"].values()
    }
    if len(t600_aurocs) != 1:
        raise ValueError(
            "The manuscript's shared T=600 AUROC claim is no longer valid: "
            f"{sorted(t600_aurocs)}"
        )

    return {
        "schema_version": "1.0",
        "source_artifacts": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                REAL_DATA_PATH,
                INTERVENTION_PATH,
                SEMANTICS_PATH,
                NONLINEARITY_PATH,
            )
        },
        "real_benchmark": real_values,
        "intervention": intervention_values,
        "within_method_holm_p": within_method_holm_values.pop(),
        "intervention_semantics": {
            "significant_positive_min": min(significant_positive_semantics),
            "significant_positive_max": max(significant_positive_semantics),
            "constant_max": max(constant_effects),
        },
        "granger_minus_bottleneck": {
            "K10": {
                "mean": contrast["K=10_T=300"]["mean_difference"],
                "holm_p": contrast["K=10_T=300"]["inference"]["holm_adjusted_p"],
            },
            "K20": {
                "mean": contrast["K=20_T=300"]["mean_difference"],
                "holm_p": contrast["K=20_T=300"]["inference"]["holm_adjusted_p"],
            },
        },
        "nonlinearity": {
            "status": nonlinearity["regime_outcome"]["status"],
            "K10_T300_delta": nonlinear_reference["K10_T300_mix0"]["paired_effects"][
                "flat_lag_auroc"
            ],
            "K20_T300_delta": nonlinear_reference["K20_T300_mix0"]["paired_effects"][
                "flat_lag_auroc"
            ],
            "K10_T600_bn": nonlinear_reference["K10_T600_mix0"]["means"]["flat_lag_auroc"][
                "Linear-BN"
            ],
            "K10_T600_lasso": nonlinear_reference["K10_T600_mix0"]["means"][
                "flat_lag_auroc"
            ]["Lasso"],
            "K20_T600_bn": nonlinear_reference["K20_T600_mix0"]["means"]["flat_lag_auroc"][
                "Linear-BN"
            ],
            "K20_T600_lasso": nonlinear_reference["K20_T600_mix0"]["means"][
                "flat_lag_auroc"
            ]["Lasso"],
            "T600_shared_auroc": t600_aurocs.pop(),
        },
    }


def render_tex(values: dict[str, Any]) -> str:
    real = values["real_benchmark"]
    intervention = values["intervention"]
    within_method_holm_p = values["within_method_holm_p"]
    semantics = values["intervention_semantics"]
    contrast = values["granger_minus_bottleneck"]
    nonlinearity = values["nonlinearity"]

    real_labels = {
        "Lasso": "Lasso",
        "Ridge": "Ridge",
        "Lagged-Bottleneck": "Lagged-BN (3 seeds)",
        "PCMCI": "PCMCI",
        "PCMCI+": "PCMCI+",
        "Granger": "Bivariate Granger",
    }
    real_rows = [
        f"{real_labels[method]} & "
        + " & ".join(
            _render_real_cell(real[method][dataset])
            for dataset in ("climate", "finance", "lorenz96")
        )
        + r" \\"
        for method in real_labels
    ]
    intervention_rows = [
        f"{label} & {_render_intervention_cell(intervention[method]['K10'])} & "
        f"{_render_intervention_cell(intervention[method]['K20'])} \\\\"
        for method, label in zip(
            ("Bottleneck", "Lasso", "PCMCI", "Granger"),
            ("Lagged-BN", "Lasso", "PCMCI", "Granger"),
            strict=True,
        )
    ]

    lines = [
        "% Generated by paper/generate_workshop_results.py; do not edit by hand.",
        r"\newcommand{\RealBenchmarkTableRows}{%",
        *real_rows,
        "}",
        r"\newcommand{\InterventionTableRows}{%",
        *intervention_rows,
        "}",
        (
            r"\newcommand{\CorrectedPCMCI}{"
            + _format_decimal(real["PCMCI"]["lorenz96"]["mean"])
            + "}"
        ),
        (
            r"\newcommand{\CorrectedLaggedBNMean}{"
            + _format_decimal(real["Lagged-Bottleneck"]["lorenz96"]["mean"])
            + "}"
        ),
        (
            r"\newcommand{\CorrectedLaggedBNStd}{"
            + _format_decimal(real["Lagged-Bottleneck"]["lorenz96"]["std"])
            + "}"
        ),
        (
            r"\newcommand{\CorrectedGranger}{"
            + _format_decimal(real["Granger"]["lorenz96"]["mean"])
            + "}"
        ),
        (
            r"\newcommand{\CorrectedLasso}{"
            + _format_decimal(real["Lasso"]["lorenz96"]["mean"])
            + "}"
        ),
        (
            r"\newcommand{\CorrectedRidge}{"
            + _format_decimal(real["Ridge"]["lorenz96"]["mean"])
            + "}"
        ),
        (
            r"\newcommand{\WithinMethodHolmP}{"
            + _format_p(within_method_holm_p)
            + "}"
        ),
        (
            r"\newcommand{\SemanticsPositiveRange}{"
            + _format_decimal(semantics["significant_positive_min"])
            + "--"
            + _format_decimal(semantics["significant_positive_max"])
            + "}"
        ),
        (
            r"\newcommand{\ConstantClampMaximum}{"
            + _format_decimal(semantics["constant_max"])
            + "}"
        ),
        (
            r"\newcommand{\GrangerBNKTenP}{"
            + _format_p(contrast["K10"]["holm_p"])
            + "}"
        ),
        (
            r"\newcommand{\GrangerBNKTwentyDelta}{"
            + _format_decimal(contrast["K20"]["mean"])
            + "}"
        ),
        (
            r"\newcommand{\GrangerBNKTwentyP}{"
            + _format_p(contrast["K20"]["holm_p"])
            + "}"
        ),
        (
            r"\newcommand{\NonlinearityKTenDelta}{"
            + _format_decimal(nonlinearity["K10_T300_delta"], 2)
            + "}"
        ),
        (
            r"\newcommand{\NonlinearityKTwentyDelta}{"
            + _format_decimal(nonlinearity["K20_T300_delta"], 2)
            + "}"
        ),
        (
            r"\newcommand{\NonlinearityStatus}{\texttt{"
            + str(nonlinearity["status"]).replace("_", r"\_")
            + "}}"
        ),
        (
            r"\newcommand{\NonlinearityTSixHundredAUROC}{"
            + _format_decimal(nonlinearity["T600_shared_auroc"], 2)
            + "}"
        ),
        "",
    ]
    return "\n".join(lines)


def render_figure(values: dict[str, Any], output: Path = FIGURE_OUTPUT) -> None:
    """Render the corrected benchmark and intervention evidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    real = values["real_benchmark"]
    intervention = values["intervention"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
        }
    )

    fig, (ax_real, ax_intervention) = plt.subplots(
        1,
        2,
        figsize=(5.5, 2.6),
        gridspec_kw={"width_ratios": (0.95, 1.25)},
        constrained_layout=True,
    )

    real_methods = (
        "PCMCI",
        "PCMCI+",
        "Lagged-Bottleneck",
        "Lasso",
        "Ridge",
        "Granger",
    )
    real_labels = ("PCMCI", "PCMCI+", "Lagged-BN", "Lasso", "Ridge", "Granger")
    real_values = np.array(
        [real[method]["lorenz96"]["mean"] for method in real_methods]
    )
    real_errors = np.array(
        [real[method]["lorenz96"].get("std", 0.0) for method in real_methods]
    )
    real_colors = [
        "#777777" if method != "Lagged-Bottleneck" else "#0072B2"
        for method in real_methods
    ]
    y_positions = np.arange(len(real_methods))

    for y, value, error, color in zip(
        y_positions, real_values, real_errors, real_colors, strict=True
    ):
        ax_real.errorbar(
            value,
            y,
            xerr=error if error else None,
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.2,
            capsize=2.0,
            markeredgewidth=0.5,
            markeredgecolor="white",
            zorder=3,
        )
    ax_real.set_yticks(y_positions, real_labels)
    ax_real.set_xlim(0.80, 1.005)
    ax_real.set_xticks((0.80, 0.85, 0.90, 0.95, 1.00))
    ax_real.set_xlabel("AUROC (lag horizon 3)")
    ax_real.set_title("(a) Exact Lorenz-96 graph", loc="left", fontweight="bold")
    ax_real.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    ax_real.set_axisbelow(True)
    ax_real.spines["top"].set_visible(False)
    ax_real.spines["right"].set_visible(False)

    methods = ("Bottleneck", "Lasso", "PCMCI", "Granger")
    method_labels = ("Lagged-BN", "Lasso", "PCMCI", "Granger")
    x_positions = np.arange(len(methods))
    conditions = (
        ("K10", r"$K=10$", "#0072B2", "o", -0.10),
        ("K20", r"$K=20$", "#D55E00", "s", 0.10),
    )
    for condition, label, color, marker, offset in conditions:
        means = np.array(
            [intervention[method][condition]["mean"] for method in methods]
        )
        lows = np.array(
            [intervention[method][condition]["low"] for method in methods]
        )
        highs = np.array(
            [intervention[method][condition]["high"] for method in methods]
        )
        asymmetric_errors = np.vstack((means - lows, highs - means))
        ax_intervention.errorbar(
            x_positions + offset,
            means,
            yerr=asymmetric_errors,
            fmt=marker,
            color=color,
            ecolor=color,
            label=label,
            markersize=4.0,
            capsize=2.0,
            markeredgewidth=0.5,
            markeredgecolor="white",
            zorder=3,
        )

    ax_intervention.axhline(0.0, color="#444444", linewidth=0.7)
    ax_intervention.set_xticks(x_positions, method_labels)
    ax_intervention.set_ylim(-0.005, 0.115)
    ax_intervention.set_yticks((0.00, 0.03, 0.06, 0.09))
    ax_intervention.set_ylabel(r"AUROC gain $\Delta_m$")
    ax_intervention.set_title(
        "(b) Matched random forcing",
        loc="left",
        fontweight="bold",
    )
    ax_intervention.legend(
        loc="upper left",
        frameon=False,
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    ax_intervention.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax_intervention.set_axisbelow(True)
    ax_intervention.spines["top"].set_visible(False)
    ax_intervention.spines["right"].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", metadata={"CreationDate": None})
    plt.close(fig)


def main() -> None:
    values = build_values()
    TEX_OUTPUT.write_text(render_tex(values))
    JSON_OUTPUT.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    render_figure(values)
    print(f"Wrote {TEX_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {JSON_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {FIGURE_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
