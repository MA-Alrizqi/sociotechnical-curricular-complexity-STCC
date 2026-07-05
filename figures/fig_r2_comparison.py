"""
Figure: R² Model Comparison (Structure-only vs. STCC)
=====================================================
Generates bar charts comparing the structure-only baseline and STCC-based
models for LMM Marginal R², LMM Conditional R², and TTD R².

Reads results from gpa_model_comparison.csv and ttd_model_comparison.csv,
produced by src/gpa_lmm.py and src/ttd_ols.py.

Usage:
    python figures/fig_r2_comparison.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

BASELINE = "Structure-only model"
STCC = "STCC-based models"

STYLE = {
    BASELINE: {"color": "#95A5A6", "hatch": "///", "edge": "#2C3E50"},
    STCC:     {"color": "#27AE60", "hatch": None,  "edge": "#1E8449"},
}


def plot_r2(programs, baseline_vals, stcc_vals, ylabel, outpath):
    """Create one grouped bar chart."""
    # Sort by improvement
    delta = np.array(stcc_vals) - np.array(baseline_vals)
    order = np.argsort(-delta)
    programs = [programs[i] for i in order]
    baseline_vals = [baseline_vals[i] for i in order]
    stcc_vals = [stcc_vals[i] for i in order]

    x = np.arange(len(programs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 7))

    b1 = ax.bar(x - w/2, baseline_vals, w, color=STYLE[BASELINE]["color"],
                edgecolor=STYLE[BASELINE]["edge"], hatch=STYLE[BASELINE]["hatch"],
                linewidth=1.5, alpha=0.85, label=BASELINE, zorder=3)
    b2 = ax.bar(x + w/2, stcc_vals, w, color=STYLE[STCC]["color"],
                edgecolor=STYLE[STCC]["edge"], linewidth=1.5, alpha=0.90,
                label=STCC, zorder=3)

    # Value labels
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=10.5, color="#2C3E50")

    # Mean lines
    for vals, key in [(baseline_vals, BASELINE), (stcc_vals, STCC)]:
        m = np.mean(vals)
        ax.axhline(m, linestyle="--", linewidth=2, color=STYLE[key]["edge"], alpha=0.55)
        ax.text(len(x)-0.15, m+0.015, f"Mean: {m:.2f}", fontsize=10, fontweight="bold",
                color=STYLE[key]["edge"], ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=STYLE[key]["edge"], alpha=0.8))

    ax.set_xlabel("Program", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(programs)
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, title="Model")
    ax.grid(axis="y", linestyle="--", alpha=0.2, zorder=0)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(f"{outpath}.{ext}", dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    outdir = os.path.join("results", "figures")
    os.makedirs(outdir, exist_ok=True)

    gpa_path = os.path.join("results", "gpa_model_comparison.csv")
    ttd_path = os.path.join("results", "ttd_model_comparison.csv")
    if not (os.path.exists(gpa_path) and os.path.exists(ttd_path)):
        raise SystemExit("Run src/gpa_lmm.py and src/ttd_ols.py first to produce "
                         "the model-comparison CSVs.")

    gpa = pd.read_csv(gpa_path)
    ttd = pd.read_csv(ttd_path).set_index("program")
    progs = gpa["program"].tolist()

    datasets = {
        "LMM Marginal R²": {
            "baseline": gpa["marginal_r2_baseline"].tolist(),
            "stcc":     gpa["marginal_r2_stcc"].tolist(),
        },
        "LMM Conditional R²": {
            "baseline": gpa["conditional_r2_baseline"].tolist(),
            "stcc":     gpa["conditional_r2_stcc"].tolist(),
        },
        "TTD R²": {
            "baseline": ttd.reindex(progs)["r2_baseline"].fillna(0.0).tolist(),
            "stcc":     ttd.reindex(progs)["r2_stcc"].fillna(0.0).tolist(),
        },
    }

    for name, vals in datasets.items():
        safe = name.replace(" ", "_").replace("²", "2")
        plot_r2(progs, vals["baseline"], vals["stcc"], "R²",
                os.path.join(outdir, safe))
        print("  done:", name)

    print("Saved to", outdir)
