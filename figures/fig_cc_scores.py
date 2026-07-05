"""
Figure: CC Score by Program (Forest Plot)
=========================================
Horizontal forest plot showing program-level CC scores with 95% CIs
and median markers.

Reads results/stcc_scores_by_program.csv (produced by src/cc_score.py).

Usage:
    python figures/fig_cc_scores.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({"font.size": 14})

COLORS = {
    "program_1": "#1F77B4",
    "program_2": "#D67228",
    "program_3": "#2CA02C",
    "program_4": "#9467BD",
    "program_5": "#BCBD22",
    "program_6": "#17BECF",
    "program_7": "#D62728",
}
MARKERS = {
    "program_1": "o",
    "program_2": "s",
    "program_3": "^",
    "program_4": "v",
    "program_5": "D",
    "program_6": "H",
    "program_7": "p",
}


def load_data():
    path = os.path.join("results", "stcc_scores_by_program.csv")
    if not os.path.exists(path):
        raise SystemExit("Run src/cc_score.py first to produce "
                         "results/stcc_scores_by_program.csv.")
    return pd.read_csv(path)


if __name__ == "__main__":
    outdir = os.path.join("results", "figures")
    os.makedirs(outdir, exist_ok=True)

    df = load_data().sort_values("mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.3))
    for i, row in df.iterrows():
        p = row["program"]
        ax.errorbar(row["mean"], i,
                     xerr=[[row["mean"]-row["ci_lower"]], [row["ci_upper"]-row["mean"]]],
                     fmt=MARKERS.get(p, "o"), color=COLORS.get(p, "gray"),
                     capsize=6, elinewidth=2, markersize=10, markeredgewidth=1.5)
        ax.plot(row["median"], i, "|", color="#2F4F4F", markersize=18, markeredgewidth=2.5)
        ax.text(row["ci_upper"]+0.005, i, f"{row['mean']:.2f}", va="center", fontsize=13,
                bbox=dict(fc="white", alpha=0.7, ec="none", pad=1.5))

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["program"], fontweight="semibold")
    ax.set_xlabel("Program-level STCC (0-100) with 95% CI")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    ax.errorbar([], [], xerr=1, fmt="o", color="gray", capsize=6, label="Mean (95% CI)")
    ax.plot([], [], "|", color="#2F4F4F", markersize=18, markeredgewidth=2.5, label="Median")
    ax.legend(loc="lower right", fontsize=11)

    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(os.path.join(outdir, f"fig_CC_by_program.{ext}"), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"✓ CC forest plot saved to {outdir}/")
