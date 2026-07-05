"""
Prerequisite Network Analysis: Delay and Blocking Factors
=========================================================
Computes the Curricular Architecture (CA) metrics from Heileman et al. (2018)
using prerequisite graphs for each program.

This script takes a simple prerequisite list and computes:
  - Delay factor: longest prerequisite chain through each course
  - Blocking factor: number of downstream courses depending on each course
  - Course complexity: delay + blocking (Heileman et al., 2018, Eq. 1)
  - Program complexity (α_c): sum of all course complexities (Eq. 2)

Input format
------------
A CSV file per program with two columns:
  course, prerequisite

  e.g.:
    PHYS201, MATH101
    PHYS201, PHYS101
    THERMO, PHYS201
    THERMO, MATH201

Courses with no prerequisites should have an empty prerequisite column.

Usage:
    python src/prereq_network.py --input data/prereqs/program_1_prereqs.csv
"""

import argparse
import pandas as pd
import networkx as nx


def build_graph(prereq_df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from a prerequisite list."""
    G = nx.DiGraph()
    for _, row in prereq_df.iterrows():
        course = str(row["course"]).strip()
        prereq = str(row.get("prerequisite", "")).strip()
        G.add_node(course)
        if prereq and prereq.lower() not in ("", "nan", "none"):
            G.add_node(prereq)
            G.add_edge(prereq, course)  # prereq → course
    return G


def compute_delay(G: nx.DiGraph) -> dict:
    """Delay factor: length of the longest path ending at each node."""
    delay = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            delay[node] = 1
        else:
            delay[node] = max(delay[p] for p in preds) + 1
    return delay


def compute_blocking(G: nx.DiGraph) -> dict:
    """Blocking factor: number of descendants (direct and indirect)."""
    return {n: len(nx.descendants(G, n)) for n in G.nodes()}


def analyze_program(prereq_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all CA metrics for a program."""
    G = build_graph(prereq_df)

    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        raise ValueError(f"Prerequisite graph has cycles: {cycles[:3]}")

    delay = compute_delay(G)
    blocking = compute_blocking(G)

    results = []
    for course in G.nodes():
        d = delay[course]
        b = blocking[course]
        results.append({
            "course": course,
            "delay": d,
            "blocking": b,
            "complexity": d + b,   # Eq. 1: v_n = d_n + b_n
        })

    df = pd.DataFrame(results).sort_values("complexity", ascending=False)

    # Program-level complexity (Eq. 2): α_c = Σ v_n
    alpha_c = df["complexity"].sum()
    print(f"  Courses: {len(df)}")
    print(f"  Program complexity (α_c): {alpha_c}")
    print(f"  Longest chain (max delay): {df['delay'].max()}")
    print(f"  Highest blocking: {df.loc[df['blocking'].idxmax(), 'course']} "
          f"(blocks {df['blocking'].max()} courses)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CA metrics from prerequisite lists")
    parser.add_argument("--input", required=True, help="Path to prerequisite CSV")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    prereqs = pd.read_csv(args.input)
    results = analyze_program(prereqs)

    out = args.output or args.input.replace(".csv", "_ca_metrics.csv")
    results.to_csv(out, index=False)
    print(f"\n  Saved to {out}")
