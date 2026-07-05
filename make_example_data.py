"""
make_example_data.py
====================
Generate SYNTHETIC, IDEALIZED example data so the STCC pipeline runs out of
the box. The data is deliberately simplified and stylized -- NOT a realistic
simulation of student behavior.

*** THIS DATA IS FABRICATED. It contains NO real students and is for
    demonstration only. It is NOT the data used in the paper and will NOT
    reproduce the paper's numbers. ***

It writes, for every program in config.PROGRAMS:
  data/<prog>_gpa.csv          one row per student x course x term
  data/<prog>_ttd.csv          one row per student
  data/prereqs/<prog>_prereqs.csv   course, prerequisite

Run from the repository root:
    python make_example_data.py
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

import sys
sys.path.insert(0, "src")
from config import PROGRAMS, LEQ_VARS, DATA_DIR

RNG = np.random.default_rng(42)
N_STUDENTS = 110                     # per program
TERMS = ["2015-Fall", "2016-Spring", "2016-Fall", "2017-Spring"]
ETHNICITIES = ["White", "Asian", "Black", "Hispanic", "International", "Other"]
ETH_P =        [0.42,   0.20,    0.12,    0.12,       0.10,            0.04]


def prereq_edges(seed: int, dens: int = 3) -> list:
    """Build a small acyclic prerequisite graph; density (dens = max prereqs
    per course) varies by program so programs differ in structural complexity."""
    r = np.random.default_rng(seed)
    courses = [f"C{i}" for i in range(1, 9)]            # C1..C8
    edges = []
    for j, c in enumerate(courses):
        # each course may require 0..dens earlier courses (keeps graph acyclic)
        n_pre = r.integers(0, dens + 1) if j > 0 else 0
        if n_pre:
            for p in r.choice(courses[:j], size=min(n_pre, j), replace=False):
                edges.append((p, c))
    return courses, edges


def delay_blocking(courses, edges) -> pd.DataFrame:
    """Compute Delay/Blocking per course (same definitions as prereq_network.py)."""
    G = nx.DiGraph()
    G.add_nodes_from(courses)
    G.add_edges_from(edges)
    delay = {}
    for n in nx.topological_sort(G):
        preds = list(G.predecessors(n))
        delay[n] = 1 if not preds else max(delay[p] for p in preds) + 1
    blocking = {n: len(nx.descendants(G, n)) for n in G.nodes()}
    return pd.DataFrame({"course_id": courses,
                         "delay": [delay[c] for c in courses],
                         "blocking": [blocking[c] for c in courses]})


def likert(mean, n):
    """Draw Likert-style 1-5 values."""
    return np.clip(np.round(RNG.normal(mean, 0.6, n), 1), 1, 5)


# Per-program layer profiles: CA (structural) and LEQ (experience) are
# deliberately ANTI-CORRELATED so different programs are led by different
# layers -- this mirrors the framework's central claim and makes the demo
# report visibly vary. (edge-density cap, DFW base, LEQ climate shift, prep cap)
# (dens, DFW, LEQ shift, precollege cap, first-gen, male, White, units_base).
#  Designed so each LAYER leads in some programs: P1,P5 CA-led | P2,P4 LEQ-led
#  | P3,P6 IL-led | P7 balanced. LEQ is forced low in the CA/IL programs and
#  course load (units_base) gives CA a real ceiling.
PROFILES = [
    (4, 0.34, -1.6, 12, 0.12, 0.55, 0.72, 21),  # program_1: CA-led
    (2, 0.05, +0.9, 12, 0.12, 0.50, 0.65, 13),  # program_2: LEQ-led
    (2, 0.06, -1.6, 65, 0.90, 0.90, 0.15, 12),  # program_3: IL-led
    (2, 0.05, +0.9, 16, 0.30, 0.50, 0.55, 13),  # program_4: LEQ-led
    (4, 0.32, -1.6, 10, 0.10, 0.52, 0.72, 20),  # program_5: CA-led
    (2, 0.06, -1.6, 60, 0.85, 0.88, 0.18, 12),  # program_6: IL-led
    (3, 0.18, -0.4, 30, 0.45, 0.60, 0.45, 16),  # program_7: balanced
]


def make_program(prog: str, idx: int) -> tuple:
    dens, dfw_base, leq_delta, pre_max, fg_p, male_p, white_p, units_base = PROFILES[idx % len(PROFILES)]
    courses, edges = prereq_edges(seed=100 + idx, dens=dens)
    cb = delay_blocking(courses, edges)
    base_dfw = dfw_base
    # per-program LEQ climate and demographic/prep mix so all three layers vary
    leq_shift = leq_delta
    # White share from the profile; remaining mass split across the other
    # categories in proportion to the base mix
    rest = np.array(ETH_P[1:]); rest = rest / rest.sum()
    eth_p = np.concatenate([[white_p], (1 - white_p) * rest])
    first_gen_p = fg_p
    gpa_rows, ttd_rows = [], []

    for s in range(N_STUDENTS):
        sid = f"{prog}_S{s:03d}"
        eth = RNG.choice(ETHNICITIES, p=eth_p)
        gender = RNG.choice(["Male", "Female"], p=[male_p, 1 - male_p])
        first_gen = int(RNG.random() < first_gen_p)
        precollege = int(RNG.integers(0, pre_max + 1))  # per-program prep mix
        ability = RNG.normal(0, 0.35)                   # latent student ability

        taken = RNG.choice(courses, size=RNG.integers(5, 8), replace=False)
        s_gpas, s_delay, s_block, s_dfw, s_units = [], [], [], [], []
        leq_base = {v: b + leq_shift
                    for v, b in zip(LEQ_VARS, np.linspace(3.0, 4.3, len(LEQ_VARS)))}
        leq_accum = {v: [] for v in LEQ_VARS}
        for c in taken:
            term = RNG.choice(TERMS)
            d = int(cb.loc[cb.course_id == c, "delay"].iloc[0])
            b = int(cb.loc[cb.course_id == c, "blocking"].iloc[0])
            dfw = float(np.clip(RNG.normal(base_dfw + 0.01 * d, 0.03), 0, 0.6))
            units = int(np.clip(RNG.normal(units_base, 2), 9, 24))
            gpa = float(np.clip(3.1 + ability - 0.05 * d - 1.5 * dfw + RNG.normal(0, 0.3), 0, 4))
            row = dict(student_id=sid, course_id=c, term=term, gpa=round(gpa, 2),
                       delay=d, blocking=b, dfw_rate=round(dfw, 3), units_taken=units,
                       time_to_degree=None,  # filled below
                       ethnicity=eth, gender=gender, first_gen=first_gen,
                       precollege_credits=precollege)
            for v in LEQ_VARS:
                # leq_11 is the study-hours-like item, tied to course delay
                m = (3.0 + 0.2 * d) if v == "leq_11" else (leq_base[v] + 0.15 * ability)
                val = float(likert(m, 1)[0])
                row[v] = val
                leq_accum[v].append(val)
            gpa_rows.append(row)
            s_gpas.append(gpa); s_delay.append(d); s_block.append(b)
            s_dfw.append(dfw); s_units.append(units)

        ttd = float(np.clip(3.9 + 0.04 * np.mean(s_delay) - 0.1 * ability + RNG.normal(0, 0.25), 3.2, 5.5))
        for r in gpa_rows:
            if r["student_id"] == sid:
                r["time_to_degree"] = round(ttd, 2)

        ttd_row = dict(student_id=sid, time_to_degree=round(ttd, 2),
                       delay=round(np.mean(s_delay), 2), blocking=round(np.mean(s_block), 2),
                       dfw_rate=round(np.mean(s_dfw), 3), units_taken=round(np.mean(s_units), 2),
                       ethnicity=eth, gender=gender, first_gen=first_gen,
                       precollege_credits=precollege)
        for v in LEQ_VARS:
            ttd_row[v] = round(float(np.mean(leq_accum[v])), 2)
        ttd_rows.append(ttd_row)

    prereq_df = pd.DataFrame(
        [(c, p) for (p, c) in edges] +
        [(c, "") for c in courses if c not in {ch for (_, ch) in edges}],
        columns=["course", "prerequisite"],
    )
    return pd.DataFrame(gpa_rows), pd.DataFrame(ttd_rows), prereq_df


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "prereqs").mkdir(parents=True, exist_ok=True)
    print("Generating SYNTHETIC example data (fabricated; not real students)...")
    for i, prog in enumerate(PROGRAMS):
        gpa_df, ttd_df, prereq_df = make_program(prog, i)
        gpa_df.to_csv(DATA_DIR / PROGRAMS[prog]["gpa_file"], index=False)
        ttd_df.to_csv(DATA_DIR / PROGRAMS[prog]["ttd_file"], index=False)
        prereq_df.to_csv(DATA_DIR / "prereqs" / f"{prog}_prereqs.csv", index=False)
        print(f"  {prog}: {len(gpa_df)} enrollment rows, {len(ttd_df)} students")
    with open(DATA_DIR / "_SYNTHETIC_EXAMPLE_DATA.txt", "w") as f:
        f.write("The *_gpa.csv, *_ttd.csv and prereqs/*.csv files here are FABRICATED,\n"
                "IDEALIZED synthetic example data generated by make_example_data.py.\n"
                "synthetic example data generated by make_example_data.py.\n"
                "They contain NO real students and do NOT reproduce the paper's results.\n"
                "Replace them with your own institutional data to run a real analysis.\n")
    print("Done. Files written to", DATA_DIR)
