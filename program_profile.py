"""
program_profile.py -- single-program diagnostic deep dive
=========================================================
Usage:
    python program_profile.py --program program_3

Produces results/profile_<program>.html: a self-contained report with
  1. the program's STCC score, rank, and layer sub-scores (0-100)
  2. a course-level diagnosis table (delay, blocking, DFW, mean LEQ)
  3. the program's prerequisite network (node size = blocking)
  4. its fitted model coefficients, if run_all.py has been run
  5. how its indicators compare to the cross-program average
  6. an EXPLORATORY entry-cohort trend of student-level STCC (Eq. 4)
  7. the reflective questions from the paper

Works in single-program mode: comparison sections collapse gracefully and
the 0-100 score is relabeled (within-program normalization only).
"""

import argparse
import base64
import io
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import config
from config import OUTPUT_DIR, DATA_DIR, STUDENT_ID
from cc_score import compute_stcc_index, build_il_indicators
from data_preparation import prepare_gpa

LAYER_COLORS = {"CA": "#0072B2", "LEQ": "#009E73", "IL": "#CC79A7"}
NAVY = "#10314f"
FONT = "Calibri,'Segoe UI',Arial,Helvetica,sans-serif"

REFLECTIVE = {
    "CA": ["Why do the required courses and their sequence matter in student "
           "development and learning?",
           "Where are key concepts introduced, reinforced, and expected for "
           "students to fully achieve?",
           "Does the current complexity support student success and outcomes, "
           "or does it provide burdensome complexity because of historic norms or traditions?"],
    "LEQ": ["What classroom practices are supporting student success and in which programs?",
            "How can specific exemplar programs be used as positive case studies to "
            "improve practices across engineering?",
            "How do student perceptions shape their engagement with learning?"],
    "IL": ["Where does the system produce differential outcomes, and which curricular structures "
           "and experiences contribute to those patterns?",
           "What differential outcomes might unintentionally emerge from a lack of consideration of "
           "curricula as a sociotechnical system?",
           "What efforts can be leveraged in programs to better support students, particularly those "
           "whom the system's current design underserves?"],
}

# ---------------------------------------------------------------- helpers
def _ensure_scores():
    csv = OUTPUT_DIR / "stcc_scores_by_program.csv"
    if not csv.exists():
        print("Scores not found -- running stcc_score.py first...")
        r = subprocess.run([sys.executable, str(ROOT / "stcc_score.py")])
        if r.returncode != 0 or not csv.exists():
            raise SystemExit("Could not produce STCC scores; fix errors above.")
    return pd.read_csv(csv)


def _pooled_scored():
    """Enrollment-level frame with STCC and layer sub-scores (Eq. 1-3)."""
    frames = []
    for prog in config.PROGRAMS:
        try:
            df = prepare_gpa(prog)
        except Exception:
            continue
        df["program"] = prog
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    pooled, il_cols = build_il_indicators(pooled)
    layers = {"CA": config.CA_VARS, "LEQ": config.LEQ_VARS, "IL": il_cols}
    return compute_stcc_index(pooled, layers)


def _course_table(df: pd.DataFrame) -> pd.DataFrame:
    leq_cols = [c for c in config.LEQ_VARS if c in df.columns]
    g = df.groupby("course_id").agg(
        enrollments=("gpa", "size"),
        delay=("delay", "first"),
        blocking=("blocking", "first"),
        dfw_rate=("dfw_rate", "mean"),
    )
    g["mean_LEQ"] = df.groupby("course_id")[leq_cols].mean().mean(axis=1)
    return g.sort_values(["blocking", "delay"], ascending=False).reset_index()


def _network_png(prog: str) -> str:
    """Base64 PNG of the prerequisite network, node size by blocking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    pre = DATA_DIR / "prereqs" / f"{prog}_prereqs.csv"
    if not pre.exists():
        return ""
    edges = pd.read_csv(pre).dropna(subset=["course"])
    G = nx.DiGraph()
    G.add_nodes_from(edges["course"].astype(str))
    real = edges.dropna(subset=["prerequisite"])
    G.add_edges_from(real[["prerequisite", "course"]].astype(str)
                     .itertuples(index=False))
    delay = {}
    for n in nx.topological_sort(G):
        preds = list(G.predecessors(n))
        delay[n] = 1 if not preds else max(delay[p] for p in preds) + 1
    blocking = {n: len(nx.descendants(G, n)) for n in G.nodes()}
    # layered layout: x = delay, y spreads nodes within a layer
    pos, counts = {}, {}
    for n in sorted(G.nodes()):
        d = delay[n]
        counts[d] = counts.get(d, 0) + 1
        pos[n] = (d, -counts[d])
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    sizes = [300 + 260 * blocking[n] for n in G.nodes()]
    nx.draw_networkx(G, pos, ax=ax, node_size=sizes, node_color="#dbe7f3",
                     edgecolors=NAVY, font_size=8, font_color=NAVY,
                     edge_color="#8aa5bd", arrowsize=12, width=1.2)
    ax.set_title("Prerequisite network (node size = blocking factor)",
                 fontsize=9, color=NAVY)
    ax.axis("off")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _coef_section(prog: str) -> str:
    f = OUTPUT_DIR / f"{prog}_gpa_coefficients.csv"
    if not f.exists():
        return ('<p class="note">Model coefficients not shown -- run '
                '<code>python run_all.py</code> to fit this program\'s GPA '
                'model, then rebuild this profile.</p>')
    d = pd.read_csv(f)
    d = d[(d["variable"] != "Intercept") & (d["pval"] < 0.05)]
    if d.empty:
        return '<p class="note">No coefficients reached p &lt; .05.</p>'

    def layer_of(v):
        s = v.lower()
        if s.startswith("leq"): return "LEQ"
        if any(k in s for k in ("ethnicity", "gender", "first_gen",
                                "precollege", "male")): return "IL"
        return "CA"

    d["layer"] = d["variable"].map(layer_of)
    d = d.sort_values(["layer", "coef"])
    rows = ""
    for _, r in d.iterrows():
        color = LAYER_COLORS[r["layer"]]
        sign = "#0a6c4f" if r["coef"] > 0 else "#a1352a"
        rows += (f'<tr><td><span class="chip" style="background:{color}1a;'
                 f'color:{color}">{r["layer"]}</span></td>'
                 f'<td>{r["variable"]}</td>'
                 f'<td style="text-align:right;color:{sign};font-weight:700">'
                 f'{r["coef"]:+.3f}</td>'
                 f'<td style="text-align:right;color:#6b7c8f">'
                 f'[{r["ci_lower"]:.3f}, {r["ci_upper"]:.3f}]</td></tr>')
    return ('<table class="tbl"><tr><th>Layer</th><th>Variable</th>'
            '<th style="text-align:right">Estimate</th>'
            '<th style="text-align:right">95% CI</th></tr>' + rows +
            '</table><p class="note">Significant (p &lt; .05) fixed effects '
            'from this program\'s GPA mixed model, fitted on the data in '
            '<code>data/</code>. Associations, not causal effects.</p>')


def _cohort_section(scored: pd.DataFrame, prog: str) -> str:
    sub = scored[scored["program"] == prog].copy()
    first_term = (sub.assign(_k=sub["term"].str.split("-").map(
        lambda p: (int(p[0]), 0 if p[1] == "Spring" else 1)))
        .sort_values("_k").groupby(STUDENT_ID)["term"].first())
    student = sub.groupby(STUDENT_ID)["STCC"].mean().to_frame("stcc")
    student["cohort"] = first_term
    g = student.groupby("cohort")["stcc"].agg(["mean", "sem", "size"])
    g = g.sort_index(key=lambda ix: [
        (int(t.split("-")[0]), 0 if t.split("-")[1] == "Spring" else 1)
        for t in ix])
    rows = ""
    for term, r in g.iterrows():
        ci = 1.96 * (r["sem"] if pd.notna(r["sem"]) else 0)
        rows += (f'<div class="crow"><span class="clab">{term}</span>'
                 f'<span class="ctrack"><span class="cfill" '
                 f'style="width:{min(r["mean"],100):.1f}%"></span></span>'
                 f'<span class="cval">{r["mean"]:.1f} &plusmn;{ci:.1f} '
                 f'<span style="color:#8aa5bd">(n={int(r["size"])})</span>'
                 f'</span></div>')
    return rows


# ---------------------------------------------------------------- build
def build_profile(prog: str) -> Path:
    scores = _ensure_scores()
    if prog not in set(scores["program"]):
        raise SystemExit(f"Unknown program '{prog}'. Available: "
                         f"{', '.join(scores['program'])}")
    single = len(scores) == 1
    scores = scores.sort_values("mean", ascending=False).reset_index(drop=True)
    row = scores[scores["program"] == prog].iloc[0]
    rank = int(scores.index[scores["program"] == prog][0]) + 1

    df = prepare_gpa(prog)
    df["program"] = prog
    scored = _pooled_scored()
    courses = _course_table(df)
    net64 = _network_png(prog)

    demo = (DATA_DIR / "_SYNTHETIC_EXAMPLE_DATA.txt").exists()

    # layer bars
    bars = ""
    for k, c in LAYER_COLORS.items():
        v = float(row[f"STCC_{k}"])
        bars += (f'<div class="lrow"><span class="lname" style="color:{c}">'
                 f'{k}</span><span class="lbar"><span class="lfill" '
                 f'style="width:{min(v,100):.1f}%;background:{c}"></span>'
                 f'</span><span class="lval">{v:.0f}</span></div>')

    rank_line = (
        "Single-program mode: this 0&ndash;100 score is normalized within "
        "your own program's variation and is <b>not comparable</b> to "
        "multi-program or published STCC scores."
        if single else
        f"Rank {rank} of {len(scores)} programs by STCC (0&ndash;100, "
        f"normalized across all programs in this run).")

    # course table
    dfw_hi = courses["dfw_rate"].quantile(0.75)
    crows = ""
    for _, r in courses.iterrows():
        flag = (' <span class="flag">high DFW</span>'
                if r["dfw_rate"] >= max(dfw_hi, 0.15) else "")
        crows += (f'<tr><td>{r["course_id"]}{flag}</td>'
                  f'<td style="text-align:right">{int(r["delay"])}</td>'
                  f'<td style="text-align:right">{int(r["blocking"])}</td>'
                  f'<td style="text-align:right">{r["dfw_rate"]:.2f}</td>'
                  f'<td style="text-align:right">{r["mean_LEQ"]:.2f}</td>'
                  f'<td style="text-align:right;color:#6b7c8f">'
                  f'{int(r["enrollments"])}</td></tr>')

    # indicator comparison (skip in single mode)
    comp_html = ""
    if not single:
        inds = (config.CA_VARS + config.LEQ_VARS +
                ["precollege_credits", "first_gen"])
        rows = ""
        for v in inds:
            if v not in scored.columns:
                continue
            mine = scored.loc[scored["program"] == prog, v].mean()
            allm = scored[v].mean()
            d = mine - allm
            col = "#a1352a" if d > 0 else "#0a6c4f"
            rows += (f'<tr><td>{v}</td>'
                     f'<td style="text-align:right">{mine:.2f}</td>'
                     f'<td style="text-align:right;color:#6b7c8f">{allm:.2f}</td>'
                     f'<td style="text-align:right;color:{col}">{d:+.2f}</td></tr>')
        comp_html = (
            '<h2>Indicators vs. cross-program average</h2>'
            '<table class="tbl"><tr><th>Indicator</th>'
            '<th style="text-align:right">This program</th>'
            '<th style="text-align:right">All programs</th>'
            '<th style="text-align:right">&Delta;</th></tr>' + rows +
            '</table><p class="note">Raw (unnormalized) means. A positive '
            '&Delta; means this program sits above the cross-program average '
            'on that indicator.</p>')

    refl = ""
    for k, qs in REFLECTIVE.items():
        items = "".join(f"<li>{q}</li>" for q in qs)
        refl += (f'<p class="rq" style="color:{LAYER_COLORS[k]}">{k}</p>'
                 f'<ul class="rql">{items}</ul>')

    demo_line = ('<p class="demo">Demonstration data &mdash; synthetic '
                 'example; no real students; not citable as results.</p>'
                 if demo else "")
    net_html = (f'<img src="data:image/png;base64,{net64}" '
                f'style="width:100%;max-width:760px">' if net64 else
                '<p class="note">No prerequisite file found for this '
                'program.</p>')

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STCC profile &mdash; {prog}</title>
<style>
* {{ box-sizing:border-box; }}
body {{ max-width:860px; margin:0 auto; padding:2.4rem 2.2rem 3rem;
  background:#fff; font-family:{FONT}; color:#1f2d3d; line-height:1.55; }}
.eyebrow {{ margin:0; font-size:.78rem; letter-spacing:.14em;
  text-transform:uppercase; color:#1668b0; font-weight:700; }}
h1 {{ margin:.3rem 0 .15rem; font-size:1.6rem; color:{NAVY}; }}
.meta {{ margin:0; font-size:.9rem; color:#6b7c8f; }}
.rule {{ border-top:3px solid {NAVY}; margin:.9rem 0 0; }}
.demo {{ margin:.9rem 0 0; font-size:.86rem; letter-spacing:.05em;
  color:#b23a2a; font-weight:700; text-transform:uppercase; }}
h2 {{ margin:1.8rem 0 .6rem; font-size:1rem; color:{NAVY}; }}
.score {{ display:flex; gap:26px; align-items:center; margin-top:1.1rem; }}
.big {{ font-size:3rem; font-weight:700; color:{NAVY};
  font-variant-numeric:tabular-nums; }}
.lrow {{ display:flex; align-items:center; gap:8px; margin:4px 0;
  font-size:.82rem; width:340px; }}
.lname {{ width:34px; font-weight:700; }}
.lbar {{ flex:1; height:11px; background:#eef3f9; position:relative; }}
.lfill {{ position:absolute; left:0; top:0; bottom:0; }}
.lval {{ width:26px; text-align:right; }}
.note {{ font-size:.84rem; color:#6b7c8f; }}
.tbl {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
.tbl th {{ background:{NAVY}; color:#fff; padding:.4rem .55rem;
  text-align:left; font-weight:600; }}
.tbl td {{ padding:.38rem .55rem; border-bottom:1px solid #eef2f7; }}
.tbl tr:nth-child(even) {{ background:#f7fafd; }}
.flag {{ font-size:.72rem; color:#a1352a; background:#fdecea;
  padding:1px 7px; border-radius:9px; }}
.chip {{ font-size:.72rem; padding:1px 8px; border-radius:9px;
  font-weight:700; }}
.crow {{ display:flex; align-items:center; gap:10px; margin:.4rem 0;
  font-size:.84rem; }}
.clab {{ width:92px; }}
.ctrack {{ flex:1; height:13px; background:#eef3f9; position:relative; }}
.cfill {{ position:absolute; left:0; top:0; bottom:0; background:{NAVY}; }}
.cval {{ width:150px; text-align:right;
  font-variant-numeric:tabular-nums; }}
.expl {{ display:inline-block; font-size:.72rem; letter-spacing:.06em;
  color:#8a5a00; background:#fff4dd; padding:1px 8px; border-radius:9px;
  font-weight:700; text-transform:uppercase; vertical-align:2px; }}
.rq {{ margin:.9rem 0 .2rem; font-weight:700; font-size:.85rem; }}
.rql {{ margin:.1rem 0 .4rem 1.2rem; font-size:.88rem; color:#33424f; }}
.foot {{ margin-top:1.6rem; font-size:.8rem; color:#93a3b2;
  border-top:1px solid #eef2f7; padding-top:.7rem; }}
</style></head><body>
<p class="eyebrow">Program diagnostic profile</p>
<h1>{prog}</h1>
<p class="meta">Socio-technical curricular complexity &middot; generated
{date.today().isoformat()}</p>
<div class="rule"></div>
{demo_line}
<div class="score">
  <div><div class="big">{row["mean"]:.1f}</div>
  <div class="note">STCC (0&ndash;100)<br>95% CI {row["ci_lower"]:.1f}&ndash;{row["ci_upper"]:.1f}</div></div>
  <div>{bars}</div>
</div>
<p class="note">{rank_line}</p>

<h2>Course-level diagnosis</h2>
<table class="tbl"><tr><th>Course</th>
<th style="text-align:right">Delay</th>
<th style="text-align:right">Blocking</th>
<th style="text-align:right">DFW rate</th>
<th style="text-align:right">Mean LEQ (1&ndash;5)</th>
<th style="text-align:right">Enrollments</th></tr>{crows}</table>
<p class="note">Sorted by blocking, so gateway courses appear first. High-DFW
flags mark courses at or above this program's 75th percentile (and at least
0.15). Descriptive diagnostics &mdash; not causal effects.</p>

<h2>Prerequisite network</h2>
{net_html}

<h2>Model coefficients (this program)</h2>
{_coef_section(prog)}

{comp_html}

<h2>Entry-cohort trend <span class="expl">exploratory</span></h2>
{_cohort_section(scored, prog)}
<p class="note">Student-level STCC (Eq. 4) averaged by entry cohort, &plusmn;95%
CI. This view is not reported in the accompanying paper. Adjacent cohorts
typically differ by less than their uncertainty; it is most informative
around known curricular changes or across longer spans.</p>

<h2>Reflective questions</h2>
{refl}

<p class="foot">Generated by <code>program_profile.py</code> &middot; STCC
framework. Scores and sub-scores are on a 0&ndash;100 scale.</p>
</body></html>"""

    out = OUTPUT_DIR / f"profile_{prog}.html"
    out.write_text(html, encoding="utf-8")
    print(f"Profile written to {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Single-program STCC deep dive")
    ap.add_argument("--program", required=True,
                    help="Program key as in config.PROGRAMS, e.g. program_3")
    args = ap.parse_args()
    build_profile(args.program)
