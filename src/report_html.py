"""
STCC HTML Report (institutional-brief style)
============================================
Generates a single, self-contained HTML report (results/stcc_report.html):
a dot-and-whisker chart of program STCC scores with 95% CIs, followed by a
three-column layer decomposition (CA / LEQ / IL) with value labels.
Pure standard library + pandas -- no new dependencies, no server.

Reads:  results/stcc_scores_by_program.csv  (produced by src/cc_score.py)
Writes: results/stcc_report.html

All user-facing wording lives in the WORDING block below -- edit there,
then re-run `python src/report_html.py` (or `python stcc_score.py`).
"""

from datetime import date
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR, DATA_DIR

# ---------------------------------------------------------------------------
# WORDING -- edit the report's text here
# ---------------------------------------------------------------------------
WORDING = {
    "eyebrow": "Institutional research brief",
    "title": "Socio-technical curricular complexity",
    "subtitle": "Program-level scores, 0\u2013100",
    "demo_note": ("Demonstration data \u2014 synthetic example; "
                  "no real students; not citable as results."),
    "how_to_read": ("How to read: the STCC score averages three layer "
                    "sub-scores with equal weight. Higher means more "
                    "complexity carried by the program \u2014 context-"
                    "dependent, not inherently good or bad. All scores, "
                    "including the three layer sub-scores, are on a "
                    "0\u2013100 scale; the layer profile shows where that "
                    "complexity concentrates."),
    "fig_heading": "STCC by program (0\u2013100), with 95% CI",
    "decomp_heading": "Layer decomposition (0\u2013100)",
    "footnote": ("CA = curricular architecture \u00b7 LEQ = learning-"
                 "experience quality \u00b7 IL = individual-level. Exact "
                 "values with CIs in stcc_scores_by_program.csv."),
}

# Okabe-Ito colorblind-safe palette (matches the manuscript figures)
LAYER = {
    "CA":  "#0072B2",
    "LEQ": "#009E73",
    "IL":  "#CC79A7",
}
NAVY = "#10314f"

FONT = "Calibri,'Segoe UI',Arial,Helvetica,sans-serif"

CSS = f"""
* {{ box-sizing: border-box; }}
body {{ max-width: 860px; margin: 0 auto; padding: 2.6rem 2.2rem 3rem;
       background:#ffffff; font-family:{FONT}; color:#1f2d3d; line-height:1.55; }}
.eyebrow {{ margin:0; font-size:.78rem; letter-spacing:.14em; text-transform:uppercase;
           color:#1668b0; font-weight:700; }}
h1 {{ margin:.3rem 0 .15rem; font-size:1.7rem; font-weight:700; color:{NAVY}; }}
.meta {{ margin:0; font-size:.9rem; color:#6b7c8f; }}
.rule {{ border-top:3px solid {NAVY}; margin:.9rem 0 0; }}
.demo {{ margin:.9rem 0 0; font-size:.86rem; letter-spacing:.05em; color:#b23a2a;
        font-weight:700; text-transform:uppercase; }}
.how {{ margin:1.1rem 0 0; font-size:.95rem; color:#33424f; }}
h2 {{ margin:1.7rem 0 .6rem; font-size:1rem; font-weight:700; color:{NAVY}; }}
.chartrow {{ display:flex; align-items:center; gap:12px; margin:.5rem 0; font-size:.86rem;
            color:#33424f; }}
.chartrow .lab {{ width:96px; }}
.track {{ flex:1; height:16px; background:#eef3f9; position:relative; }}
.fillbar {{ position:absolute; left:0; top:0; bottom:0; background:{NAVY}; }}
.whisker {{ position:absolute; top:2px; height:2px; background:#5b7a99; }}
.dot {{ position:absolute; top:4px; width:8px; height:8px; border-radius:50%;
       background:#b8860b; }}
.val {{ width:44px; text-align:right; font-weight:700; color:{NAVY};
       font-variant-numeric:tabular-nums; }}
.decomp {{ display:grid; grid-template-columns:96px repeat(3, 1fr); gap:8px 18px;
          align-items:center; font-size:.82rem; color:#33424f; }}
.decomp .hd {{ font-weight:700; }}
.cell {{ display:flex; align-items:center; gap:6px; }}
.cell .bar {{ flex:1; height:12px; background:#eef3f9; position:relative; }}
.cell .fill {{ position:absolute; left:0; top:0; bottom:0; }}
.cell .num {{ width:26px; text-align:right; font-variant-numeric:tabular-nums; }}
.foot {{ margin:1.2rem 0 0; font-size:.82rem; color:#6b7c8f; }}
"""


def _chart_row(row: pd.Series) -> str:
    mean = float(row["mean"])
    lo = float(row.get("ci_lower", mean))
    hi = float(row.get("ci_upper", mean))
    return (
        f'<div class="chartrow"><span class="lab">{row["program"]}</span>'
        f'<span class="track">'
        f'<span class="fillbar" style="width:{mean:.1f}%"></span>'
        f'<span class="whisker" style="left:{lo:.1f}%;width:{max(hi - lo, 0.6):.1f}%"></span>'
        f'<span class="dot" style="left:calc({mean:.1f}% - 4px)"></span>'
        f'</span><span class="val">{mean:.1f}</span></div>'
    )


def _decomp_cell(value, color: str) -> str:
    if pd.isna(value):
        return '<span class="cell">&ndash;</span>'
    v = max(0.0, min(100.0, float(value)))
    return (f'<span class="cell"><span class="bar">'
            f'<span class="fill" style="width:{v:.1f}%;background:{color}"></span>'
            f'</span><span class="num">{v:.0f}</span></span>')


def build_report(csv_path=None, out_path=None):
    csv_path = csv_path or OUTPUT_DIR / "stcc_scores_by_program.csv"
    out_path = out_path or OUTPUT_DIR / "stcc_report.html"
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} not found. Run `python src/cc_score.py` (or "
            "`python stcc_score.py`) first."
        )

    df = pd.read_csv(csv_path).sort_values("mean", ascending=False).reset_index(drop=True)
    single_note = ""
    if len(df) == 1:
        single_note = ('<p class="demo" style="color:#8a5a00">Single-program '
                       'mode: the 0&ndash;100 score is normalized within this '
                       'program\'s own variation and is not comparable to '
                       'multi-program or published STCC scores. For the full '
                       'diagnosis, run program_profile.py.</p>')

    chart = "".join(_chart_row(r) for _, r in df.iterrows())

    head_cells = ('<span></span>' +
                  "".join(f'<span class="hd" style="color:{c}">{k}</span>'
                          for k, c in LAYER.items()))
    body_cells = ""
    for _, r in df.iterrows():
        body_cells += f'<span>{r["program"]}</span>'
        for k, c in LAYER.items():
            body_cells += _decomp_cell(r.get(f"STCC_{k}"), c)

    demo_line = ""
    if (DATA_DIR / "_SYNTHETIC_EXAMPLE_DATA.txt").exists():
        demo_line = f'<p class="demo">{WORDING["demo_note"]}</p>'

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STCC report</title>
<style>{CSS}</style></head><body>
<p class="eyebrow">{WORDING["eyebrow"]}</p>
<h1>{WORDING["title"]}</h1>
<p class="meta">{WORDING["subtitle"]} &middot; generated {date.today().isoformat()}</p>
<div class="rule"></div>
{demo_line}
{single_note}
<p class="how">{WORDING["how_to_read"]}</p>
<h2>{WORDING["fig_heading"]}</h2>
{chart}
<h2>{WORDING["decomp_heading"]}</h2>
<div class="decomp">{head_cells}{body_cells}</div>
<p class="foot">{WORDING["footnote"]}</p>
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"HTML report written to {out_path}")
    return out_path


if __name__ == "__main__":
    build_report()
