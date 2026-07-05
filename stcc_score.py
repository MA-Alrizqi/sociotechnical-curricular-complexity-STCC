"""
stcc_score.py — ONE command to get your STCC scores
===================================================
This is the shortest path from data to the final STCC composite index:

    python stcc_score.py

It will, in order:
  1. Generate synthetic example data if no data is present (so it always runs).
  2. Compute Delay/Blocking from the prerequisite files (CA metrics).
  3. Compute the STCC composite score per program (Table 4 of the manuscript).
  4. Write results/stcc_scores_by_program.csv,
           results/stcc_report.html          <- open this in your browser,
           results/figures/fig_CC_by_program.png  (forest plot).

For the full analysis pipeline (mixed models, TTD regression, validation),
run `python run_all.py` instead.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(args, what: str):
    print(f"\n>>> {what}")
    print("   ", " ".join(str(a) for a in args))
    result = subprocess.run([PY, *map(str, args)], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(
            f"\nERROR: step failed -> {what}\n"
            "Fix the error above and re-run `python stcc_score.py`.\n"
            "Most common cause: your data files do not match the template "
            "(see data/STCC_data_format_template.xlsx and data/data_dictionary.md)."
        )


def main():
    sys.path.insert(0, str(ROOT / "src"))
    import config

    print("=" * 70)
    print("STCC — Socio-Technical Curricular Complexity: composite score")
    print("=" * 70)

    # 1) Data present? If not, create the synthetic example set.
    have_data = any(
        (config.DATA_DIR / v["gpa_file"]).exists() for v in config.PROGRAMS.values()
    )
    if not have_data:
        run(["make_example_data.py"],
            "No data found -> generating SYNTHETIC example data (demo only)")

    # 2) CA metrics (Delay/Blocking) from prerequisite graphs.
    for prog in config.PROGRAMS:
        pre = config.DATA_DIR / "prereqs" / f"{prog}_prereqs.csv"
        if pre.exists():
            run(["src/prereq_network.py", "--input", pre],
                f"CA metrics (Delay/Blocking) for {prog}")

    # 3) The composite score itself.
    run(["src/cc_score.py"], "STCC composite score (Table 4)")

    # 4) Deliverables: HTML report (always) + forest plot (best effort).
    run(["src/report_html.py"], "HTML report")
    fig = subprocess.run([PY, "figures/fig_cc_scores.py"], cwd=ROOT)
    if fig.returncode != 0:
        print("  (forest-plot figure skipped — see error above; "
              "the CSV and HTML report are unaffected)")

    print("\n" + "=" * 70)
    print("DONE. Your results:")
    print(f"  Scores table : {config.OUTPUT_DIR / 'stcc_scores_by_program.csv'}")
    print(f"  Report       : {config.OUTPUT_DIR / 'stcc_report.html'}  <- open in browser")
    first = next(iter(config.PROGRAMS))
    print(f"  Deep dive    : python program_profile.py --program {first}")
    print("=" * 70)


if __name__ == "__main__":
    main()
