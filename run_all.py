"""
run_all.py — one-command pipeline runner
========================================
Runs the full STCC analysis end to end. If no program data is present (or with
--example), it first generates SYNTHETIC example data so the pipeline runs out
of the box.

    python run_all.py            # run on data in data/ (makes example data if none)
    python run_all.py --example  # (re)generate synthetic example data, then run

All outputs are written to results/.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(args):
    print("\n>>>", " ".join(str(a) for a in args))
    subprocess.run([PY, *map(str, args)], cwd=ROOT, check=False)


def load_config():
    sys.path.insert(0, str(ROOT / "src"))
    import config
    return config


if __name__ == "__main__":
    cfg = load_config()
    example = "--example" in sys.argv
    have_data = any((cfg.DATA_DIR / v["gpa_file"]).exists() for v in cfg.PROGRAMS.values())

    if example or not have_data:
        print("Generating synthetic example data (no real data found)."
              if not example else "Regenerating synthetic example data.")
        run(["make_example_data.py"])

    # Step 0: CA metrics from prerequisite graphs (per program)
    for prog in cfg.PROGRAMS:
        pre = cfg.DATA_DIR / "prereqs" / f"{prog}_prereqs.csv"
        if pre.exists():
            run(["src/prereq_network.py", "--input", pre])

    # Steps 1-5: data check, GPA LMM, TTD OLS, validation, STCC score
    for step in ["data_preparation.py", "gpa_lmm.py", "ttd_ols.py",
                 "validation.py", "cc_score.py"]:
        run([f"src/{step}"])

    print("\n" + "=" * 70)
    print("Pipeline complete. See the results/ folder for all outputs.")
    print("=" * 70)
