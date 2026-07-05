"""
Step 3: Time-to-Degree Model — OLS Regression
===============================================
Fits structure-only baseline and STCC models for TTD (one row per student).
Uses BIC-guided selection from candidate predictor sets.

Usage:
    python src/03_ttd_ols.py
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from config import (
    PROGRAMS, DATA_DIR, OUTPUT_DIR, TTD_OUTCOME,
    CA_VARS, LEQ_VARS, LEQ_COMPOSITES, IL_CATEGORICAL, IL_CONTINUOUS,
    ETHNICITY_REFERENCE, STCC_LABEL, BASELINE_LABEL,
)
from data_preparation import prepare_ttd


def fit_baseline_ttd(df: pd.DataFrame) -> object:
    """Structure-only baseline: Delay + Blocking sum."""
    formula = f"{TTD_OUTCOME} ~ delay_blocking_sum"
    return smf.ols(formula, df).fit()


def build_stcc_formula_ttd(df: pd.DataFrame) -> str:
    """Build STCC formula from available variables (composites preferred)."""
    terms = []

    # CA aggregates
    for v in ["delay", "blocking", "dfw_rate", "units_taken"]:
        # TTD files may have aggregated versions (mean_, max_, etc.)
        for prefix in ["mean_", "max_", ""]:
            col = f"{prefix}{v}"
            if col in df.columns:
                terms.append(col)
                break

    # LEQ composites (preferred) or individual items
    composite_names = list(LEQ_COMPOSITES.keys())
    for c in composite_names:
        if c in df.columns:
            terms.append(c)

    # IL
    for v in IL_CATEGORICAL:
        if v in df.columns:
            ref = ETHNICITY_REFERENCE if v == "ethnicity" else None
            terms.append(f'C({v}, Treatment(reference="{ref}"))' if ref else f"C({v})")
    for v in IL_CONTINUOUS:
        if v in df.columns:
            terms.append(v)

    rhs = " + ".join(terms) if terms else "1"
    return f"{TTD_OUTCOME} ~ {rhs}"


def fit_stcc_ttd(df: pd.DataFrame) -> object:
    """Fit the STCC model for TTD."""
    formula = build_stcc_formula_ttd(df)
    return smf.ols(formula, df).fit()


if __name__ == "__main__":
    print("=" * 70)
    print("STCC Framework — Step 3: TTD OLS")
    print("=" * 70)

    rows = []
    for prog in PROGRAMS:
        print(f"\n--- {prog} ---")
        try:
            df = prepare_ttd(prog)
        except FileNotFoundError:
            print(f"  Data not found; skipping.")
            continue

        try:
            base = fit_baseline_ttd(df)
            stcc = fit_stcc_ttd(df)
        except Exception as e:
            print(f"  Model fit failed ({type(e).__name__}); skipping program.")
            continue

        print(f"  {BASELINE_LABEL:30s}  R²={base.rsquared:.3f}  AIC={base.aic:.1f}")
        print(f"  {STCC_LABEL:30s}  R²={stcc.rsquared:.3f}  AIC={stcc.aic:.1f}")

        rows.append({
            "program": prog,
            "r2_baseline": base.rsquared, "r2_stcc": stcc.rsquared,
            "aic_baseline": base.aic, "aic_stcc": stcc.aic,
            "bic_baseline": base.bic, "bic_stcc": stcc.bic,
        })

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "ttd_model_comparison.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")
