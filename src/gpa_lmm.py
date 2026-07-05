"""
Step 2: GPA Model — Linear Mixed Models (LMM)
==============================================
Fits two models per program and compares them:

  (a) Structure-only baseline  — fixed: Delay + Blocking; random: student intercept
  (b) STCC model               — fixed: CA + LEQ + IL;    random: student intercept

Reports marginal R² (fixed effects) and conditional R² (fixed + random).
AIC/BIC are computed from maximum-likelihood (ML) refits, since these criteria
are not defined under REML (the paper refits under ML for AIC/BIC).

Usage:
    python src/gpa_lmm.py
"""

import warnings
import re
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from config import (
    PROGRAMS, DATA_DIR, OUTPUT_DIR, STUDENT_ID,
    CA_VARS, LEQ_VARS, IL_CATEGORICAL, IL_CONTINUOUS,
    ETHNICITY_REFERENCE, GPA_OUTCOME, STCC_LABEL, BASELINE_LABEL,
)
from data_preparation import prepare_gpa

warnings.filterwarnings("ignore")


# ============================================================================
# R² HELPERS  (Nakagawa & Schielzeth, 2013)
# ============================================================================

def marginal_r2(model) -> float:
    """Variance explained by fixed effects only."""
    yhat_fixed = np.asarray(model.model.exog) @ np.asarray(model.fe_params)
    ss_total = np.var(model.model.endog)
    return float(np.var(yhat_fixed) / ss_total) if ss_total > 0 else 0.0


def conditional_r2(model) -> float:
    """Variance explained by fixed effects plus the estimated random-intercept
    variance, over total outcome variance."""
    yhat_fixed = np.asarray(model.model.exog) @ np.asarray(model.fe_params)
    var_fixed = np.var(yhat_fixed)
    var_re = (float(model.cov_re.iloc[0, 0])
              if getattr(model, "cov_re", None) is not None and model.cov_re.size else 0.0)
    ss_total = np.var(model.model.endog)
    return float((var_fixed + var_re) / ss_total) if ss_total > 0 else 0.0


# ============================================================================
# COEFFICIENT LABELS (for figures/fig_coefficients.py)
# ============================================================================

LABEL_MAP = {
    "delay_c": "Delay", "blocking_c": "Blocking Factor",
    "dfw_rate_scaled_c": "DFW Rate", "dfw_rate_c": "DFW Rate",
    "units_taken_c": "Number of Units", "time_to_degree_c": "Time to Degree",
    "leq_01_c": "LEQ 01", "leq_02_c": "LEQ 02",
    "leq_03_c": "LEQ 03", "leq_04_c": "LEQ 04",
    "leq_05_c": "LEQ 05", "leq_06_c": "LEQ 06",
    "leq_07_c": "LEQ 07", "leq_08_c": "LEQ 08",
    "leq_09_c": "LEQ 09", "leq_10_c": "LEQ 10",
    "leq_11_c": "LEQ 11", "precollege_credits_c": "Pre-College Credits",
}


def clean_label(name: str) -> str:
    """Map a raw model parameter name to a readable display label."""
    if name in LABEL_MAP:
        return LABEL_MAP[name]
    m = re.search(r"ethnicity.*\[T\.(.+?)\]", name)
    if m:
        return f"Ethnicity_{m.group(1)}"
    m = re.search(r"gender.*\[T\.(.+?)\]", name)
    if m:
        return m.group(1)
    if "first_gen" in name and "[T." in name:
        return "First-Generation"
    return name


def tidy_coefficients(model, program: str) -> pd.DataFrame:
    """Return fixed-effect coefficients as [variable, coef, ci_lower, ci_upper, pval, program]."""
    ci = model.conf_int()
    params = model.params
    rows = []
    for name in params.index:
        rows.append({
            "variable": clean_label(name),
            "coef": float(params[name]),
            "ci_lower": float(ci.loc[name, 0]),
            "ci_upper": float(ci.loc[name, 1]),
            "pval": float(model.pvalues.get(name, float("nan"))),
            "program": program,
        })
    return pd.DataFrame(rows)


# ============================================================================
# FORMULAS & FITTING
# ============================================================================

def baseline_formula() -> str:
    return f"{GPA_OUTCOME} ~ delay_c + blocking_c"


def build_stcc_formula(df: pd.DataFrame) -> str:
    """Build a formula from whichever CA, LEQ, IL variables are present."""
    def _varies(col):  # skip constant predictors -> avoids singular design
        return col in df.columns and df[col].nunique(dropna=True) > 1

    terms = []
    for v in CA_VARS:                                   # CA (centered)
        c = "dfw_rate_scaled_c" if (v == "dfw_rate" and "dfw_rate_scaled_c" in df.columns) else f"{v}_c"
        if _varies(c):
            terms.append(c)
    for v in LEQ_VARS:                                  # LEQ (centered)
        if _varies(f"{v}_c"):
            terms.append(f"{v}_c")
    for v in IL_CATEGORICAL:                            # IL categorical
        if v in df.columns:
            ref = ETHNICITY_REFERENCE if v == "ethnicity" else None
            terms.append(f'C({v}, Treatment(reference="{ref}"))' if ref else f"C({v})")
    for v in IL_CONTINUOUS:                             # IL continuous (centered)
        if _varies(f"{v}_c"):
            terms.append(f"{v}_c")
    rhs = " + ".join(terms) if terms else "1"
    return f"{GPA_OUTCOME} ~ {rhs}"


def _fit(formula: str, df: pd.DataFrame, reml: bool = True):
    """REML fit with optimizer fallback for numerical stability.

    The fixed-effects design is full rank; a Singular matrix here comes from
    the REML Hessian when a variance component sits near its boundary, which
    can happen under a single optimizer on some datasets. Trying a sequence
    of optimizers returns identical estimates whenever the first converges.
    """
    model = smf.mixedlm(formula, df, groups=df[STUDENT_ID])
    last_err = None
    for method in ("lbfgs", "bfgs", "powell", "nm", "cg"):
        try:
            return model.fit(reml=reml, method=method)
        except Exception as err:
            last_err = err
    raise last_err


def fit_baseline(df: pd.DataFrame, reml: bool = True):
    """Structure-only baseline (Delay + Blocking)."""
    return _fit(baseline_formula(), df, reml)


def fit_stcc(df: pd.DataFrame, reml: bool = True):
    """Full STCC model (CA + LEQ + IL)."""
    return _fit(build_stcc_formula(df), df, reml)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STCC Framework — Step 2: GPA LMM")
    print("=" * 70)

    rows = []
    coef_rows = []
    for prog in PROGRAMS:
        print(f"\n--- {prog} ---")
        try:
            df = prepare_gpa(prog)
        except FileNotFoundError:
            print("  Data not found; skipping.")
            continue
        try:
            base, stcc = fit_baseline(df), fit_stcc(df)             # REML (R²)
            base_ml, stcc_ml = fit_baseline(df, reml=False), fit_stcc(df, reml=False)  # ML (AIC/BIC)
        except Exception as e:
            print(f"  Model fit failed ({type(e).__name__}: {e}); skipping program.")
            continue

        mr2_b, cr2_b = marginal_r2(base), conditional_r2(base)
        mr2_s, cr2_s = marginal_r2(stcc), conditional_r2(stcc)
        print(f"  {BASELINE_LABEL:30s}  Marg R²={mr2_b:.3f}  Cond R²={cr2_b:.3f}  AIC={base_ml.aic:.1f}")
        print(f"  {STCC_LABEL:30s}  Marg R²={mr2_s:.3f}  Cond R²={cr2_s:.3f}  AIC={stcc_ml.aic:.1f}")

        rows.append({
            "program": prog,
            "marginal_r2_baseline": mr2_b, "marginal_r2_stcc": mr2_s,
            "conditional_r2_baseline": cr2_b, "conditional_r2_stcc": cr2_s,
            "aic_baseline": base_ml.aic, "aic_stcc": stcc_ml.aic,
            "bic_baseline": base_ml.bic, "bic_stcc": stcc_ml.bic,
        })
        try:
            tidy = tidy_coefficients(stcc, prog)
            tidy.to_csv(OUTPUT_DIR / f"{prog}_gpa_coefficients.csv", index=False)
            coef_rows.append(tidy)
        except Exception:
            pass

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "gpa_model_comparison.csv", index=False)
    if coef_rows:
        pd.concat(coef_rows, ignore_index=True).to_csv(
            OUTPUT_DIR / "merged_coefficients.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")
