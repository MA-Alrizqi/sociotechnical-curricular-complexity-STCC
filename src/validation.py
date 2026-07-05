"""
Step 4: Model Validation
========================
Cross-validation, holdout testing, and permutation tests for STCC models.

Methods (following the paper):
  - 5-fold grouped cross-validation (GroupKFold by student_id)
  - 80/20 student-level holdout split
  - Permutation test (100 shuffles) for robustness

Usage:
    python src/04_validation.py
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from config import (
    PROGRAMS, OUTPUT_DIR, STUDENT_ID, GPA_OUTCOME,
    CV_N_FOLDS, HOLDOUT_FRACTION, HOLDOUT_SEED, PERMUTATION_ITERS,
)
from data_preparation import prepare_gpa
from gpa_lmm import build_stcc_formula, marginal_r2


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def grouped_cv(df: pd.DataFrame, n_folds: int = CV_N_FOLDS) -> dict:
    """k-fold CV grouped by student so no student appears in both train and test."""
    formula = build_stcc_formula(df)
    groups = df[STUDENT_ID].values
    gkf = GroupKFold(n_splits=n_folds)

    r2_folds = []
    for train_idx, test_idx in gkf.split(df, groups=groups):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        try:
            model = smf.mixedlm(formula, train, groups=train[STUDENT_ID])
            fit = _fit_mixed(model)
            y_pred = fit.predict(exog=test)
            ss_res = np.sum((test[GPA_OUTCOME].values - y_pred.values) ** 2)
            ss_tot = np.sum((test[GPA_OUTCOME].values - test[GPA_OUTCOME].mean()) ** 2)
            r2_folds.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        except Exception:
            r2_folds.append(np.nan)

    return {"cv_r2_mean": np.nanmean(r2_folds), "cv_r2_sd": np.nanstd(r2_folds)}



def _fit_mixed(model):
    """REML fit with optimizer fallback for numerical stability.

    Tries a sequence of optimizers; estimates are identical whenever the
    first converges. This guards against boundary variance components
    (singular Hessian under a single optimizer) on some data splits.
    """
    last_err = None
    for method in ("lbfgs", "bfgs", "powell", "nm"):
        try:
            return model.fit(reml=True, method=method)
        except Exception as err:  # LinAlgError, ConvergenceWarning-as-error
            last_err = err
    raise last_err

# ============================================================================
# HOLDOUT
# ============================================================================

def holdout_test(df: pd.DataFrame) -> dict:
    """80/20 student-level holdout; returns R² and RMSE on held-out students."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=HOLDOUT_FRACTION, random_state=HOLDOUT_SEED,
    )
    train_idx, test_idx = next(splitter.split(df, groups=df[STUDENT_ID]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    formula = build_stcc_formula(train)
    model = smf.mixedlm(formula, train, groups=train[STUDENT_ID])
    fit = _fit_mixed(model)

    y_pred = fit.predict(exog=test)
    y_true = test[GPA_OUTCOME].values
    ss_res = np.sum((y_true - y_pred.values) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((y_true - y_pred.values) ** 2))
    return {"holdout_r2": r2, "holdout_rmse": rmse}


# ============================================================================
# PERMUTATION TEST
# ============================================================================

def permutation_test(df: pd.DataFrame, n_iter: int = PERMUTATION_ITERS) -> dict:
    """Shuffle GPA across students and refit to build null distribution."""
    formula = build_stcc_formula(df)

    # Observed R²
    fit_obs = smf.mixedlm(formula, df, groups=df[STUDENT_ID]).fit(reml=True, method="lbfgs")
    r2_obs = marginal_r2(fit_obs)

    r2_null = []
    for _ in range(n_iter):
        df_perm = df.copy()
        df_perm[GPA_OUTCOME] = np.random.permutation(df_perm[GPA_OUTCOME].values)
        try:
            fit_p = smf.mixedlm(formula, df_perm, groups=df_perm[STUDENT_ID]).fit(
                reml=True, method="lbfgs"
            )
            r2_null.append(marginal_r2(fit_p))
        except Exception:
            pass

    p_value = np.mean([r >= r2_obs for r in r2_null]) if r2_null else np.nan
    return {"observed_r2": r2_obs, "permutation_p": p_value}


# ============================================================================
# ICC
# ============================================================================

def compute_icc(df: pd.DataFrame) -> float:
    """Intraclass correlation from a null (intercept-only) mixed model."""
    null = smf.mixedlm(f"{GPA_OUTCOME} ~ 1", df, groups=df[STUDENT_ID])
    fit = null.fit(reml=True)
    var_student = fit.cov_re.iloc[0, 0]
    var_resid = fit.scale
    return float(var_student / (var_student + var_resid))


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STCC Framework — Step 4: Validation")
    print("=" * 70)

    results = []
    for prog in PROGRAMS:
        print(f"\n--- {prog} ---")
        try:
            df = prepare_gpa(prog)
        except FileNotFoundError:
            print(f"  Data not found; skipping.")
            continue

        try:
            cv = grouped_cv(df)
            ho = holdout_test(df)
            perm = permutation_test(df)
            icc = compute_icc(df)
        except Exception as e:
            print(f"  Validation failed ({type(e).__name__}); skipping program.")
            continue

        print(f"  CV R²   = {cv['cv_r2_mean']:.3f} ± {cv['cv_r2_sd']:.3f}")
        print(f"  Holdout = {ho['holdout_r2']:.3f}  RMSE = {ho['holdout_rmse']:.3f}")
        print(f"  Perm p  = {perm['permutation_p']:.4f}")
        print(f"  ICC     = {icc:.3f}")

        results.append({"program": prog, **cv, **ho, **perm, "icc": icc})

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "validation_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")
