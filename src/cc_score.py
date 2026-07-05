"""
Step 5: Program-Level STCC Composite Score  (equal-weight, min-max)
==================================================================
Implements the STCC composite index defined in the manuscript
(Table 4, "Construction of the STCC Composite Score").

This score is DESCRIPTIVE and independent of the LMM. The LMM (gpa_lmm.py)
validates that the three layers explain variance; this index summarizes how
much complexity each program carries on a 0-100 scale and is replicable in a
spreadsheet. Equal weighting follows OECD (2008), Sec. 3.3.

Procedure (equation numbers from Table 4):
  1. (Eq. 1) Min-max normalize each indicator to 0-INDEX_SCALE (default 100), pooled across ALL
             programs so 0 = lowest and 100 = highest observed value.
             Binary 0/1 indicators map to 0/100.
  2. (Eq. 2) Equal-weight average of normalized indicators within each layer.
  3. (Eq. 3) Equal-weight average of the three layer sub-scores.
  4. (Eq. 4) Average across courses per student.
  5. (Eq. 5) Average across students per program (mean, median, 95% CI).

NOTE on variable set: layers are built from config (CA_VARS, LEQ_VARS, IL),
following the layer-to-variable mapping in the manuscript (Table 2), which
includes time_to_degree in the CA layer. Course GPA is intentionally NOT in
the IL layer (it is the GPA outcome; it enters only the TTD model).

Usage:
    python src/cc_score.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from config import (
    PROGRAMS, OUTPUT_DIR, STUDENT_ID,
    CA_VARS, LEQ_VARS, IL_CONTINUOUS, IL_CATEGORICAL,
    ETHNICITY_REFERENCE, INDEX_SCALE,
)
from data_preparation import prepare_gpa

GENDER_REFERENCE = "Female"   # reference level for the gender dummy (is_male)


def build_il_indicators(df: pd.DataFrame):
    """Expand IL categoricals into 0/1 indicators (dropping reference levels)
    and combine with continuous IL variables. Run on the POOLED frame so all
    category levels are coded consistently. Returns (df, il_indicator_cols)."""
    df = df.copy()
    il_cols = [c for c in IL_CONTINUOUS if c in df.columns]
    references = {"ethnicity": ETHNICITY_REFERENCE, "gender": GENDER_REFERENCE}

    for col in IL_CATEGORICAL:
        if col not in df.columns:
            continue
        if df[col].dropna().isin([0, 1]).all():          # already binary (e.g. first_gen)
            il_cols.append(col)
            continue
        dummies = pd.get_dummies(df[col], prefix=col).astype(int)
        ref_col = f"{col}_{references.get(col)}"
        if ref_col in dummies.columns:                    # drop reference level
            dummies = dummies.drop(columns=[ref_col])
        df = pd.concat([df, dummies], axis=1)
        il_cols.extend(dummies.columns.tolist())
    return df, il_cols


def normalize_minmax(s: pd.Series) -> pd.Series:
    """Eq. 1: rescale to 0-100 on the pooled sample. Constant columns -> 0."""
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo) * INDEX_SCALE


def compute_stcc_index(df: pd.DataFrame, layers: dict) -> pd.DataFrame:
    """Eq. 1-3 at the enrollment level. Pass the POOLED (all-program) frame."""
    out = df.copy()
    sub_cols = []
    for layer, cols in layers.items():
        present = [c for c in cols if c in out.columns]
        if not present:
            raise ValueError(f"Layer {layer}: no indicator columns present.")
        normed = pd.DataFrame({c: normalize_minmax(out[c]) for c in present},
                              index=out.index)                 # Eq. 1
        out[f"STCC_{layer}"] = normed.mean(axis=1, skipna=True)  # Eq. 2 (nanmean)
        sub_cols.append(f"STCC_{layer}")
    out["STCC"] = out[sub_cols].mean(axis=1, skipna=True)      # Eq. 3
    return out


def program_summary(student_scores: pd.Series) -> dict:
    """Eq. 5: program-level mean, median, SE, 95% CI from student scores."""
    s = student_scores.dropna()
    n = len(s)
    m = s.mean()
    se = s.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    ci = stats.t.interval(0.95, df=n - 1, loc=m, scale=se) if n > 1 else (np.nan, np.nan)
    return {"n_students": n, "mean": m, "median": s.median(),
            "se": se, "ci_lower": ci[0], "ci_upper": ci[1]}


if __name__ == "__main__":
    print("=" * 70)
    print("STCC Framework — Step 5: Composite Score by Program (Table 4)")
    print("=" * 70)

    # 1) Pool every available program so Eq. 1 normalization is cross-program.
    frames = []
    for prog in PROGRAMS:
        try:
            d = prepare_gpa(prog)
        except FileNotFoundError:
            print(f"  [{prog}] data not found; skipping.")
            continue
        d["program"] = prog
        frames.append(d)
    if not frames:
        raise SystemExit("No program data found in data/.")

    pooled = pd.concat(frames, ignore_index=True)
    pooled, il_cols = build_il_indicators(pooled)

    layers = {"CA": CA_VARS, "LEQ": LEQ_VARS, "IL": il_cols}
    print("\nIndicators per layer:")
    for k, v in layers.items():
        print(f"  {k}: {v}")

    scored = compute_stcc_index(pooled, layers)            # Eq. 1-3
    sub_cols = [f"STCC_{k}" for k in layers]               # STCC_CA, STCC_LEQ, STCC_IL
    student = (scored.groupby(["program", STUDENT_ID])[["STCC", *sub_cols]]
               .mean())                                    # Eq. 4 (per student)

    rows = []
    for prog, grp in student.groupby(level="program"):     # Eq. 5
        summ = program_summary(grp["STCC"]); summ["program"] = prog
        for c in sub_cols:                                 # layer means (program level)
            summ[c] = grp[c].mean()
        rows.append(summ)
        print(f"  {prog}: STCC = {summ['mean']:.1f} "
              f"(95% CI {summ['ci_lower']:.1f}-{summ['ci_upper']:.1f}), "
              f"median = {summ['median']:.1f} | "
              + " ".join(f"{c.replace('STCC_','')}={summ[c]:.1f}" for c in sub_cols))

    out = (pd.DataFrame(rows).sort_values("mean", ascending=False)
           [["program", "n_students", "mean", "median", "se",
             "ci_lower", "ci_upper", *sub_cols]])
    out.to_csv(OUTPUT_DIR / "stcc_scores_by_program.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR}/stcc_scores_by_program.csv")
