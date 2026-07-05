"""
Step 1: Data Preparation
========================
Generic functions for loading, cleaning, centering, and structuring
data for the STCC framework analysis. Adapt the loading section to
match your institution's file formats and column naming conventions.

Usage:
    python src/01_data_preparation.py

Required input
--------------
For each program listed in config.PROGRAMS, provide two CSV (or Excel) files:

  1. GPA file (one row per student × course × term):
     student_id | course_id | term | gpa | delay | blocking | dfw_rate |
     units_taken | <LEQ vars> | ethnicity | gender | first_gen | precollege_credits

  2. TTD file (one row per student):
     student_id | time_to_degree | <aggregated CA/LEQ means> | <IL vars>

See data/data_dictionary.md for full variable definitions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    PROGRAMS, DATA_DIR, OUTPUT_DIR,
    COLUMN_RENAMES, STUDENT_ID,
    CA_VARS, LEQ_VARS, IL_CONTINUOUS,
    LEQ_COMPOSITES, ETHNICITY_REFERENCE, DFW_SCALE,
)


# ============================================================================
# CLEANING
# ============================================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip whitespace, and apply user-defined renames."""
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)
    df.rename(columns=COLUMN_RENAMES, inplace=True)
    return df


def collapse_rare_categories(
    df: pd.DataFrame,
    col: str = "ethnicity",
    min_n: int = 5,
    other_label: str = "Other",
) -> pd.DataFrame:
    """Replace ethnicity categories with < min_n students by `other_label`."""
    if col not in df.columns:
        return df
    counts = df[col].value_counts()
    rare = counts[counts < min_n].index.tolist()
    df[col] = df[col].replace(rare, other_label)
    return df


def center_continuous(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Grand-mean center continuous variables, creating new *_c columns."""
    for col in cols:
        if col in df.columns:
            df[f"{col}_c"] = df[col] - df[col].mean()
    # Rescale DFW so 1 unit = 10 percentage-point change
    if "dfw_rate_c" in df.columns:
        df["dfw_rate_scaled_c"] = df["dfw_rate_c"] * DFW_SCALE
    return df


def make_leq_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Average constituent LEQ items into composite scores."""
    for name, items in LEQ_COMPOSITES.items():
        present = [c for c in items if c in df.columns]
        if present:
            df[name] = df[present].mean(axis=1)
    return df


# ============================================================================
# LOADING
# ============================================================================

def load_file(path: Path) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame."""
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def prepare_gpa(program: str) -> pd.DataFrame:
    """Load and prepare the GPA (student × course × term) dataset."""
    path = DATA_DIR / PROGRAMS[program]["gpa_file"]
    df = standardize_columns(load_file(path))
    df = collapse_rare_categories(df)

    # Center continuous predictors
    to_center = [c for c in CA_VARS + LEQ_VARS + IL_CONTINUOUS if c in df.columns]
    df = center_continuous(df, to_center)

    print(f"  [{program}] GPA data: {len(df):,} rows, "
          f"{df[STUDENT_ID].nunique():,} students")
    return df


def prepare_ttd(program: str) -> pd.DataFrame:
    """Load and prepare the collapsed student-level TTD dataset."""
    path = DATA_DIR / PROGRAMS[program]["ttd_file"]
    df = standardize_columns(load_file(path))
    df = collapse_rare_categories(df)
    df = make_leq_composites(df)

    # Create structure-only sum for baseline comparison
    if "delay" in df.columns and "blocking" in df.columns:
        df["delay_blocking_sum"] = df["delay"] + df["blocking"]

    print(f"  [{program}] TTD data: {len(df):,} students")
    return df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STCC Framework — Step 1: Data Preparation")
    print("=" * 70)

    for prog in PROGRAMS:
        print(f"\n--- {prog} ---")
        try:
            prepare_gpa(prog)
        except FileNotFoundError as e:
            print(f"  GPA file not found: {e}")
        try:
            prepare_ttd(prog)
        except FileNotFoundError as e:
            print(f"  TTD file not found: {e}")

    print("\nDone.")
