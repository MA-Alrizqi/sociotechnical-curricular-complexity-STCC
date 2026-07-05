"""
STCC Framework Configuration
=============================
Socio-Technical Curricular Complexity (STCC) Framework.

This file defines the STCC three-layer variable structure. Adapt the
PROGRAMS dictionary, column names, and LEQ items to your own institutional
data before running the analysis pipeline.

Reference
---------
Mohammed A. Alrizqi & Allison Godwin (2026). Engineering Curricular Analytics as 
Complex Systems: Introducing, Modeling, and Empirically Testing the Socio-Technical 
Curricular Complexity (STCC) Framework. Journal of Engineering Education.
"""

from pathlib import Path

# ============================================================================
# PATHS — Adapt to your directory layout
# ============================================================================

DATA_DIR = Path("data/")
OUTPUT_DIR = Path("results/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PROGRAMS — One entry per program you wish to analyze
# ============================================================================
# Each key is a display name; the value dict points to the data file(s)
# that hold the merged student-by-course-term records (for GPA) and the
# collapsed student-level records (for TTD).

# Replace these placeholder names with your own programs (1, 3, 20 — any number).
PROGRAMS = {
    "program_1": {"gpa_file": "program_1_gpa.csv", "ttd_file": "program_1_ttd.csv"},
    "program_2": {"gpa_file": "program_2_gpa.csv", "ttd_file": "program_2_ttd.csv"},
    "program_3": {"gpa_file": "program_3_gpa.csv", "ttd_file": "program_3_ttd.csv"},
    "program_4": {"gpa_file": "program_4_gpa.csv", "ttd_file": "program_4_ttd.csv"},
    "program_5": {"gpa_file": "program_5_gpa.csv", "ttd_file": "program_5_ttd.csv"},
    "program_6": {"gpa_file": "program_6_gpa.csv", "ttd_file": "program_6_ttd.csv"},
    "program_7": {"gpa_file": "program_7_gpa.csv", "ttd_file": "program_7_ttd.csv"},
}

# ============================================================================
# STANDARD COLUMN NAMES
# ============================================================================
# The code expects these column names in your data files after any renaming.
# Use COLUMN_RENAMES to map your institution's raw column names to these.

STUDENT_ID = "student_id"       # Unique per student (used for random intercepts)
COURSE_ID = "course_id"         # Course identifier
TERM_ID = "term"                # Academic term (e.g., "2017-Fall")

# Rename map: {your raw column name: standard name}
# Add entries here so that your data matches the standard names above.
COLUMN_RENAMES = {
    # "YourRawGPAColumn": "gpa",
    # "YourRawTTDColumn": "time_to_degree",
}

# ============================================================================
# MODEL TERMINOLOGY
# ============================================================================

STCC_LABEL = "STCC-based models"          # Label for the proposed model
BASELINE_LABEL = "Structure-only model"    # Label for the Heileman et al. baseline

# ============================================================================
# THREE-LAYER VARIABLE DEFINITIONS
# ============================================================================
# Adapt these lists to match the variables available at your institution.
# The names below are descriptive; your data files should use these names
# (or map to them via COLUMN_RENAMES).
#
# The STCC framework groups predictors into three layers. Within each layer,
# include whichever variables your institution can provide.  At minimum,
# the CA layer needs Delay and Blocking (from prerequisite network analysis)
# and the IL layer needs at least one demographic or preparation variable.

# --- Curricular Architecture (CA) ---
CA_VARS = [
    "delay",            # Longest prerequisite chain through this course
    "blocking",         # Number of downstream courses depending on this course
    "dfw_rate",         # Proportion D/F/W in the course-term
    "units_taken",      # Credit units the student took that term
    "time_to_degree",   # (Also used as outcome for TTD models)
]

# --- Learning-Experience Quality (LEQ) ---
# Generic slots derived from student course evaluations. Any number of
# columns is fine -- name them however you like, list them here, and the
# pipeline adapts. The comments show the worked example from the original
# study's 11-item instrument (see data/data_dictionary.md for the mapping).
LEQ_VARS = [
    "leq_01",   # e.g., value of assigned readings
    "leq_02",   # e.g., value of homework / assignments
    "leq_03",   # e.g., instructor stimulated interest
    "leq_04",   # e.g., lecture organization and clarity
    "leq_05",   # e.g., instructor willingness/ability to help
    "leq_06",   # e.g., recitation organization and clarity
    "leq_07",   # e.g., recitation instructor availability
    "leq_08",   # e.g., recitation instructor command of material
    "leq_09",   # e.g., overall recitation quality
    "leq_10",   # e.g., course quality vs. other courses
    "leq_11",   # e.g., weekly study hours outside class
]

# Optional composites for TTD models (averages of constituent items).
# Useful to reduce multicollinearity when all items enter one OLS model.
LEQ_COMPOSITES = {
    "instructor_quality":   ["leq_04", "leq_05", "leq_08"],
    "recitation_composite": ["leq_06", "leq_07", "leq_09"],
    "engagement":           ["leq_03", "leq_10"],
}

# --- Individual-Level (IL) ---
IL_CATEGORICAL = [
    "ethnicity",        # Race/ethnicity category
    "gender",           # Gender
    "first_gen",        # First-generation college student (0/1)
]
IL_CONTINUOUS = [
    "precollege_credits",   # Credits earned before matriculation (AP, IB, etc.)
]

# Ethnicity reference category for dummy coding
ETHNICITY_REFERENCE = "White"

# ============================================================================
# MODEL SETTINGS
# ============================================================================

GPA_OUTCOME = "gpa"
TTD_OUTCOME = "time_to_degree"

# STCC composite-index scale (cc_score.py). The manuscript reports the index on
# a 0-100 percentage scale (Table 4 multiplies normalized values by 100). Set to
# 10 for a 0-10 scale or 1 for a 0-1 proportion, to match your paper.
INDEX_SCALE = 100

CV_N_FOLDS = 5                  # k-fold cross-validation
HOLDOUT_FRACTION = 0.20         # Fraction reserved for holdout test
HOLDOUT_SEED = 42
PERMUTATION_ITERS = 100         # Number of permutation-test shuffles

# DFW scaling: express coefficient per 10 percentage-point change
DFW_SCALE = 10


# ============================================================================
# SYNTHETIC-DATA NOTICE
# ============================================================================
# If the bundled synthetic example data is in use, print a clear reminder on
# every run so results are never mistaken for real findings. When you switch to
# your own data, delete the marker file data/_SYNTHETIC_EXAMPLE_DATA.txt and
# this notice disappears automatically.
_SYNTH_MARKER = DATA_DIR / "_SYNTHETIC_EXAMPLE_DATA.txt"
if _SYNTH_MARKER.exists():
    import sys as _sys
    print(
        "\n" + "!" * 72 +
        "\n!!  NOTE: running on SYNTHETIC, IDEALIZED example data (fabricated)." +
        "\n!!  Any numbers are for DEMONSTRATION ONLY -- not real results and not" +
        "\n!!  the paper's findings. Replace data/ with your own institutional data" +
        "\n!!  and delete data/_SYNTHETIC_EXAMPLE_DATA.txt to remove this notice." +
        "\n" + "!" * 72 + "\n",
        file=_sys.stderr,
    )
