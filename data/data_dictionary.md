# Data Dictionary

This document describes the standardized variable names expected by the STCC analysis pipeline. Map your institution's raw column names to these before running the code (use `COLUMN_RENAMES` in `config.py`).

## Contents

- [Example Rows (exact file format)](#example-rows-exact-file-format)
- [Outcome Variables (Dependent Variables)](#outcome-variables-dependent-variables)
- [Identifiers](#identifiers)
- [Curricular Architecture (CA) Layer](#curricular-architecture-ca-layer)
- [Learning-Experience Quality (LEQ) Layer](#learning-experience-quality-leq-layer)
- [Individual-Level (IL) Layer](#individual-level-il-layer)
- [Preprocessing Notes](#preprocessing-notes)

---

## Example Rows (exact file format)

> 💡 A ready-to-fill **Excel template** with these exact columns (colour-coded by layer) and example rows is provided at **`data/STCC_data_format_template.xlsx`**. Fill it in, then save each sheet as a CSV.

Each program needs three CSV files. Below is the exact column order with two illustrative rows each. (Values are illustrative only.)

**1. GPA file — `<program>_gpa.csv`** (one row per student × course × term):

```csv
student_id,course_id,term,gpa,delay,blocking,dfw_rate,units_taken,time_to_degree,leq_01,leq_02,leq_03,leq_04,leq_05,leq_06,leq_07,leq_08,leq_09,leq_10,leq_11,ethnicity,gender,first_gen,precollege_credits
S001,C1,2016-Fall,3.7,2,5,0.08,15,4.1,4,3,4,4,5,4,4,4,4,3,10,White,Female,0,12
S001,C2,2017-Spring,3.3,3,2,0.12,16,4.1,3,4,3,4,4,3,4,4,3,3,12,White,Female,0,12
```

**2. TTD file — `<program>_ttd.csv`** (one row per student; CA/LEQ columns are that student's averages):

```csv
student_id,time_to_degree,delay,blocking,dfw_rate,units_taken,leq_01,leq_02,leq_03,leq_04,leq_05,leq_06,leq_07,leq_08,leq_09,leq_10,leq_11,ethnicity,gender,first_gen,precollege_credits
S001,4.1,2.5,3.5,0.10,15.5,3.6,3.5,3.6,4.0,4.2,3.7,3.9,4.0,3.7,3.1,11.0,White,Female,0,12
S002,4.5,3.1,2.8,0.14,15.0,3.4,3.7,3.5,3.8,4.0,3.6,3.8,3.9,3.6,3.0,12.5,Asian,Male,1,6
```

**3. Prerequisite file — `prereqs/<program>_prereqs.csv`** (`course, prerequisite`; one row per edge; leave `prerequisite` blank for a course with none):

```csv
course,prerequisite
C2,C1
C3,C1
C4,C2
C1,
```


## Outcome Variables (Dependent Variables)

`gpa` and `time_to_degree` are the **dependent variables (DVs)** — the outcomes the models explain. They must be present in your data because this is **historical data on students who already graduated**: the models learn how the CA/LEQ/IL columns account for these *known, already-observed* outcomes (this is variance explained, not forecasting of unknown values).

| Variable | Description | Type | Role |
|----------|-------------|------|------|
| `gpa` | Course-level grade point average | Continuous | DV of the GPA mixed model |
| `time_to_degree` | Time from matriculation to degree (years) | Continuous | DV of the TTD model; also a CA predictor in the GPA model |

## Identifiers

| Variable | Description |
|----------|-------------|
| `student_id` | Unique student identifier (used for random intercepts and grouping) |
| `course_id` | Course identifier |
| `term` | Academic term (e.g., "2017-Fall") |

## Curricular Architecture (CA) Layer

These structural features come from prerequisite network analysis and student transcripts.

| Variable | Description | How to compute |
|----------|-------------|----------------|
| `delay` | Length of the longest prerequisite chain through this course | Network analysis of prerequisite graph (see Heileman et al., 2018) |
| `blocking` | Number of downstream courses that depend on this course | Network analysis of prerequisite graph |
| `dfw_rate` | Proportion of D/F/Withdrawal grades in the course-term | From transcript data: count(D,F,W) / total enrollments |
| `units_taken` | Total credit units the student enrolled in during the term | From student transcripts |
| `time_to_degree` | (Also used as CA predictor in some models) | From student records |


## Column roles (read this before mapping your data)

| Role | Columns | What it means |
|------|---------|---------------|
| **Fixed** | `student_id`, `course_id`, `term`, `gpa`, `time_to_degree`; Prereqs sheet (`course`, `prerequisite`) | The pipeline's contract -- keep these names exactly. |
| **Computed** | `delay`, `blocking` | Produced by `src/prereq_network.py` from the Prereqs sheet; you do not supply them. |
| **Replaceable** | `dfw_rate`, `units_taken`, all `LEQ_*`, `ethnicity`, `gender`, `first_gen`, `precollege_credits` | Substitute your institution's equivalents. Add or drop columns freely -- then register your column names in the matching lists in `src/config.py` (`CA_VARS`, `LEQ_VARS`, `IL_CATEGORICAL`, `IL_CONTINUOUS`). |

`src/config.py` is the single place where column names are registered; the models and the STCC index are built from those lists, so the pipeline is variable-agnostic.

## Learning-Experience Quality (LEQ) Layer

These capture how content is delivered and experienced. In the original study, 11 items from student course evaluations were used (see Table A1 in the Supplementary Materials). **Replace these with whatever instructional quality data your institution has available** — the framework is flexible.

| Variable | Original Item | Example survey question |
|----------|--------------|------------------------|
| `leq_01` | Item 1 | How valuable were the assigned readings? |
| `leq_02` | Item 2 | How valuable were homework assignments? |
| `leq_03` | Item 3 | Did the instructor stimulate your interest? |
| `leq_04` | Item 4 | Was the lecture organized and clear? |
| `leq_05` | Item 5 | Was the instructor willing and able to help? |
| `leq_06` | Item 6 | Was the recitation organized and clear? |
| `leq_07` | Item 7 | Was the recitation instructor available to help? |
| `leq_08` | Item 8 | How would you rate the instructor's command of material? |
| `leq_09` | Item 9 | Overall quality of recitations? |
| `leq_10` | Item 10 | How does this course compare to others? |
| `leq_11` | Item 11 | Weekly hours spent outside class? |

**Note**: Not all institutions will have all of these items. Include whichever LEQ variables you have. The code dynamically builds models from available variables.

## Individual-Level (IL) Layer

| Variable | Description | Type | Notes |
|----------|-------------|------|-------|
| `ethnicity` | Race/ethnicity category | Categorical | Reference: White (adjustable in config) |
| `gender` | Gender | Categorical | As available in institutional records |
| `first_gen` | First-generation college student | Binary (0/1) | |
| `precollege_credits` | Credits earned before matriculation | Continuous | AP, IB, dual enrollment, etc. |

## Preprocessing Notes

- Continuous predictors are **grand-mean centered** before modeling (suffix `_c` added automatically)
- DFW rate is additionally scaled × 10 so that a 1-unit change = 10 percentage-point increase
- Categorical variables are dummy-coded with the specified reference category
- Rare ethnicity categories (< 5 students) are collapsed into "Other"
