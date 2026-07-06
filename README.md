# STCC: Socio-Technical Curricular Complexity Framework

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License: MIT](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/status-research%20code-informational)

A reusable, **runs-out-of-the-box** toolkit for quantifying curricular
complexity as a multi-layered socio-technical system, and for computing a single
**STCC composite score (0–100)** per program. Accompanies the paper:

> Mohammed A. Alrizqi & Allison Godwin (2026). Engineering Curricular Analytics as Complex Systems: Introducing, Modeling, and Empirically Testing the Sociotechnical Curricular Complexity (STCC) Framework. *Journal of Engineering Education*. 

## Your STCC score in 3 commands

```bash
pip install -r requirements.txt
python stcc_score.py
# then open results/stcc_report.html in your browser
```

`stcc_score.py` is the **one canonical way to get the final score**. It prepares
the data (or generates a synthetic demo set if you have none), computes the
Delay/Blocking metrics, builds the composite index, and writes three deliverables:

| Output | What it is |
|---|---|
| `results/stcc_report.html` | **Score card per program**: STCC score, 95% CI, and CA / LEQ / IL layer bars showing *where* complexity concentrates |
| `results/stcc_scores_by_program.csv` | The same numbers, machine-readable |
| `results/figures/fig_CC_by_program.png` | Publication-style forest plot |

Example console output (synthetic demo data):

```text
program_3: STCC = 43.2 (95% CI 42.5-44.0), median = 43.0 | CA=44.2 LEQ=61.3 IL=24.2
program_6: STCC = 41.9 (95% CI 41.2-42.7), median = 41.8 | CA=41.4 LEQ=60.4 IL=24.0
program_7: STCC = 41.2 (95% CI 40.4-42.0), median = 41.3 | CA=39.8 LEQ=60.6 IL=23.2
...
```

The layer columns answer the *why*: here `program_3` ranks highest because of
its Curricular Architecture (CA), not its classroom experience (LEQ) or individual level (IL).
For the **full analysis pipeline** (mixed models, TTD regression, validation,
baseline comparisons), run `python run_all.py` instead.

> ## ⚠️ About the bundled data
>
> The data in this repository is **synthetic and idealized** — it is **fabricated by `make_example_data.py`** and exists **only** so the pipeline runs out of the box. It contains **no real students data**, does **not** model real student behavior, and does **not** reproduce the results reported in the paper. Any numbers it produces are for **demonstration of the workflow only**. Real individuals data cannot be shared due to FERPA/IRB restrictions; to obtain genuine results, supply your own institutional data (see *Using Your Own Data* below). 


## Table of Contents

- [What This Framework Does](#what-this-framework-does)
- [Your STCC score in 3 commands](#your-stcc-score-in-3-commands)
- [Quick Start](#quick-start-no-data-needed)
- [How the Example Data Is Generated](#how-the-example-data-is-generated-and-its-limitations)
- [Repository Structure](#repository-structure)
- [Using Your Own Data](#using-your-own-data)
- [Running Steps Individually](#running-steps-individually)
- [The STCC Composite Index](#the-stcc-composite-index)
- [Robustness](#robustness)
- [Data Availability](#data-availability)
- [Citation](#citation)
- [License](#license)

---

## What This Framework Does

Traditional curricular complexity metrics focus on structural features alone (prerequisite chains, bottleneck courses). The STCC framework integrates three layers:

| Layer | What it captures | Example variables |
|-------|-----------------|-------------------|
| **Curricular Architecture (CA)** | The macro-level design of the academic program: how courses are organized, sequenced, and interconnected within the formal curriculum structure | e.g.,  DFW rate and course load |
| **Learning-Experience Quality (LEQ)** | How content is delivered and experienced by students in classroom environments (pedagogy, instructional quality, supports) | e.g., course qulaity and support availability | 
| **Individual Level (IL)** | Characteristics of the individuals navigating the system | e.g., Demographics, and first-gen status |

The toolkit fits linear mixed models (GPA) and OLS (time-to-degree), compares STCC models against structure-only baselines, and computes a program-level **STCC composite index** (0–100 scale).

## Quick Start: full pipeline (no data needed)

The repository ships with **synthetic example data** so you can see and get an idea of the full pipeline run before adapting it to your own institution:

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` generates fabricated example data (if none is present) and runs every step, writing all outputs to `results/`. The example data is **synthesised and fabricated** and does **not** reproduce the paper's numbers; it only demonstrates that the pipeline works.

> ⏱️ The validation step runs 100 permutation refits per program and can take a few minutes. Lower `PERMUTATION_ITERS` in `src/config.py` for a faster check.

## How the Example Data Is Synthesised (and its limitations)

`make_example_data.py` produces deliberately **simple, idealized** data — not a realistic simulation:

- Each synthetic student is given a random latent "ability" value.
- A course grade is a simple function of that ability minus small effects of Delay and DFW, plus random noise.
- Course-evaluation (LEQ) items are random Likert-style values drawn around fixed per-item means.
- Prerequisite graphs are small random acyclic graphs; Delay/Blocking are computed from them.
- Demographics are assigned at random.

Because the relationships are hand-built, the example **cannot** reproduce the paper's findings and should **never** be interpreted as empirical evidence. When you run any step, a banner is printed reminding you that synthetic data is in use.

## Repository Structure

```
stcc-curricular-complexity/
├── stcc_score.py                  # ← THE one-command score: index + HTML report
├── program_profile.py             # Single-program deep dive (courses, network, coefficients)
├── run_all.py                     # Full pipeline runner (models + validation)
├── make_example_data.py           # Generates synthetic example data
├── requirements.txt
├── src/
│   ├── config.py                  # ← Start here: define your programs and variables
│   ├── prereq_network.py          # Compute Delay/Blocking from prerequisite lists
│   ├── data_preparation.py        # Data loading, cleaning, centering
│   ├── gpa_lmm.py                 # GPA linear mixed models (marginal/conditional R², AIC/BIC)
│   ├── ttd_ols.py                 # Time-to-degree OLS regression
│   ├── validation.py              # Cross-validation, holdout, permutation tests
│   ├── cc_score.py                # STCC composite index (0–100; paper Table 4)
│   └── report_html.py             # Self-contained HTML score report
├── figures/                       # Plotting scripts for the paper figures
├── data/
│   ├── data_dictionary.md         # Variable definitions and expected formats
│   └── prereqs/                   # Prerequisite lists (course, prerequisite)
└── results/                       # Auto-created output directory
```

## Single-program deep dive

 In case of only one program — including when you only have **your own department's data and no other programs** — run:


```bash
python program_profile.py --program <program_key>
```

This writes `results/profile_<program>.html`: the program's STCC score and
layer profile, a course-level diagnosis table (delay, blocking, DFW, mean
LEQ) sorted so gateway courses surface first, the prerequisite network, the
program's own fitted model coefficients (after `run_all.py`), an exploratory
entry-cohort trend, and the framework's reflective questions. In
single-program mode the 0–100 score is normalized within your program's own
variation and is not comparable to multi-program or published scores (the
report says so explicitly).

## Using Your Own Data

The quickest way to see the required format is the fill-in template **`data/STCC_data_format_template.xlsx`** — an Excel workbook with one sheet per file, columns colour-coded by layer, and example rows. Fill it in, then save each sheet as a CSV (the pipeline reads CSV).

Follow these steps to run the analysis on your institution's data:

1. **Format your files:** Replace the synthetic files in `data/` with your own CSVs and delete `data/_SYNTHETIC_EXAMPLE_DATA.txt`. Each program needs:
   - **Prerequisite list** — `data/prereqs/<Program>_prereqs.csv` with `course, prerequisite` columns (feeds `prereq_network.py` to compute Delay and Blocking).
   - **GPA file** — `data/<Program>_gpa.csv`, one row per student × course × term, with GPA, structural metrics, LEQ items, and demographics.
   - **TTD file** — `data/<Program>_ttd.csv`, one row per student, with aggregated metrics and time-to-degree.

2. **Register your variables:** See `data/data_dictionary.md` for exact column names, then edit `src/config.py` to:
   - Define your programs (`PROGRAMS`) and map raw column names (`COLUMN_RENAMES`).
   - List the LEQ variables you actually have (`LEQ_VARS`) — the models include only what's present.
   - Set the IL variables you have (`IL_CATEGORICAL`, `IL_CONTINUOUS`).
   - Choose the index scale (`INDEX_SCALE`): 100 for a 0–100 index, 10 for 0–10, 1 for a 0–1 proportion.

3. **Run the pipeline:** Execute `python stcc_score.py` (for the score only) or `python run_all.py` (for the full analysis).

> **Note:** *The variables listed above and in the provided templates are variables used in the paper. Institutions/departments may substitute them with their own available data that aligns with the STCC framework layers as explained in the above "What This Framework Does" section.*

## Running Steps Individually

Run from the **repository root** (so `data/` resolves correctly):

```bash
python src/prereq_network.py --input data/prereqs/program_1_prereqs.csv  # CA metrics
python src/gpa_lmm.py        # GPA models + comparison
python src/ttd_ols.py        # TTD models + comparison
python src/validation.py     # CV, holdout, permutation tests
python src/cc_score.py       # STCC composite index by program
python src/report_html.py    # HTML score report from the index CSV
```

## The STCC Composite Index

`cc_score.py` implements the paper's Table 4 exactly and is **independent of the LMM** — it is a transparent, spreadsheet-replicable index:

1. Min–max normalize each indicator to 0–`INDEX_SCALE` (pooled across all programs).
2. Equal-weight average within each layer (CA, LEQ, IL).
3. Equal-weight average across the three layers.
4. Average across courses per student, then across students per program (mean, median, 95% CI).

Equal weighting follows the OECD (2008) Handbook on Constructing Composite Indicators (Sec. 3.3).

## Robustness

The pipeline is built to tolerate messy real-world data: programs whose model is singular or fails to converge are **skipped with a message** rather than crashing the whole run, and missing LEQ items are handled gracefully (the index and models use whatever variables are available).

## Data Availability

**Real data cannot be shared** due to IRB/FERPA restrictions — standard in curricular analytics research using institutional records. The code itself is a methodological contribution: it documents the full analytical pipeline so researchers can apply the STCC framework to their own data. The bundled example data is synthetic and exists only to make the pipeline runnable.


## Citation

This repository accompanies a research paper. **Cite the paper** for the STCC
framework, methods, and findings. **Cite the software** (via its DOI) if you
use or adapt the code. If you use the code in your own study, please cite both.

**Paper:**
> Alrizqi, M. A., & Godwin, A. (2026). Engineering Curricular Analytics as
> Complex Systems: Introducing, Modeling, and Empirically Testing the
> Sociotechnical Curricular Complexity (STCC) Framework. *Journal of
> Engineering Education*.

**Software:**
> Alrizqi, M. A., & Godwin, A. (2026). *STCC: Sociotechnical Curricular
> Complexity Framework* (Version 1.0.0) [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.XXXXXXX

For formatted citations (APA, BibTeX), see [`CITATION.cff`](CITATION.cff) or
click **"Cite this repository"** in the sidebar.


## License

MIT License. See LICENSE for details.
