# Dirty, Shameful, Not My Own? Traumatized Sexuality as a Pathway From Childhood Sexual Abuse to Attitudes Toward Women’s Genitals

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Official statistical analysis and figure-reproduction repository for:

> **Sperber, S., & Gewirtz-Meydan, A.** (2026). *Dirty, shameful, not my own? Traumatized sexuality as a pathway from childhood sexual abuse to attitudes toward women’s genitals.* School of Social Work, University of Haifa.

---

## Overview & Key Contributions

Although childhood sexual abuse (CSA) is often perpetrated directly on and through the genitals, little is known about how women survivors evaluate the female genitals in adulthood—or which psychological mechanisms link early abuse to adult genital evaluations. Grounded in the traumagenic dynamics model (Finkelhor & Browne, 1985) and objectification theory (Fredrickson & Roberts, 1997), this repository implements the full statistical pipeline examining $N = 346$ Israeli women CSA survivors (drawn from $N = 675$ eligible women):

1. **Missing-Data Audit & Little’s (1988) MCAR Test**: Compares completers ($n = 222$) and non-completers ($n = 124$, including $121$ [35.0%] missing the ATWGS) and implements Little’s (1988) multivariate MCAR test ($\chi^2(3) = 9.00, p = .029$).
2. **Full Information Maximum Likelihood (FIML) via Expectation–Maximization (EM)**: Maximizes the observed-data multivariate normal log-likelihood across all $N = 346$ survivors so participants with partial data contribute all observed information, paired with 5,000-resample percentile bootstrap confidence intervals.
3. **Simple Mediation Suite (Table 2 & Supplementary Table S1)**: Estimates `CSA Severity (CTQ-SA) -> Traumatized Sexuality (PT-SEX) -> Genital Attitudes (ATWGS)` controlling for age across FIML ($N = 346$; $\beta_{\text{ind}} = -.19$), Multiple Imputation by Chained Equations (MICE; $M = 30$), complete-case OLS with HC3 robust standard errors ($n = 222$), strict 100% item-complete cases ($n = 199$), Cook’s $D$ outlier-screened cases ($n = 205$), Huber M-estimator and Tukey biweight robust regressions, inverse-probability of attrition weighting (IPW), single-indicator latent SEM correcting for measurement error ($\beta_{\text{ind}} = -.23$), covariate checks (controlling for age at abuse onset; omitting age), and alternative sample definitions ($N = 402, 527, 675$).
4. **Parallel Multiple Mediation & Shapley Dominance Analysis (Table 3 & Supplementary Table S2)**:Simultaneously tests all six PT-SEX dimensions (`Shame/Guilt`, `Pleasing the Other`, `Dissociation`, `Hypervigilance`, `Interpersonal Distress`, `Intrusiveness`), exact Shapley value dominance across all $2^6 = 64$ subset models, 10-fold cross-validated Ridge and Lasso regressions, and leave-one-item-out (LOIO) sensitivity checks.
5. **Developmental Timing Mediation**: Tests whether traumatized sexuality mediates the link between age at abuse onset and genital attitudes controlling for CSA severity and current age ($\beta_{\text{ind}} = .10$).
6. **Publication Figures (Figures 1–3)**: Generates `Figure_1_Simple_Mediation.png`, `Figure_2_Parallel_Mediation.png`, and `Figure_3_Scatter_PTSEX_ATWGS.png`.

---

## Repository Structure

```text
csa-genital-attitudes-reproducibility/
├── .gitignore                        # Excludes private raw data and build artifacts
├── .zenodo.json                      # Zenodo metadata for automatic DOI archiving
├── CITATION.cff                      # Citation File Format metadata for GitHub & Zenodo
├── LICENSE                           # MIT License
├── pyproject.toml                    # Package definition and CLI entry point
├── requirements.txt                  # Python dependencies
├── README.md                         # Repository documentation
├── data/
│   └── README_DATA.md                # Codebook, item mapping, and IRB data access policy
├── src/
│   └── csa_genital_attitudes/
│       ├── __init__.py               # Public package API
│       ├── config.py                 # CTQ-SA, PT-SEX (6 subscales), and ATWGS item definitions
│       ├── scoring.py                # >=80% and 100% scale scoring and reliability functions
│       ├── fiml.py                   # EM algorithm for FIML, Little's MCAR test, & path solver
│       ├── models.py                 # All mediation, robustness, dominance, and LOIO models
│       ├── figures.py                # High-resolution Figure 1, 2, and 3 generators
│       ├── synthetic.py              # Privacy-preserving synthetic dataset generator
│       └── cli.py                    # Command-line runner
├── scripts/
│   └── run_all_analyses.py           # One-command reproduction script
└── tests/
    └── test_pipeline.py              # Automated pytest verification suite
```

---

## Installation & Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<YOUR_GITHUB_USERNAME>/csa-genital-attitudes-reproducibility.git
cd csa-genital-attitudes-reproducibility

# 2. Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .
```

### Running the Full Analysis Pipeline

- **With the study dataset** (pass `--data-path` to the de-identified CSV):
  ```bash
  python scripts/run_all_analyses.py --data-path path/to/analysis_data.csv --boot-fiml 5000 --boot-cc 5000
  ```
- **With the built-in privacy-preserving synthetic dataset** (runs anywhere without private data):
  ```bash
  python scripts/run_all_analyses.py --synthetic --boot-fiml 1000 --boot-cc 1000
  ```
- **Run the automated test suite**:
  ```bash
  pytest
  ```

All numeric outputs are saved to `outputs/reproduction_results.json` and publication figures to `outputs/figures/`.

---

## Data and Code Availability Statement (Ready to Paste into the Manuscript)

> **Data and Code Availability:** To protect participant privacy in accordance with Institutional Review Board approval, data is available from the corresponding author upon reasonable academic request. Complete Python code for data preprocessing, Full Information Maximum Likelihood (FIML) expectation–maximization path modeling, Little’s (1988) MCAR test, Multiple Imputation, outlier-robust regressions, Shapley value dominance analysis, figure generation, and a synthetic dataset for code verification is openly archived on GitHub (`https://github.com/shulisrael/Dirty-Shameful-Not-My-Own`) and Zenodo (`https://doi.org/10.5281/zenodo.24081175`).
