# Data Dictionary, Scoring Rules, and Data Access Statement

## 1. Ethical & Privacy Note
To protect participant confidentiality in accordance with the University of Haifa School of Social Work Institutional Review Board (IRB) approval, raw participant-level survey data (`PRIVATE_*.csv`) are not stored in this public repository. 

- **De-identified Data Requests**: De-identified data are available from the corresponding author upon reasonable academic request and subject to institutional ethical approval.
- **Out-of-the-Box Code Verification**: This package includes a privacy-preserving synthetic data generator (`src/csa_genital_attitudes/synthetic.py`) that matches the exact variable names, item ranges, covariance structure, and monotone attrition pattern of the study sample ($N = 675$ eligible women; $N = 346$ women CSA survivors). Running `python scripts/run_all_analyses.py --synthetic` executes the entire analytical and figure-generation pipeline out of the box.

---

## 2. Variable Codebook & Item Mappings

| Construct / Variable | Column Name(s) | Scale / Coding | Scoring Rule |
|---|---|---|---|
| **Gender** | `gender` | Categorical (`"woman"`) | Eligibility filter ($N = 675$ women out of $N = 1,100$ eligible adults) |
| **Survivor Status (Self-Reported CSA)** | `csa_direct` | Binary (`1` = Yes, `0` = No) | Primary analytic sample filter ($N = 346$ women CSA survivors) |
| **Age** | `age` | Continuous (years, $\ge 18$) | Covariate in all primary models ($M = 41.41, SD = 12.18$, range $18\text{--}73$) |
| **Age at Abuse Onset** | `abuse_onset` (`onset`) | Continuous (years, `< 18`) | Auxiliary variable in FIML and predictor in onset mediation ($M = 8.66, SD = 4.42, n = 322$) |
| **CSA Severity (CTQ-SA)** | `Q18_1`–`Q18_5` (`ctq_80`, `ctq_100`) | 5 items, rated `1` (*never true*) to `5` (*very often true*) | Prorated sum (`mean * 5`, range `5–25`) when $\ge 80\%$ ($\ge 4/5$) items answered ($n = 345$; $\alpha = .81$) |
| **Traumatized Sexuality Total (PT-SEX)** | 25 core items (`ptsex_80`, `ptsex_100`) | Rated `1` (*not at all*) to `5` (*very much*) | Mean of answered items when $\ge 80\%$ ($\ge 20/25$) items answered ($n = 248$; $\alpha = .95$) |
| **PT-SEX: Dissociation** | `PTSEX_19, 34, 18, 20, 22, 17, 46` | 7 items (`1–5`) | Mean when $\ge 80\%$ ($\ge 6/7$) items answered ($\alpha = .94$) |
| **PT-SEX: Intrusiveness** | `PTSEX_1, 2, 3, 5, 40` | 5 items (`1–5`) | Mean when $\ge 80\%$ ($\ge 4/5$) items answered ($\alpha = .92$) |
| **PT-SEX: Shame/Guilt** | `PTSEX_14, 15, 13, 12, 10` | 5 items (`1–5`) | Mean when $\ge 80\%$ ($\ge 4/5$) items answered ($\alpha = .89$) |
| **PT-SEX: Pleasing the Other** | `PTSEX_42, 41, 43, 45` | 4 items (`1–5`) | Mean when $\ge 80\%$ ($\ge 4/4$) items answered ($\alpha = .87$) |
| **PT-SEX: Interpersonal Distress** | `PTSEX_33, 32` | 2 items (`1–5`) | Mean when $2/2$ items answered ($\alpha = .83, \rho_{\text{SB}} = .83, r = .71$) |
| **PT-SEX: Hypervigilance** | `PTSEX_27, 23` | 2 items (`1–5`) | Mean when $2/2$ items answered ($\alpha = .64, \rho_{\text{SB}} = .64, r = .47$) |
| **Attitudes Toward Women's Genitals (ATWGS)** | `Q151_1`–`Q151_10` (`atwgs_80`, `atwgs_100`) | 10 items, rated `1` (*strongly disagree*) to `4` (*strongly agree*) | Reverse-score `Q151_3, Q151_5, Q151_6, Q151_8` via `5 - x`, then mean when $\ge 80\%$ ($\ge 8/10$) items answered ($n = 225$; $\alpha = .91$) |
