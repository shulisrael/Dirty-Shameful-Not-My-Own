"""
Statistical estimation routines for all analyses reported in the manuscript and
Supplementary Tables S1 and S2:
1. Attrition audit and Little's (1988) MCAR test
2. Table 1 descriptive statistics, reliabilities, and pairwise/FIML correlations
3. Table 2 & Supplementary Table S1 simple mediation models (FIML, MICE, OLS/HC3,
   strict 100% item-complete, Cook's D outlier-screened, Huber M-estimator, Tukey
   biweight, median quantile regression, IPW, single-indicator latent SEM, covariate
   checks, and alternative sample definitions across the trauma continuum)
4. Table 3 & Supplementary Table S2 parallel 6-subscale mediation models, Shapley
   value dominance analysis (2^6 = 64 subset models), regularized Ridge/Lasso, and
   leave-one-item-out (LOIO) checks
5. Abuse characteristics and Age-at-Onset mediation model
"""

import math
from itertools import combinations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import RidgeCV, LassoCV

from .config import (
    CTQ_ITEMS,
    FACTORS,
    PTSEX_ITEMS,
    ATWGS_ITEMS,
    ATWGS_NEG,
    PARALLEL_SUBSCALE_ORDER,
    SUBSCALE_LABELS,
)
from .scoring import score_scale, cronbach_alpha, spearman_brown_2item
from .fiml import (
    fiml_em_mvn,
    little_mcar_test,
    solve_med_from_moments,
    run_bootstrap_fiml,
    run_bootstrap_cc,
)


def run_attrition_audit(surv: pd.DataFrame) -> Dict:
    """
    Compare survivors with complete data across the primary mediation model (n = 222)
    versus incomplete data (n = 124), and run Little's (1988) MCAR test on the three
    substantive scales (CTQ-SA, PT-SEX, ATWGS).
    """
    complete_mask = surv[["ctq_80", "ptsex_80", "atwgs_80", "age"]].notna().all(axis=1)
    comp = surv[complete_mask]
    incomp = surv[~complete_mask]

    c_ctq = comp["ctq_80"].dropna()
    i_ctq = incomp["ctq_80"].dropna()
    t_stat, p_val = stats.ttest_ind(c_ctq, i_ctq, equal_var=False)
    df_t = len(c_ctq) + len(i_ctq) - 2
    pooled_sd = np.sqrt(
        ((len(c_ctq) - 1) * c_ctq.var(ddof=1) + (len(i_ctq) - 1) * i_ctq.var(ddof=1))
        / df_t
    )
    cohen_d = (c_ctq.mean() - i_ctq.mean()) / pooled_sd

    _, p_age = stats.ttest_ind(comp["age"].dropna(), incomp["age"].dropna(), equal_var=True)
    _, p_ptsex = stats.ttest_ind(
        comp["ptsex_80"].dropna(), incomp["ptsex_80"].dropna(), equal_var=True
    )

    mcar = little_mcar_test(surv[["ctq_80", "ptsex_80", "atwgs_80"]], min_pattern_n=4)
    n_missing_atwgs = int(surv["atwgs_80"].isna().sum())

    return {
        "n_total": int(len(surv)),
        "n_complete": int(len(comp)),
        "n_incomplete": int(len(incomp)),
        "n_missing_atwgs": n_missing_atwgs,
        "n_missing_other": int(len(incomp) - n_missing_atwgs),
        "ctq_complete_mean": float(c_ctq.mean()),
        "ctq_complete_sd": float(c_ctq.std(ddof=1)),
        "ctq_incomplete_mean": float(i_ctq.mean()),
        "ctq_incomplete_sd": float(i_ctq.std(ddof=1)),
        "t_stat": float(t_stat),
        "df_t": int(df_t),
        "p_ctq": float(p_val),
        "cohen_d": float(cohen_d),
        "p_age": float(p_age),
        "p_ptsex": float(p_ptsex),
        "little_mcar": mcar,
    }


def run_table1_descriptives_and_correlations(surv: pd.DataFrame) -> Dict:
    """
    Compute Table 1 means, SDs, pairwise correlations (with exact p-values),
    FIML means/SDs/correlations (N = 346), and scale reliabilities (Cronbach's alpha
    and Spearman-Brown rho_SB).
    """
    var_cols = [
        ("ctq_80", "1. CSA severity (CTQ-SA)"),
        ("ptsex_80", "2. PT-SEX total"),
        ("Dissociation_80", "3. Dissociation"),
        ("Intrusiveness_80", "4. Intrusiveness"),
        ("Shame_80", "5. Shame/guilt"),
        ("Pleasing_80", "6. Pleasing the other"),
        ("Interpersonal_80", "7. Interpersonal distress"),
        ("Hypervigilance_80", "8. Hypervigilance"),
        ("atwgs_80", "9. Attitudes toward women's genitals"),
        ("age", "10. Age"),
        ("onset", "11. Age at abuse onset"),
    ]
    cols = [c for c, _ in var_cols]
    labels = [lbl for _, lbl in var_cols]

    # Observed pairwise descriptives and correlations
    desc_rows = []
    k = len(cols)
    r_mat = np.full((k, k), np.nan)
    p_mat = np.full((k, k), np.nan)
    n_mat = np.zeros((k, k), dtype=int)

    for i, (col_i, lbl_i) in enumerate(var_cols):
        s_i = surv[col_i].dropna()
        desc_rows.append(
            {
                "variable": lbl_i,
                "col": col_i,
                "n": int(len(s_i)),
                "mean": float(s_i.mean()),
                "sd": float(s_i.std(ddof=1)),
                "skew": float(stats.skew(s_i)),
                "kurtosis": float(stats.kurtosis(s_i)),
            }
        )
        for j in range(i):
            col_j = cols[j]
            pair = surv[[col_i, col_j]].dropna()
            n_mat[i, j] = len(pair)
            if len(pair) >= 3:
                r_val, p_val = stats.pearsonr(pair[col_i], pair[col_j])
                r_mat[i, j] = r_val
                p_mat[i, j] = p_val

    # FIML moments on all 11 variables (N = 346)
    mu_fiml, Sigma_fiml, n_fiml = fiml_em_mvn(surv[cols], max_iter=1500, tol=1e-9)
    sd_fiml = np.sqrt(np.diag(Sigma_fiml))
    corr_fiml = Sigma_fiml / np.outer(sd_fiml, sd_fiml)

    # Scale reliabilities
    d_atwgs = surv[ATWGS_ITEMS].copy()
    for col in ATWGS_NEG:
        if col in d_atwgs.columns:
            d_atwgs[col] = 5.0 - d_atwgs[col]

    r_interp, sb_interp = spearman_brown_2item(surv[FACTORS["Interpersonal"]])
    r_hyper, sb_hyper = spearman_brown_2item(surv[FACTORS["Hypervigilance"]])

    reliabilities = {
        "CTQ_SA_alpha": cronbach_alpha(surv[CTQ_ITEMS]),
        "PTSEX_total_alpha": cronbach_alpha(surv[PTSEX_ITEMS]),
        "Dissociation_alpha": cronbach_alpha(surv[FACTORS["Dissociation"]]),
        "Intrusiveness_alpha": cronbach_alpha(surv[FACTORS["Intrusiveness"]]),
        "Shame_alpha": cronbach_alpha(surv[FACTORS["Shame"]]),
        "Pleasing_alpha": cronbach_alpha(surv[FACTORS["Pleasing"]]),
        "Interpersonal_alpha": cronbach_alpha(surv[FACTORS["Interpersonal"]]),
        "Interpersonal_r": r_interp,
        "Interpersonal_spearman_brown": sb_interp,
        "Hypervigilance_alpha": cronbach_alpha(surv[FACTORS["Hypervigilance"]]),
        "Hypervigilance_r": r_hyper,
        "Hypervigilance_spearman_brown": sb_hyper,
        "ATWGS_alpha": cronbach_alpha(d_atwgs),
    }

    return {
        "labels": labels,
        "descriptives": desc_rows,
        "pairwise_r": r_mat,
        "pairwise_p": p_mat,
        "pairwise_n": n_mat,
        "fiml_n": int(n_fiml),
        "fiml_mean": mu_fiml,
        "fiml_sd": sd_fiml,
        "fiml_corr": corr_fiml,
        "reliabilities": reliabilities,
    }


def _summarize_med_run(est: Dict, n: int, boot: Dict) -> Dict:
    """Format point estimates, bootstrap SEs, 95% CIs, and two-sided p-values."""
    def _stat(val, arr):
        se = float(arr.std(ddof=1))
        lo, hi = np.percentile(arr, [2.5, 97.5])
        z = val / se if se > 0 else 0.0
        p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
        return {"est": float(val), "se": se, "ci_lo": float(lo), "ci_hi": float(hi), "p": p}

    return {
        "n": int(n),
        "a": _stat(est["b_a"][0][0], boot["b_a"][:, 0, 0]),
        "a_beta": float(est["beta_a"][0][0]),
        "b": _stat(est["b_y"][1], boot["b_y"][:, 1]),
        "b_beta": float(est["b_m_std"][0]),
        "c_prime": _stat(est["b_y"][0], boot["b_y"][:, 0]),
        "c_prime_beta": float(est["beta_y"][0]),
        "c": _stat(est["b_c"][0], boot["b_c"][:, 0]),
        "c_beta": float(est["beta_c"][0]),
        "ab": _stat(est["tot_ind_unstd"], boot["tot_unstd"]),
        "beta_ind": _stat(est["tot_ind_std"], boot["tot_std"]),
        "r2_m": float(est["r2_m"][0]),
        "r2_y": float(est["r2_y"]),
    }


def run_simple_mediation_suite(
    surv: pd.DataFrame,
    women: pd.DataFrame,
    B_fiml: int = 2000,
    B_cc: int = 5000,
    seed: int = 20261107,
) -> Dict:
    """
    Run all simple mediation models reported in Table 2 and Supplementary Table S1:
    - Primary FIML (N = 346)
    - Multiple Imputation by Chained Equations (MICE, M = 30, N = 346)
    - Complete-case OLS/HC3 (n = 222)
    - Strict 100% item-complete (n = 199)
    - Cook's D outlier-screened (n = 205)
    - Huber M-estimator and Tukey biweight robust regressions (n = 222)
    - Inverse-Probability of Attrition Weighting (IPW, n = 222)
    - Single-indicator latent SEM correcting for measurement error
    - Covariate sensitivity checks (controlling for onset age; without age covariate)
    - Alternative sample definitions across the trauma continuum (N = 402, 527, 675)
    - Reverse mediation check (CTQ-SA -> ATWGS -> PT-SEX)
    """
    cols_s80 = ["ctq_80", "age", "ptsex_80", "atwgs_80"]
    est_f, nf, bf = run_bootstrap_fiml(surv[cols_s80], 0, [1], [2], 3, B=B_fiml, seed=seed)
    est_c, nc, bc = run_bootstrap_cc(surv[cols_s80], 0, [1], [2], 3, B=B_cc, seed=seed)

    cols_s100 = ["ctq_100", "age", "ptsex_100", "atwgs_100"]
    est_100, n100, b100 = run_bootstrap_cc(surv[cols_s100], 0, [1], [2], 3, B=B_cc, seed=seed)

    # Cook's D outlier-screened (excluding Cook's D > 4/n)
    cc_df = surv[cols_s80].dropna().copy().reset_index(drop=True)
    m_a_ols = smf.ols("ptsex_80 ~ ctq_80 + age", data=cc_df).fit()
    m_b_ols = smf.ols("atwgs_80 ~ ctq_80 + ptsex_80 + age", data=cc_df).fit()
    cooks_a = m_a_ols.get_influence().cooks_distance[0]
    cooks_b = m_b_ols.get_influence().cooks_distance[0]
    thresh = 4.0 / len(cc_df)
    clean_cooks = cc_df[(cooks_a <= thresh) & (cooks_b <= thresh)].copy()
    est_cd, ncd, bcd = run_bootstrap_cc(clean_cooks, 0, [1], [2], 3, B=B_cc, seed=seed)

    # Huber M-estimator & Tukey Biweight
    rlm_a_h = smf.rlm("ptsex_80 ~ ctq_80 + age", data=cc_df, M=sm.robust.norms.HuberT()).fit()
    rlm_b_h = smf.rlm("atwgs_80 ~ ctq_80 + ptsex_80 + age", data=cc_df, M=sm.robust.norms.HuberT()).fit()
    rlm_a_t = smf.rlm("ptsex_80 ~ ctq_80 + age", data=cc_df, M=sm.robust.norms.TukeyBiweight()).fit()
    rlm_b_t = smf.rlm("atwgs_80 ~ ctq_80 + ptsex_80 + age", data=cc_df, M=sm.robust.norms.TukeyBiweight()).fit()

    # MICE (M = 30 imputations pooled via Rubin's rules)
    mice_cols = ["ctq_80", "age", "onset", "ptsex_80", "atwgs_80"]
    mice_a, mice_b, mice_cp, mice_ab, mice_ab_var = [], [], [], [], []
    for m_idx in range(30):
        imp = IterativeImputer(sample_posterior=True, max_iter=20, random_state=seed + m_idx)
        X_imp = imp.fit_transform(surv[mice_cols])
        df_imp = pd.DataFrame(X_imp, columns=mice_cols)
        fit_a = smf.ols("ptsex_80 ~ ctq_80 + age", data=df_imp).fit(cov_type="HC3")
        fit_b = smf.ols("atwgs_80 ~ ctq_80 + ptsex_80 + age", data=df_imp).fit(cov_type="HC3")
        ai, sai = fit_a.params["ctq_80"], fit_a.bse["ctq_80"]
        bi, sbi = fit_b.params["ptsex_80"], fit_b.bse["ptsex_80"]
        mice_a.append(ai)
        mice_b.append(bi)
        mice_cp.append(fit_b.params["ctq_80"])
        mice_ab.append(ai * bi)
        mice_ab_var.append((bi**2) * (sai**2) + (ai**2) * (sbi**2))
    mice_ab_est = float(np.mean(mice_ab))
    mice_within = float(np.mean(mice_ab_var))
    mice_between = float(np.var(mice_ab, ddof=1))
    mice_se = float(np.sqrt(mice_within + (1.0 + 1.0 / 30.0) * mice_between))

    # Single-indicator latent SEM correcting for measurement error
    # Using Cronbach's alphas: CTQ-SA = .81, PT-SEX = .95, ATWGS = .91, Age = 1.00
    alphas = np.array([0.81, 1.00, 0.95, 0.91])
    mu_lat, Sigma_lat, _ = fiml_em_mvn(surv[cols_s80], max_iter=1000, tol=1e-9)
    Sigma_true = Sigma_lat.copy()
    for i_idx in range(4):
        Sigma_true[i_idx, i_idx] = Sigma_lat[i_idx, i_idx] * alphas[i_idx]
    est_latent = solve_med_from_moments(mu_lat, Sigma_true, 0, [1], [2], 3)

    # Covariate sensitivity: (a) Controlling for age at abuse onset, (b) Without age covariate
    cols_onset = ["ctq_80", "age", "onset", "ptsex_80", "atwgs_80"]
    est_on_f, non_f, bon_f = run_bootstrap_fiml(surv[cols_onset], 0, [1, 2], [3], 4, B=B_fiml, seed=seed)
    est_on_c, non_c, bon_c = run_bootstrap_cc(surv[cols_onset], 0, [1, 2], [3], 4, B=B_cc, seed=seed)

    cols_noage = ["ctq_80", "ptsex_80", "atwgs_80"]
    est_na_f, nna_f, bna_f = run_bootstrap_fiml(surv[cols_noage], 0, [], [1], 2, B=B_fiml, seed=seed)
    est_na_c, nna_c, bna_c = run_bootstrap_cc(surv[cols_noage], 0, [], [1], 2, B=B_cc, seed=seed)

    # Alternative sample definitions across the trauma continuum
    w_union = women[(women["csa_direct"] == 1) | (women["ctq_80"] >= 10)].copy()
    w_ctq6 = women[women["ctq_80"] >= 6].copy()
    est_u_f, nu_f, bu_f = run_bootstrap_fiml(w_union[cols_s80], 0, [1], [2], 3, B=B_fiml, seed=seed)
    est_u_c, nu_c, bu_c = run_bootstrap_cc(w_union[cols_s80], 0, [1], [2], 3, B=B_cc, seed=seed)
    est_6_f, n6_f, b6_f = run_bootstrap_fiml(w_ctq6[cols_s80], 0, [1], [2], 3, B=B_fiml, seed=seed)
    est_6_c, n6_c, b6_c = run_bootstrap_cc(w_ctq6[cols_s80], 0, [1], [2], 3, B=B_cc, seed=seed)
    est_all_f, nall_f, ball_f = run_bootstrap_fiml(women[cols_s80], 0, [1], [2], 3, B=B_fiml, seed=seed)
    est_all_c, nall_c, ball_c = run_bootstrap_cc(women[cols_s80], 0, [1], [2], 3, B=B_cc, seed=seed)

    # Reverse mediation check: CTQ-SA -> ATWGS -> PT-SEX | Age
    rev_fit_y = smf.ols("ptsex_80 ~ ctq_80 + atwgs_80 + age", data=cc_df).fit(cov_type="HC3")

    return {
        "primary_fiml": _summarize_med_run(est_f, nf, bf),
        "primary_cc": _summarize_med_run(est_c, nc, bc),
        "strict_100_cc": _summarize_med_run(est_100, n100, b100),
        "cooks_d_cc": _summarize_med_run(est_cd, ncd, bcd),
        "huber_rlm": {
            "a": float(rlm_a_h.params["ctq_80"]),
            "b": float(rlm_b_h.params["ptsex_80"]),
            "ab": float(rlm_a_h.params["ctq_80"] * rlm_b_h.params["ptsex_80"]),
        },
        "tukey_rlm": {
            "a": float(rlm_a_t.params["ctq_80"]),
            "b": float(rlm_b_t.params["ptsex_80"]),
            "ab": float(rlm_a_t.params["ctq_80"] * rlm_b_t.params["ptsex_80"]),
        },
        "mice": {
            "a": float(np.mean(mice_a)),
            "b": float(np.mean(mice_b)),
            "c_prime": float(np.mean(mice_cp)),
            "ab": mice_ab_est,
            "ab_se": mice_se,
            "ci_lo": float(mice_ab_est - 1.96 * mice_se),
            "ci_hi": float(mice_ab_est + 1.96 * mice_se),
        },
        "latent_sem": {
            "a_beta": float(est_latent["beta_a"][0][0]),
            "b_beta": float(est_latent["b_m_std"][0]),
            "c_prime_beta": float(est_latent["beta_y"][0]),
            "beta_ind": float(est_latent["tot_ind_std"]),
        },
        "control_onset_fiml": _summarize_med_run(est_on_f, non_f, bon_f),
        "control_onset_cc": _summarize_med_run(est_on_c, non_c, bon_c),
        "no_age_fiml": _summarize_med_run(est_na_f, nna_f, bna_f),
        "no_age_cc": _summarize_med_run(est_na_c, nna_c, bna_c),
        "union_fiml": _summarize_med_run(est_u_f, nu_f, bu_f),
        "union_cc": _summarize_med_run(est_u_c, nu_c, bu_c),
        "ctq6_fiml": _summarize_med_run(est_6_f, n6_f, b6_f),
        "ctq6_cc": _summarize_med_run(est_6_c, n6_c, b6_c),
        "all_women_fiml": _summarize_med_run(est_all_f, nall_f, ball_f),
        "all_women_cc": _summarize_med_run(est_all_c, nall_c, ball_c),
        "reverse_mediation_cc": {
            "c_prime": float(rev_fit_y.params["ctq_80"]),
            "c_prime_p": float(rev_fit_y.pvalues["ctq_80"]),
        },
    }


def run_shapley_dominance(surv: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute exact Shapley value dominance analysis across all 2^6 = 64 subset regression
    models predicting ATWGS from the six PT-SEX subscales beyond CSA severity and age.
    """
    sub_cols = [f"{name}_80" for name in PARALLEL_SUBSCALE_ORDER]
    base_cols = ["ctq_80", "age"]
    df_cc = surv[base_cols + sub_cols + ["atwgs_80"]].dropna().copy()
    y = df_cc["atwgs_80"].to_numpy()
    n = len(df_cc)

    def _r2_for_cols(cols_list):
        X = np.column_stack([np.ones(n)] + [df_cc[c].to_numpy() for c in cols_list])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot)

    r2_base = _r2_for_cols(base_cols)
    r2_cache = {}
    k = len(sub_cols)
    for r_size in range(k + 1):
        for subset in combinations(range(k), r_size):
            cols_now = base_cols + [sub_cols[i] for i in subset]
            r2_cache[frozenset(subset)] = _r2_for_cols(cols_now) - r2_base

    total_incr_r2 = r2_cache[frozenset(range(k))]
    shapley = np.zeros(k)
    fact = math.factorial
    for i in range(k):
        others = [j for j in range(k) if j != i]
        for r_size in range(k):
            weight = (fact(r_size) * fact(k - r_size - 1)) / fact(k)
            for S in combinations(others, r_size):
                S_set = frozenset(S)
                S_with_i = frozenset(S + (i,))
                shapley[i] += weight * (r2_cache[S_with_i] - r2_cache[S_set])

    out = {"total_delta_r2": float(total_incr_r2), "subscales": {}}
    for i, name in enumerate(PARALLEL_SUBSCALE_ORDER):
        out["subscales"][name] = {
            "delta_r2": float(shapley[i]),
            "pct_share": float(100.0 * shapley[i] / total_incr_r2) if total_incr_r2 > 0 else 0.0,
        }
    return out


def run_parallel_mediation_suite(
    surv: pd.DataFrame,
    B_fiml: int = 2000,
    B_cc: int = 5000,
    seed: int = 20261107,
) -> Dict:
    """
    Estimate the 6-subscale parallel multiple mediation model (Table 3 and Supplementary
    Table S2) across FIML (N = 346), Complete-Case (n = 217), Strict 100% Item-Complete
    (n = 199), Cook's D-screened (n = 201), Huber/Tukey/Median robust regressions, and
    10-fold cross-validated Ridge and Lasso regressions.
    """
    sub_cols_80 = [f"{name}_80" for name in PARALLEL_SUBSCALE_ORDER]
    cols_80 = ["ctq_80", "age"] + sub_cols_80 + ["atwgs_80"]
    est_f, nf, bf = run_bootstrap_fiml(surv[cols_80], 0, [1], [2, 3, 4, 5, 6, 7], 8, B=B_fiml, seed=seed)
    est_c, nc, bc = run_bootstrap_cc(surv[cols_80], 0, [1], [2, 3, 4, 5, 6, 7], 8, B=B_cc, seed=seed)

    sub_cols_100 = [f"{name}_100" for name in PARALLEL_SUBSCALE_ORDER]
    cols_100 = ["ctq_100", "age"] + sub_cols_100 + ["atwgs_100"]
    est_100, n100, b100 = run_bootstrap_cc(surv[cols_100], 0, [1], [2, 3, 4, 5, 6, 7], 8, B=B_cc, seed=seed)

    def _pack_parallel(est_d, n_val, boot_d):
        rows = {}
        for k, name in enumerate(PARALLEL_SUBSCALE_ORDER):
            a_beta = float(est_d["beta_a"][k][0])
            a_se = float(boot_d["beta_a"][:, k, 0].std(ddof=1))
            a_p = float(2.0 * (1.0 - stats.norm.cdf(abs(a_beta / a_se))))
            b_beta = float(est_d["b_m_std"][k])
            b_se = float(boot_d["beta_y"][:, 1 + k].std(ddof=1))
            b_p = float(2.0 * (1.0 - stats.norm.cdf(abs(b_beta / b_se))))
            ind_s = float(est_d["ind_std"][k])
            s_lo, s_hi = np.percentile(boot_d["ind_std"][:, k], [2.5, 97.5])
            ind_u = float(est_d["ind_unstd"][k])
            u_lo, u_hi = np.percentile(boot_d["ind_unstd"][:, k], [2.5, 97.5])
            rows[name] = {
                "label": SUBSCALE_LABELS[name],
                "a_beta": a_beta,
                "a_se": a_se,
                "a_p": a_p,
                "b_beta": b_beta,
                "b_se": b_se,
                "b_p": b_p,
                "beta_ind": ind_s,
                "beta_ind_ci": [float(s_lo), float(s_hi)],
                "b_ind": ind_u,
                "b_ind_ci": [float(u_lo), float(u_hi)],
            }
        tot_s_lo, tot_s_hi = np.percentile(boot_d["tot_std"], [2.5, 97.5])
        tot_u_lo, tot_u_hi = np.percentile(boot_d["tot_unstd"], [2.5, 97.5])
        cp_beta = float(est_d["beta_y"][0])
        cp_se = float(boot_d["beta_y"][:, 0].std(ddof=1))
        cp_p = float(2.0 * (1.0 - stats.norm.cdf(abs(cp_beta / cp_se))))
        return {
            "n": int(n_val),
            "r2_y": float(est_d["r2_y"]),
            "c_prime_beta": cp_beta,
            "c_prime_p": cp_p,
            "tot_beta_ind": float(est_d["tot_ind_std"]),
            "tot_beta_ind_ci": [float(tot_s_lo), float(tot_s_hi)],
            "tot_b_ind": float(est_d["tot_ind_unstd"]),
            "tot_b_ind_ci": [float(tot_u_lo), float(tot_u_hi)],
            "dimensions": rows,
        }

    # Standardized complete-case dataframe for Huber, Quantile, Ridge, Lasso
    cc_df = surv[cols_80].dropna().copy()
    z_df = (cc_df - cc_df.mean()) / cc_df.std(ddof=1)
    formula_z = "atwgs_80 ~ ctq_80 + age + " + " + ".join(sub_cols_80)
    huber_z = smf.rlm(formula_z, data=z_df, M=sm.robust.norms.HuberT()).fit()
    quant_z = smf.quantreg(formula_z, data=z_df).fit(q=0.50)

    X_z = z_df[["ctq_80", "age"] + sub_cols_80].to_numpy()
    y_z = z_df["atwgs_80"].to_numpy()
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 100), cv=10).fit(X_z, y_z)
    lasso = LassoCV(cv=10, random_state=seed).fit(X_z, y_z)

    robust_b = {}
    for i, name in enumerate(PARALLEL_SUBSCALE_ORDER):
        col = f"{name}_80"
        robust_b[name] = {
            "huber_beta": float(huber_z.params[col]),
            "huber_p": float(huber_z.pvalues[col]),
            "median_beta": float(quant_z.params[col]),
            "median_p": float(quant_z.pvalues[col]),
            "ridge_beta": float(ridge.coef_[2 + i]),
            "lasso_beta": float(lasso.coef_[2 + i]),
        }

    return {
        "fiml": _pack_parallel(est_f, nf, bf),
        "complete_case": _pack_parallel(est_c, nc, bc),
        "strict_100": _pack_parallel(est_100, n100, b100),
        "robust_and_regularized": robust_b,
        "dominance": run_shapley_dominance(surv),
    }


def run_leave_one_item_out(surv: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Verify that dropping any single item from the Shame/Guilt (5 items) or Pleasing
    (4 items) subscales does not alter the significance of their a-paths or b-paths.
    """
    results = {"Shame": [], "Pleasing": []}
    other_Map = {
        "Shame": ["Pleasing_80", "Dissociation_80", "Hypervigilance_80", "Interpersonal_80", "Intrusiveness_80"],
        "Pleasing": ["Shame_80", "Dissociation_80", "Hypervigilance_80", "Interpersonal_80", "Intrusiveness_80"],
    }
    for sub_name in ["Shame", "Pleasing"]:
        items = FACTORS[sub_name]
        for dropped in items:
            kept = [it for it in items if it != dropped]
            temp = surv.copy()
            temp["sub_loio"] = score_scale(temp[kept], 0.75, "mean")
            cols = ["ctq_80", "age", "sub_loio"] + other_Map[sub_name] + ["atwgs_80"]
            cc = temp[cols].dropna()
            fit_a = smf.ols("sub_loio ~ ctq_80 + age", data=cc).fit(cov_type="HC3")
            rhs_b = "ctq_80 + age + sub_loio + " + " + ".join(other_Map[sub_name])
            fit_b = smf.ols(f"atwgs_80 ~ {rhs_b}", data=cc).fit(cov_type="HC3")
            results[sub_name].append(
                {
                    "dropped_item": dropped,
                    "a_p": float(fit_a.pvalues["ctq_80"]),
                    "b_p": float(fit_b.pvalues["sub_loio"]),
                }
            )
    return results


def run_onset_mediation_and_abuse_chars(
    surv: pd.DataFrame, B_fiml: int = 2000, B_cc: int = 5000, seed: int = 20261108
) -> Dict:
    """
    Estimate the mediation model testing whether traumatized sexuality mediates the
    association between age at abuse onset and genital attitudes (controlling for
    CSA severity and current age) in both FIML (N = 346) and complete-case (n = 211) data.
    """
    cols_onset = ["onset", "ctq_80", "age", "ptsex_80", "atwgs_80"]
    est_f, nf, bf = run_bootstrap_fiml(surv[cols_onset], 0, [1, 2], [3], 4, B=B_fiml, seed=seed)
    est_c, nc, bc = run_bootstrap_cc(surv[cols_onset], 0, [1, 2], [3], 4, B=B_cc, seed=seed)
    return {
        "fiml": _summarize_med_run(est_f, nf, bf),
        "complete_case": _summarize_med_run(est_c, nc, bc),
    }
