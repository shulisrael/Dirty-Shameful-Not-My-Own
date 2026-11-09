"""
Full Information Maximum Likelihood (FIML) multivariate normal estimation via the
Expectation-Maximization (EM) algorithm, Little's (1988) MCAR test, saturated path
model solver, and percentile / BCa bootstrap inference.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def fiml_em_mvn(
    X_in,
    max_iter: int = 1000,
    tol: float = 1e-9,
    init_mu: Optional[np.ndarray] = None,
    init_Sigma: Optional[np.ndarray] = None,
    ridge: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Estimate the multivariate normal mean vector (mu) and covariance matrix (Sigma)
    by maximizing the observed-data log-likelihood via the Expectation-Maximization
    (EM) algorithm across all cases with at least one observed variable.

    Parameters
    ----------
    X_in : array-like or pd.DataFrame of shape (n_samples, n_features)
        Input data matrix containing NaNs for missing entries.
    max_iter : int, default 1000
        Maximum number of EM iterations.
    tol : float, default 1e-9
        Convergence threshold on maximum absolute parameter change.
    init_mu, init_Sigma : optional warm-start arrays
        Initial mean vector and covariance matrix (used to accelerate bootstrap draws).
    ridge : float, default 1e-8
        Small diagonal ridge term for numerical stability during bootstrap resampling.

    Returns
    -------
    mu : np.ndarray of shape (p,)
        Maximum-likelihood mean vector.
    Sigma : np.ndarray of shape (p, p)
        Maximum-likelihood covariance matrix.
    n : int
        Number of valid cases contributing to the log-likelihood.
    """
    X = np.asarray(X_in, dtype=float)
    obs_mask = ~np.isnan(X)
    valid_rows = obs_mask.any(axis=1)
    X = X[valid_rows]
    obs_mask = obs_mask[valid_rows]
    n, p = X.shape

    if init_mu is not None and init_Sigma is not None:
        mu = init_mu.copy()
        Sigma = init_Sigma.copy()
    else:
        mu = np.nanmean(X, axis=0)
        X_filled = np.where(obs_mask, X, mu)
        Sigma = np.cov(X_filled, rowvar=False, ddof=0) + np.eye(p) * 1e-4

    patterns = {}
    for i in range(n):
        pat = tuple(obs_mask[i])
        patterns.setdefault(pat, []).append(i)
    pat_groups = [
        (np.array(pat, dtype=bool), np.array(idxs, dtype=int))
        for pat, idxs in patterns.items()
    ]

    for _ in range(max_iter):
        mu_old = mu.copy()
        Sigma_old = Sigma.copy()
        T1 = np.zeros(p)
        T2 = np.zeros((p, p))
        for pat, idxs in pat_groups:
            m_i = len(idxs)
            o = np.where(pat)[0]
            m = np.where(~pat)[0]
            X_sub = X[idxs]
            if len(m) == 0:
                T1 += X_sub.sum(axis=0)
                T2 += X_sub.T @ X_sub
            else:
                X_o = X_sub[:, o]
                mu_o = mu[o]
                mu_m = mu[m]
                S_oo = Sigma[np.ix_(o, o)]
                S_mo = Sigma[np.ix_(m, o)]
                S_mm = Sigma[np.ix_(m, m)]
                B = np.linalg.solve(S_oo, S_mo.T).T
                V_m_given_o = S_mm - B @ S_mo.T
                E_X = np.zeros((m_i, p))
                E_X[:, o] = X_o
                E_X[:, m] = mu_m + (X_o - mu_o) @ B.T
                T1 += E_X.sum(axis=0)
                T2 += E_X.T @ E_X
                T2[np.ix_(m, m)] += m_i * V_m_given_o
        mu = T1 / n
        Sigma = (T2 / n) - np.outer(mu, mu) + np.eye(p) * ridge
        if np.max(np.abs(mu - mu_old)) < tol and np.max(np.abs(Sigma - Sigma_old)) < tol:
            break
    return mu, Sigma, n


def little_mcar_test(df_vars: pd.DataFrame, min_pattern_n: int = 4) -> Dict[str, float]:
    """
    Compute Little's (1988) global chi-square test of Missing Completely at Random (MCAR):
        d^2 = sum_j m_j (x_bar_{o_j} - mu_hat_{o_j})^T Sigma_hat_{o_j}^{-1} (x_bar_{o_j} - mu_hat_{o_j})
        df  = (sum_j k_j) - p
    where mu_hat and Sigma_hat are the EM maximum-likelihood estimates across all cases.

    Parameters
    ----------
    df_vars : pd.DataFrame
        Substantive study variables (e.g., ['ctq_80', 'ptsex_80', 'atwgs_80']).
    min_pattern_n : int, default 4
        Minimum pattern cell count for main monotone attrition patterns (set to 1 to
        include singleton sporadic patterns).
    """
    X = df_vars.to_numpy(dtype=float)
    obs_mask = ~np.isnan(X)
    valid_rows = obs_mask.any(axis=1)
    X = X[valid_rows]
    obs_mask = obs_mask[valid_rows]
    n, p = X.shape

    mu_ml, Sigma_ml, _ = fiml_em_mvn(df_vars, max_iter=2000, tol=1e-10, ridge=0.0)

    patterns = {}
    for i in range(n):
        pat = tuple(obs_mask[i])
        patterns.setdefault(pat, []).append(i)

    d2 = 0.0
    sum_kj = 0
    n_patterns = 0
    for pat, idxs in patterns.items():
        if len(idxs) < min_pattern_n:
            continue
        m_j = len(idxs)
        o = np.where(pat)[0]
        sum_kj += len(o)
        n_patterns += 1
        X_o = X[idxs][:, o]
        diff = X_o.mean(axis=0) - mu_ml[o]
        d2 += m_j * float(diff.T @ np.linalg.inv(Sigma_ml[np.ix_(o, o)]) @ diff)

    df = sum_kj - p
    pval = float(1.0 - stats.chi2.cdf(d2, df)) if df > 0 else np.nan
    return {"chi2": float(d2), "df": int(df), "p": pval, "n_patterns": int(n_patterns)}


def solve_med_from_moments(
    mu: np.ndarray,
    Sigma: np.ndarray,
    idx_x: int,
    idx_z: List[int],
    idx_m_list: List[int],
    idx_y: int,
) -> Dict:
    """
    Derive exact saturated path model coefficients (total effect c, a-paths, b-paths,
    direct effect c', specific and total unstandardized/standardized indirect effects,
    and R^2 values) from a mean vector `mu` and covariance matrix `Sigma`.
    """
    sd = np.sqrt(np.diag(Sigma))
    preds_c = [idx_x] + idx_z
    b_c = np.linalg.solve(Sigma[np.ix_(preds_c, preds_c)], Sigma[preds_c, idx_y])
    r2_c = (b_c @ Sigma[preds_c, idx_y]) / Sigma[idx_y, idx_y]
    beta_c = b_c * (sd[preds_c] / sd[idx_y])

    a_unstd, a_std, r2_m = [], [], []
    for idx_m in idx_m_list:
        b_a = np.linalg.solve(Sigma[np.ix_(preds_c, preds_c)], Sigma[preds_c, idx_m])
        a_unstd.append(b_a)
        a_std.append(b_a * (sd[preds_c] / sd[idx_m]))
        r2_m.append((b_a @ Sigma[preds_c, idx_m]) / Sigma[idx_m, idx_m])

    preds_y = [idx_x] + idx_m_list + idx_z
    b_y = np.linalg.solve(Sigma[np.ix_(preds_y, preds_y)], Sigma[preds_y, idx_y])
    r2_y = (b_y @ Sigma[preds_y, idx_y]) / Sigma[idx_y, idx_y]
    beta_y = b_y * (sd[preds_y] / sd[idx_y])

    k_m = len(idx_m_list)
    b_m_unstd = b_y[1 : 1 + k_m]
    b_m_std = beta_y[1 : 1 + k_m]
    ind_unstd = np.array([a_unstd[k][0] * b_m_unstd[k] for k in range(k_m)])
    ind_std = np.array([a_std[k][0] * b_m_std[k] for k in range(k_m)])

    return {
        "mu": mu,
        "sd": sd,
        "corr": Sigma / np.outer(sd, sd),
        "b_c": b_c,
        "beta_c": beta_c,
        "r2_c": float(r2_c),
        "b_a": a_unstd,
        "beta_a": a_std,
        "r2_m": [float(r) for r in r2_m],
        "b_y": b_y,
        "beta_y": beta_y,
        "r2_y": float(r2_y),
        "b_m_unstd": b_m_unstd,
        "b_m_std": b_m_std,
        "ind_unstd": ind_unstd,
        "ind_std": ind_std,
        "tot_ind_unstd": float(ind_unstd.sum()),
        "tot_ind_std": float(ind_std.sum()),
    }


def run_bootstrap_fiml(
    df_vars: pd.DataFrame,
    idx_x: int,
    idx_z: List[int],
    idx_m_list: List[int],
    idx_y: int,
    B: int = 5000,
    seed: int = 20261107,
) -> Tuple[Dict, int, Dict[str, np.ndarray]]:
    """
    Estimate a saturated mediation path model via FIML (EM algorithm) and compute
    bootstrap distributions across `B` resamples of all cases (including incomplete cases).
    """
    X = df_vars.to_numpy(dtype=float)
    mu0, Sigma0, n = fiml_em_mvn(X, max_iter=1000, tol=1e-9)
    est0 = solve_med_from_moments(mu0, Sigma0, idx_x, idx_z, idx_m_list, idx_y)
    rng = np.random.default_rng(seed)

    boot_b_c, boot_beta_c = [], []
    boot_b_a, boot_beta_a = [], []
    boot_b_y, boot_beta_y = [], []
    boot_ind_unstd, boot_ind_std = [], []
    boot_tot_unstd, boot_tot_std = [], []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        mu_b, Sigma_b, _ = fiml_em_mvn(
            X[idx], max_iter=150, tol=1e-5, init_mu=mu0, init_Sigma=Sigma0
        )
        res_b = solve_med_from_moments(mu_b, Sigma_b, idx_x, idx_z, idx_m_list, idx_y)
        boot_b_c.append(res_b["b_c"])
        boot_beta_c.append(res_b["beta_c"])
        boot_b_a.append(np.array(res_b["b_a"]))
        boot_beta_a.append(np.array(res_b["beta_a"]))
        boot_b_y.append(res_b["b_y"])
        boot_beta_y.append(res_b["beta_y"])
        boot_ind_unstd.append(res_b["ind_unstd"])
        boot_ind_std.append(res_b["ind_std"])
        boot_tot_unstd.append(res_b["tot_ind_unstd"])
        boot_tot_std.append(res_b["tot_ind_std"])

    return est0, n, {
        "b_c": np.array(boot_b_c),
        "beta_c": np.array(boot_beta_c),
        "b_a": np.array(boot_b_a),
        "beta_a": np.array(boot_beta_a),
        "b_y": np.array(boot_b_y),
        "beta_y": np.array(boot_beta_y),
        "ind_unstd": np.array(boot_ind_unstd),
        "ind_std": np.array(boot_ind_std),
        "tot_unstd": np.array(boot_tot_unstd),
        "tot_std": np.array(boot_tot_std),
    }


def run_bootstrap_cc(
    df_vars: pd.DataFrame,
    idx_x: int,
    idx_z: List[int],
    idx_m_list: List[int],
    idx_y: int,
    B: int = 5000,
    seed: int = 20261107,
) -> Tuple[Dict, int, Dict[str, np.ndarray]]:
    """
    Estimate a saturated mediation path model on listwise complete cases and compute
    bootstrap distributions across `B` resamples.
    """
    cc = df_vars.dropna()
    X = cc.to_numpy(dtype=float)
    n = len(X)
    mu0 = X.mean(axis=0)
    Sigma0 = np.cov(X, rowvar=False, ddof=1)
    est0 = solve_med_from_moments(mu0, Sigma0, idx_x, idx_z, idx_m_list, idx_y)
    rng = np.random.default_rng(seed)

    boot_b_c, boot_beta_c = [], []
    boot_b_a, boot_beta_a = [], []
    boot_b_y, boot_beta_y = [], []
    boot_ind_unstd, boot_ind_std = [], []
    boot_tot_unstd, boot_tot_std = [], []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        mub = Xb.mean(axis=0)
        Sigmab = np.cov(Xb, rowvar=False, ddof=1)
        res_b = solve_med_from_moments(mub, Sigmab, idx_x, idx_z, idx_m_list, idx_y)
        boot_b_c.append(res_b["b_c"])
        boot_beta_c.append(res_b["beta_c"])
        boot_b_a.append(np.array(res_b["b_a"]))
        boot_beta_a.append(np.array(res_b["beta_a"]))
        boot_b_y.append(res_b["b_y"])
        boot_beta_y.append(res_b["beta_y"])
        boot_ind_unstd.append(res_b["ind_unstd"])
        boot_ind_std.append(res_b["ind_std"])
        boot_tot_unstd.append(res_b["tot_ind_unstd"])
        boot_tot_std.append(res_b["tot_ind_std"])

    return est0, n, {
        "b_c": np.array(boot_b_c),
        "beta_c": np.array(boot_beta_c),
        "b_a": np.array(boot_b_a),
        "beta_a": np.array(boot_beta_a),
        "b_y": np.array(boot_b_y),
        "beta_y": np.array(boot_beta_y),
        "ind_unstd": np.array(boot_ind_unstd),
        "ind_std": np.array(boot_ind_std),
        "tot_unstd": np.array(boot_tot_unstd),
        "tot_std": np.array(boot_tot_std),
    }
