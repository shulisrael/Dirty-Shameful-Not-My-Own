"""
Privacy-preserving synthetic data generator matching the empirical variable structure,
item ranges, reliability, and missingness patterns (N = 675 eligible women; N = 346
women CSA survivors).

Allows external researchers, peer reviewers, and continuous-integration pipelines
to verify the entire statistical and figure-generation codebase when the private
clinical dataset is not present locally.
"""

import numpy as np
import pandas as pd

from .config import CTQ_ITEMS, FACTORS, PTSEX_ITEMS, ATWGS_ITEMS, ATWGS_NEG


def generate_synthetic_dataset(n_women: int = 675, n_survivors: int = 346, seed: int = 20261108) -> pd.DataFrame:
    """
    Generate a synthetic item-level dataset with the same columns, item scales,
    covariance structure, and monotone survey attrition pattern as the study sample.
    """
    rng = np.random.default_rng(seed)
    n_non_surv = n_women - n_survivors

    # Latent constructs for survivors: [CTQ,Dissoc,Intrus,Shame,Please,Interp,Hyper,ATWGS,Age,Onset]
    corr_surv = np.array([
        [ 1.00,  0.25,  0.47,  0.34,  0.22,  0.23,  0.32, -0.16, -0.07, -0.33],
        [ 0.25,  1.00,  0.52,  0.55,  0.58,  0.60,  0.69, -0.39, -0.22, -0.20],
        [ 0.47,  0.52,  1.00,  0.58,  0.37,  0.51,  0.58, -0.29, -0.21, -0.25],
        [ 0.34,  0.55,  0.58,  1.00,  0.44,  0.50,  0.55, -0.47, -0.14, -0.27],
        [ 0.22,  0.58,  0.37,  0.44,  1.00,  0.54,  0.43, -0.39, -0.20, -0.10],
        [ 0.23,  0.60,  0.51,  0.50,  0.54,  1.00,  0.64, -0.34, -0.05, -0.22],
        [ 0.32,  0.69,  0.58,  0.55,  0.43,  0.64,  1.00, -0.39, -0.17, -0.19],
        [-0.16, -0.39, -0.29, -0.47, -0.39, -0.34, -0.39,  1.00,  0.04,  0.18],
        [-0.07, -0.22, -0.21, -0.14, -0.20, -0.05, -0.17,  0.04,  1.00, -0.05],
        [-0.33, -0.20, -0.25, -0.27, -0.10, -0.22, -0.19,  0.18, -0.05,  1.00],
    ])
    # Ensure positive definiteness
    eigvals, eigvecs = np.linalg.eigh(corr_surv)
    eigvals = np.maximum(eigvals, 0.02)
    corr_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d_inv = np.diag(1.0 / np.sqrt(np.diag(corr_pd)))
    corr_pd = d_inv @ corr_pd @ d_inv

    Z_surv = rng.multivariate_normal(np.zeros(10), corr_pd, size=n_survivors)
    Z_non = rng.multivariate_normal(np.zeros(10), np.eye(10), size=n_non_surv)
    Z_non[:, 0] -= 1.8  # lower CTQ-SA among non-survivors
    Z = np.vstack([Z_surv, Z_non])
    csa_direct = np.concatenate([np.ones(n_survivors, dtype=int), np.zeros(n_non_surv, dtype=int)])

    rows = pd.DataFrame({
        "gender": ["woman"] * n_women,
        "csa_direct": csa_direct,
        "age": np.clip(np.round(41.41 + 12.18 * Z[:, 8]), 18, 73),
        "abuse_onset": np.where(csa_direct == 1, np.clip(np.round(8.66 + 4.42 * Z[:, 9]), 1, 17), np.nan),
    })

    # Generate 5 CTQ-SA items (1..5)
    for item in CTQ_ITEMS:
        val = 3.60 + 1.09 * Z[:, 0] + rng.normal(0, 0.45, size=n_women)
        rows[item] = np.clip(np.round(val), 1, 5)

    # Generate 25 PT-SEX items (1..5)
    factor_means = [2.82, 2.46, 1.94, 2.78, 2.23, 2.45]
    factor_sds = [1.21, 1.26, 1.13, 1.30, 1.30, 1.25]
    for f_idx, (f_name, f_items) in enumerate(FACTORS.items()):
        for item in f_items:
            val = factor_means[f_idx] + factor_sds[f_idx] * Z[:, 1 + f_idx] + rng.normal(0, 0.40, size=n_women)
            rows[item] = np.clip(np.round(val), 1, 5)

    # Generate 10 ATWGS items (1..4; reverse-coding the 4 negative items)
    for item in ATWGS_ITEMS:
        pos_val = np.clip(np.round(2.89 + 0.70 * Z[:, 7] + rng.normal(0, 0.30, size=n_women)), 1, 4)
        if item in ATWGS_NEG:
            rows[item] = 5.0 - pos_val
        else:
            rows[item] = pos_val

    # Impose realistic monotone attrition matching the study (248 PT-SEX, 225 ATWGS among 346 survivors)
    # Slightly higher dropout among lower CTQ-SA scores (MAR)
    surv_idx = np.where(csa_direct == 1)[0]
    drop_prob = 1.0 / (1.0 + np.exp(0.65 + 0.25 * Z_surv[:, 0]))
    order = np.argsort(-drop_prob)
    no_ptsex = surv_idx[order[:98]]
    no_atwgs = surv_idx[order[:121]]
    rows.loc[no_ptsex, PTSEX_ITEMS] = np.nan
    rows.loc[no_atwgs, ATWGS_ITEMS] = np.nan

    return rows
