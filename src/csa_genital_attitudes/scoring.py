"""
Scale scoring (with the consistent >= 80% item-completion rule and strict 100%
item-complete rule), psychometric reliability (Cronbach's alpha and Spearman-Brown
two-item reliability), and sample filtering utilities.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from .config import CTQ_ITEMS, FACTORS, PTSEX_ITEMS, ATWGS_ITEMS, ATWGS_NEG


def score_scale(df_items: pd.DataFrame, min_frac: float = 0.80, mode: str = "mean") -> pd.Series:
    """
    Compute prorated scale scores when at least `min_frac` of a scale's items are answered.

    Parameters
    ----------
    df_items : pd.DataFrame
        DataFrame of item columns (already reverse-scored if applicable).
    min_frac : float, default 0.80
        Minimum proportion of items required (e.g., 0.80 -> >= 4/5 on CTQ-SA,
        >= 20/25 on PT-SEX, >= 8/10 on ATWGS; 1.0 -> strict 100% item completion).
    mode : {'mean', 'sum'}
        Whether to return the item mean or prorated sum (item mean * number of items).
    """
    n_items = df_items.shape[1]
    min_count = int(np.ceil(min_frac * n_items)) if min_frac > 0 else 1
    counts = df_items.notna().sum(axis=1)
    val = df_items.mean(axis=1)
    if mode == "sum":
        val = val * n_items
    return val.where(counts >= min_count)


def cronbach_alpha(df_items: pd.DataFrame) -> float:
    """Compute Cronbach's alpha on listwise-complete rows of `df_items`."""
    clean = df_items.dropna()
    k = clean.shape[1]
    if k < 2 or len(clean) < 2:
        return np.nan
    item_vars = clean.var(axis=0, ddof=1).sum()
    total_var = clean.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return np.nan
    return float((k / (k - 1.0)) * (1.0 - item_vars / total_var))


def spearman_brown_2item(df_two_items: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute inter-item Pearson r and Spearman-Brown stepped-up reliability (rho_SB)
    for a two-item subscale.
    """
    clean = df_two_items.dropna()
    if clean.shape[1] != 2 or len(clean) < 3:
        return np.nan, np.nan
    r, _ = stats.pearsonr(clean.iloc[:, 0], clean.iloc[:, 1])
    rho_sb = (2.0 * r) / (1.0 + r)
    return float(r), float(rho_sb)


def prepare_analytic_data(
    df_raw: pd.DataFrame,
    df_extra: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply item numeric coercion, ATWGS reverse-scoring (5 - x on the 1-4 scale),
    and both >= 80% and 100% scale scoring.

    Returns
    -------
    women : pd.DataFrame
        All eligible women (N = 675 in the study dataset).
    surv : pd.DataFrame
        Primary analytic sample of women survivors of CSA (csa_direct == 1; N = 346).
    """
    d = df_raw.copy()
    if df_extra is not None:
        extra_cols = [c for c in df_extra.columns if c not in d.columns]
        d = pd.concat([d, df_extra[extra_cols]], axis=1)

    numeric_cols = CTQ_ITEMS + PTSEX_ITEMS + ATWGS_ITEMS + ["age", "abuse_onset"]
    for col in numeric_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Reverse-score the 4 negatively worded ATWGS items on the 1-4 scale
    d_atwgs = d[ATWGS_ITEMS].copy()
    for col in ATWGS_NEG:
        if col in d_atwgs.columns:
            d_atwgs[col] = 5.0 - d_atwgs[col]

    new_cols = {}
    for tag, frac in [("80", 0.80), ("100", 1.00)]:
        new_cols[f"ctq_{tag}"] = score_scale(d[CTQ_ITEMS], frac, "sum")
        new_cols[f"ptsex_{tag}"] = score_scale(d[PTSEX_ITEMS], frac, "mean")
        new_cols[f"atwgs_{tag}"] = score_scale(d_atwgs, frac, "mean")
        for f_name, f_items in FACTORS.items():
            new_cols[f"{f_name}_{tag}"] = score_scale(d[f_items], frac, "mean")

    d = pd.concat([d, pd.DataFrame(new_cols, index=d.index)], axis=1)
    if "abuse_onset" in d.columns:
        d["onset"] = pd.to_numeric(d["abuse_onset"], errors="coerce")

    women = d[d["gender"] == "woman"].copy().reset_index(drop=True)
    surv = women[women["csa_direct"] == 1].copy().reset_index(drop=True)
    return women, surv
