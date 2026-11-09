"""
Reproducibility package for:
"Dirty, Shameful, Not My Own? Traumatized Sexuality as a Pathway From
Childhood Sexual Abuse to Attitudes Toward Women's Genitals"
by Shulamit Sperber & Ateret Gewirtz-Meydan (University of Haifa).
"""

__version__ = "1.0.0"

from .config import CTQ_ITEMS, FACTORS, PTSEX_ITEMS, ATWGS_ITEMS, ATWGS_NEG
from .scoring import score_scale, cronbach_alpha, spearman_brown_2item, prepare_analytic_data
from .fiml import fiml_em_mvn, little_mcar_test, solve_med_from_moments, run_bootstrap_fiml, run_bootstrap_cc
from .models import (
    run_attrition_audit,
    run_table1_descriptives_and_correlations,
    run_simple_mediation_suite,
    run_parallel_mediation_suite,
    run_shapley_dominance,
    run_leave_one_item_out,
    run_onset_mediation_and_abuse_chars,
)
from .figures import generate_all_figures
from .synthetic import generate_synthetic_dataset

__all__ = [
    "CTQ_ITEMS",
    "FACTORS",
    "PTSEX_ITEMS",
    "ATWGS_ITEMS",
    "ATWGS_NEG",
    "score_scale",
    "cronbach_alpha",
    "spearman_brown_2item",
    "prepare_analytic_data",
    "fiml_em_mvn",
    "little_mcar_test",
    "solve_med_from_moments",
    "run_bootstrap_fiml",
    "run_bootstrap_cc",
    "run_attrition_audit",
    "run_table1_descriptives_and_correlations",
    "run_simple_mediation_suite",
    "run_parallel_mediation_suite",
    "run_shapley_dominance",
    "run_leave_one_item_out",
    "run_onset_mediation_and_abuse_chars",
    "generate_all_figures",
    "generate_synthetic_dataset",
]
