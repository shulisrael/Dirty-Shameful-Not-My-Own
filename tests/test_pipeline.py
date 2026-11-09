"""
Automated unit and integration tests for the `csa_genital_attitudes` package.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from csa_genital_attitudes import (
    prepare_analytic_data,
    fiml_em_mvn,
    little_mcar_test,
    solve_med_from_moments,
    run_attrition_audit,
    run_table1_descriptives_and_correlations,
    run_shapley_dominance,
    generate_all_figures,
    generate_synthetic_dataset,
)


def test_fiml_em_and_mediation_solver():
    df_raw = generate_synthetic_dataset(n_women=675, n_survivors=346, seed=20261108)
    women, surv = prepare_analytic_data(df_raw)
    assert len(women) == 675
    assert len(surv) == 346

    cols = ["ctq_80", "age", "ptsex_80", "atwgs_80"]
    mu, Sigma, n = fiml_em_mvn(surv[cols], max_iter=500, tol=1e-7)
    assert n == 346
    assert mu.shape == (4,)
    assert Sigma.shape == (4, 4)

    est = solve_med_from_moments(mu, Sigma, 0, [1], [2], 3)
    assert est["b_a"][0][0] > 0  # More severe CSA -> higher PT-SEX
    assert est["b_y"][1] < 0     # Higher PT-SEX -> less positive ATWGS
    assert est["tot_ind_unstd"] < 0


def test_little_mcar_and_attrition_audit():
    df_raw = generate_synthetic_dataset(n_women=675, n_survivors=346, seed=20261108)
    _, surv = prepare_analytic_data(df_raw)
    audit = run_attrition_audit(surv)
    assert audit["n_total"] == 346
    assert audit["n_complete"] + audit["n_incomplete"] == 346
    mcar = audit["little_mcar"]
    assert mcar["df"] > 0
    assert 0.0 <= mcar["p"] <= 1.0


def test_table1_and_dominance_and_figures(tmp_path):
    df_raw = generate_synthetic_dataset(n_women=675, n_survivors=346, seed=20261108)
    _, surv = prepare_analytic_data(df_raw)

    t1 = run_table1_descriptives_and_correlations(surv)
    assert len(t1["descriptives"]) == 11
    assert 0.5 < t1["reliabilities"]["PTSEX_total_alpha"] < 1.0

    dom = run_shapley_dominance(surv)
    shares = [v["pct_share"] for v in dom["subscales"].values()]
    assert np.isclose(sum(shares), 100.0, atol=1e-4)

    figs = generate_all_figures(surv, tmp_path)
    for p in figs.values():
        assert Path(p).exists()
        assert Path(p).stat().st_size > 10_000


if __name__ == "__main__":
    import tempfile
    test_fiml_em_and_mediation_solver()
    test_little_mcar_and_attrition_audit()
    with tempfile.TemporaryDirectory() as tmpd:
        test_table1_and_dominance_and_figures(Path(tmpd))
    print("ALL UNIT AND INTEGRATION TESTS PASSED.")

