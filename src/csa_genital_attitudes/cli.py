"""
Command-line interface to reproduce all analyses, tables, and figures.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

from .scoring import prepare_analytic_data
from .models import (
    run_attrition_audit,
    run_table1_descriptives_and_correlations,
    run_simple_mediation_suite,
    run_parallel_mediation_suite,
    run_leave_one_item_out,
    run_onset_mediation_and_abuse_chars,
)
from .figures import generate_all_figures
from .synthetic import generate_synthetic_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce statistical analyses and figures for Sperber & Gewirtz-Meydan (CSA -> PT-SEX -> ATWGS)."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to primary CSV dataset (if omitted, checks local study path or uses synthetic data).",
    )
    parser.add_argument(
        "--extra-data-path",
        type=Path,
        default=None,
        help="Optional path to supplementary item-level CSV.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run pipeline on privacy-preserving synthetic dataset (N = 675 women, N = 346 CSA survivors).",
    )
    parser.add_argument(
        "--boot-fiml",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for FIML models (default: 5000).",
    )
    parser.add_argument(
        "--boot-cc",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for complete-case models (default: 5000).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for JSON results and PNG figures (default: ./outputs).",
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    default_root = Path(r"c:\Users\hagay\.codex\.chatgpt-projects\g-p-68fc9d8fd2c081918ec73bb2f5abde18")
    default_d1 = default_root / "outputs/Genital_Attitudes_Discovery_2026-11-02/PRIVATE_analysis_data.csv"
    default_d2 = default_root / "outputs/Clustering_Method_Assessment_2026-10-23/PRIVATE_all_eligible_values.csv"

    if args.synthetic:
        print("[INFO] Generating privacy-preserving synthetic dataset (N = 675 women, N = 346 survivors)...")
        df_raw = generate_synthetic_dataset()
        df_extra = None
        data_source = "synthetic"
    elif args.data_path is not None and args.data_path.exists():
        print(f"[INFO] Loading dataset from {args.data_path}...")
        df_raw = pd.read_csv(args.data_path)
        df_extra = pd.read_csv(args.extra_data_path) if (args.extra_data_path and args.extra_data_path.exists()) else None
        data_source = str(args.data_path)
    elif default_d1.exists():
        print("[INFO] Loading primary study dataset...")
        df_raw = pd.read_csv(default_d1)
        df_extra = pd.read_csv(default_d2) if default_d2.exists() else None
        data_source = "primary_study_dataset"
    else:
        print("[INFO] Private dataset not found locally; falling back to synthetic dataset...")
        df_raw = generate_synthetic_dataset()
        df_extra = None
        data_source = "synthetic_fallback"

    women, surv = prepare_analytic_data(df_raw, df_extra)
    print(f"[INFO] Prepared analytic samples: Eligible women N = {len(women)}, CSA survivors N = {len(surv)}")

    print("[1/6] Running missing-data & attrition audit + Little's (1988) MCAR test...")
    attrition = run_attrition_audit(surv)
    mcar = attrition["little_mcar"]
    print(
        f"      Complete n = {attrition['n_complete']} vs Incomplete n = {attrition['n_incomplete']} "
        f"(Missing ATWGS = {attrition['n_missing_atwgs']})\n"
        f"      CTQ-SA t({attrition['df_t']}) = {attrition['t_stat']:.2f}, p = {attrition['p_ctq']:.3f}, d = {attrition['cohen_d']:.2f}\n"
        f"      Little's MCAR: chi2({mcar['df']}) = {mcar['chi2']:.2f}, p = {mcar['p']:.3f}"
    )

    print("[2/6] Computing Table 1 descriptives, reliabilities, and pairwise + FIML correlations...")
    table1 = run_table1_descriptives_and_correlations(surv)

    print(f"[3/6] Estimating Table 2 & Supplementary Table S1 simple mediation suite (B_fiml={args.boot_fiml}, B_cc={args.boot_cc})...")
    simple_med = run_simple_mediation_suite(surv, women, B_fiml=args.boot_fiml, B_cc=args.boot_cc)
    pf = simple_med["primary_fiml"]
    print(
        f"      Primary FIML (N = {pf['n']}): a = {pf['a']['est']:.3f}, b = {pf['b']['est']:.3f}, "
        f"c' = {pf['c_prime']['est']:.3f} (p = {pf['c_prime']['p']:.3f}), "
        f"ab = {pf['ab']['est']:.3f} [{pf['ab']['ci_lo']:.3f}, {pf['ab']['ci_hi']:.3f}], "
        f"beta_ind = {pf['beta_ind']['est']:.2f} [{pf['beta_ind']['ci_lo']:.2f}, {pf['beta_ind']['ci_hi']:.2f}]"
    )

    print(f"[4/6] Estimating Table 3 & Supplementary Table S2 6-subscale parallel mediation + Shapley dominance...")
    parallel_med = run_parallel_mediation_suite(surv, B_fiml=args.boot_fiml, B_cc=args.boot_cc)
    loio = run_leave_one_item_out(surv)
    dom = parallel_med["dominance"]
    print(
        f"      Shapley Dominance (Total Delta R^2 = {dom['total_delta_r2']:.3f}): "
        f"Shame/guilt = {dom['subscales']['Shame']['pct_share']:.1f}%, "
        f"Pleasing = {dom['subscales']['Pleasing']['pct_share']:.1f}%"
    )

    print("[5/6] Estimating Abuse Characteristics & Age-at-Onset mediation model...")
    onset_med = run_onset_mediation_and_abuse_chars(surv, B_fiml=args.boot_fiml, B_cc=args.boot_cc)

    print("[6/6] Generating Figures 1, 2, and 3...")
    fig_dir = args.out_dir / "figures"
    fig_paths = generate_all_figures(surv, fig_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "data_source": data_source,
        "attrition_audit": attrition,
        "table1_reliabilities": table1["reliabilities"],
        "table1_descriptives": table1["descriptives"],
        "simple_mediation_table2_s1": simple_med,
        "parallel_mediation_table3_s2": parallel_med,
        "leave_one_item_out": loio,
        "onset_mediation": onset_med,
        "figures": {k: str(v) for k, v in fig_paths.items()},
    }
    json_path = args.out_dir / "reproduction_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[DONE] Saved full JSON results to {json_path} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
