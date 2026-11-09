#!/usr/bin/env python3
"""
One-command script to reproduce all statistical analyses, tables, and figures for:
"Dirty, Shameful, Not My Own? Traumatized Sexuality as a Pathway From Childhood
Sexual Abuse to Attitudes Toward Women's Genitals"
(Shulamit Sperber & Ateret Gewirtz-Meydan, University of Haifa).

Usage:
    # Run on local study dataset (or automatic synthetic fallback if private data is absent):
    python scripts/run_all_analyses.py

    # Explicitly run on privacy-preserving synthetic dataset:
    python scripts/run_all_analyses.py --synthetic --boot-fiml 500 --boot-cc 1000
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from csa_genital_attitudes.cli import main

if __name__ == "__main__":
    main()
