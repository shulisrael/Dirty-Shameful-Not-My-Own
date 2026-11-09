"""
Item and subscale definitions for the Childhood Trauma Questionnaire Sexual Abuse
subscale (CTQ-SA), the Post-Traumatic Sexuality Scale (PT-SEX), and the Attitudes
Toward Women's Genitals Scale (ATWGS).
"""

# 5-item Childhood Trauma Questionnaire Sexual Abuse subscale (CTQ-SF; Bernstein et al., 2003)
CTQ_ITEMS = [f"Q18_{j}" for j in range(1, 6)]

# 25-item Post-Traumatic Sexuality Scale (PT-SEX; Gewirtz-Meydan & Lassri, 2023)
FACTORS = {
    "Dissociation":   [f"PTSEX_{j}" for j in [19, 34, 18, 20, 22, 17, 46]],
    "Intrusiveness":  [f"PTSEX_{j}" for j in [1, 2, 3, 5, 40]],
    "Shame":          [f"PTSEX_{j}" for j in [14, 15, 13, 12, 10]],
    "Pleasing":       [f"PTSEX_{j}" for j in [42, 41, 43, 45]],
    "Interpersonal":  [f"PTSEX_{j}" for j in [33, 32]],
    "Hypervigilance": [f"PTSEX_{j}" for j in [27, 23]],
}

PTSEX_ITEMS = [item for sublist in FACTORS.values() for item in sublist]

# Ordered subscale names matching Table 3 and Figure 2
PARALLEL_SUBSCALE_ORDER = [
    "Shame",
    "Pleasing",
    "Dissociation",
    "Hypervigilance",
    "Interpersonal",
    "Intrusiveness",
]

SUBSCALE_LABELS = {
    "Shame": "Shame/guilt",
    "Pleasing": "Pleasing the other",
    "Dissociation": "Dissociation",
    "Hypervigilance": "Hypervigilance",
    "Interpersonal": "Interpersonal distress",
    "Intrusiveness": "Intrusiveness",
}

# 10-item Attitudes Toward Women's Genitals Scale (ATWGS; Herbenick, 2009)
ATWGS_ITEMS = [f"Q151_{j}" for j in range(1, 11)]

# 4 negatively worded ATWGS items reverse-scored on the 1-4 scale (5 - x)
ATWGS_NEG = ["Q151_3", "Q151_5", "Q151_6", "Q151_8"]
ATWGS_POS = [c for c in ATWGS_ITEMS if c not in ATWGS_NEG]
