"""
Publication-quality figure generator for Figures 1, 2, and 3.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def make_figure1(out_dir: Path) -> Path:
    """Generate Figure 1: Simple Mediation Path Diagram."""
    fig, ax = plt.subplots(figsize=(15.71, 8.22), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.23)
    ax.axis("off")

    b_left = FancyBboxPatch(
        (0.3, 0.8), 2.8, 0.9,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor="white", edgecolor="black", linewidth=3.2,
    )
    ax.add_patch(b_left)
    ax.text(
        1.7, 1.25, "Childhood sexual abuse\nseverity (CTQ-SA)",
        ha="center", va="center", fontsize=23, color="black", linespacing=1.25,
    )

    b_top = FancyBboxPatch(
        (3.65, 3.75), 2.7, 0.9,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor="#E5E5E5", edgecolor="black", linewidth=3.2,
    )
    ax.add_patch(b_top)
    ax.text(
        5.0, 4.2, "Traumatized sexuality\n(PT-SEX)",
        ha="center", va="center", fontsize=23, color="black", linespacing=1.25,
    )

    b_right = FancyBboxPatch(
        (6.9, 0.7), 2.85, 1.1,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor="white", edgecolor="black", linewidth=3.2,
    )
    ax.add_patch(b_right)
    ax.text(
        8.325, 1.25, "Attitudes toward\nwomen’s genitals\n(ATWGS)",
        ha="center", va="center", fontsize=23, color="black", linespacing=1.25,
    )

    arr_a = FancyArrowPatch(
        (2.25, 1.75), (3.92, 3.68),
        arrowstyle="-|>,head_length=13,head_width=9", color="#1B365D", linewidth=4.2,
    )
    ax.add_patch(arr_a)
    ax.text(2.25, 2.88, "a = 0.066***", ha="center", va="center", fontsize=22, color="black")

    arr_b = FancyArrowPatch(
        (6.08, 3.68), (7.72, 1.82),
        arrowstyle="-|>,head_length=13,head_width=9", color="#1B365D", linewidth=4.2,
    )
    ax.add_patch(arr_b)
    ax.text(7.85, 2.88, "b = −0.369***", ha="center", va="center", fontsize=22, color="black")

    arr_c = FancyArrowPatch(
        (3.18, 1.25), (6.80, 1.25),
        arrowstyle="-|>,head_length=11,head_width=8",
        linestyle="--", color="#8C8C8C", linewidth=2.6,
    )
    ax.add_patch(arr_c)
    ax.text(5.0, 1.54, "c′ = 0.005", ha="center", va="center", fontsize=22, color="black")
    ax.text(5.0, 0.90, "(c = −0.019*)", ha="center", va="center", fontsize=22, color="black")

    ax.text(
        5.0, 0.28,
        "Indirect effect = −0.024, 95% CI [−0.036, −0.015]; standardized = −.19",
        ha="center", va="center", fontsize=21, color="black",
    )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path = Path(out_dir) / "Figure_1_Simple_Mediation.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)
    return out_path


def make_figure2(out_dir: Path) -> Path:
    """Generate Figure 2: Parallel Multiple Mediation Model of the Six PT-SEX Dimensions."""
    fig, ax = plt.subplots(figsize=(23.85, 16.77), dpi=100)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.44)
    ax.axis("off")

    b_left = FancyBboxPatch(
        (0.35, 4.25), 2.15, 1.0,
        boxstyle="round,pad=0.06,rounding_size=0.07",
        facecolor="white", edgecolor="black", linewidth=2.6,
    )
    ax.add_patch(b_left)
    ax.text(
        1.425, 4.75, "CSA severity\n(CTQ-SA)",
        ha="center", va="center", fontsize=22, fontweight="bold", color="black", linespacing=1.25,
    )

    b_right = FancyBboxPatch(
        (9.58, 4.25), 2.15, 1.0,
        boxstyle="round,pad=0.06,rounding_size=0.07",
        facecolor="white", edgecolor="black", linewidth=2.6,
    )
    ax.add_patch(b_right)
    ax.text(
        10.655, 4.75, "Attitudes toward\nwomen’s genitals\n(ATWGS)",
        ha="center", va="center", fontsize=22, fontweight="bold", color="black", linespacing=1.2,
    )

    mediators = [
        {"name": "Shame / guilt",          "y": 7.45, "sig": True,  "a": ".32***", "b": "−.32***", "ind": "indirect β = −.10 [−.17, −.05]"},
        {"name": "Pleasing the other",     "y": 6.22, "sig": True,  "a": ".21***", "b": "−.21**",  "ind": "indirect β = −.04 [−.09, −.01]"},
        {"name": "Dissociation",           "y": 5.00, "sig": False, "a": ".23***", "b": "−.07",    "ind": "indirect β = −.01 [−.06, .03]"},
        {"name": "Hypervigilance",         "y": 3.78, "sig": False, "a": ".30***", "b": "−.17",    "ind": "indirect β = −.05 [−.12, .01]"},
        {"name": "Interpersonal distress", "y": 2.55, "sig": False, "a": ".22***", "b": ".06",     "ind": "indirect β = .01 [−.03, .06]"},
        {"name": "Intrusiveness",          "y": 1.33, "sig": False, "a": ".45***", "b": ".06",     "ind": "indirect β = .03 [−.05, .11]"},
    ]

    for m in mediators:
        yc = m["y"]
        fc = "#E5E5E5" if m["sig"] else "white"
        lw = 2.6 if m["sig"] else 2.2
        box = FancyBboxPatch(
            (4.82, yc - 0.31), 2.40, 0.64,
            boxstyle="round,pad=0.05,rounding_size=0.07",
            facecolor=fc, edgecolor="black", linewidth=lw,
        )
        ax.add_patch(box)
        ax.text(6.02, yc + 0.01, m["name"], ha="center", va="center", fontsize=22, color="black")

        ind_col = "#13294B" if m["sig"] else "#555555"
        ind_wt = "bold" if m["sig"] else "normal"
        ax.text(6.02, yc - 0.52, m["ind"], ha="center", va="center", fontsize=18, color=ind_col, fontweight=ind_wt)

        arr_a = FancyArrowPatch(
            (2.56, 4.75), (4.76, yc + 0.01),
            arrowstyle="-|>,head_length=11,head_width=7", color="#222222", linewidth=2.2,
        )
        ax.add_patch(arr_a)
        t_pos = 0.72
        ax_x = 2.56 + t_pos * (4.76 - 2.56) + (0.08 if yc < 4.75 else -0.05)
        ax_y = 4.75 + t_pos * ((yc + 0.01) - 4.75) + (0.14 if yc >= 3.78 else 0.08)
        ax.text(
            ax_x, ax_y, m["a"], ha="center", va="center", fontsize=19, color="black",
            bbox=dict(boxstyle="square,pad=0.10", facecolor="white", edgecolor="none"),
        )

        if m["sig"]:
            arr_b = FancyArrowPatch(
                (7.28, yc + 0.05), (9.52, 4.75 + (0.06 if yc > 6.5 else 0.02)),
                arrowstyle="-|>,head_length=12,head_width=8", color="#13294B", linewidth=4.2,
            )
            ax.add_patch(arr_b)
            bx_x = 7.28 + 0.22 * (9.52 - 7.28)
            bx_y = (yc + 0.05) + 0.22 * (4.78 - (yc + 0.05))
            ax.text(
                bx_x, bx_y, m["b"], ha="center", va="center", fontsize=19, color="#13294B",
                bbox=dict(boxstyle="square,pad=0.10", facecolor="white", edgecolor="none"),
            )
        else:
            arr_b = FancyArrowPatch(
                (7.28, yc + 0.01), (9.52, 4.75 - (4.75 - yc) * 0.03),
                arrowstyle="-|>,head_length=10,head_width=6.5",
                linestyle="--", color="#999999", linewidth=2.2,
            )
            ax.add_patch(arr_b)
            bx_x = 7.28 + 0.20 * (9.52 - 7.28)
            bx_y = (yc + 0.01) + 0.20 * (4.75 - (yc + 0.01)) + 0.14
            ax.text(
                bx_x, bx_y, m["b"], ha="center", va="center", fontsize=18.5, color="#666666",
                bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none"),
            )

    ax.plot([1.425, 1.425, 10.655], [4.18, 0.56, 0.56], linestyle="--", color="#888888", linewidth=2.1)
    arr_direct = FancyArrowPatch(
        (10.655, 0.56), (10.655, 4.18),
        arrowstyle="-|>,head_length=10,head_width=7", linestyle="--", color="#888888", linewidth=2.1,
    )
    ax.add_patch(arr_direct)
    ax.text(
        6.04, 0.34,
        "Direct effect c′ = .02 (n.s.)     Total indirect β = −.17 [−.27, −.08]     R² (ATWGS) = .28",
        ha="center", va="center", fontsize=20, color="black",
    )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path = Path(out_dir) / "Figure_2_Parallel_Mediation.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)
    return out_path


def make_figure3(surv: pd.DataFrame, out_dir: Path, seed: int = 20261107) -> Path:
    """Generate Figure 3: Scatterplot of Traumatized Sexuality and Genital Attitudes."""
    pair = surv[["ptsex_80", "atwgs_80"]].dropna()
    x = pair["ptsex_80"].to_numpy()
    y = pair["atwgs_80"].to_numpy()

    rng = np.random.default_rng(seed)
    x_jit = x + rng.normal(0, 0.025, size=len(x))
    y_jit = y + rng.normal(0, 0.025, size=len(y))

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=200)
    ax.scatter(x_jit, y_jit, alpha=0.55, s=36, color="#1B365D", edgecolors="none")

    x_grid = np.linspace(1.0, 5.0, 200)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = intercept + slope * x_grid
    n = len(x)
    s_err = np.sqrt(np.sum((y - (intercept + slope * x)) ** 2) / (n - 2))
    ci = 1.97 * s_err * np.sqrt(1.0 / n + (x_grid - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    ax.plot(x_grid, y_hat, color="#1B365D", linewidth=2.4)
    ax.fill_between(x_grid, y_hat - ci, y_hat + ci, color="#1B365D", alpha=0.18)

    ax.set_xlabel("Traumatized Sexuality (PT-SEX; 1–5)", fontsize=13)
    ax.set_ylabel("Attitudes Toward Women’s Genitals (ATWGS; 1–4)", fontsize=13)
    ax.set_xlim(0.9, 5.1)
    ax.set_ylim(0.9, 4.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = Path(out_dir) / "Figure_3_Scatter_PTSEX_ATWGS.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def generate_all_figures(surv: pd.DataFrame, out_dir: Path):
    """Generate Figures 1, 2, and 3 into `out_dir`."""
    p1 = make_figure1(out_dir)
    p2 = make_figure2(out_dir)
    p3 = make_figure3(surv, out_dir)
    return {"figure1": p1, "figure2": p2, "figure3": p3}
