#!/usr/bin/env python3
"""
12_fig8_conceptual.py — Conceptual schematic of thermodynamic intensification.

Two panels show the same circulation-moisture pattern under a weak (early,
1960-1991) vs enhanced/warmer (late, 1992-2024) background: pattern frequency
is unchanged while pattern-conditioned precipitation rises (thermodynamic /
Clausius-Clapeyron), not the dynamic/frequency channel. The inset bar uses
verified decomposition numbers.

Output: figures/fig8_conceptual.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
from matplotlib.patches import FancyArrow, Ellipse, FancyBboxPatch
import matplotlib.patheffects as pe
import config

FIG = config.FIGDIR
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 11.5})

# verified decomposition numbers (combined CNN, 1960-1991 vs 1992-2024)
FP_E, FP_L = 0.0678, 0.0675          # pattern frequency f_p  (unchanged)
MUP_E, MUP_L = 6.99, 7.55            # pattern-day mean precip mu_p (mm/d, up)
COLD, WARM = "#d6e6f2", "#f6e2d6"    # background tints


def draw_panel(ax, title, bg, n_drops, drop_size, mois_w, temp_lbl, rain_lbl):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 10, 10, color=bg, zorder=0))
    # basin ellipse (stylized Yellow River middle+lower)
    basin = Ellipse((5.2, 4.3), 4.6, 2.1, angle=-18, fill=False,
                    edgecolor="#7a5c3e", lw=2.2, zorder=2)
    ax.add_patch(basin)
    ax.text(5.2, 3.0, "Yellow R.\nmiddle+lower", ha="center", va="top",
            fontsize=9.2, color="#7a5c3e", zorder=2)
    # circulation pattern (Z500 low) — same in both panels
    for r, a in [(0.9, 0.35), (0.6, 0.6), (0.3, 0.9)]:
        ax.add_patch(plt.Circle((4.6, 6.0), r, fill=False, edgecolor="#2c6fbf",
                                lw=1.6, alpha=a, zorder=3))
    ax.text(4.6, 6.0, "L", ha="center", va="center", fontsize=12.6, fontweight="bold",
            color="#2c6fbf", zorder=3)
    # moisture inflow arrow from the SE (EASM) — thickness encodes moisture
    ax.add_patch(FancyArrow(8.6, 1.4, -2.7, 3.0, width=mois_w, head_width=mois_w*4,
                            length_includes_head=True, color="#1a7d8a", alpha=0.75,
                            zorder=2))
    ax.text(8.9, 1.0, "EASM moisture\ninflow", ha="right", va="top", fontsize=8.6,
            color="#1a7d8a", zorder=2)
    # rain over the basin
    rng = np.random.RandomState(7)
    xs = rng.uniform(3.2, 7.2, n_drops); ys = rng.uniform(3.6, 5.0, n_drops)
    ax.scatter(xs, ys, s=drop_size, c="#2c3e9a", alpha=0.7, zorder=4, marker=r"$\bullet$")
    ax.text(5.2, 5.7, rain_lbl, ha="center", fontsize=10.3, color="#2c3e9a",
            fontweight="bold", zorder=4,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    # background-state label
    ax.text(0.3, 9.6, title, ha="left", va="top", fontsize=12.1, fontweight="bold")
    ax.text(0.3, 8.9, temp_lbl, ha="left", va="top", fontsize=9.8, color="#444")


def draw_inset(fig):
    """Mini bar: f_p unchanged vs mu_p up (the thermodynamic point)."""
    ax = fig.add_axes([0.30, 0.02, 0.40, 0.085])
    x = np.array([0, 1]); w = 0.36
    fp = [FP_E, FP_L]; mu = [MUP_E, MUP_L]
    # normalize each pair for side-by-side comparison on one axis (two groups)
    ax.bar(x - w/2, [v*100 for v in fp], w, color="#e67e22", alpha=0.85, label="pattern freq ×100")
    ax.bar(x + w/2, mu, w, color="#2c3e9a", alpha=0.85, label="pattern-day μ_p (mm/d)")
    ax.set_xticks(x); ax.set_xticklabels(["early\n1960–91", "late\n1992–24"], fontsize=9.2)
    for xi, v in zip(x - w/2, [v*100 for v in fp]):
        ax.text(xi, v + 0.2, f"{v/100:.1%}", ha="center", fontsize=8)
    for xi, v in zip(x + w/2, mu):
        ax.text(xi, v + 0.2, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 9.5); ax.set_yticks([])
    ax.set_title("pattern frequency ≈ unchanged  ·  pattern-conditioned intensity ↑  (thermodynamic)",
                 fontsize=9.8, color="#333")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)


def main():
    fig = plt.figure(figsize=(12, 6.4))
    axL = fig.add_axes([0.03, 0.20, 0.45, 0.74])
    axR = fig.add_axes([0.52, 0.20, 0.45, 0.74])
    draw_panel(axL, "(a) Early 1960–1991 — weaker background", COLD,
               n_drops=26, drop_size=20, mois_w=0.10,
               temp_lbl="cooler background, less column moisture",
               rain_lbl="μ$_p$ = 6.99 mm/d")
    draw_panel(axR, "(b) Late 1992–2024 — enhanced (warmer) background", WARM,
               n_drops=58, drop_size=42, mois_w=0.20,
               temp_lbl="+ΔT  →  +column moisture (Clausius–Clapeyron)",
               rain_lbl="μ$_p$ = 7.55 mm/d")
    # big arrow between panels
    fig.patches.append(FancyArrow(0.485, 0.55, 0.035, 0.0, transform=fig.transFigure,
                                  width=0.012, head_width=0.04, head_length=0.012,
                                  length_includes_head=True, color="#c0392b", alpha=0.8))
    draw_inset(fig)
    fig.text(0.5, 0.985, "Fig 8 — Conceptual model: the SAME circulation–moisture pattern produces more "
             "precipitation under a warmer background (thermodynamic), not more often (dynamic).",
             ha="center", va="top", fontsize=13.2, fontweight="bold")
    p = os.path.join(FIG, "fig8_conceptual.png")
    save_figure(plt.gcf(), p, bbox_inches="tight"); plt.close(); print(f"[save] {p}")


if __name__ == "__main__":
    main()
