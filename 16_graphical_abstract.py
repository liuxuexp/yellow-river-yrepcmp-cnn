#!/usr/bin/env python3
"""
16_graphical_abstract.py — Graphical abstract (Elsevier TOC/search image).

Left  — concept: the same circulation-moisture pattern produces more rain under
        a warmer background (mu_p 6.99->7.55 mm/d).
Right — the window-dependent decomposition: over 1960-2024 the intensity/
        thermodynamic term is robust (CI [+0.0006,+0.0053]); over 1979-2024 the
        frequency/dynamic term is robust (CI [+0.0007,+0.0051]). Both CIs are
        year-block bootstrap.
Foot  — one-line takeaway + key verified numbers + method.

Reads: out/fig5d_window_dependence_data.csv
Output: figures/graphical_abstract.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
from matplotlib.patches import FancyArrow, Ellipse, FancyBboxPatch
import matplotlib.patheffects as pe
import config

OUT = config.OUT
FIG = config.FIGDIR
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 220, "font.size": 12.6})
COLD, WARM = "#dce8f2", "#f6dcc6"
DYN, THM = "#c0392b", "#1a7d1a"


def concept(ax, title, bg, ndrop, dropsize, mw, mulbl):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 10, 10, color=bg))
    ax.add_patch(Ellipse((5.0, 4.2), 4.6, 2.0, angle=-18, fill=False, edgecolor="#7a5c3e", lw=2.2))
    for r, a in [(0.85, 0.35), (0.55, 0.65), (0.28, 0.9)]:
        ax.add_patch(plt.Circle((4.5, 6.0), r, fill=False, edgecolor="#2c6fbf", lw=1.5, alpha=a))
    ax.text(4.5, 6.0, "L", ha="center", va="center", fontsize=13.8, fontweight="bold", color="#2c6fbf")
    ax.add_patch(FancyArrow(8.4, 1.5, -2.6, 2.9, width=mw, head_width=mw*4.5,
                            length_includes_head=True, color="#1a7d8a", alpha=0.8))
    rng = np.random.RandomState(7)
    ax.scatter(rng.uniform(3.2, 6.8, ndrop), rng.uniform(3.5, 4.9, ndrop),
               s=dropsize, c="#2c3e9a", alpha=0.75, marker=r"$\bullet$")
    ax.text(5.0, 5.6, mulbl, ha="center", fontsize=13.8, color="#2c3e9a", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.text(0.3, 9.6, title, ha="left", va="top", fontsize=13.2, fontweight="bold")


def main():
    dat = pd.read_csv(os.path.join(OUT, "fig5d_window_dependence_data.csv"))
    def get(win, term):
        r = dat[(dat.window == win) & (dat.term == term)].iloc[0]
        return r.point, r.ci_low, r.ci_high
    w60 = "1960–2024 (1960–91 vs 1992–24)"
    w79 = "1979–2024 (1979–2000 vs 2001–24)"

    fig = plt.figure(figsize=(13.5, 8.4))
    fig.text(0.5, 0.965, "Why is extreme precipitation intensifying over the middle & lower Yellow River?",
             ha="center", va="top", fontsize=17.8, fontweight="bold")
    fig.text(0.5, 0.928, "the dynamic–thermodynamic answer is time-window-dependent",
             ha="center", va="top", fontsize=13.8, color="#444", style="italic")

    # LEFT: concept (the thermodynamic mechanism over 1960-2024)
    axL = fig.add_axes([0.025, 0.34, 0.235, 0.55])
    axR = fig.add_axes([0.275, 0.34, 0.235, 0.55])
    concept(axL, "Early (1960–91)", COLD, 22, 26, 0.09, r"$\mu_p$ = 6.99 mm/d")
    concept(axR, "Late (1992–2024)", WARM, 56, 52, 0.20, r"$\mu_p$ = 7.55 mm/d")
    fig.patches.append(FancyArrow(0.247, 0.60, 0.032, 0.0, transform=fig.transFigure,
                                  width=0.012, head_width=0.04, head_length=0.012,
                                  length_includes_head=True, color="#c0392b", alpha=0.85))
    fig.text(0.263, 0.66, "warmer\nbackground", ha="center", va="center", fontsize=9.8,
             color="#c0392b", fontweight="bold")
    fig.text(0.267, 0.31, "1960–2024: more rain per pattern day\n(thermodynamic, μ_p rises)",
             ha="center", va="top", fontsize=10.6, color="#1a7d1a", fontweight="bold")

    # RIGHT: window-dependent decomposition bars
    axB = fig.add_axes([0.58, 0.36, 0.40, 0.52])
    xs = [0, 1]; width = 0.32
    for i, (win, lab) in enumerate([(w60, "1960–2024"), (w79, "1979–2024")]):
        fp, flo, fhi = get(win, "freq_dynamic")
        ip, ilo, ihi = get(win, "inten_thermo")
        axB.bar(i - width/2, fp, width, color=DYN, edgecolor="white", zorder=3)
        axB.errorbar(i - width/2, fp, [[fp-flo], [fhi-fp]], fmt="none", ecolor="k", capsize=4, lw=1.3, zorder=4)
        axB.bar(i + width/2, ip, width, color=THM, edgecolor="white", zorder=3)
        axB.errorbar(i + width/2, ip, [[ip-ilo], [ihi-ip]], fmt="none", ecolor="k", capsize=4, lw=1.3, zorder=4)
        axB.text(i - width/2, max(fhi, 0) + 0.0007, "★" if (flo > 0 or fhi < 0) else "ns",
                 ha="center", fontsize=12.6, color=DYN, fontweight="bold")
        axB.text(i + width/2, max(ihi, 0) + 0.0007, "★" if (ilo > 0 or ihi < 0) else "ns",
                 ha="center", fontsize=12.6, color=THM, fontweight="bold")
    axB.axhline(0, color="k", lw=0.7)
    axB.set_xticks(xs); axB.set_xticklabels(["1960–2024\n→ thermodynamic", "1979–2024\n→ dynamic"], fontsize=11.5)
    axB.set_ylabel("ΔPr(E) annual term", fontsize=11.5)
    axB.set_title("which term is robustly signed depends on the window\n(year-block bootstrap 95% CI; ★ = CI excludes 0)",
                  fontsize=11.5)
    axB.set_ylim(-0.0015, 0.0066)
    axB.text(0.98, 0.97, "■ dynamic (freq)", transform=axB.transAxes, color=DYN, fontsize=10.3, ha="right", va="top")
    axB.text(0.98, 0.90, "■ thermodynamic (inten)", transform=axB.transAxes, color=THM, fontsize=10.3, ha="right", va="top")
    for sp in ["top", "right"]:
        axB.spines[sp].set_visible(False)

    # footnote
    box = FancyBboxPatch((0.025, 0.02), 0.95, 0.205, boxstyle="round,pad=0.012",
                         transform=fig.transFigure, facecolor="#f6f8fa", edgecolor="#bbb", lw=1)
    fig.patches.append(box)
    fig.text(0.05, 0.185, "Result", fontsize=13.8, fontweight="bold", color="#c0392b")
    fig.text(0.05, 0.135,
             "Over 1960–2024 the intensification is thermodynamic — pattern-conditioned extreme-event probability rose 0.250→0.287\n"
             "(ΔPr(E) annual intensity +0.0043, year-block 95% CI [+0.0006,+0.0054], positive in all 8 CNN seeds), pattern frequency flat.\n"
             "Over 1979–2024 the OPPOSITE term is robust — the frequency/dynamic term (CI [+0.0007,+0.0051]) — because pattern frequency\n"
             "dipped in 1979–2000 and recovered by 2001–2024. The robust sign is window-dependent; point estimates need uncertainty quantification.",
             fontsize=11.3, va="top")
    fig.text(0.05, 0.032,
             "Method: 5-channel circulation–moisture CNN (PR-AUC 0.460) on NCEP/NCAR-R1 + CHM_PRE, interpreted with LRP + a "
             "closed-form frequency/intensity decomposition with year-block + 8-seed uncertainty.",
             fontsize=10.1, va="top", color="#555")

    p = os.path.join(FIG, "graphical_abstract.png")
    save_figure(plt.gcf(), p, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"[save] {p}")


if __name__ == "__main__":
    main()
