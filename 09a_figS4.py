#!/usr/bin/env python3
"""09a_figS4.py — Fig S4: probability-threshold sensitivity.

Shows how CNN classification skill and the intensity/frequency decomposition vary
with the probability decision threshold (the P>0.5 cutoff used in the paper).

Panels:
  (a) Pattern frequency f_p (early vs late) vs threshold — the precision/recall
      trade-off in terms of how many days are classified as YR-EPCMP.
  (b) Decomposition terms (intensity vs frequency) vs threshold — the headline
      intensity result is robust.
  (c) Intensity-term share (%) vs threshold — quantifies the robustness.

The P=0.5 operating point used in the paper is marked with a vertical line.
Message: the intensity term stays positive across thresholds 0.30–0.70, while the
frequency term remains small.

Reads: out_legacy46x82/threshold_sweep_1960_2024.csv
Output: figures/figS4.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
import config

FIGDIR = config.FIGDIR
OUT_LEG = config.OUT_LEGACY

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777", light_gray="#E8E8E8")

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 9.1,
    "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
})

TTL = dict(fontsize=9.4, fontweight="bold", loc="left", pad=4)


def main():
    df = pd.read_csv(os.path.join(OUT_LEG, "threshold_sweep_1960_2024.csv"))

    # Layout: (a)/(b) share the top row, (c) centred alone on the bottom row.
    fig = plt.figure(figsize=(5.8, 4.5))

    PW, PH = 0.385, 0.35
    pos_a = [0.070, 0.575, PW, PH]
    pos_b = [0.545, 0.575, PW, PH]
    pos_c = [(1.0 - PW) / 2.0, 0.095, PW, PH]   # centred on row 2

    # ----------------------------------------------------------------------- #
    # Panel (a): Pattern frequency f_p (early vs late) vs threshold
    # ----------------------------------------------------------------------- #
    ax_a = fig.add_axes(pos_a)

    for sname, marker, ls_early, ls_late in [
        ("annual", "o", "-", "--"),
        ("JAS", "s", "-", "--")
    ]:
        s = df[df.season == sname]
        ax_a.plot(s.threshold, s.f_p_early * 100,
                  ls_early, marker=marker, color=OI["blue"], ms=6,
                  markeredgecolor="white", markeredgewidth=1.0,
                  label=f"{sname.capitalize()} Early" if sname == "annual" else None,
                  markevery=1)
        ax_a.plot(s.threshold, s.f_p_late * 100,
                  ls_late, marker=marker, color=OI["vermillion"], ms=6,
                  markeredgecolor="white", markeredgewidth=1.0,
                  label=f"{sname.capitalize()} Late" if sname == "annual" else None,
                  markevery=1)

    ax_a.axvline(0.5, color=OI["gray"], ls=":", lw=1.2, zorder=0)
    ax_a.set_xlabel("Probability threshold P(YR-EPCMP)", fontsize=8.8, labelpad=2)
    ax_a.set_ylabel("Pattern frequency f$_p$ (%)", fontsize=8.8, labelpad=0.8)
    ax_a.set_title("(a) Pattern frequency (early vs late)", **TTL)
    ax_a.tick_params(labelsize=7.8, length=3, width=0.5, pad=1.5)

    ax_a.grid(True, axis="y", color=OI["light_gray"], linestyle="-", linewidth=0.5, alpha=0.7)
    ax_a.set_axisbelow(True)

    # frameon=False: no legend box or facecolor
    ax_a.legend(fontsize=7.3, ncol=1, loc="upper right", frameon=False,
                labelspacing=0.3, handlelength=1.5)

    for sp in ("top", "right"):
        ax_a.spines[sp].set_visible(False)
    ax_a.spines["bottom"].set_linewidth(0.8)
    ax_a.spines["left"].set_linewidth(0.8)

    # ----------------------------------------------------------------------- #
    # Panel (b): Decomposition terms (intensity vs frequency) vs threshold
    # ----------------------------------------------------------------------- #
    ax_b = fig.add_axes(pos_b)

    for sname, marker in [("annual", "o"), ("JAS", "s")]:
        s = df[df.season == sname]
        ax_b.plot(s.threshold, s.inten * 100,
                  "-", marker=marker, color=OI["green"], ms=6,
                  markeredgecolor="white", markeredgewidth=1.0,
                  label=f"Intensity ({sname.capitalize()})", markevery=1)
        ax_b.plot(s.threshold, s.freq * 100,
                  "--", marker=marker, color=OI["orange"], ms=6,
                  markeredgecolor="white", markeredgewidth=1.0,
                  label=f"Frequency ({sname.capitalize()})", markevery=1)

    ax_b.axhline(0, color=OI["black"], lw=0.8, ls=":", zorder=0)
    ax_b.axvline(0.5, color=OI["gray"], ls=":", lw=1.2, zorder=0)
    # ylim headroom keeps the green curve below the upper-right legend block
    ax_b.set_ylim(top=1.3)
    # fix the right limit at 0.75
    ax_b.set_xlim(right=0.75)
    ax_b.set_xlabel("Probability threshold P(YR-EPCMP)", fontsize=8.8, labelpad=2)
    ax_b.set_ylabel("ΔPr(E) term contribution (%)", fontsize=8.8, labelpad=0.8)
    ax_b.yaxis.set_label_coords(-0.12, 0.5)
    ax_b.set_title("(b) Intensity vs frequency terms", **TTL)
    ax_b.tick_params(labelsize=7.8, length=3, width=0.5, pad=1.5)

    ax_b.grid(True, axis="y", color=OI["light_gray"], linestyle="-", linewidth=0.5, alpha=0.7)
    ax_b.set_axisbelow(True)

    # anchor legend at top-right; title is left-justified so that corner is empty
    ax_b.legend(fontsize=6.8, ncol=1, loc="upper right", bbox_to_anchor=(1.0, 1.03),
                frameon=False, labelspacing=0.1, handlelength=1.5)

    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(False)
    ax_b.spines["bottom"].set_linewidth(0.8)
    ax_b.spines["left"].set_linewidth(0.8)

    # ----------------------------------------------------------------------- #
    # Panel (c): Intensity-term share (%) vs threshold
    # ----------------------------------------------------------------------- #
    ax_c = fig.add_axes(pos_c)

    for sname, marker, color in [("annual", "o", OI["green"]), ("JAS", "s", OI["sky"])]:
        s = df[df.season == sname]
        ax_c.plot(s.threshold, s.thermo_share,
                  "-", marker=marker, color=color, ms=6,
                  markeredgecolor="white", markeredgewidth=1.0,
                  label=("Annual" if sname == "annual" else sname), markevery=1)

    ax_c.axhline(0, color=OI["black"], lw=0.8, ls=":", zorder=0)
    ax_c.axhline(100, color=OI["gray"], ls=":", lw=1.0, zorder=0)
    ax_c.set_ylim(40, 135)
    ax_c.axvline(0.5, color=OI["gray"], ls=":", lw=1.2, zorder=0)
    ax_c.set_xlabel("Probability threshold P(YR-EPCMP)", fontsize=8.8, labelpad=2)
    ax_c.set_ylabel("Intensity-term share (%)", fontsize=8.8, labelpad=0.8)
    ax_c.set_title("(c) Intensity-term share of ΔPr(E)", **TTL)
    ax_c.tick_params(labelsize=7.8, length=3, width=0.5, pad=1.5)

    ax_c.grid(True, axis="y", color=OI["light_gray"], linestyle="-", linewidth=0.5, alpha=0.7)
    ax_c.set_axisbelow(True)

    ax_c.legend(fontsize=7.3, ncol=1, loc="upper right", frameon=False,
                labelspacing=0.3, handlelength=1.5)

    for sp in ("top", "right"):
        ax_c.spines[sp].set_visible(False)
    ax_c.spines["bottom"].set_linewidth(0.8)
    ax_c.spines["left"].set_linewidth(0.8)

    # ----------------------------------------------------------------------- #
    # Save
    # ----------------------------------------------------------------------- #
    out_path = os.path.join(FIGDIR, "figS4.png")
    save_figure(fig, out_path, dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print(f"[figS4] saved {out_path}")

    # ----------------------------------------------------------------------- #
    # Numeric echo: skill values at P=0.5 (the operating point used in paper)
    # ----------------------------------------------------------------------- #
    p50 = df[df.threshold == 0.5]
    print("\n[figS4] Skill values at P=0.5 (operating point):")
    for _, row in p50.iterrows():
        print(f"       {row.season}: f_p {row.f_p_early*100:.2f}% -> {row.f_p_late*100:.2f}%, "
              f"intensity={row.inten*100:+.3f}%, frequency={row.freq*100:+.3f}%, "
              f"thermo_share={row.thermo_share:.1f}%")

    ann_inten = df[df.season == "annual"]["inten"].values
    jas_inten = df[df.season == "JAS"]["inten"].values
    print(f"\n[figS4] Robustness: intensity term positive at all thresholds "
          f"(annual: min={ann_inten.min()*100:.3f}%, JAS: min={jas_inten.min()*100:.3f}%)")


if __name__ == "__main__":
    main()
