#!/usr/bin/env python3
"""09a_figS8.py — Fig S8: middle vs lower reach partition.

Applies the decomposition separately to each reach to test whether the robust
intensity finding holds per-reach and whether the mechanism/amplitude differ
(middle = Loess-Plateau orographic; lower = North-China-Plain moisture-transport).
Partition = basin-wide YR-EPCMP labels (P>=0.5), identical to the headline
decomposition, so reach results are directly comparable to the basin result.

Three panels:
  (a) Climatology — mean P̄ + reach-extreme days/yr, middle vs lower (lower is
      wetter).
  (b) ΔP̄ decomposition — freq/inten/inter for {middle,lower}×{annual,JAS}; lower
      reach dominates the basin change (largest in JAS).
  (c) ΔPr(E) intensity with year-block bootstrap CIs — only the BASIN-pooled
      annual intensity is robustly signed (CI excludes 0); each reach alone spans
      0 (halving the sample inflates CIs), with the lower reach carrying the
      larger point estimate.

Reads: out/reach_Pbar_1960_2024.csv, out/reach_decomposition_1960_2024.csv,
       out/reach_boot_1960_2024.csv,
       out_legacy46x82/decomposition_boot_full_1960_2024.csv
Output: figures/figS8.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import config

FIGDIR = config.FIGDIR
OUT    = config.OUT
OUT_LEG = config.OUT_LEGACY

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777")

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 9.1,
                     "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6})
TTL = dict(fontsize=9.1, fontweight="bold", pad=3)   # no loc= -- it overrides set_x at render
TITLE_X = -0.12   # nudge panel titles left (may extend past the plot's left edge)

MID, LOW = OI["orange"], OI["pink"]          # reach identity colors
FREQ, INTEN, INTER = OI["vermillion"], OI["blue"], OI["gray"]   # decomposition-term colors
ROBUST, SPANS0 = OI["green"], OI["gray"]


# --------------------------------------------------------------------------- #
def panel_clim(ax):
    """(a) Reach climatology: mean P̄ + reach-extreme days/yr (lower wetter)."""
    d = pd.read_csv(os.path.join(OUT, "reach_Pbar_1960_2024.csv"))
    n_yr = d.date.nunique() / 65.0
    x = np.arange(2)
    lbl = ["Middle\n(Loess Plateau)", "Lower\n(N-China Plain)"]
    pbar = [d.Pbar_middle.mean(), d.Pbar_lower.mean()]
    ext_per_yr = [d.reach_extreme_middle.sum() / 65.0, d.reach_extreme_lower.sum() / 65.0]

    ax2 = ax.twinx()
    b1 = ax.bar(x - 0.2, pbar, 0.38, color=[MID, LOW], alpha=0.9, edgecolor="white", lw=0.4)
    b2 = ax2.bar(x + 0.2, ext_per_yr, 0.38, color=[MID, LOW], alpha=0.45,
                 edgecolor="white", lw=0.4, hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=8.1)
    # mathtext bar for the macron (U+0304 mis-anchors in Noto CJK)
    ax.set_ylabel("Mean $\\bar{P}$ (mm d$^{-1}$)", fontsize=8.3)
    ax2.set_ylabel("Reach-extreme days yr$^{-1}$", fontsize=8.3, labelpad=2)
    ax.tick_params(labelsize=7.5); ax2.tick_params(labelsize=7.5)
    ax.set_ylim(0, max(pbar) * 1.35); ax2.set_ylim(0, max(ext_per_yr) * 1.4)
    for xi, v in zip(x - 0.2, pbar):
        ax.text(xi, v + max(pbar) * 0.03, f"{v:.2f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + 0.2, ext_per_yr):
        ax2.text(xi, v + max(ext_per_yr) * 0.04, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.legend([b1, b2], ["Mean P̄", "Extreme days yr$^{-1}$"], fontsize=6.3,
              loc="upper left", framealpha=0.9)
    for sp in ("top",):
        ax.spines[sp].set_visible(False); ax2.spines[sp].set_visible(False)
    ax.set_title("(a)  Reach climatology (1960–2024)\n     lower reach wetter (monsoon inflow)",
                 **TTL)
    ax.title.set_ha("left"); ax.title.set_x(TITLE_X)


def panel_dPbar(ax, d):
    """(b) ΔP̄ decomposition by reach: stacked freq/inten/inter, {mid,low}×{annual,JAS}."""
    cases = [("dPbar_middle", "Mid\nannual", MID), ("dPbar_middle_JAS", "Mid\nJAS", MID),
             ("dPbar_lower", "Low\nannual", LOW), ("dPbar_lower_JAS", "Low\nJAS", LOW)]
    x = np.arange(len(cases))
    for i, (case, lbl, col) in enumerate(cases):
        f = float(d[(d.case == case) & (d.term == "freq")].value.iloc[0])
        n = float(d[(d.case == case) & (d.term == "inten")].value.iloc[0])
        it = float(d[(d.case == case) & (d.term == "inter")].value.iloc[0])
        hatch = "//" if "JAS" in case else None
        ax.bar(i, f, 0.62, color=FREQ, alpha=0.85, hatch=hatch, edgecolor="white", lw=0.3)
        ax.bar(i, n, 0.62, bottom=f, color=INTEN, alpha=0.85, hatch=hatch, edgecolor="white", lw=0.3)
        ax.bar(i, it, 0.62, bottom=f + n, color=INTER, alpha=0.7, hatch=hatch, edgecolor="white", lw=0.3)
        tot = f + n + it
        ax.text(i, tot + (0.003 if tot >= 0 else -0.009), f"Δ={tot:+.3f}",
                ha="center", fontsize=7.5, fontweight="bold")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-0.035, 0.095)
    ax.set_yticks(np.arange(-0.03, 0.0901, 0.03))
    ax.set_xticks(x); ax.set_xticklabels([c[1] for c in cases], fontsize=7.8)
    # mathtext bar for the macron
    ax.set_ylabel("Δ$\\bar{P}$ contribution (mm d$^{-1}$)", fontsize=8.3, labelpad=-2)
    ax.tick_params(labelsize=7.5)
    ax.legend([Patch(facecolor=FREQ, alpha=0.85), Patch(facecolor=INTEN, alpha=0.85),
               Patch(facecolor=INTER, alpha=0.7)],
              ["Frequency term", "Intensity term", "Interaction"], fontsize=6.8,
              loc="upper left", framealpha=0.9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # Cleaner single subtitle line; "largest in JAS" is backed by the bars
    # themselves (low-JAS Δ=+0.083 is the tallest stack). Mathtext bar for the
    # macron (U+0304 mis-anchors in Noto CJK); Δ stays unicode to match the
    # (a)/(c) title glyphs.
    ax.set_title("(b)  Δ$\\mathbf{\\bar{P}}$ Decomposition by reach\n     lower reach dominates, largest in JAS",
                 **TTL)
    ax.title.set_ha("left"); ax.title.set_x(TITLE_X)


def panel_dPrE_boot(ax, d, boot):
    """(c) ΔPr(E) intensity term with year-block bootstrap CIs.

    Basin-pooled (robust, CI excludes 0) vs the less-robust per-reach partitions
    (each spans 0 because halving the sample inflates CIs). Lower reach carries
    the larger point estimate."""
    bp = pd.read_csv(os.path.join(OUT_LEG, "decomposition_boot_full_1960_2024.csv"))
    basin = bp[(bp.case == "dPrE_annual") & (bp.term == "inten") & (bp.bootstrap == "year")].iloc[0]
    items = [("middle", "annual"), ("middle", "JAS"), ("lower", "annual"), ("lower", "JAS")]
    casekey = {"middle": {"annual": "dPrE_middle", "JAS": "dPrE_middle_JAS"},
               "lower": {"annual": "dPrE_lower", "JAS": "dPrE_lower_JAS"}}

    yvals, labels = [], []
    # basin pooled (top) — robust reference
    y = 0.0
    ax.errorbar(basin.point, y,
                yerr=[[basin.point - basin.ci_low], [basin.ci_high - basin.point]],
                fmt="D", color=ROBUST, ecolor=ROBUST, capsize=3, ms=6, lw=1.8, zorder=4)
    labels.append("Basin pooled\n(robust)"); yvals.append(y); y += 0.85
    for reach, seas in reversed(items):       # lower-annual nearest basin
        case = casekey[reach][seas]
        point = float(d[(d.case == case) & (d.term == "inten")].value.iloc[0])
        br = boot[(boot.reach == reach) & (boot.case == case)].iloc[0]
        robust = bool(br.ci_excludes_0)
        col = ROBUST if robust else SPANS0
        mk = "s" if seas == "JAS" else "o"
        ax.errorbar(point, y, yerr=[[point - br.ci_low], [br.ci_high - point]],
                    fmt=mk, color=col, ecolor=col, capsize=3, ms=5.5, lw=1.5, zorder=3)
        labels.append(f"{reach.capitalize()} {seas}"); yvals.append(y); y += 0.7
    ax.set_xlim(0, 0.021)
    ax.set_xticks([0, 0.005, 0.010, 0.015, 0.020])
    ax.set_yticks(yvals); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.7, max(yvals) + 0.7)
    ax.invert_yaxis()
    ax.set_xlabel("ΔPr(E) Intensity term (pattern-conditioned)", fontsize=8.3)
    ax.tick_params(labelsize=7.5)
    ax.legend([Line2D([0], [0], color=ROBUST, lw=1.8, marker="D"),
               Line2D([0], [0], color=SPANS0, lw=1.8, marker="o"),
               Line2D([0], [0], color=SPANS0, lw=1.8, marker="s")],
              ["CI excludes 0", "Spans 0 · annual", "Spans 0 · JAS"], fontsize=6.5,
              loc="upper right", framealpha=0.9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("(c)  ΔPr(E) Intensity, year-block bootstrap\n     reaches alone span 0; lower carries more",
                 **TTL)
    ax.title.set_ha("left"); ax.title.set_x(TITLE_X)


def main():
    d = pd.read_csv(os.path.join(OUT, "reach_decomposition_1960_2024.csv"))
    boot = pd.read_csv(os.path.join(OUT, "reach_boot_1960_2024.csv"))

    # Two-row layout: (a)/(b) on top, (c) centred below and slightly wider to host
    # its categorical y-labels + in-axes legend. The a–b gutter keeps the twin
    # right ylabel of (a) clear of (b)'s left ylabel.
    fig = plt.figure(figsize=(6.4, 5.5))
    ax_a = fig.add_axes([0.070, 0.575, 0.360, 0.349])
    ax_b = fig.add_axes([0.570, 0.575, 0.360, 0.349])
    ax_c = fig.add_axes([0.300, 0.085, 0.400, 0.349])   # centred on row 2

    panel_clim(ax_a)
    panel_dPbar(ax_b, d)
    panel_dPrE_boot(ax_c, d, boot)

    out_path = os.path.join(FIGDIR, "figS8.png")
    save_figure(fig, out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[figS8] saved {out_path} ({os.path.getsize(out_path):,} bytes)")

    # Key numbers echo (CSV-grounded)
    def gv(case, term):
        return float(d[(d.case == case) & (d.term == term)].value.iloc[0])
    print("       === Fig S8 key numbers ===")
    for case, nm in [("dPrE_middle", "middle"), ("dPrE_lower", "lower")]:
        inten = gv(case, "inten"); freq = gv(case, "freq"); delta = gv(case, "delta")
        sh = float(d[(d.case == case) & (d.term == "thermo_share_pct")].value.iloc[0])
        print(f"       ΔPr(E) {nm:6s}: inten={inten:+.4f}  freq={freq:+.4f}  "
              f"delta={delta:+.4f}  thermo_share={sh:.0f}%")
    print("       Per-reach year-block bootstrap CIs all span 0 (sample halved); "
          "only BASIN-pooled is robust.")


if __name__ == "__main__":
    main()
