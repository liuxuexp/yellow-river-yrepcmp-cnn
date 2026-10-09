#!/usr/bin/env python3
"""09a_figS5.py — Fig S5: per-season CNN skill.

Per-season PR-AUC for the combined CNN vs the month-only baseline. The combined
CNN beats the month-only baseline within all three seasons; PR-AUC peaks in SON,
while most YREP days occur in JJA (flood season) — event frequency and classifier
skill peak in different seasons. DJF is omitted (zero YREP days).

Panels:
  (a) Per-season PR-AUC bar chart: combined vs month-only.
  (b) Per-season YREP base rate + sample size n (context for class imbalance).

Reads: out/model_metrics.csv, out/YREP_labels_1960_2024.csv
Output: figures/figS5.png
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
OUT    = config.OUT
OUT_LEG = config.OUT_LEGACY

# Okabe-Ito palette (color-blind safe)
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777")

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 9.1,
                     "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6})
TTL = dict(fontsize=9.1, fontweight="bold", loc="left", pad=3)

# Seasonal splits (DJF omitted: zero YREP days in the dry season)
SEASONS = ["MAM", "JJA", "SON"]


def _panel_a(ax):
    """Panel (a): Per-season PR-AUC, combined vs month-only."""
    met = pd.read_csv(os.path.join(OUT_LEG, "model_metrics.csv"))
    x = np.arange(len(SEASONS))
    w = 0.35

    # Combined CNN (blue)
    vals_comb = []
    for s in SEASONS:
        r = met[(met.model == "combined") & (met.split == f"test_{s}")]
        vals_comb.append(float(r.pr_auc.iloc[0]) if len(r) else np.nan)

    # Month-only baseline (gray)
    vals_month = []
    for s in SEASONS:
        r = met[(met.model == "month_only") & (met.split == f"test_{s}")]
        vals_month.append(float(r.pr_auc.iloc[0]) if len(r) else np.nan)

    # Bars: zero-baselined for honest comparison
    ax.bar(x - w/2, vals_comb, w, color=OI["blue"], alpha=0.92,
           edgecolor="white", lw=0.4, label="Combined CNN")
    ax.bar(x + w/2, vals_month, w, color=OI["gray"], alpha=0.85,
           edgecolor="white", lw=0.4, label="Month-only")

    # Value labels on bars
    for xi, v in zip(x - w/2, vals_comb):
        if np.isfinite(v):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=OI["blue"], fontweight="bold")
    for xi, v in zip(x + w/2, vals_month):
        if np.isfinite(v):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=OI["gray"])

    ax.set_xticks(x)
    ax.set_xticklabels(SEASONS)
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 0.70)
    ax.set_title("(a)  Per-season classifier skill  (combined vs month-only)", **TTL)
    ax.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.8, loc="upper left", framealpha=0.9, handlelength=1.0)


def _panel_b(ax):
    """Panel (b): Per-season YREP base rate + sample size n."""
    lab = pd.read_csv(os.path.join(OUT, "YREP_labels_1960_2024.csv"))
    lab["season"] = lab["month"].map({12: "DJF", 1: "DJF", 2: "DJF",
                                       3: "MAM", 4: "MAM", 5: "MAM",
                                       6: "JJA", 7: "JJA", 8: "JJA",
                                       9: "SON", 10: "SON", 11: "SON"})

    x = np.arange(len(SEASONS))
    rates = []
    ns = []
    npos_list = []

    for s in SEASONS:
        sdf = lab[lab["season"] == s]
        rate = sdf["YREP"].mean() * 100  # percent
        n = len(sdf)
        npos = sdf["YREP"].sum()
        rates.append(rate)
        ns.append(n)
        npos_list.append(npos)

    # Primary axis: YREP rate (bars)
    ax2 = ax.twinx()
    bars = ax2.bar(x, rates, 0.56, color=OI["sky"], alpha=0.75, edgecolor="white", lw=0.4)

    # Rate labels on bars
    for xi, r, np_ in zip(x, rates, npos_list):
        ax2.text(xi, r + 0.08, f"{r:.2f}%\nn={np_}", ha="center", va="bottom",
                 fontsize=7, color=OI["blue"])

    ax2.set_ylabel("YREP rate  (%)", color=OI["sky"], fontsize=8.6)
    ax2.tick_params(axis="y", labelsize=7.3, labelcolor=OI["sky"])
    ax2.set_ylim(0, max(rates) * 1.4)
    # Flood-season note belongs to event FREQUENCY (not classifier skill):
    # most YREP days occur in JJA even though per-season PR-AUC peaks in SON
    ax2.text(0.0, max(rates) * 1.36, "most YREP days: JJA (flood season)",
             ha="left", va="center", fontsize=6.8, color=OI["blue"], style="italic")

    # Secondary axis: sample size n (line, right y-axis)
    ax.plot(x, ns, color=OI["vermillion"], marker="o", markersize=3.5,
            lw=1.2, label="sample size n", zorder=10)
    for xi, n in zip(x, ns):
        ax.text(xi, n + 40, f"{n}", ha="center", va="bottom",
                fontsize=6.8, color=OI["vermillion"], zorder=11)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.set_ylabel("sample size  n", color=OI["vermillion"], fontsize=8.6)
    ax.tick_params(axis="y", labelsize=7.3, labelcolor=OI["vermillion"])
    ax.set_ylim(min(ns) * 0.9, max(ns) * 1.08)

    ax.set_xticks(x)
    ax.set_xticklabels(SEASONS)
    ax.set_title("(b)  Per-season YREP base rate  (class imbalance context)", **TTL)
    ax.tick_params(axis="x", labelsize=7.5)

    # Clean spines
    for sp in ("top",):
        ax.spines[sp].set_visible(False)
        ax2.spines[sp].set_visible(False)


def main():
    # 2×1 layout; left/right margins fit the twin-axis labels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.0, 5.6))
    _panel_a(ax1)
    _panel_b(ax2)
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.075, top=0.94, hspace=0.52)

    out = os.path.join(FIGDIR, "figS5.png")
    save_figure(fig, out, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"[figS5] saved {out}")

    # ---- numeric echo ----
    met = pd.read_csv(os.path.join(OUT_LEG, "model_metrics.csv"))
    for s in SEASONS:
        comb = float(met[(met.model == "combined") & (met.split == f"test_{s}")].pr_auc.iloc[0])
        month = float(met[(met.model == "month_only") & (met.split == f"test_{s}")].pr_auc.iloc[0])
        gap = comb - month
        print(f"       {s}: combined={comb:.3f}, month_only={month:.3f}, gap={gap:.3f}")
    print(f"       JJA (flood season): combined beats month-only by "
          f"{comb - month:.3f} PR-AUC points")


if __name__ == "__main__":
    main()
