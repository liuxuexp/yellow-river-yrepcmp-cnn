#!/usr/bin/env python3
"""09a_figS3.py — Fig S3: temporal-block cross-validation.

The temporal-block split (train ≤2005 / val 2006–2014 / test ≥2015) tests whether
the CNN generalizes in time rather than memorizing the calendar. PR-AUC of four
models (combined/circulation/moisture/month-only) is shown across three regimes:
  - random stratified test (interleaved years)
  - temporal-block val (2006–2014)
  - temporal-block test (≥2015, future block)

Reads: out/model_metrics.csv
Output: figures/figS3.png
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

# Okabe-Ito palette (color-blind safe)
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777")

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 9.1,
                     "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6})
TTL = dict(fontsize=9.1, fontweight="bold", loc="left", pad=3)

# Model names and display labels
MODELS = ["combined", "circulation", "moisture", "month_only"]
MODEL_LABELS = ["Combined\n(main)", "Circulation", "Moisture", "Month-only\nbaseline"]
MODEL_COLORS = [OI["blue"], OI["orange"], OI["green"], OI["gray"]]

# Three evaluation regimes
SPLITS = [
    ("test", "Random test (stratified)"),
    ("tb_val", "Temporal-block val (2006–2014)"),
    ("tb_test", "Temporal-block test (≥2015)")
]
SPLIT_COLORS = [OI["gray"], OI["sky"], OI["blue"]]


def main():
    # Load metrics
    m = pd.read_csv(os.path.join(OUT_LEG, "model_metrics.csv"))

    # Extract PR-AUC values for each model × split
    pr_auc = {}
    for model in MODELS:
        pr_auc[model] = {}
        for split, _ in SPLITS:
            row = m[(m.model == model) & (m.split == split)]
            if len(row) > 0:
                pr_auc[model][split] = float(row.pr_auc.iloc[0])
            else:
                pr_auc[model][split] = np.nan

    # Get base rate (climatology) from tb_test for combined model
    tb = m[(m.model == "combined") & (m.split == "tb_test")]
    if len(tb) > 0:
        base_rate = tb.n_pos.iloc[0] / tb.n.iloc[0]
    else:
        base_rate = np.nan

    # Create figure (tightened figsize for single-panel bar chart)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    n_model = len(MODELS)
    n_split = len(SPLITS)
    width = 0.22
    x = np.arange(n_model)

    # Anti-collision for the adjacent tb_val/tb_test labels (bar centres are only
    # 0.22 apart but a "0.xx" label is wider): when the two tb bars are within
    # 0.045 in height, keep the LOWER bar's label at the standard offset and place
    # the HIGHER bar's label at the fixed y = 0.71 tier, so every label stays
    # above its bar in one uniform style. tb_raise maps (model, split) -> absolute
    # y for the raised label.
    tb_raise = {}
    for mdl in MODELS:
        v_val, v_test = pr_auc[mdl]["tb_val"], pr_auc[mdl]["tb_test"]
        if np.isfinite(v_val) and np.isfinite(v_test) and abs(v_val - v_test) <= 0.045:
            v_hi = max(v_val, v_test)
            higher = "tb_val" if v_val >= v_test else "tb_test"
            tb_raise[(mdl, higher)] = max(v_hi + 0.01, 0.71)

    # Plot grouped bars
    label_objs = {}   # (model, split) -> Text, for the post-draw de-collision pass
    for j, (split, lbl) in enumerate(SPLITS):
        vals = [pr_auc[model][split] for model in MODELS]
        offs = (j - (n_split - 1) / 2) * width
        bars = ax.bar(x + offs, vals, width, label=lbl, color=SPLIT_COLORS[j],
                      edgecolor="white", linewidth=0.6)
        # Value labels: all identical style above each bar; the higher label of a
        # near-equal tb pair is anchored at the precomputed raised y tier.
        for b, v, mdl in zip(bars, vals, MODELS):
            if np.isfinite(v):
                y = tb_raise.get((mdl, split), v + 0.015)
                label_objs[(mdl, split)] = ax.text(
                    b.get_x() + b.get_width() / 2, y, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#222")

    # Post-draw de-collision: with the higher label pinned at y = 0.71 the lower
    # label at its standard offset would touch it. Drop the lower label just
    # enough to leave a vertical gap; it still sits above its own bar.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    GAP = 0.016   # data units
    for mdl in MODELS:
        hi_split = next((s for (m, s) in tb_raise if m == mdl), None)
        if hi_split is None:
            continue
        lo_split = "tb_test" if hi_split == "tb_val" else "tb_val"
        thi, tlo = label_objs[(mdl, hi_split)], label_objs[(mdl, lo_split)]
        bhi = thi.get_window_extent(rend); blo = tlo.get_window_extent(rend)
        hi_bot = inv.transform((0, bhi.y0))[1]
        lo_top = inv.transform((0, blo.y1))[1]
        v_lo = pr_auc[mdl][lo_split]
        if lo_top > hi_bot - GAP:
            new_y = max(hi_bot - GAP - (lo_top - tlo.get_position()[1]), v_lo)
            tlo.set_position((tlo.get_position()[0], new_y))

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=8.1)
    ax.set_ylabel("PR-AUC  (positive = YREP day)", fontsize=8.6)

    # Y-axis (zero-baselined for honesty); headroom so the upper-right legend
    # clears the tallest bar label
    ax.set_ylim(0, 1.08)
    ax.axhline(0, color=OI["black"], lw=0.7)

    # Climatology reference (honest chance level)
    if np.isfinite(base_rate):
        ax.axhline(base_rate, ls="--", lw=0.9, color=OI["vermillion"],
                   label=f"Climatology base rate ({base_rate:.3f})")

    # Legend inside the axes at upper right, 2 columns
    ax.legend(loc="upper right", fontsize=5.6, framealpha=0.92, handlelength=1.1,
              ncol=2, columnspacing=1.0)

    # Grid and spines
    ax.grid(axis="y", ls=":", alpha=0.35)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # No panel title; keep only the interpretive caption. Numbers come from the
    # plotted values.
    ax.text(-0.16, 1.009,
            f"Combined: future block ({pr_auc['combined']['tb_test']:.2f}) > "
            f"random test ({pr_auc['combined']['test']:.2f}) → not calendar memorization",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="left")

    # Tight layout with minimal margins; bbox_inches='tight' trims outer whitespace
    plt.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.12)
    out = os.path.join(FIGDIR, "figS3.png")
    save_figure(fig, out, dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"[figS3] saved {out}")

    # Echo numeric table
    print("\n       PR-AUC by model × split:")
    print("                    random test    tb_val (2006-14)    tb_test (≥2015)")
    for model, label in zip(MODELS, MODEL_LABELS):
        vals = [pr_auc[model][split] for split, _ in SPLITS]
        print(f"    {label:18s}     {vals[0]:.3f}              {vals[1]:.3f}              {vals[2]:.3f}")
    print(f"\n       climatology base rate (tb_test): {base_rate:.4f}")
    print(f"       combined: tb_test > random test by {pr_auc['combined']['tb_test'] - pr_auc['combined']['test']:+.3f}")


if __name__ == "__main__":
    main()
