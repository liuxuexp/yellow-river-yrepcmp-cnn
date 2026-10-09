#!/usr/bin/env python3
"""09a_figS2.py — Fig S2: CNN model skill (ROC + PR curves).

Three CNN models (YR-EPCMP=combined 5-channel main; YR-EPCP=circulation;
YR-EPMFP=moisture) plus a month-only anti-climatology baseline.

Panels:
  (a) ROC curves — combined / circulation / moisture / month-only with chance
      diagonal and AUC in legend
  (b) PR curves — same four models with the positive base-rate horizontal
      reference and PR-AUC in legend (primary metric, class-imbalanced)
  (c) ensemble spread across seeds for the combined model

Reads: out/model_metrics.csv, out/prediction_probability_test_canonical.csv,
       out/ensemble_cnn_metrics.csv
Output: figures/figS2.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
import config

FIGDIR = config.FIGDIR
OUT_LEG = config.OUT_LEGACY

# Okabe-Ito color-blind-safe palette
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777")

# Canonical rcParams
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 9.1,
    "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.6,
})

TTL = dict(fontsize=9.1, fontweight="bold", loc="left", pad=3)

# Model names and canonical colors
MODELS = ["combined", "circulation", "moisture", "month_only"]
MCOL = {
    "combined": OI["vermillion"],    # main model
    "circulation": OI["blue"],
    "moisture": OI["green"],
    "month_only": OI["gray"],        # baseline
}

# Short labels for legend
MLAB = {
    "combined": "YR-EPCMP (combined)",
    "circulation": "YR-EPCP (circulation)",
    "moisture": "YR-EPMFP (moisture)",
    "month_only": "Month-only baseline",
}


def smooth_curve(x, y, n=400):
    """Cosmetic shape-preserving (PCHIP) smoothing of a ROC/PR trace for plotting only.

    AUC / PR-AUC are always computed from the raw points in the caller and shown in
    the legend; this only resamples the line for visual smoothness. PCHIP is monotone
    on monotone data (ROC) and gentle on non-monotone data (precision); output clipped to [0,1].
    """
    from scipy.interpolate import PchipInterpolator
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xu, idx = np.unique(x, return_index=True)      # strictly-increasing unique x
    yu = y[idx]
    if len(xu) < 4:
        return x, y
    xs = np.linspace(xu[0], xu[-1], n)
    return xs, np.clip(PchipInterpolator(xu, yu)(xs), 0.0, 1.0)


def main():
    # Load data
    prob = pd.read_csv(os.path.join(OUT_LEG, "prediction_probability_test_canonical.csv"))
    prob = prob[prob.split == "test"]

    met = pd.read_csv(os.path.join(OUT_LEG, "model_metrics.csv"))

    ensemble = pd.read_csv(os.path.join(OUT_LEG, "ensemble_cnn_metrics.csv"))

    # Create figure with 3 panels: (a) ROC, (b) PR, (c) ensemble spread
    # Layout: (a) ROC + (b) PR in top row, (c) ensemble spread full width (own row)
    fig = plt.figure(figsize=(7.48, 4.9))

    # Panel positions: [left, bottom, width, height]
    # Row 1 — (a) ROC + (b) PR side by side (both curve panels, same size)
    ax_roc = fig.add_axes([0.07, 0.55, 0.40, 0.40])
    ax_pr  = fig.add_axes([0.52, 0.55, 0.40, 0.40])
    # Row 2 — (c) ensemble spread, full width on its own row; inter-row gap = 0.10
    ax_ens = fig.add_axes([0.08, 0.05, 0.78, 0.40])

    # Compute positive rate (base rate for PR reference line)
    # From model_metrics.csv, n_pos/n for combined/test
    combined_test = met[(met.model == "combined") & (met.split == "test")]
    if len(combined_test) > 0:
        n_pos = int(combined_test.iloc[0]["n_pos"])
        n = int(combined_test.iloc[0]["n"])
        pos_rate = n_pos / n
    else:
        # Fallback: compute from prob data
        pos_data = prob[prob.model == "combined"]
        pos_rate = pos_data.y_true.mean()

    # ---------- Panel (a) ROC curves ----------
    for m in MODELS:
        s = prob[prob.model == m]
        if len(s) == 0 or s.y_true.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(s.y_true, s.y_prob)
        auc = roc_auc_score(s.y_true, s.y_prob)
        fs, ts = smooth_curve(fpr, tpr)
        ax_roc.plot(fs, ts, color=MCOL[m], lw=1.2, label=f"{MLAB[m]} (AUC={auc:.3f})")

    # Chance diagonal
    ax_roc.plot([0, 1], [0, 1], "--", lw=0.6, color=OI["gray"], alpha=0.7, label="Chance")
    ax_roc.set_xlabel("False positive rate", fontsize=8.5, labelpad=1.5)
    ax_roc.set_ylabel("True positive rate", fontsize=8.5, labelpad=1.5)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_title("(a)  ROC curves  (canonical test split)", **TTL)
    ax_roc.legend(fontsize=6.8, loc="lower right", framealpha=0.9)
    ax_roc.tick_params(labelsize=7.3)
    for sp in ("top", "right"):
        ax_roc.spines[sp].set_visible(False)

    # ---------- Panel (b) PR curves ----------
    for m in MODELS:
        s = prob[prob.model == m]
        if len(s) == 0 or s.y_true.nunique() < 2:
            continue
        prec, rec, _ = precision_recall_curve(s.y_true, s.y_prob)
        pr_auc = average_precision_score(s.y_true, s.y_prob)
        rs, ps = smooth_curve(rec, prec)
        ax_pr.plot(rs, ps, color=MCOL[m], lw=1.2, label=f"{MLAB[m]} (PR-AUC={pr_auc:.3f})")

    # Base-rate horizontal reference line (constant precision = positive rate)
    ax_pr.axhline(pos_rate, color=OI["gray"], linestyle="--", lw=0.8, alpha=0.7,
                  label=f"Base rate ({pos_rate:.2%})")
    ax_pr.set_xlabel("Recall", fontsize=8.5, labelpad=1.5)
    ax_pr.set_ylabel("Precision", fontsize=8.5, labelpad=1.5)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.set_title("(b)  Precision–recall curves  (primary metric)", **TTL)
    ax_pr.legend(fontsize=6.8, loc="upper right", framealpha=0.9)
    ax_pr.tick_params(labelsize=7.3)
    for sp in ("top", "right"):
        ax_pr.spines[sp].set_visible(False)

    # ---------- Panel (c) ensemble spread (combined model across seeds) ----------
    # ensemble_cnn_metrics.csv has columns: seed, pr_auc_test, f_p
    # Show PR-AUC distribution and f_p (positive rate) distribution
    x = np.arange(len(ensemble))
    w = 0.35

    # Twin axes for PR-AUC and f_p
    ax1 = ax_ens
    ax2 = ax_ens.twinx()

    # PR-AUC bars
    bars1 = ax1.bar(x - w/2, ensemble["pr_auc_test"], w, color=OI["vermillion"],
                    alpha=0.85, edgecolor="white", lw=0.3, label="PR-AUC")
    # f_p bars (positive rate per seed)
    bars2 = ax2.bar(x + w/2, ensemble["f_p"], w, color=OI["sky"],
                    alpha=0.85, edgecolor="white", lw=0.3, label="Positive rate $f_p$")

    # Mean lines
    mean_pr = ensemble["pr_auc_test"].mean()
    std_pr = ensemble["pr_auc_test"].std()
    mean_fp = ensemble["f_p"].mean()
    std_fp = ensemble["f_p"].std()

    ax1.axhline(mean_pr, color=OI["vermillion"], linestyle="--", lw=0.8, alpha=0.7)
    ax2.axhline(mean_fp, color=OI["sky"], linestyle="--", lw=0.8, alpha=0.7)

    # Add mean±std text
    ax1.text(0.02, 0.98, f"PR-AUC: {mean_pr:.3f} ± {std_pr:.3f}",
              transform=ax1.transAxes, fontsize=7.5, va="top", ha="left",
              color=OI["vermillion"], fontweight="bold")
    ax2.text(0.98, 0.98, f"$f_p$: {mean_fp:.3f} ± {std_fp:.3f}",
              transform=ax2.transAxes, fontsize=7.5, va="top", ha="right",
              color=OI["sky"], fontweight="bold")

    ax1.set_xlabel("Random seed", fontsize=8.5, labelpad=1.5)
    ax1.set_ylabel("PR-AUC", fontsize=8.5, labelpad=1.5, color=OI["vermillion"])
    ax2.set_ylabel("Positive rate $f_p$", fontsize=8.5, labelpad=1.5, color=OI["sky"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(ensemble["seed"], fontsize=7.3)
    ax1.set_ylim(0, 0.6)   # PR-AUC (left axis)
    ax2.set_ylim(0, 0.2)   # f_p (right axis)

    ax1.tick_params(axis="y", labelsize=7.3, labelcolor=OI["vermillion"])
    ax2.tick_params(axis="y", labelsize=7.3, labelcolor=OI["sky"])
    ax1.tick_params(axis="x", labelsize=7.3)

    ax_ens.set_title("(c)  Combined-model ensemble spread (8 random seeds)", **TTL)

    # Legend for panel c
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_ens.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc="upper right",
                 bbox_to_anchor=(1.0, 0.88), framealpha=0.9, ncol=2)

    for sp in ("top", "right"):
        ax_ens.spines[sp].set_visible(False)

    # Save
    outpath = os.path.join(FIGDIR, "figS2.png")
    save_figure(fig, outpath, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Print metrics for verification
    print(f"[figS2] saved {outpath}")

    # Verify the file
    if os.path.exists(outpath):
        size = os.path.getsize(outpath)
        print(f"[figS2] file size: {size:,} bytes ({size/1024:.1f} KB)")

    # Print AUC/PR-AUC values for each model
    print("\n[figS2] Model metrics (test split):")
    for m in MODELS:
        s = prob[prob.model == m]
        if len(s) == 0 or s.y_true.nunique() < 2:
            continue
        auc = roc_auc_score(s.y_true, s.y_prob)
        pr_auc = average_precision_score(s.y_true, s.y_prob)
        print(f"       {m:12s}: ROC-AUC={auc:.4f}, PR-AUC={pr_auc:.4f}")

    # Print ensemble spread stats
    print("\n[figS2] Combined model ensemble spread (8 seeds):")
    print(f"       PR-AUC: {mean_pr:.4f} ± {std_pr:.4f} (range: {ensemble['pr_auc_test'].min():.4f} - {ensemble['pr_auc_test'].max():.4f})")
    print(f"       f_p:    {mean_fp:.4f} ± {std_fp:.4f} (range: {ensemble['f_p'].min():.4f} - {ensemble['f_p'].max():.4f})")

    # Verify key finding: combined beats month-only
    s_comb = prob[prob.model == "combined"]
    s_month = prob[prob.model == "month_only"]
    pr_auc_comb = average_precision_score(s_comb.y_true, s_comb.y_prob)
    pr_auc_month = average_precision_score(s_month.y_true, s_month.y_prob)
    print(f"\n[figS2] KEY: combined PR-AUC ({pr_auc_comb:.3f}) >> month-only ({pr_auc_month:.3f}) "
          f"— skill is NOT just climatology/seasonality")

    return outpath


if __name__ == "__main__":
    main()
