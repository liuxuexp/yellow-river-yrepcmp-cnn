#!/usr/bin/env python3
"""13_figS4_threshold_sweep.py — Probability-threshold sensitivity of the decomposition.

Re-binarizes the canonical combined-CNN probabilities at {0.30,0.40,0.50,0.60,0.70},
recomputes f_p and the full ΔPr(E) decomposition (annual + JAS) at each, and shows
the intensity (thermodynamic) term stays positive across thresholds while the
frequency term remains small. PNG only.

Reads: prediction_probability_full.csv, precip cache (P̄/YREP)
Writes: threshold_sweep_1960_2024.csv, figures/figS4_threshold_sweep.png
"""
import os, importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

HERE = os.path.dirname(__file__)
OUT = config.OUT
FIG = config.FIGDIR
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 600,
    "font.size": 9.5,
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none"
})

COLORS = {
    "blue": "#2c6fbf",
    "red": "#c0392b",
    "green": "#1a7d1a",
    "orange": "#E69F00",
    "gray": "#666666",
    "light_gray": "#DDDDDD"
}

spec = importlib.util.spec_from_file_location("decomp", os.path.join(HERE, "decomposition.py"))
dc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc)
EARLY_MAIN, LATE_MAIN = 1991, 1992
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def load_all():
    cache_path = os.path.join(OUT, "precip_daily_region_1960_2024.nc")
    import xarray as xr
    ds = xr.open_dataset(cache_path)
    pbar = ds["Pbar"].values.astype(float); yrep = ds["YREP"].values.astype(float)
    dates = pd.to_datetime(ds.time.values).normalize(); ds.close()
    prob = pd.read_csv(os.path.join(OUT, "prediction_probability_full.csv"), parse_dates=["date"])
    prob = prob[prob.model == "combined"][["date", "y_prob"]].copy()
    prob["date"] = prob["date"].dt.normalize()
    lab = pd.DataFrame({"date": dates}).reset_index().rename(columns={"index": "i"})
    lab = lab.merge(prob, on="date", how="inner")
    assert len(lab) == len(dates)
    return pbar, yrep, dates, lab["y_prob"].values


def main():
    pbar, yrep, dates, yprob = load_all()
    yr = dates.year.values; mo = dates.month.values
    jas = np.isin(mo, [7, 8, 9]); early = yr <= EARLY_MAIN; late = yr >= LATE_MAIN
    rows = []
    for t in THRESHOLDS:
        labels = (yprob >= t).astype(int)
        for sname, sel in [("annual", np.ones(len(yr), bool)), ("JAS", jas)]:
            r = dc.decompose(yrep, labels, early & sel, late & sel)
            tot = r["term_freq"] + r["term_intensity"] + r["term_interaction"]
            rows.append(dict(threshold=t, season=sname, f_p_early=float(labels[early & sel].mean()),
                             f_p_late=float(labels[late & sel].mean()),
                             inten=r["term_intensity"], freq=r["term_freq"],
                             thermo_share=100 * r["term_intensity"] / tot if abs(tot) > 1e-12 else np.nan))
            print(f"  t={t:.2f} {sname:6s}: f_p {labels[early&sel].mean():.3f}->{labels[late&sel].mean():.3f} "
                  f"ΔPr(E) inten={r['term_intensity']:+.4f} freq={r['term_freq']:+.4f}")
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "threshold_sweep_1960_2024.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) f_p early vs late vs threshold
    ax_a = axes[0]
    for sname, mk in [("annual", "o"), ("JAS", "s")]:
        s = d[d.season == sname]
        ax_a.plot(s.threshold, s.f_p_early * 100, "-" + mk, color=COLORS["blue"], ms=8,
                  markeredgecolor="white", markeredgewidth=1.2,
                  label=f"{sname} early" if sname == "annual" else None)
        ax_a.plot(s.threshold, s.f_p_late * 100, "--" + mk, color=COLORS["red"], ms=8,
                  markeredgecolor="white", markeredgewidth=1.2,
                  label=f"{sname} late" if sname == "annual" else None)

    ax_a.set_xlabel("Pattern-day threshold on P(YREP)", fontsize=9.5)
    ax_a.set_ylabel("Pattern frequency f$_p$ (%)", fontsize=9.5)
    ax_a.set_title("(a) Pattern frequency (early vs late)", fontsize=10.5, fontweight="bold", pad=8)
    ax_a.axvline(0.5, color=COLORS["gray"], ls=":", lw=1.2)

    ax_a.grid(True, axis="y", color="#E0E0E0", linestyle="-", linewidth=0.6, alpha=0.7)
    ax_a.set_axisbelow(True)

    leg_a = ax_a.legend(loc="upper right", framealpha=0.95, edgecolor="#CCCCCC",
                         borderpad=0.4, labelspacing=0.3, handlelength=1.5)
    for spine in ax_a.spines.values():
        spine.set_linewidth(0.8)
    ax_a.tick_params(length=4, width=0.6, pad=2)

    # (b) ΔPr(E) intensity + frequency terms vs threshold
    ax_b = axes[1]
    for sname, mk in [("annual", "o"), ("JAS", "s")]:
        s = d[d.season == sname]
        ax_b.plot(s.threshold, s.inten, "-" + mk, color=COLORS["green"], ms=8,
                  markeredgecolor="white", markeredgewidth=1.2,
                  label=f"inten/thermo ({sname})")
        ax_b.plot(s.threshold, s.freq, "--" + mk, color=COLORS["red"], ms=8,
                  markeredgecolor="white", markeredgewidth=1.2,
                  label=f"freq/dynamic ({sname})")

    ax_b.axhline(0, color="black", lw=0.8)
    ax_b.axvline(0.5, color=COLORS["gray"], ls=":", lw=1.2)

    ax_b.set_xlabel("Threshold", fontsize=9.5)
    ax_b.set_ylabel("ΔPr(E) term (probability)", fontsize=9.5)
    ax_b.set_title("(b) Decomposition terms vs threshold", fontsize=10.5, fontweight="bold", pad=8)

    ax_b.grid(True, axis="y", color="#E0E0E0", linestyle="-", linewidth=0.6, alpha=0.7)
    ax_b.set_axisbelow(True)

    leg_b = ax_b.legend(loc="upper left", framealpha=0.95, edgecolor="#CCCCCC",
                         borderpad=0.4, labelspacing=0.3, handlelength=1.5)
    for spine in ax_b.spines.values():
        spine.set_linewidth(0.8)
    ax_b.tick_params(length=4, width=0.6, pad=2)

    # (c) thermo_share vs threshold
    ax_c = axes[2]
    for sname, mk in [("annual", "o"), ("JAS", "s")]:
        s = d[d.season == sname]
        ax_c.plot(s.threshold, s.thermo_share, "-" + mk, color=COLORS["green"], ms=8,
                  markeredgecolor="white", markeredgewidth=1.2, label=sname)

    ax_c.axhline(0, color="black", lw=0.8)
    ax_c.axhline(100, color=COLORS["gray"], ls=":", lw=1.0)
    ax_c.axvline(0.5, color=COLORS["gray"], ls=":", lw=1.2)

    ax_c.set_xlabel("Threshold", fontsize=9.5)
    ax_c.set_ylabel("Thermodynamic share (%)", fontsize=9.5)
    ax_c.set_title("(c) Thermodynamic share vs threshold", fontsize=10.5, fontweight="bold", pad=8)

    ax_c.grid(True, axis="y", color="#E0E0E0", linestyle="-", linewidth=0.6, alpha=0.7)
    ax_c.set_axisbelow(True)

    leg_c = ax_c.legend(loc="lower left", framealpha=0.95, edgecolor="#CCCCCC",
                         borderpad=0.4, labelspacing=0.3, handlelength=1.5)
    for spine in ax_c.spines.values():
        spine.set_linewidth(0.8)
    ax_c.tick_params(length=4, width=0.6, pad=2)

    fig.suptitle("Fig S4 — Probability-threshold sensitivity: the ΔPr(E) intensity term stays positive "
                 "(thermodynamic) across thresholds 0.30–0.70",
                 fontsize=11.5, fontweight="bold", y=1.02, x=0.5)

    plt.tight_layout()

    p = os.path.join(FIG, "figS4_threshold_sweep.png")
    plt.savefig(p, bbox_inches="tight", facecolor="white", dpi=600)
    plt.close()
    print(f"[save] {p}")


if __name__ == "__main__":
    main()
