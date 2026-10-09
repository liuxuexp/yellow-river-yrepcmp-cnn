#!/usr/bin/env python3
"""09a_figS1.py — Fig S1: sensitivity (threshold/definition/region/product).

Five panels covering four sensitivity dimensions:
  (a) YREP annual-rate heatmap across the 18-combo threshold grid {Q90,Q95,Q99}×
      {{5,10,15}% cov}×{P90,P95} (canonical cell marked)
  (b) rate per combo as bars + the ROBUST ΔPr(E) thermodynamic-share reference
  (c) extreme-day rate by definition axis (wet-day vs all-day+2σ, vs pillar papers)
  (d) YREP rate when the spatial domain is the whole basin vs middle vs lower reach
      (Q95 held at basin-wide; P90 gate per domain) -> tests robustness to boundary
  (e) YREP rate under CHM_PRE vs CN05.1 over the 1961–2023 overlap, identical
      definition -> tests robustness to the precipitation product

Note (panel b): the ΔP̄ annual thermodynamic split is seed-unstable (its frequency
term flips sign across the 8 CNN seeds, ensemble frac_positive≈0.38) and is
deliberately NOT quoted. The ΔPr(E) annual intensity term is robust (positive in
8/8 seeds, bootstrap CI excludes 0) and is the headline quantity cited here.

Reads: out/sensitivity_grid_1960_2024.csv, out/sensitivity_allday_1960_2024.csv,
       out/sensitivity_region_1960_2024.csv, out/sensitivity_product_1961_2023.csv,
       out_legacy46x82/decomposition_headline_1960_2024.csv,
       out_legacy46x82/decomposition_ensemble_summary_1960_2024.csv
Output: figures/figS1.png
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

# --------------------------------------------------------------------------- #
# Canonical style
# --------------------------------------------------------------------------- #
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777")

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 9.1,
                     "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6})
TTL = dict(fontsize=9.1, fontweight="bold", loc="left", pad=3)

# row geometry (3 rows, figsize height 5.8): h=0.22, inter-row gap 0.08
Y1, Y2, Y3, H = 0.69, 0.39, 0.09, 0.22


# --------------------------------------------------------------------------- #
def _rate_matrix(g):
    """18-combo wet-day grid {Q90,Q95,Q99}×{{5,10,15}%/P90,P95} → (3,6) class-1 rate (%)."""
    mat = np.full((3, 6), np.nan)
    for _, r in g.iterrows():
        i = {90: 0, 95: 1, 99: 2}[int(r.intensity_q)]
        j = {5: {90: 0, 95: 1}, 10: {90: 2, 95: 3}, 15: {90: 4, 95: 5}}[
            int(r.coverage_pct)][int(r.areamean_p)]
        mat[i, j] = r.class1_pct
    return mat


# --------------------------------------------------------------------------- #
# Panel (a) — YREP rate heatmap across the 18-combo threshold grid
# --------------------------------------------------------------------------- #
def panel_a(fig, rect, cax_rect, g):
    ax = fig.add_axes(rect)
    cax = fig.add_axes(cax_rect)
    combos = [f"{c}/P{p}" for c in (5, 10, 15) for p in (90, 95)]
    mat = _rate_matrix(g)

    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=3.0)
    # single-line combo labels: dropping the redundant "%" ("10/P95" ≈ 0.35 in)
    # fits the 0.40-in column pitch at 7.5 pt — "%" lives in the xlabel instead
    # (rotation stays rejected: rotated labels descend into (b)'s title band).
    ax.set_xticks(range(6))
    ax.set_xticklabels(combos, fontsize=7.5)
    ax.tick_params(axis="x", pad=1.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(["Q90", "Q95", "Q99"], fontsize=7.5)
    ax.set_ylabel("Local intensity percentile", fontsize=7.8, labelpad=-1)
    # labelpad<0 eats Noto CJK's tall line-box padding so the xlabel ink sits
    # ~2 pt below the tick labels instead of a full text height (digits have
    # no descenders, so nothing collides).
    ax.set_xlabel("Coverage % / area-mean percentile", fontsize=7.8, labelpad=-1)
    ax.set_title("(a)  YREP annual rate across threshold grid  (%)", **TTL)

    for i in range(3):
        for j in range(6):
            if np.isfinite(mat[i, j]):
                col = "white" if mat[i, j] > 1.8 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.3, color=col)
    # mark canonical cell (Q95, 10%/P90 = row1, col2)
    ax.add_patch(plt.Rectangle((2 - 0.5, 1 - 0.5), 1, 1, fill=False,
                               edgecolor=OI["blue"], lw=1.8))

    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel("Class-1 rate (%)", fontsize=7.8)
    cb.ax.tick_params(labelsize=7)
    return mat


# --------------------------------------------------------------------------- #
# Panel (b) — rate per combo (bars) + robust ΔPr(E) thermo reference
# --------------------------------------------------------------------------- #
def panel_b(fig, rect, g, decomp, decomp_ens):
    ax = fig.add_axes(rect)
    rates = _rate_matrix(g).flatten()        # row-major: Q90 row, then Q95, then Q99
    x = np.arange(18)

    # ROBUST headline: ΔPr(E) annual thermodynamic share (intensity term + in 8/8 seeds).
    # ΔP̄ annual is NOT quoted — its frequency term is seed-unstable (frac_positive≈0.38).
    row = decomp[decomp.decomposition.str.contains("Pr") &
                 decomp.decomposition.str.contains("annual")].iloc[0]
    thermo_pct = 100 * row.term_intensity_thermo / (row.term_freq_dynamic + row.term_intensity_thermo)
    inten_frac_pos = float(decomp_ens[(decomp_ens.case == "dPrE_annual") &
                                      (decomp_ens.term == "inten")].frac_positive.iloc[0])
    n_seeds = int(round(inten_frac_pos * 8))

    bars = ax.bar(x, rates, color=OI["sky"], alpha=0.85, edgecolor="white", lw=0.4)
    # canonical combo = Q95 / 10% / P90 → row1(=1)*6 + col2(=2) = flat index 8
    bars[8].set_color(OI["vermillion"]); bars[8].set_alpha(0.95)
    # anchored ha="right" — centered, the enlarged label's right tail grazed the
    # top-right italic annotation's first line.
    ax.text(8.2, rates[8] * 1.18, f"ΔPr(E)\nthermo {thermo_pct:.0f}%\n({n_seeds}/8 seeds)",
            ha="right", fontsize=7.3, color=OI["vermillion"], fontweight="bold")

    ax.set_xticks(x)
    # 18 two-line labels at 0.39-in pitch — 6.0 pt fits without rotation (rotating
    # two-line labels would hit the row-3 titles). linespacing 0.85 closes the
    # "Q90"↔"5%/P90" gap (same Noto CJK fix as panel (c)).
    ax.set_xticklabels([f"Q{q}\n{c}%/P{p}" for q in (90, 95, 99)
                        for c in (5, 10, 15) for p in (90, 95)], fontsize=6.0,
                       rotation=0, linespacing=0.85)
    ax.tick_params(axis="x", pad=1.5)
    ax.set_ylabel("YREP rate (%)", fontsize=8.6)
    ax.set_ylim(0, 3.2)
    ax.set_title("(b)  Rate per threshold combo  (robust ΔPr(E) thermo dominance)", **TTL)
    ax.tick_params(axis="y", labelsize=7.3)   # y only; x tick fontsize set above
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.text(0.98, 0.98,
            f"Thermo dominance robust on ΔPr(E): intensity term + in {n_seeds}/8 seeds,\n"
            "bootstrap CI excludes 0. ΔP̄ split is seed-unstable (freq sign flips), not quoted.\n"
            "Per-combo decomposition across the 18-grid is not computed; rate robustness is.",
            transform=ax.transAxes, fontsize=6.5, color=OI["gray"],
            ha="right", va="top", style="italic")


# --------------------------------------------------------------------------- #
# Panel (c) — wet-day vs all-day definition axis (vs pillar papers)
# --------------------------------------------------------------------------- #
def panel_c(fig, rect, a):
    ax = fig.add_axes(rect)
    a = a.sort_values("pct").reset_index(drop=True)

    cols, short = [], []
    for d in a.definition:
        if "primary" in d:
            cols.append(OI["blue"]);    short.append("Primary\n(wet Q95, 10%/P90)")
        elif "Mouassom" in d:
            cov = d.split("F>=")[1].split("%")[0]            # F>=5 / 10 / 15
            cols.append(OI["orange"]);  short.append(f"Mouassom all-day\n+2σ, F≥{cov}%")
        else:                           # Davenport — distinguish (all) vs (wet) variants
            suff = d[d.index("(") + 1:d.index(")")]
            cols.append(OI["vermillion"]); short.append(f"Davenport\nP̄≥p95 ({suff})")

    y = np.arange(len(a))
    ax.barh(y, a.pct, height=0.6, color=cols, alpha=0.85, edgecolor="white", lw=0.5)
    for i, (v, c) in enumerate(zip(a.pct, cols)):
        ax.text(v + 0.15, i, f"{v:.2f}%", va="center", fontsize=7.3, color=c, fontweight="bold")

    ax.axvspan(5.0, 7.28, color=OI["orange"], alpha=0.08)
    ax.axvline(7.28, color=OI["orange"], ls="--", lw=0.8, alpha=0.7)
    ax.axvline(5.0,  color=OI["vermillion"], ls="--", lw=0.8, alpha=0.7)
    ax.text(7.4, 0.4, "Mouassom 7.28%", fontsize=7, color=OI["orange"], ha="left", va="center")
    ax.text(4.9, len(a) - 0.6, "Davenport ~5%", fontsize=7, color=OI["vermillion"], ha="right", va="center")

    # 6 rows at 0.21-in pitch — 6.8 pt with 0.9 line spacing fits both the pitch
    # and the a–c gutter.
    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=6.8, linespacing=0.9)
    ax.set_xlabel("Extreme-day rate (%)", fontsize=8.6)
    ax.set_xlim(0, 18)
    ax.set_title("(c)  Wet-day vs all-day definition axis", x=-0.20, **TTL)
    ax.tick_params(axis="x", labelsize=7.3)   # x only; y tick fontsize set above
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# --------------------------------------------------------------------------- #
# Panel (d) — region sensitivity: YREP rate by spatial domain
# --------------------------------------------------------------------------- #
def panel_d(fig, rect, reg):
    ax = fig.add_axes(rect)
    core = reg[reg.domain.isin(["basin (whole)", "middle (Loess Plateau)",
                                "lower (N-China Plain)"])].reset_index(drop=True)
    cols = [OI["blue"], OI["orange"], OI["pink"]]          # basin / MID / LOW (figS8 convention)
    labels = ["Basin\n(whole)", "Middle\n(Loess Plateau)", "Lower\n(N-China Plain)"]

    x = np.arange(len(core))
    ax.bar(x, core.rate_pct, color=cols, alpha=0.9, edgecolor="white", lw=0.4, width=0.62)
    for xi, v in zip(x, core.rate_pct):
        ax.text(xi, v + 0.06, f"{v:.2f}%", ha="center", fontsize=7.8, fontweight="bold")
    ax.axhline(core.rate_pct.iloc[0], color=OI["blue"], ls=":", lw=0.8, alpha=0.7)  # basin ref

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("YREP rate (%)", fontsize=8.6)
    ax.set_ylim(1, 3.0); ax.set_yticks([1, 1.5, 2, 2.5, 3])          # shared y-axis with panel (e)
    ax.set_title("(d)  Region sensitivity — rate by spatial domain", **TTL)
    ax.tick_params(labelsize=7.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    any_r = float(reg[reg.domain.str.contains("OR")].rate_pct.iloc[0])
    synch = float(reg[reg.domain.str.contains("AND")].rate_pct.iloc[0])
    ax.text(0.02, 0.98,
            f"Q95 held at basin-wide (fixed threshold);\nP90 gate per domain.\n"
            f"any-reach {any_r:.2f}%  •  synchronous {synch:.2f}%\n"
            "→ ~2% stable across domains",
            transform=ax.transAxes, fontsize=6.8, color=OI["gray"],
            ha="left", va="top", style="italic")
    return ax


# --------------------------------------------------------------------------- #
# Panel (e) — product sensitivity: CHM_PRE vs CN05.1
# --------------------------------------------------------------------------- #
def panel_e(fig, rect, prod, sharey=None):
    ax = fig.add_axes(rect, sharey=sharey)
    chm_ov = prod[(prod["product"] == "CHM_PRE") & (prod.span == "1961-2023")].rate_pct.iloc[0]
    cn05   = prod[(prod["product"] == "CN05.1") & (prod.span == "1961-2023")].rate_pct.iloc[0]
    chm_full = prod[(prod["product"] == "CHM_PRE") & (prod.span == "1960-2024")].rate_pct.iloc[0]

    x = np.arange(2)
    vals = [chm_ov, cn05]
    cols = [OI["blue"], OI["vermillion"]]                  # CHM_PRE / CN05.1 (figS6 convention)
    ax.bar(x, vals, color=cols, alpha=0.9, edgecolor="white", lw=0.4, width=0.5)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.06, f"{v:.2f}%", ha="center", fontsize=8.1, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(["CHM_PRE\n(0.1°, gauge-fused)", "CN05.1\n(0.25°, station)"],
                                         fontsize=7.8)
    ax.set_ylabel("YREP rate (%)", fontsize=8.6)
    ax.set_xlim(-0.55, 1.55)
    ax.set_title("(e)  Product sensitivity — CHM_PRE vs CN05.1", **TTL)
    ax.tick_params(labelsize=7.3)
    if sharey is not None:                              # shared with (d): inherit [1,3]/0.5 scale, hide labels
        ax.tick_params(labelleft=False); ax.set_ylabel("")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.text(0.02, 0.98,
            f"1961–2023 overlap, identical definition\n"
            f"(wet Q95, F≥10%, Pbar≥P90_wet).\n"
            f"CHM_PRE headline 1960–2024: {chm_full:.2f}%.\n"
            "Both ≈2%, far below pillar 5–7.3% (off scale ↑)\n"
            "→ semi-arid ~2% rate survives the product swap",
            transform=ax.transAxes, fontsize=6.2, color=OI["gray"],
            ha="left", va="top", style="italic")


# --------------------------------------------------------------------------- #
def main():
    g          = pd.read_csv(os.path.join(OUT, "sensitivity_grid_1960_2024.csv"))
    a          = pd.read_csv(os.path.join(OUT, "sensitivity_allday_1960_2024.csv"))
    reg        = pd.read_csv(os.path.join(OUT, "sensitivity_region_1960_2024.csv"))
    prod       = pd.read_csv(os.path.join(OUT, "sensitivity_product_1961_2023.csv"))
    decomp     = pd.read_csv(os.path.join(OUT_LEG, "decomposition_headline_1960_2024.csv"))
    decomp_ens = pd.read_csv(os.path.join(OUT_LEG, "decomposition_ensemble_summary_1960_2024.csv"))

    fig = plt.figure(figsize=(7.48, 5.8))

    # row 1 — (a) heatmap (narrow) + (c) definition axis (narrower, larger gap a–c)
    mat = panel_a(fig, [0.04, Y1, 0.32, H], [0.37, Y1, 0.010, H], g)
    panel_c(fig, [0.565, Y1, 0.385, H], a)
    # row 2 — (b) 18-bar threshold + robust thermo (full width, own row)
    panel_b(fig, [0.04, Y2, 0.94, H], g, decomp, decomp_ens)
    # row 3 — region + product (d master, e shares its y-axis)
    ax_d = panel_d(fig, [0.04, Y3, 0.46, H], reg)
    panel_e(fig, [0.54, Y3, 0.44, H], prod, sharey=ax_d)

    out_path = os.path.join(FIGDIR, "figS1.png")
    save_figure(fig, out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figS1] saved {out_path} ({os.path.getsize(out_path):,} bytes)")

    # key-numbers echo
    print("       === Fig S1 key numbers ===")
    print(f"       threshold grid: canonical Q95/10%/P90 = {mat[1, 2]:.2f}%  "
          f"(range {np.nanmin(_rate_matrix(g)):.2f}–{np.nanmax(_rate_matrix(g)):.2f}%)")
    row = decomp[decomp.decomposition.str.contains("Pr") &
                 decomp.decomposition.str.contains("annual")].iloc[0]
    tp = 100 * row.term_intensity_thermo / (row.term_freq_dynamic + row.term_intensity_thermo)
    print(f"       ΔPr(E) annual thermo share (robust): {tp:.1f}%")
    for _, r in reg[reg.domain.isin(["basin (whole)", "middle (Loess Plateau)",
                                     "lower (N-China Plain)"])].iterrows():
        print(f"       region {r.domain:26s}: {r.rate_pct:.3f}%")
    print(f"       product CHM_PRE 1961-2023: {prod[(prod['product']=='CHM_PRE')&(prod.span=='1961-2023')].rate_pct.iloc[0]:.3f}%   "
          f"CN05.1 1961-2023: {prod[(prod['product']=='CN05.1')&(prod.span=='1961-2023')].rate_pct.iloc[0]:.3f}%")


if __name__ == "__main__":
    main()
