#!/usr/bin/env python3
"""09a_fig2.py — Fig 2: YREP-day definition + trends + seasonal distribution.

Layout (5 panels, Elsevier double-column 7.48 in, dpi 600):
  (a) YREP-day definition flow — dual-gate schematic (full width, top)
  (b) YREP-day annual frequency        — Theil-Sen + block-boot 95% CI + MK p
  (c) YREP-day intensity (area-mean P)  — Theil-Sen + MK p
  (d) YREP-day affected area (F(d))    — Theil-Sen + MK p
  (e) seasonal distribution            — monthly counts, JAS share annotated

Reads cached outputs (no recompute of the label).

Output: figures/fig2.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from scipy.stats import theilslopes, kendalltau
import config

FIGDIR = config.FIGDIR
OUT    = config.OUT

# Okabe-Ito palette
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777")
C_DATA   = OI["blue"]        # YREP-day time-series markers/lines
C_TREND  = OI["black"]       # Theil-Sen line
C_SEAS   = OI["vermillion"]  # seasonal bars (extreme = warm)
C_GATE1  = OI["orange"]      # coverage gate
C_GATE2  = OI["green"]       # intensity gate
C_OUT    = OI["vermillion"]  # YREP output

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 9.1,
                     "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6})
TTL = dict(fontsize=9.8, fontweight="bold", loc="left", pad=3)


# --------------------------------------------------------------------------- #
def _mk_p(x, y):
    """Two-sided Mann-Kendall p via Kendall tau (scipy)."""
    return float(kendalltau(x, y).pvalue)


def _load():
    yt = pd.read_csv(os.path.join(OUT, "YREP_day_table_1960_2024.csv"), parse_dates=["date"])
    ts = pd.read_csv(os.path.join(OUT, "trend_statistics.csv"))
    # P90 area-mean threshold from the precip dataset attrs (for the schematic)
    import xarray as xr
    ds = xr.open_dataset(os.path.join(OUT, "precip_daily_region_1960_2024.nc"))
    p90 = float(ds.attrs["areamean_p90_mm_d"])
    q95_mean = float(ds.Q95.mean())
    n_cells = int(ds.mask.sum())
    ds.close()
    return yt, ts, p90, q95_mean, n_cells


# --------------------------------------------------------------------------- #
# Panel (a) — YREP-day definition flow (dual gate)
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, fc, ec="#333", fs=7.0, fw="normal", tc="#222", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=fw, color=tc, zorder=4, linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1, color="#444", lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=9, color=color, lw=lw,
                                 shrinkA=2, shrinkB=2, zorder=2))


def _panel_a(ax, p90, q95_mean, n_yrep, n_days, pct, jas_pct, ja_pct):
    """Panel (a) is the dual-gate flowchart artwork
    (data/schematics/fig2a_flowchart.png). Its baked-in numbers (Q95 grid-mean ~26.1
    mm d-1, P90 = 8.72 mm d-1, 457/23742 = 1.92%, JAS 78%) match the cached
    YREP table this script reads, so no content is altered.
    The p90/q95_mean/... arguments are kept for signature compatibility but
    unused (artwork carries the text)."""
    img = plt.imread(str(config.SCHEMATICS / "fig2a_flowchart.png"))
    # extent= maps the raster onto the 0-1 axes box; the axes rect itself carries the
    # artwork aspect, so aspect="auto" fills the box without distortion.
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.imshow(img, extent=(0, 1, 0, 1), aspect="auto", interpolation="none", zorder=3)
    ax.set_title("(a)  YREP-day definition — dual gate "
                 "(spatial coverage AND area intensity)", **TTL)


# --------------------------------------------------------------------------- #
# Generic trend panel
# --------------------------------------------------------------------------- #
def _trend_panel(ax, yr, y, ylabel, title, slope, p, unit_fmt, ci=None,
                 legend_loc="upper left"):
    """Time-series + Theil-Sen line, annotated with slope, MK p, and (optional) CI."""
    good = ~np.isnan(y)
    ax.plot(yr[good], y[good], "o-", color=C_DATA, ms=2.2, lw=0.8, mfc=C_DATA, mec=C_DATA)
    tsl = theilslopes(y[good], yr[good])
    ax.plot(yr, tsl.intercept + tsl.slope * yr, "--", color=C_TREND, lw=1.0)
    slope_txt = unit_fmt % slope if "%" in unit_fmt else unit_fmt.format(slope)
    lbl = "Theil–Sen %s/yr\n(MK p=%.2f)" % (slope_txt, p)
    if ci is not None:
        lbl += "\nblock-boot 95%% CI [%+.3f, %+.3f]" % (ci[0], ci[1])
    ax.plot([], [], "--", color=C_TREND, lw=1.0, label=lbl)
    # headroom so the legend clears the data
    rng = np.nanmax(y) - np.nanmin(y)
    ax.set_ylim(top=np.nanmax(y) + 0.24 * rng)
    ax.legend(fontsize=7.5, loc=legend_loc, framealpha=0.9, handlelength=1.6)
    ax.set_ylabel(ylabel, fontsize=8.6, labelpad=1.5)
    ax.set_xlabel("Year", fontsize=8.6, labelpad=1.5)
    ax.set_title(title, **TTL)
    ax.tick_params(labelsize=7.8)
    tk = np.arange(int(yr.min() // 10) * 10, int(yr.max()) + 1, 20)
    ax.set_xticks(tk)
    ax.set_xticklabels([str(int(t)) for t in tk], fontsize=7.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
def main():
    yt, ts, p90, q95_mean, n_cells = _load()
    yrep = yt[yt.YREP_label == 1]
    yr = np.arange(int(yt.year.min()), int(yt.year.max()) + 1)

    cnt = yrep.groupby("year").size().reindex(yr, fill_value=0).values
    inten = yrep.groupby("year").area_mean_precip_mm_d.mean().reindex(yr).values
    area = yrep.groupby("year").extreme_area_fraction.mean().reindex(yr).values

    # authoritative frequency trend (trend_statistics.csv)
    rfreq = ts[ts.metric == "YREP_freq"].iloc[0]
    f_ci = [float(v) for v in str(rfreq.boot95).strip("[]").split(",")]
    f_slope = float(rfreq.theilsen_per_yr); f_p = float(rfreq.mk_p)
    # intensity / area trends (YREP-day level, computed here — no CNN)
    i_slope = float(theilslopes(inten[~np.isnan(inten)], yr[~np.isnan(inten)]).slope)
    i_p = _mk_p(yr[~np.isnan(inten)], inten[~np.isnan(inten)])
    a_slope = float(theilslopes(area[~np.isnan(area)], yr[~np.isnan(area)]).slope)
    a_p = _mk_p(yr[~np.isnan(area)], area[~np.isnan(area)])

    # seasonal
    mcounts = yrep.month.value_counts().reindex(range(1, 13), fill_value=0)
    n_yrep, n_days = int(len(yrep)), int(len(yt))
    pct = 100.0 * n_yrep / n_days
    jas_pct = int(round(100 * mcounts.loc[[7, 8, 9]].sum() / n_yrep))
    ja_pct = int(round(100 * mcounts.loc[[7, 8]].sum() / n_yrep))

    # ---- figure ----
    fig = plt.figure(figsize=(7.48, 6.4))

    ax_a = fig.add_axes([0.0400, 0.7259, 0.9200, 0.2309])  # schematic artwork, full width
    _panel_a(ax_a, p90, q95_mean, n_yrep, n_days, pct, jas_pct, ja_pct)

    ax_b = fig.add_axes([0.0400, 0.4006, 0.4325, 0.2784])      # frequency
    _trend_panel(ax_b, yr, cnt, "YREP days per year",
                 "(b)  YREP-day frequency", f_slope, f_p, "%.3f", ci=f_ci)

    ax_c = fig.add_axes([0.5275, 0.4006, 0.4325, 0.2784])      # intensity
    _trend_panel(ax_c, yr, inten, "YREP-day mean $\\bar P$ (mm d$^{-1}$)",
                 "(c)  YREP-day intensity", i_slope, i_p, "%.4f",
                 legend_loc="upper right")

    ax_d = fig.add_axes([0.0400, 0.0238, 0.4325, 0.2784])      # affected area
    _trend_panel(ax_d, yr, area, "YREP-day mean $F(d)$ (fraction)",
                 "(d)  YREP-day affected area", a_slope, a_p, "%.5f",
                 legend_loc="upper right")

    ax_e = fig.add_axes([0.5275, 0.0238, 0.4325, 0.2784])      # seasonal
    ax_e.bar(mcounts.index, mcounts.values, color=C_SEAS, alpha=0.85, width=0.8)
    ax_e.set_xticks(range(1, 13))
    ax_e.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=7.5)
    ax_e.set_ylabel("YREP days", fontsize=8.6, labelpad=1.5)
    ax_e.set_xlabel("Month", fontsize=8.6, labelpad=1.5)
    ax_e.set_title("(e)  Seasonal distribution", **TTL)
    ax_e.tick_params(labelsize=7.8)
    for s in ("top", "right"):
        ax_e.spines[s].set_visible(False)
    # JAS shading + annotation
    ax_e.axvspan(6.5, 9.5, color=OI["yellow"], alpha=0.25, zorder=0)
    ax_e.text(8.5, ax_e.get_ylim()[1] * 0.90, "JAS %d%%" % jas_pct, ha="center", fontsize=8.1,
              color="#7a5c00", fontweight="bold")

    out = os.path.join(FIGDIR, "fig2.png")
    save_figure(fig, out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig2] saved {out}")
    print(f"       YREP days={n_yrep}/{n_days}={pct:.2f}%  JAS={jas_pct}% Jul+Aug={ja_pct}%")
    print(f"       freq {f_slope:+.4f}/yr p={f_p:.2f} CI[{f_ci[0]:+.3f},{f_ci[1]:+.3f}]  "
          f"inten {i_slope:+.4f} p={i_p:.2f}  area {a_slope:+.5f} p={a_p:.2f}")
    print(f"       mean YREP-day intensity {np.nanmean(inten):.2f} mm/d, area {np.nanmean(area):.4f}")


if __name__ == "__main__":
    main()
