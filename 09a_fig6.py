#!/usr/bin/env python3
"""09a_fig6.py — Fig 6: early/late IVT + late-early ΔIVT/ΔVIMFC + moisture budget.

Layout (6 panels):
  (a) early (1960–91) YR-EPCMP-day IVT — magnitude + vectors (the moisture pathway)
  (b) late  (1992–24) YR-EPCMP-day IVT — shared scale with (a)
  (c) late − early ΔIVT (YR-EPCMP days) — magnitude + vectors
  (d) late − early ΔVIMFC (mm d⁻¹)
  (e) four-boundary moisture inflow (box 105–120°E/34–41°N): S/N/W/E, early/late × pattern/non
  (f) moisture budget  P = E_app + VIMFC − ∂W/∂t  (closes to ~0; E_app solved as residual)

Spatial fields are RAW pattern-day composites (composite_fields_1960_2024.nc);
the late−early DIFFERENCE is within-pattern-day so the seasonal cycle cancels.
Tibet-interior (lon<=100 & lat>=32) masked on all moisture fields.

Reads: out_legacy46x82/composite_fields_1960_2024.nc,
       out/moisture_budget_summary.csv
Output: figures/fig6.png
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import geopandas as gpd
import config

FIGDIR = config.FIGDIR
OUT    = config.OUT
OUT_LEG = config.OUT_LEGACY

PROJ   = ccrs.PlateCarree()
DISPLAY = [60, 180, -10, 70]
GEO_ASP = (DISPLAY[1]-DISPLAY[0])/(DISPLAY[3]-DISPLAY[2])   # 120/80 = 1.50
SH_DB   = config.SHAPES
CNTRY   = config.CNTRY_SHP
NINE    = config.NINEDASH_SHP
BASIN   = config.BASIN_SHP

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777")

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 9.1,
                     "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6})
TTL = dict(fontsize=9.4, fontweight="bold", loc="left", pad=3)

# cache shapefile geometries
_G = {}
def _geoms(path):
    if path not in _G:
        _G[path] = gpd.read_file(path).to_crs(4326).geometry
    return _G[path]


def _lines(geom):
    """Plotable line(s) of a geometry: Polygon->outline, LineString->itself.
    geom.boundary returns only the 2 endpoints for a LineString, which silently
    dropped the nine-dashed-line (10 LineStrings)."""
    gt = geom.geom_type
    if gt == "Polygon":         return [geom.boundary]
    if gt == "MultiPolygon":    return [p.boundary for p in geom.geoms]
    if gt == "LineString":      return [geom]
    if gt == "MultiLineString": return list(geom.geoms)
    return []


def _borders(ax):
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5)
    for geom in _geoms(CNTRY):
        for ln in _lines(geom):
            ax.plot(ln.xy[0], ln.xy[1], transform=PROJ, color="#555", lw=0.6)
    for geom in _geoms(NINE):
        for ln in _lines(geom):
            ax.plot(ln.xy[0], ln.xy[1], transform=PROJ, color="#222", lw=0.9)


def _basin(ax):
    g = _geoms(BASIN).union_all().boundary
    for ln in (list(g.geoms) if hasattr(g, "geoms") else [g]):
        ax.plot(ln.xy[0], ln.xy[1], transform=PROJ, color=OI["vermillion"], lw=0.9, alpha=0.9)


def _geo(ax, extent=DISPLAY, lon_lbl=True, lat_lbl=True):
    """fig4-style geo axes: °E/°N formatters, tight tick pad, and per-edge
    label/mark suppression for the shared-axis block (unlabeled edges also drop
    their tick marks)."""
    _borders(ax); _basin(ax)
    ax.set_extent(extent, crs=PROJ)
    ax.set_xticks(range(60, 181, 30), crs=PROJ)
    ax.set_yticks(range(-10, 71, 20), crs=PROJ)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=8.2, pad=1.0)
    if not lon_lbl:
        ax.tick_params(axis="x", length=0, labelbottom=False)
    if not lat_lbl:
        ax.tick_params(axis="y", length=0, labelleft=False)


def _mask_tibet(da):
    return da


def _clip(da):
    # Pad one 2.5° grid cell BEYOND DISPLAY so contourf fills to the frame edge.
    # composite_fields are on the NCEP 2.5° grid; DISPLAY lon 82/138 fall
    # mid-cell, so an exact clip makes contourf fill only 82.5–137.5 and leaves
    # a ~0.5° blank strip each side. Padding covers it and set_extent (DISPLAY)
    # crops the extra out of view. Display-only; data/science unchanged.
    # (lat is descending here: slice(hi, lo).)
    _p = 2.5
    return da.sel(lat=slice(DISPLAY[3] + _p, DISPLAY[2] - _p),
                  lon=slice(DISPLAY[0] - _p, DISPLAY[1] + _p))


# --------------------------------------------------------------------------- #
# IVT map panel (magnitude fill + vectors)
# --------------------------------------------------------------------------- #
def _ivt_map(fig, rect, cax_rect, mag, ux, uy, title, vmax, scale, keyval, cmap="YlGnBu",
             divergent=False, lon_lbl=True, lat_lbl=True, unit="", show_quiverkey=True,
             arrow_width=0.0032, vmin=0):
    ax = fig.add_axes(rect, projection=PROJ)
    cax = fig.add_axes(cax_rect)
    mag = _mask_tibet(_clip(mag))
    ux = _mask_tibet(_clip(ux)); uy = _mask_tibet(_clip(uy))
    mag_vals = mag.values[~np.isnan(mag.values)]
    if divergent:
        pcm = ax.contourf(mag.lon, mag.lat, mag.values, transform=PROJ, cmap=cmap,
                          levels=np.linspace(-vmax, vmax, 41), extend='both')
    else:
        pcm = ax.contourf(mag.lon, mag.lat, mag.values, transform=PROJ, cmap=cmap,
                          levels=np.linspace(vmin, vmax, 41), extend='both')
    _geo(ax, lon_lbl=lon_lbl, lat_lbl=lat_lbl)
    skip = 2   # 49x33-cell domain at this panel width needs every 2nd arrow
    lon = ux.lon.values; lat = ux.lat.values
    U = ux.values[::skip, ::skip]; V = uy.values[::skip, ::skip]
    q = ax.quiver(lon[::skip], lat[::skip], U, V, transform=PROJ, scale=scale,
                  width=arrow_width, color="k", alpha=0.85, minshaft=1.6)
    if show_quiverkey:
        ax.quiverkey(q, 0.78, 1.045, keyval, f"{keyval:g} kg m$^{{-1}}$s$^{{-1}}$",
                     labelpos="E", fontproperties=dict(size=7), labelsep=0.05)
    ax.set_title(title, **TTL)
    cb = fig.colorbar(pcm, cax=cax)
    if divergent:
        cb.set_ticks(np.round(np.linspace(-vmax, vmax, 3), 0))
    else:
        cb.set_ticks(np.round(np.linspace(vmin, vmax, 3), 0))
    cb.ax.tick_params(labelsize=8.5, length=2.2, pad=1.5)
    if unit:
        cb.ax.set_title(unit, fontsize=8.4, pad=2.5)
    return ax


# --------------------------------------------------------------------------- #
# Panel (e) — four-boundary moisture inflow
# --------------------------------------------------------------------------- #
def _panel_e(ax, s):
    bounds = [("F_S_kgs", "South\n(in)"), ("F_N_kgs", "North\n(out)"),
              ("F_W_kgs", "West\n(in)"), ("F_E_kgs", "East\n(out)")]
    combos = [(("early", "YREPCMP"), OI["blue"],       "Early · pattern"),
              (("early", "non"),      OI["sky"],        "Early · non-pattern"),
              (("late",  "YREPCMP"), OI["vermillion"], "Late · pattern"),
              (("late",  "non"),      OI["pink"],       "Late · non-pattern")]
    x = np.arange(len(bounds)); w = 0.15
    for i, ((per, cat), col, lbl) in enumerate(combos):
        vals = [s.loc[(per, cat), k] / 1e8 for k, _ in bounds]
        ax.bar(x + (i - 1.5) * w, vals, w, color=col, alpha=0.92, edgecolor="white",
               lw=0.3, label=lbl)
    ax.axhline(0, color=OI["black"], lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([b[1] for b in bounds], fontsize=9.4)
    ax.set_ylabel("Mean boundary flux  (10$^8$ kg s$^{-1}$)", fontsize=9.3)
    ax.set_title("(e)  Four-boundary moisture inflow", **TTL)
    # box definition demoted from the title to a small gray note
    ax.text(0.995, 0.955, "105–120°E, 34–41°N", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.9, color="#666")
    ax.tick_params(labelsize=9.1, pad=1.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8.1, ncol=1, loc="lower right", framealpha=0.9, handlelength=1.2,
              columnspacing=0.8, labelspacing=0.25)
    # headline: pattern-day south inflow declines
    fse = s.loc[("early", "YREPCMP"), "F_S_kgs"] / 1e8
    fsl = s.loc[("late",  "YREPCMP"), "F_S_kgs"] / 1e8
    ax.annotate(f"pattern-day south inflow\n{fse:.2f} → {fsl:.2f}",
                xy=(0, fse), xytext=(0.55, ax.get_ylim()[1] * 0.80), fontsize=8.5,
                color=OI["vermillion"], ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=OI["vermillion"], lw=0.7))


# --------------------------------------------------------------------------- #
# Panel (f) — moisture budget P = E_app + VIMFC − ∂W/∂t  (stacked, closes to ~0)
# --------------------------------------------------------------------------- #
def _panel_f(ax, s):
    cats = [(("early", "YREPCMP"), "Early\npattern"), (("late", "YREPCMP"), "Late\npattern"),
            (("early", "non"),      "Early\nnon"),      (("late",  "non"),      "Late\nnon")]
    x = np.arange(len(cats)); w = 0.45
    E = np.array([s.loc[k, "E_app_mmday"]     for k, _ in cats])
    V = np.array([s.loc[k, "VIMFC_mmday"]     for k, _ in cats])
    W = np.array([s.loc[k, "neg_dWdt_mmday"]  for k, _ in cats])
    P = np.array([s.loc[k, "Pbar"]            for k, _ in cats])
    ax.bar(x, E, w, color=OI["green"],   edgecolor="white", lw=0.3, label="$E_{app}$ (evap, residual)")
    ax.bar(x, V, w, bottom=E, color=OI["blue"], edgecolor="white", lw=0.3, label="VIMFC (convergence)")
    ax.bar(x, W, w, bottom=E + V, color=OI["orange"], edgecolor="white", lw=0.3,
           label="$-\\partial W/\\partial t$ (storage release)")
    ax.scatter(x, P, color=OI["black"], zorder=5, s=20, label="$P$ (observed precip)")
    # explicit ylim (autoscale mishandles mixed-sign stacks -> pattern-day totals ~7 mm/d
    # were clipped); give headroom for the P dots + close-err labels
    stack = E + V + W
    ax.set_ylim(min(0.0, float(stack.min())) - 0.6, float(max(P.max(), stack.max())) * 1.18)
    for i in range(len(cats)):
        close = P[i] - (E[i] + V[i] + W[i])
        # close labels float on the white panel ABOVE the stacks; one dark color
        # for all four reads cleaner than mixing series colours.
        ax.text(x[i], max(P[i], E[i] + V[i] + W[i]) + 0.25,
                f"close\n{close:+.2f}", ha="center", fontsize=8.3, color=OI["black"])
    ax.axhline(0, color=OI["black"], lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([c[1] for c in cats], fontsize=9.2)
    ax.set_ylabel("mm d$^{-1}$", fontsize=9.3)
    ax.set_title("(f)  Moisture budget  $P = E_{app} + \\mathrm{VIMFC} - \\partial W/\\partial t$", **TTL)
    ax.tick_params(labelsize=9.1, pad=1.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8.1, loc="upper right", framealpha=0.9, handlelength=1.1)


# --------------------------------------------------------------------------- #
def main():
    c = xr.open_dataset(os.path.join(OUT_LEG, "composite_fields_1960_2024.nc"))
    s = pd.read_csv(os.path.join(OUT, "moisture_budget_summary.csv")).set_index(["period", "cat"])

    fig = plt.figure(figsize=(7.48, 8.4))

    # ---- fig4-style aspect-matched grid: GEO_ASP=1.40 so every map FILLS its ----
    # ---- cell (no letterbox). Slim colorbars with top-unit (fig4 _cb).        ----
    LM, RM = 0.050, 0.040
    MAP_W, CB_W, CB_PAD, GUT_X = 0.4175, 0.0075, 0.010, 0.040
    _step = MAP_W + CB_PAD + CB_W + GUT_X
    col_x = [LM, LM + _step]                        # map left x per column
    cb_x  = [cx + MAP_W + CB_PAD for cx in col_x]   # colorbar left x per column
    TM, BM = 0.960, 0.035
    # Vertical gaps: row1(a,b) and row2(c,d) share the lon axis -> row-1 bottom to
    # row-2 title top only has the title ink zone + minimum breathing; row-2->row-3
    # (e,f are bar panels, x differs) leaves more whitespace below the lon labels.
    TITLE_H  = 0.026   # title ink zone above a map (frac of 8.4 in)
    GUT_12   = 0.006   # breathing between row-1 map bottom and row-2 title ink (shared lon)
    XLABEL_H = 0.028   # lon tick labels under row-2 maps
    GUT_23   = 0.030   # whitespace between row-2 lon labels and row-3 title ink
    MAP_H = MAP_W * 7.48 / (GEO_ASP * 8.4)          # rendered map height; fills cell at GEO_ASP=1.40
    row_top = [TM - TITLE_H,
               TM - TITLE_H - TITLE_H - GUT_12 - MAP_H,
               TM - TITLE_H - TITLE_H - GUT_12 - MAP_H - XLABEL_H - GUT_23 - MAP_H]
    row_y   = [rt - MAP_H for rt in row_top]        # map bottom y per row
    bar_w   = MAP_W + CB_PAD + CB_W                 # bar panel spans map+colorbar slot (column-aligned)
    IVT_U   = "kg m$^{-1}$s$^{-1}$"

    # ---- row 1: absolute early/late IVT ----
    _ivt_map(fig, [col_x[0], row_y[0], MAP_W, MAP_H], [cb_x[0], row_y[0], CB_W, MAP_H],
             c["early_YREPCMP__IVT_mag"], c["early_YREPCMP__IVT_x"], c["early_YREPCMP__IVT_y"],
             "(a)  Early 1960–91 IVT  (YR-EPCMP days)", vmax=500, scale=7.0e3, keyval=200,
             lon_lbl=False, lat_lbl=True, unit=IVT_U, show_quiverkey=False, cmap="Blues", vmin=100)
    _ivt_map(fig, [col_x[1], row_y[0], MAP_W, MAP_H], [cb_x[1], row_y[0], CB_W, MAP_H],
             c["late_YREPCMP__IVT_mag"], c["late_YREPCMP__IVT_x"], c["late_YREPCMP__IVT_y"],
             "(b)  Late 1992–24 IVT  (YR-EPCMP days)", vmax=500, scale=7.0e3, keyval=200,
             lon_lbl=False, lat_lbl=False, unit=IVT_U, show_quiverkey=False, cmap="Blues", vmin=100)
    # ---- row 2: late − early ΔIVT / ΔVIMFC ----
    dx = c["late_YREPCMP__IVT_x"] - c["early_YREPCMP__IVT_x"]
    dy = c["late_YREPCMP__IVT_y"] - c["early_YREPCMP__IVT_y"]
    dm = c["late_YREPCMP__IVT_mag"] - c["early_YREPCMP__IVT_mag"]
    _ivt_map(fig, [col_x[0], row_y[1], MAP_W, MAP_H], [cb_x[0], row_y[1], CB_W, MAP_H],
             dm, dx, dy, "(c)  ΔIVT  late − early  (YR-EPCMP days)",
             vmax=130, scale=5.0e3, keyval=50, cmap="RdBu_r", divergent=True,
             lon_lbl=True, lat_lbl=True, unit=IVT_U, show_quiverkey=False,
             arrow_width=0.0045)
    # ΔVIMFC (scalar, no vectors)
    dvim = (c["late_YREPCMP__VIMFC"] - c["early_YREPCMP__VIMFC"]) * 86400.0
    ax_d = fig.add_axes([col_x[1], row_y[1], MAP_W, MAP_H], projection=PROJ)
    dvim_m = _mask_tibet(_clip(dvim))
    dvim_vals = dvim_m.values[~np.isnan(dvim_m.values)]
    dvim_vmax = np.percentile(np.abs(dvim_vals), 95)
    pcm = ax_d.contourf(dvim_m.lon, dvim_m.lat, dvim_m.values, transform=PROJ,
                        cmap="BrBG", levels=np.linspace(-dvim_vmax, dvim_vmax, 41), extend='both')
    _geo(ax_d, lon_lbl=True, lat_lbl=False)
    ax_d.set_title("(d)  ΔVIMFC  late − early", **TTL)
    cb = fig.colorbar(pcm, cax=fig.add_axes([cb_x[1], row_y[1], CB_W, MAP_H]))
    cb.set_ticks(np.round(np.linspace(-dvim_vmax, dvim_vmax, 3), 0))
    cb.ax.tick_params(labelsize=8.5, length=2.2, pad=1.5)
    cb.ax.set_title("mm d$^{-1}$", fontsize=8.4, pad=2.5)
    # ---- row 3: bars (column-aligned with the maps above; same row height) ----
    ax_e = fig.add_axes([col_x[0], row_y[2], bar_w, MAP_H]); _panel_e(ax_e, s)
    ax_f = fig.add_axes([col_x[1], row_y[2], bar_w, MAP_H]); _panel_f(ax_f, s)

    out = os.path.join(FIGDIR, "fig6.png")
    save_figure(fig, out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig6] saved {out}")

    # ---- numeric echo ----
    for per in ["early", "late"]:
        r = s.loc[(per, "YREPCMP")]
        print(f"       {per} pattern: P={r.Pbar:.2f}  E_app={r.E_app_mmday:.2f}  "
              f"VIMFC={r.VIMFC_mmday:+.2f}  -dWdt={r.neg_dWdt_mmday:.2f}  "
              f"F_S={r.F_S_kgs/1e8:.2f}  close={r.Pbar-(r.E_app_mmday+r.VIMFC_mmday+r.neg_dWdt_mmday):+.3f}")
    de = s.loc[("late", "YREPCMP")].Pbar - s.loc[("early", "YREPCMP")].Pbar
    dfs = s.loc[("late", "YREPCMP")].F_S_kgs - s.loc[("early", "YREPCMP")].F_S_kgs
    print(f"       late−early pattern: ΔP=+{de:.2f} mm/d (thermo ↑) but "
          f"ΔF_S(south inflow)={dfs/1e8:+.2f}×10⁸ kg/s (transport ↓)")


if __name__ == "__main__":
    main()
