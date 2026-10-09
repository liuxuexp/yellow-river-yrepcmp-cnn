#!/usr/bin/env python3
"""09a_fig7.py — Fig 7: coverage area + connected-object trend + sync days + case.

Four panels:
  (a) coverage-area distribution — histogram of the fraction of basin grid cells
      exceeding local wet-day Q95 on YREP days
  (b) max connected-object area trend — annual time series of largest contiguous
      precip-exceedance object area, with Theil-Sen slope + Mann-Kendall p
  (c) mid/lower synchronous extreme days — days where BOTH middle and lower
      reaches are extreme simultaneously (annual count/trend)
  (d) typical large-scale event — spatial precip map (CHM_PRE) of the largest
      basin-wide extreme day (2012-09-01), with basin outline

Reads: out/footprint_yearly_1960_2024.csv, out/footprint_daily_1960_2024.csv,
       CHM_PRE_V2_daily_{YYYY}.nc
Output: figures/fig7.png
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import geopandas as gpd
from scipy.stats import kendalltau
from scipy.stats import theilslopes
import config

FIGDIR = config.FIGDIR
OUT    = config.OUT

PROJ    = ccrs.PlateCarree()
DISPLAY = [82, 138, 15, 55]
GEO_ASP = (DISPLAY[1]-DISPLAY[0])/(DISPLAY[3]-DISPLAY[2])
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

# Panel-(d) precipitation colormap: start from white, blending into YlGnBu so the
# lowest values (near 0 mm/d) read as white rather than pale yellow.
_base_ygb = plt.get_cmap("YlGnBu")
_cols = [np.array(_base_ygb(i)) for i in np.linspace(0.0, 1.0, _base_ygb.N)]
_n_blend = 40  # first ~15% blends white -> YlGnBu
for _k in range(_n_blend):
    _t = _k / (_n_blend - 1)
    _cols[_k] = (1 - _t) * np.array([1, 1, 1, 1]) + _t * _cols[_k]
CMAP_D = LinearSegmentedColormap.from_list("W_YlGnBu", _cols)

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
    """fig4/fig6-style geo axes: °E/°N formatters, tight tick pad, and per-edge
    label/mark suppression for the shared-axis block."""
    _borders(ax); _basin(ax)
    ax.set_extent(extent, crs=PROJ)
    ax.set_xticks(range(85, 139, 15), crs=PROJ)   # 85,100,115,130 °E within lon 82-138
    ax.set_yticks(range(20, 56, 10), crs=PROJ)    # 20,30,40,50 °N within lat 15-55
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=7, pad=1.0)
    if not lon_lbl:
        ax.tick_params(axis="x", length=0, labelbottom=False)
    if not lat_lbl:
        ax.tick_params(axis="y", length=0, labelleft=False)


def _trend_label(x, y, ax, units="", color=OI["vermillion"], slope_fmt="+.1f", ci_line=None):
    """Compute Theil-Sen slope + Mann-Kendall p, plot the trend line, build a label."""
    slope, intercept, _, _ = theilslopes(y, x, alpha=0.95)
    tau, p_mk = kendalltau(x, y)
    # trend line
    y_trend = slope * x + intercept
    ax.plot(x, y_trend, "--", color=color, lw=1.0, alpha=0.8, zorder=3)
    # label with significance
    sig_text = "p<0.05" if p_mk < 0.05 else "p≥0.05"
    lines = [f"{slope:{slope_fmt}} {units}/yr"]
    if ci_line:
        lines.append(ci_line)
    lines.append(sig_text)
    return "\n".join(lines), slope, p_mk


def _bootstrap_ci(x, y, n_boot=2000, seed=42, alpha=0.05):
    """Bootstrap 95% CI for the Theil-Sen trend.

    Resamples the (x, y) pairs with replacement, refits Theil-Sen each draw, and
    returns both (i) a band = pointwise 2.5/97.5-percentile of the predicted trend
    line across x (for fill_between) and (ii) the 2.5/97.5-percentile of the slope
    itself (for the text label). Same draws -> band and slope-CI are consistent."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    slopes = np.empty(n_boot)
    preds = np.empty((n_boot, n))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sl, ic, _, _ = theilslopes(y[idx], x[idx])
        slopes[b] = sl
        preds[b] = sl * x + ic
    lo_band = np.quantile(preds, alpha / 2, axis=0)
    hi_band = np.quantile(preds, 1 - alpha / 2, axis=0)
    sl_lo = float(np.quantile(slopes, alpha / 2))
    sl_hi = float(np.quantile(slopes, 1 - alpha / 2))
    return lo_band, hi_band, sl_lo, sl_hi


# --------------------------------------------------------------------------- #
# Panel (a) — coverage-area distribution
# --------------------------------------------------------------------------- #
def _panel_a(ax, da):
    """Histogram of extreme area fraction on YREP days."""
    # Using total_extreme_area_km2 normalized by basin area (≈694,868 km²)
    basin_area_km2 = 694868.0
    coverage_frac = da.total_extreme_area_km2 / basin_area_km2

    ax.hist(coverage_frac, bins=30, color=OI["blue"], alpha=0.85, edgecolor="white", lw=0.4)
    ax.axvline(coverage_frac.median(), color=OI["vermillion"], lw=1.2, ls="--",
               label=f"Median {coverage_frac.median():.2f}")
    ax.axvline(0.1, color=OI["black"], lw=0.8, ls=":", alpha=0.6, label="10% threshold")

    ax.set_xlabel("Basin extreme area fraction", fontsize=8.6, labelpad=1.5)
    ax.set_ylabel("YREP days", fontsize=8.6)
    ax.set_title("(a)  Coverage-area distribution", **TTL)
    ax.tick_params(labelsize=7.5)
    # legend anchor moved left of the right spine, still right of the bars.
    ax.legend(fontsize=6.5, loc="upper right", bbox_to_anchor=(0.96, 0.94),
              framealpha=0.9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# --------------------------------------------------------------------------- #
# Panel (b) — max connected-object area trend
# --------------------------------------------------------------------------- #
def _panel_b(ax, yr):
    """Annual time series of largest contiguous object area with Theil-Sen + MK test."""
    x = yr.year.values
    y = yr.max_object_km2.values / 1000.0  # convert to 1000 km² for readability

    ax.scatter(x, y, color=OI["blue"], s=14, alpha=0.75, edgecolor="white", lw=0.3, zorder=2)
    # bootstrap 95% CI band (light vermillion) + slope CI in the label
    lo, hi, sl_lo, sl_hi = _bootstrap_ci(x, y)
    ax.fill_between(x, lo, hi, color=OI["vermillion"], alpha=0.18, lw=0, zorder=1.5)
    ci_line = f"95% CI [{sl_lo:+.2f}, {sl_hi:+.2f}]"
    label_text, slope, p_mk = _trend_label(x, y, ax, units="×10³ km²", slope_fmt="+.2f", ci_line=ci_line)

    # Add label in upper left area
    ax.text(0.1, 0.97, label_text, transform=ax.transAxes, fontsize=6.8,
            color=OI["vermillion"], va="top", ha="left", alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", lw=0, alpha=0.5))

    ax.set_xlabel("Year", fontsize=8.6, labelpad=1.5)
    ax.set_ylabel("Max object area (10³ km²)", fontsize=8.6)
    ax.set_title("(b)  Largest connected-object area trend", **TTL)
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(1959.5, 2024.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    return slope, p_mk


# --------------------------------------------------------------------------- #
# Panel (c) — mid/lower synchronous extreme days
# --------------------------------------------------------------------------- #
def _panel_c(ax, yr):
    """Annual count of days where BOTH middle and lower reaches are extreme."""
    x = yr.year.values
    y = yr.synced_days.values

    ax.bar(x, y, color=OI["sky"], alpha=0.85, edgecolor="white", lw=0.4, width=0.7)
    # bootstrap 95% CI band (light vermillion) + slope CI in the label
    lo, hi, sl_lo, sl_hi = _bootstrap_ci(x, y)
    ax.fill_between(x, lo, hi, color=OI["vermillion"], alpha=0.18, lw=0, zorder=1.5)
    ci_line = f"95% CI [{sl_lo:+.3f}, {sl_hi:+.3f}]"
    label_text, slope, p_mk = _trend_label(x, y, ax, units="days", slope_fmt="+.3f", ci_line=ci_line)

    # Add label in upper left area
    ax.text(0.1, 0.97, label_text, transform=ax.transAxes, fontsize=6.8,
            color=OI["vermillion"], va="top", ha="left", alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", lw=0, alpha=0.5))

    ax.set_xlabel("Year", fontsize=8.6, labelpad=1.5)
    ax.set_ylabel("Synchronized days (yr⁻¹)", fontsize=8.6)
    ax.set_title("(c)  Middle–Lower synchronous extremes", **TTL)
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(1959.5, 2024.5)
    ax.set_ylim(0, max(y) * 1.15)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    return slope, p_mk


# --------------------------------------------------------------------------- #
# Panel (d) — typical large-scale event (spatial precip map)
# --------------------------------------------------------------------------- #
def _panel_d_map(ax, event_date="2012-09-01"):
    """Spatial CHM_PRE precip map of the largest basin-wide extreme day.
    Returns (pcm, vmax) so the caller can build a colorbar whose height matches
    the map AFTER layout is finalised (GeoAxes box is reliable only post-draw)."""
    vmax = 100  # mm/day cap for visualization
    ts = pd.Timestamp(event_date)
    ds = xr.open_dataset(config.chm_path(ts.year))
    prec = ds["prec"].sel(time=ts)
    ds.close()

    # Pad one grid cell beyond DISPLAY so contourf fills to the frame edge.
    # CHM_PRE lat may be ascending or descending — clip direction-agnostically.
    _p = 0.2
    lat_lo, lat_hi = DISPLAY[2] - _p, DISPLAY[3] + _p
    lon_lo, lon_hi = DISPLAY[0] - _p, DISPLAY[1] + _p
    prec_region = prec.where((prec.lat >= lat_lo) & (prec.lat <= lat_hi), drop=True)
    prec_region = prec_region.where((prec_region.lon >= lon_lo) & (prec_region.lon <= lon_hi), drop=True)

    pcm = ax.contourf(prec_region.lon, prec_region.lat, prec_region.values,
                      transform=PROJ, cmap=CMAP_D,
                      levels=np.linspace(0, vmax, 41), extend='max')

    _geo(ax, lon_lbl=True, lat_lbl=True)
    ax.set_title("(d)  Typical large-scale event", **TTL)
    # event date inside the map (title too long, move date into the panel)
    ax.text(0.03, 0.94, ts.strftime("%Y-%m-%d"), transform=ax.transAxes,
            fontsize=6.8, color="#222", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=0.7))
    return pcm, vmax


# --------------------------------------------------------------------------- #
def main():
    # Load data
    yr = pd.read_csv(os.path.join(OUT, "footprint_yearly_1960_2024.csv"))
    da = pd.read_csv(os.path.join(OUT, "footprint_daily_1960_2024.csv"))

    # Lay out a 2x2 grid whose cell aspect equals the map's display-extent aspect,
    # so the cartopy map (panel d) fills its cell with no letterboxing. All four
    # panels share one height and a colorbar built after layout matches the map
    # height exactly.
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(6.5, 4.9))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.085, right=0.905, bottom=0.10, top=0.92,
                           wspace=0.18, hspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1], projection=PROJ)

    _panel_a(ax_a, da)
    slope_b, p_b = _panel_b(ax_b, yr)
    slope_c, p_c = _panel_c(ax_c, yr)
    pcm, vmax = _panel_d_map(ax_d)

    # Finalise layout, then place the colorbar flush with the map's actual height
    # (GeoAxes box is reliable only after a draw, once cartopy has applied its aspect).
    fig.canvas.draw()
    pos = ax_d.get_position()
    cax = fig.add_axes([pos.x1 + 0.006, pos.y0, 0.012, pos.height])
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_ticks(np.round(np.linspace(0, vmax, 3), 0))
    cb.ax.tick_params(labelsize=7.8, length=2.2, pad=1.5)
    cb.ax.set_title("mm d$^{-1}$", fontsize=7.8, pad=2.5)   # directly above the bar (centered)
    # nudge slightly right of centre — in display POINTS, because an axes-fraction
    # nudge on the 2-mm-wide cax would be invisible.
    from matplotlib.transforms import ScaledTranslation
    cb.ax.title.set_transform(cb.ax.title.get_transform()
                              + ScaledTranslation(6.0/72.0, 0, fig.dpi_scale_trans))

    out = os.path.join(FIGDIR, "fig7.png")
    save_figure(fig, out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig7] saved {out}")

    # ---- numeric echo ----
    print(f"[fig7] Panel (b) max object trend: slope={slope_b:.2f}×10³ km²/yr, p={p_b:.4f}")
    print(f"[fig7] Panel (c) synced days trend: slope={slope_c:.3f} days/yr, p={p_c:.4f}")
    print(f"[fig7] Total synced days 1960-2024: {yr.synced_days.sum()} (mean {yr.synced_days.mean():.2f}/yr)")

    # Check output file size
    if os.path.exists(out):
        size = os.path.getsize(out)
        print(f"[fig7] file size: {size:,} bytes ({size/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
