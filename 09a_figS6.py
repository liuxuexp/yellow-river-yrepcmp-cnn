#!/usr/bin/env python3
"""09a_figS6.py — Fig S6: CHM_PRE vs CN05.1 extreme-precipitation cross-validation.

Validates the paper's extreme-precipitation results against the independent
CN05.1 station-based product over the 1961–2023 overlap.

Panels:
  (a) Wet-day Q95 maps side-by-side: CHM_PRE (regridded 0.25°) | CN05.1 (0.25°)
  (b) Bias map CHM_PRE − CN05.1 Q95 + inset scatter of grid-cell Q95
  (c) YREP-day product agreement: 2×2 confusion table + overlap rate, Jaccard,
      Cohen's κ
  (d) Trend agreement: annual YREP-day count from both products, Theil-Sen +
      Mann-Kendall

Reads: out/precip_daily_region_1960_2024.nc, CN05.1 daily precipitation
Output: figures/figS6.png
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import geopandas as gpd
from scipy.stats import kendalltau
from sklearn.linear_model import TheilSenRegressor
import config

FIGDIR = config.FIGDIR
OUT    = config.OUT

PROJ   = ccrs.PlateCarree()
DISPLAY = [82, 138, 15, 55]                       # East-Asia window; GEO_ASP=56/40=1.40
ZOOM    = [105, 120, 32.5, 43.5]                  # zoom to the basin so the distribution fills the panel
ZOOM_LON = [105, 110, 115, 120]
ZOOM_LAT = [35, 40]
CNTRY   = config.CNTRY_SHP
NINE    = config.NINEDASH_SHP
BASIN   = config.BASIN_SHP

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777", light_gray="#e0e0e0")

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 9.5,
                     "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6})
TTL = dict(fontsize=10.2, fontweight="bold", loc="left", pad=4)

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
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6, color="#333333")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4, color="#888888", alpha=0.6)
    for geom in _geoms(CNTRY):
        for ln in _lines(geom):
            ax.plot(ln.xy[0], ln.xy[1], transform=PROJ, color="#444", lw=0.7)
    for geom in _geoms(NINE):
        for ln in _lines(geom):
            ax.plot(ln.xy[0], ln.xy[1], transform=PROJ, color="#222", lw=1.0)

def _basin(ax):
    g = _geoms(BASIN).union_all().boundary
    for ln in (list(g.geoms) if hasattr(g, "geoms") else [g]):
        ax.plot(ln.xy[0], ln.xy[1], transform=PROJ, color=OI["vermillion"], lw=1.0, alpha=0.95, zorder=5)

def _geo(ax, extent=DISPLAY, lon_ticks=range(85, 139, 15), lat_ticks=range(20, 56, 10),
         xtick_labels=True, ytick_labels=True):
    """Geo axes with degree formatters. extent/lon_ticks/lat_ticks default to the
    full East-Asia window; the Q95 maps pass the basin ZOOM so the distribution
    fills the panel and all map panels share one tick scale.
    xtick_labels/ytick_labels hide the lon/lat labels for shared-axis layouts."""
    _borders(ax); _basin(ax)
    ax.set_extent(extent, crs=PROJ)
    ax.set_xticks(list(lon_ticks), crs=PROJ)
    ax.set_yticks(list(lat_ticks), crs=PROJ)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=8.2, pad=1.5, length=3, width=0.5)
    if not xtick_labels:
        ax.set_xticklabels([])
    if not ytick_labels:
        ax.set_yticklabels([])
    ax.set_title("")

def _clip(da):
    m = (da.lat >= DISPLAY[2]) & (da.lat <= DISPLAY[3]) & \
        (da.lon >= DISPLAY[0]) & (da.lon <= DISPLAY[1])
    return da.where(m, other=np.nan)

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
def load_data():
    """Load CHM_PRE cached data and CN05.1 data (subset to basin box)."""
    chm = xr.open_dataset(os.path.join(OUT, "precip_daily_region_1960_2024.nc"))
    # Restrict to 1961-2023 overlap
    chm_overlap = chm.sel(time=slice("1961-01-01", "2023-12-31"))

    # Basin box for CN05.1 subsetting — wider than the basin so the recomputed
    # Q95 fills the zoomed map panel; eager-load after subsetting.
    lat_box = slice(32, 44)
    lon_box = slice(104, 121)

    # Load CN05.1 (subset FIRST, then load)
    cn05 = xr.open_dataset(config.CN051_PATH)
    cn05_box = cn05["pre"].sel(lat=lat_box, lon=lon_box).load()
    cn05.close()

    return chm_overlap, cn05_box

# --------------------------------------------------------------------------- #
# Compute CN05.1 wet-day Q95 and YREP metrics
# --------------------------------------------------------------------------- #
def compute_cn05_metrics(cn05, basin_mask_chm):
    """Compute CN05.1 wet-day Q95, F(d), Pbar, YREP using same definition as CHM_PRE."""
    basin_geom = _geoms(BASIN).union_all()

    # Create mask on CN05.1 grid
    lon_2d, lat_2d = np.meshgrid(cn05.lon.values, cn05.lat.values)
    from shapely.geometry import Point
    mask_points = np.array([Point(lon, lat).within(basin_geom)
                            for lon, lat in zip(lon_2d.flat, lat_2d.flat)])
    cn05_mask = mask_points.reshape(lon_2d.shape).astype(float)

    # Get basin indices
    basin_idx = cn05_mask > 0
    basin_lats = cn05.lat.values[basin_idx[:, 0]]
    basin_lons = cn05.lon.values[basin_idx[0, :]]

    # Wet-day Q95 per grid cell, vectorised over the whole loaded box: non-wet
    # days (<1 mm) set to NaN, np.nanquantile takes the 95th pct per cell along
    # time. Computed for ALL cells (not just basin) so the Q95 MAP fills the
    # panel; basin cells are masked separately only where a basin-only quantity
    # (scatter, YREP F(d)) is needed. Identical definition to the CHM_PRE Q95.
    import warnings
    _precip = cn05.values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q95_grid = np.nanquantile(np.where(_precip >= 1.0, _precip, np.nan), 0.95, axis=0)

    cn05_q95 = xr.DataArray(q95_grid, dims=["lat", "lon"],
                            coords={"lat": cn05.lat.values, "lon": cn05.lon.values})

    # Compute daily F(d): fraction of basin cells exceeding local Q95
    basin_cells = cn05_mask > 0
    n_basin_cells = np.sum(basin_cells)

    fd_daily = []
    pbar_daily = []

    lat_weights_1d = np.cos(np.deg2rad(cn05.lat.values))
    lat_weights_2d = lat_weights_1d[:, np.newaxis] * np.ones_like(cn05_mask)

    for t in range(len(cn05.time)):
        daily_precip = cn05[t].values

        exceed = daily_precip > q95_grid
        fd = np.sum(exceed[basin_cells]) / n_basin_cells
        fd_daily.append(fd)

        valid_mask = basin_cells & (~np.isnan(daily_precip))
        if np.sum(valid_mask) > 0:
            valid_weights = lat_weights_2d[valid_mask]
            valid_precip = daily_precip[valid_mask]
            pbar = np.sum(valid_precip * valid_weights) / np.sum(valid_weights)
        else:
            pbar = np.nan
        pbar_daily.append(pbar)

    fd_daily = np.array(fd_daily)
    pbar_daily = np.array(pbar_daily)

    # P90 of wet-day area-mean (wet days: area-mean P >= 1 mm)
    wet_days = pbar_daily >= 1.0
    if np.sum(wet_days) == 0:
        print("[WARNING] No wet days found in CN05.1 basin mean, using all days for P90")
        wet_days = np.ones_like(pbar_daily, dtype=bool)
    p90_wetday = np.quantile(pbar_daily[wet_days], 0.90)

    # YREP: F(d) >= 10% AND area-mean P >= P90 of wet-day
    yrep_daily = (fd_daily >= 0.10) & (pbar_daily >= p90_wetday)

    print(f"       CN05.1: wet_day_days={np.sum(wet_days)}, P90={p90_wetday:.2f} mm/d, YREP_days={np.sum(yrep_daily)}")

    return cn05_q95, fd_daily, pbar_daily, yrep_daily, cn05_mask, p90_wetday

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("[figS6] loading data...")
    chm, cn05 = load_data()

    print("[figS6] computing CN05.1 metrics...")
    cn05_q95, cn05_fd, cn05_pbar, cn05_yrep, cn05_mask, p90_wetday = compute_cn05_metrics(cn05, chm["mask"])

    # Get CHM_PRE metrics (cached)
    chm_q95 = chm["Q95"]
    chm_fd = chm["F"].values
    chm_pbar = chm["Pbar"].values
    chm_yrep = chm["YREP"].values

    # Regrid CHM_PRE Q95 to 0.25° for comparison
    print("[figS6] regridding CHM_PRE Q95 to 0.25°...")
    chm_q95_int = chm_q95.interp(lat=cn05.lat.values, lon=cn05.lon.values, method="linear")

    # Mask both to basin
    lon_2d, lat_2d = np.meshgrid(chm_q95_int.lon.values, chm_q95_int.lat.values)
    from shapely.geometry import Point
    basin_geom = _geoms(BASIN).union_all()
    mask_chm_025 = np.array([Point(lon, lat).within(basin_geom)
                             for lon, lat in zip(lon_2d.flat, lat_2d.flat)])
    mask_chm_025 = mask_chm_025.reshape(lon_2d.shape)

    chm_q95_masked = chm_q95_int.where(mask_chm_025 > 0)
    cn05_q95_masked = cn05_q95.where(cn05_mask > 0)

    # ----------------------------------------------------------------------- #
    # Panel (a): side-by-side Q95 maps
    # ----------------------------------------------------------------------- #
    fig = plt.figure(figsize=(7.48, 8.5))
    Q95_UNIT = "mm d$^{-1}$"

    # Shared Q95 colour scale — anchored on the BASIN cells; the maps display the
    # UNMASKED field over the zoom box so the distribution fills the panel, with
    # the basin outline drawn on top.
    q95_min = min(np.nanmin(chm_q95_masked.values), np.nanmin(cn05_q95_masked.values))
    q95_max = max(np.nanmax(chm_q95_masked.values), np.nanmax(cn05_q95_masked.values))
    q95_ticks = np.round(np.linspace(q95_min, q95_max, 3), 0)

    # CHM_PRE Q95 — full field, zoomed to the basin region (basin only)
    ax_a1 = fig.add_axes([0.06, 0.699, 0.39, 0.254], projection=PROJ)
    pcm_a1 = ax_a1.contourf(chm_q95_masked.lon, chm_q95_masked.lat, chm_q95_masked.values, transform=PROJ,
                            cmap="YlGnBu", levels=np.linspace(q95_min, q95_max, 61),
                            extend="both", alpha=0.95, antialiased=True)
    _geo(ax_a1, extent=ZOOM, lon_ticks=ZOOM_LON, lat_ticks=ZOOM_LAT,
         xtick_labels=True, ytick_labels=True)
    ax_a1.set_title("(a)  CHM_PRE wet-day Q95", **TTL)
    cax_a1 = fig.add_axes([0.462, 0.699, 0.012, 0.254])
    cb_a1 = fig.colorbar(pcm_a1, cax=cax_a1)
    cb_a1.set_ticks(q95_ticks)
    cb_a1.ax.tick_params(labelsize=8.8, length=2.5, pad=1.8)
    cb_a1.ax.set_title(Q95_UNIT, fontsize=8.8, pad=4)
    cb_a1.outline.set_linewidth(0.5)

    # CN05.1 Q95 — full field, shared scale with (a) (basin only)
    ax_a2 = fig.add_axes([0.515, 0.699, 0.39, 0.254], projection=PROJ)
    pcm_a2 = ax_a2.contourf(cn05_q95_masked.lon, cn05_q95_masked.lat, cn05_q95_masked.values, transform=PROJ,
                            cmap="YlGnBu", levels=np.linspace(q95_min, q95_max, 61),
                            extend="both", alpha=0.95, antialiased=True)
    _geo(ax_a2, extent=ZOOM, lon_ticks=ZOOM_LON, lat_ticks=ZOOM_LAT,
         xtick_labels=True, ytick_labels=False)
    ax_a2.set_title("(b)  CN05.1 wet-day Q95", **TTL)
    cax_a2 = fig.add_axes([0.917, 0.699, 0.012, 0.254])
    cb_a2 = fig.colorbar(pcm_a2, cax=cax_a2)
    cb_a2.set_ticks(q95_ticks)
    cb_a2.ax.tick_params(labelsize=8.8, length=2.5, pad=1.8)
    cb_a2.ax.set_title(Q95_UNIT, fontsize=8.8, pad=4)
    cb_a2.outline.set_linewidth(0.5)

    # ----------------------------------------------------------------------- #
    # Panel (b): bias map + scatter
    # ----------------------------------------------------------------------- #
    # Bias masked to the basin so panel (c) shares the basin-only extent.
    bias = chm_q95_masked - cn05_q95_masked
    bias_max = max(abs(np.nanmin(bias.values)), abs(np.nanmax(bias.values)))
    bias_ticks = np.round(np.linspace(-bias_max, bias_max, 3), 0)

    ax_b = fig.add_axes([0.06, 0.3818, 0.39, 0.254], projection=PROJ)
    pcm_b = ax_b.contourf(bias.lon, bias.lat, bias.values, transform=PROJ,
                          cmap="RdBu_r", levels=np.linspace(-bias_max, bias_max, 61),
                          extend="both", alpha=0.95, antialiased=True)
    _geo(ax_b, extent=ZOOM, lon_ticks=ZOOM_LON, lat_ticks=ZOOM_LAT)
    ax_b.set_title("(c)  Bias CHM_PRE $-$ CN05.1 Q95", **TTL)
    cax_b = fig.add_axes([0.462, 0.3818, 0.012, 0.254])
    cb_b = fig.colorbar(pcm_b, cax=cax_b)
    cb_b.set_ticks(bias_ticks)
    cb_b.ax.tick_params(labelsize=8.8, length=2.5, pad=1.8)
    cb_b.ax.set_title(Q95_UNIT, fontsize=8.8, pad=4)
    cb_b.outline.set_linewidth(0.5)

    # Scatter inset
    ax_scat = fig.add_axes([0.515, 0.3818, 0.39, 0.254])
    valid = (~np.isnan(chm_q95_masked.values)) & (~np.isnan(cn05_q95_masked.values))
    chm_vals = chm_q95_masked.values[valid]
    cn05_vals = cn05_q95_masked.values[valid]

    ax_scat.scatter(cn05_vals, chm_vals, s=3, color=OI["blue"], alpha=0.5, edgecolors="none")
    # 1:1 line spanning the shared Q95 scale (same range as the maps' colour bar)
    ax_scat.plot([q95_min, q95_max], [q95_min, q95_max], "k--", lw=0.8, label="1:1")
    # Least squares fit
    slope, intercept, r_val, p_val, std_err = stats.linregress(cn05_vals, chm_vals)
    ax_scat.plot(cn05_vals, intercept + slope * cn05_vals, "r-", lw=1.2,
                label=f"Fit: y={slope:.2f}x+{intercept:.2f}")
    ax_scat.set_xlabel("CN05.1 Q95 (mm d$^{-1}$)", fontsize=8.6, labelpad=3)
    ax_scat.set_ylabel("CHM_PRE Q95 (mm d$^{-1}$)", fontsize=8.6, labelpad=3)
    ax_scat.yaxis.tick_right()                       # y-axis on the right
    ax_scat.yaxis.set_label_position("right")
    ax_scat.set_xlim(q95_min, q95_max); ax_scat.set_ylim(q95_min, q95_max)   # shared scale with a/b
    ax_scat.set_title("(d)  Grid-cell Q95 comparison", **TTL)
    ax_scat.tick_params(labelsize=8.2, length=3, width=0.5, pad=2)
    ax_scat.tick_params(axis="x", direction="in")   # x tick marks inward
    ax_scat.legend(fontsize=7.3, loc="upper left", framealpha=0.9)
    ax_scat.grid(True, axis="y", color=OI["light_gray"], lw=0.5, alpha=0.5)
    ax_scat.set_axisbelow(True)
    ax_scat.text(0.97, 0.05, f"Pearson r = {r_val:.3f}", transform=ax_scat.transAxes,
                 fontsize=8.2, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ----------------------------------------------------------------------- #
    # Panel (e): YREP agreement heatmap (clearer than a table)
    # ----------------------------------------------------------------------- #
    ax_conf = fig.add_axes([0.06, 0.0646, 0.39, 0.254])

    n_tp = np.sum(chm_yrep & cn05_yrep)
    n_fp = np.sum(chm_yrep & ~cn05_yrep)
    n_fn = np.sum(~chm_yrep & cn05_yrep)
    n_tn = np.sum(~chm_yrep & ~cn05_yrep)
    total = len(chm_yrep)

    overlap_rate = (n_tp + n_tn) / total
    jaccard = n_tp / (n_tp + n_fp + n_fn) if (n_tp + n_fp + n_fn) > 0 else np.nan
    # Cohen's kappa
    pe = ((n_tp + n_fp) * (n_tp + n_fn) + (n_fn + n_tn) * (n_fp + n_tn)) / (total ** 2)
    kappa = (overlap_rate - pe) / (1 - pe) if pe < 1 else np.nan

    # Confusion-matrix heatmap: rows = CHM_PRE, cols = CN05.1; diagonal = agreement
    M = np.array([[n_tp, n_fp], [n_fn, n_tn]])
    im = ax_conf.imshow(M, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            pct = 100 * M[i, j] / total
            ax_conf.text(j, i, f"{M[i, j]:,}\n({pct:.1f}%)", ha="center", va="center",
                         fontsize=11, color="white" if M[i, j] > M.max() / 2 else "black")
    ax_conf.set_xticks([0, 1]); ax_conf.set_xticklabels(["YREP", "Non-YREP"], fontsize=8.3)
    ax_conf.set_yticks([0, 1]); ax_conf.set_yticklabels(["YREP", "Non-YREP"], fontsize=8.3, rotation=90, va="center")
    ax_conf.set_xlabel("CN05.1", fontsize=8.8, labelpad=2)
    ax_conf.set_ylabel("CHM_PRE", fontsize=8.8, labelpad=2)
    ax_conf.tick_params(pad=1.2)
    ax_conf.set_title("(e)  YREP-day agreement", **TTL)

    # Agreement metrics inside the heatmap (top-centre strip)
    metrics_text = f"overlap {overlap_rate:.3f}    Jaccard {jaccard:.3f}    Cohen κ {kappa:.3f}"
    ax_conf.text(0.5, 0.965, metrics_text, transform=ax_conf.transAxes,
                 fontsize=8.0, ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, edgecolor="gray"))

    # ----------------------------------------------------------------------- #
    # Panel (d): Trend agreement (annual YREP-day count)
    # ----------------------------------------------------------------------- #
    ax_trend = fig.add_axes([0.515, 0.0646, 0.39, 0.254])

    years_chm = pd.to_datetime(chm.time.values).year
    years_cn05 = pd.to_datetime(cn05.time.values).year

    chm_annual = [np.sum(chm_yrep[years_chm == y]) for y in range(1961, 2024)]
    cn05_annual = [np.sum(cn05_yrep[years_cn05 == y]) for y in range(1961, 2024)]
    years = np.arange(1961, 2024)

    # Theil-Sen slope + Mann-Kendall
    ts_chm = TheilSenRegressor().fit(years.reshape(-1, 1), chm_annual)
    slope_chm = ts_chm.coef_[0]
    mk_chm, p_chm = kendalltau(years, chm_annual)

    ts_cn05 = TheilSenRegressor().fit(years.reshape(-1, 1), cn05_annual)
    slope_cn05 = ts_cn05.coef_[0]
    mk_cn05, p_cn05 = kendalltau(years, cn05_annual)

    ax_trend.plot(years, chm_annual, "o-", color=OI["blue"], ms=3, lw=1.0, label="CHM_PRE", alpha=0.8)
    ax_trend.plot(years, cn05_annual, "s-", color=OI["vermillion"], ms=3, lw=1.0, label="CN05.1", alpha=0.8)

    # Trend lines
    ax_trend.plot(years, ts_chm.predict(years.reshape(-1, 1)), "--", color=OI["blue"], lw=1.2, alpha=0.7)
    ax_trend.plot(years, ts_cn05.predict(years.reshape(-1, 1)), "--", color=OI["vermillion"], lw=1.2, alpha=0.7)

    ax_trend.set_xlabel("Year", fontsize=8.6, labelpad=3)
    ax_trend.set_ylabel("Annual YREP-day count", fontsize=8.6, labelpad=3)
    ax_trend.yaxis.tick_right()                       # y-axis on the right
    ax_trend.yaxis.set_label_position("right")
    ax_trend.set_title("(f)  Trend agreement: annual YREP-day count", **TTL)
    ax_trend.tick_params(labelsize=8.2, length=3, width=0.5, pad=2)
    ax_trend.legend(fontsize=7.3, loc="upper left", framealpha=0.9)
    ax_trend.grid(True, axis="y", color=OI["light_gray"], lw=0.5, alpha=0.5)
    ax_trend.set_axisbelow(True)

    # Trend info (plain text, no frame)
    trend_text = (f"CHM_PRE: slope={slope_chm:.2f} d/yr, MK p={p_chm:.3f}\n"
                  f"CN05.1: slope={slope_cn05:.2f} d/yr, MK p={p_cn05:.3f}")
    ax_trend.text(0.99, 0.97, trend_text, transform=ax_trend.transAxes,
                  fontsize=7.8, ha="right", va="top")

    # ----------------------------------------------------------------------- #
    # Save
    # ----------------------------------------------------------------------- #
    out_path = os.path.join(FIGDIR, "figS6.png")
    save_figure(fig, out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figS6] saved {out_path}")

    # ----------------------------------------------------------------------- #
    # Print numeric summary
    # ----------------------------------------------------------------------- #
    print(f"\n[figS6] Q95 correlation:")
    print(f"       Pearson r = {r_val:.3f}")
    print(f"\n[figS6] YREP agreement:")
    print(f"       TP={n_tp}, FP={n_fp}, FN={n_fn}, TN={n_tn}")
    print(f"       Overlap rate = {overlap_rate:.3f}")
    print(f"       Jaccard index = {jaccard:.3f}")
    print(f"       Cohen's κ = {kappa:.3f}")
    print(f"\n[figS6] Trends (annual YREP-day count):")
    print(f"       CHM_PRE: slope={slope_chm:.2f} d/yr, MK p={p_chm:.3f}")
    print(f"       CN05.1: slope={slope_cn05:.2f} d/yr, MK p={p_cn05:.3f}")

    # Verdict
    print(f"\n[figS6] VERDICT:")
    if r_val > 0.8 and kappa > 0.6:
        print("       Products AGREE on extreme-precipitation climatology and YREP labels.")
        print("       CHM_PRE extreme-precipitation results are validated by CN05.1.")
    elif r_val > 0.6 and kappa > 0.4:
        print("       Products show MODERATE agreement. CHM_PRE results reasonably robust.")
    else:
        print("       Products DIVERGE substantially. Caveat needed for CHM_PRE extreme tail.")

    # File info
    file_size = os.path.getsize(out_path)
    from PIL import Image
    img = Image.open(out_path)
    print(f"\n[figS6] File info:")
    print(f"       Path: {out_path}")
    print(f"       Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    print(f"       Dimensions: {img.size[0]} × {img.size[1]} pixels")


if __name__ == "__main__":
    main()
