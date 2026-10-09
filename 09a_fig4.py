#!/usr/bin/env python3
"""09a_fig4.py — Fig 4: YR-EPCMP composite + LRP relevance + typical case.

Layout (9 panels, 3 columns x 3 rows, grouped by COLUMN; each column's top-2
panels share the x-axis; every map panel carries its OWN colorbar):
  col 1: (a) Z500 anom | (b) SLP anom | (g) per-channel relevance bar
  col 2: (d) VIMFC anom | (e) 850 q+wind anom | (c) IVT-mag + IVT vectors
  col 3: (f) LRP total relevance | (i) per-day LRP | (h) precip 2016-07-19

The composite is on the CALENDAR-DAY ANOMALY (31-day rolling climatology
removed), not the raw pattern-minus-non-pattern difference, so the seasonal
cycle (JAS-concentrated days) does not dominate.

Reads: out/diagnostics_native_1960_2024.nc, out/prediction_probability_full.csv,
       out/relevance_LRP_map_combined.nc, out/relevance_IG_combined.nc,
       out/relevance_SmoothGrad_combined.nc, out/cnn_inputs_1deg_1960_2024.nc,
       out/precip_daily_region_1960_2024.nc, out/model_combined.pt,
       CHM_PRE_V2_daily_{YYYY}.nc
Output: figures/fig4.png
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import geopandas as gpd
import config

FIGDIR = config.FIGDIR
OUT    = config.OUT
sys.path.insert(0, str(config.ROOT))

BASIN_SHP   = config.BASIN_SHP
CNTRY_SHP   = config.CNTRY_SHP
NINEDASH_SHP= config.NINEDASH_SHP
CHM_DIR     = config.CHM_DIR

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777")
PROJ = ccrs.PlateCarree()
DISPLAY = [60, 180, -10, 70]
BASIN_BOX = [104.5, 119.5, 32.9, 43.6]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 9.1,
                     "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6})
TTL = dict(fontsize=9.4, fontweight="bold", loc="left", pad=3)


# --------------------------------------------------------------------------- #
# Enhanced colormaps for better visual effect
# --------------------------------------------------------------------------- #
def create_div_cmap(name, colors, N=256, symmetric=True):
    """Create a diverging colormap from a list of colors."""
    n = len(colors)
    if symmetric:
        colors_rev = colors[::-1][:-1]
        all_colors = colors_rev + colors
    else:
        all_colors = colors
    return ListedColormap(plt.cm.colors.LinearSegmentedColormap.from_list(name, all_colors, N=N)(np.linspace(0, 1, N)))


CMAP_Z500 = plt.get_cmap("RdBu_r")
CMAP_SLP = plt.get_cmap("RdBu_r")
CMAP_VIMFC = plt.get_cmap("BrBG")
CMAP_Q850 = plt.get_cmap("BrBG")
CMAP_IVT = plt.get_cmap("BrBG")
CMAP_PRECIP = plt.get_cmap("YlGnBu")

_base_mr = plt.get_cmap("magma_r")
_cmr = [np.array(_base_mr(i)) for i in np.linspace(0.0, 1.0, _base_mr.N)]
_nblend = 40
for _k in range(_nblend):
    _t = _k / (_nblend - 1)
    _cmr[_k] = (1 - _t) * np.array([1, 1, 1, 1]) + _t * _cmr[_k]
CMAP_F = LinearSegmentedColormap.from_list("W_magma_r", _cmr)

CMAP_PERDAY = plt.get_cmap("seismic")


# --------------------------------------------------------------------------- #
# map helpers
# --------------------------------------------------------------------------- #
def _plot_boundary(ax, shp, **kw):
    """Plot a shapefile's lines. Handles Polygon (-> outline), MultiPolygon,
    LineString, and MultiLineString."""
    for geom in gpd.read_file(shp).to_crs(4326).geometry:
        gt = geom.geom_type
        if gt == "Polygon":
            lines = [geom.boundary]
        elif gt == "MultiPolygon":
            lines = [p.boundary for p in geom.geoms]
        elif gt == "LineString":
            lines = [geom]
        elif gt == "MultiLineString":
            lines = list(geom.geoms)
        else:
            lines = []
        for ln in lines:
            ax.plot(ln.xy[0], ln.xy[1], transform=PROJ, **kw)


def _geo(ax, extent=DISPLAY, ticks=True, lon_lbl=True, lat_lbl=True):
    """Enhanced geographic features with improved styling."""
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6, color="#333333")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4, color="#888888", alpha=0.6)
    _plot_boundary(ax, CNTRY_SHP, color="#444444", lw=0.7)
    _plot_boundary(ax, NINEDASH_SHP, color="#222222", lw=1.0)
    ax.set_extent(extent, crs=PROJ)
    
    if ticks:
        ax.set_xticks(range(60, 181, 30), crs=PROJ)   # 60,90,120,150,180°E within lon 60-180
        ax.set_yticks(range(-10, 71, 20), crs=PROJ)   # -10,10,30,50,70°N within lat -10-70
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=7.3, pad=1.2, length=3, width=0.5)
        if not lon_lbl:
            ax.tick_params(axis="x", length=0, labelbottom=False)
        if not lat_lbl:
            ax.tick_params(axis="y", length=0, labelleft=False)
    else:
        ax.set_xticks([]); ax.set_yticks([])


def _basin(ax, color=OI["vermillion"], lw=1.0, alpha=0.95):
    """Enhanced basin boundary with better visibility."""
    g = gpd.read_file(BASIN_SHP).to_crs(4326).geometry.union_all().boundary
    for ln in (list(g.geoms) if hasattr(g, "geoms") else [g]):
        ax.plot(ln.xy[0], ln.xy[1], transform=PROJ, color=color, lw=lw, alpha=alpha, zorder=5)


def _clip(da):
    _p = 2.5
    m = (da.lat >= DISPLAY[2] - _p) & (da.lat <= DISPLAY[3] + _p) & \
        (da.lon >= DISPLAY[0] - _p) & (da.lon <= DISPLAY[1] + _p)
    return da.where(m, drop=True)


def _mask_tibet_unused(da):
    return da


def _sym_lim(comp, floor):
    m = (comp.lat >= DISPLAY[2]) & (comp.lat <= DISPLAY[3]) & \
        (comp.lon >= DISPLAY[0]) & (comp.lon <= DISPLAY[1])
    v = np.abs(comp.where(m, drop=True).values.ravel())
    v = v[np.isfinite(v)]
    return max(floor, float(np.percentile(v, 98)))


def _map(ax, da, cmap, vmin, vmax, cbar_label, title, cbar=True, lon_lbl=True, lat_lbl=True):
    """Enhanced map plotting with improved visual quality."""
    pcm = ax.contourf(da.lon, da.lat, da.values, transform=PROJ,
                      cmap=cmap, levels=np.linspace(vmin, vmax, 61), extend='both',
                      alpha=0.95, antialiased=True)
    _geo(ax, lon_lbl=lon_lbl, lat_lbl=lat_lbl)
    _basin(ax)
    ax.set_title(title, **TTL)
    if cbar:
        cb = plt.colorbar(pcm, ax=ax, fraction=0.038, pad=0.02, shrink=0.86)
        cb.set_label(cbar_label, fontsize=7.3)
        cb.ax.tick_params(labelsize=6.5)
    return pcm


def _quiver(ax, ux, uy, scale, skip, color="#1a1a1a", width=0.0035, alpha=0.9, 
            minshaft=1.5, halo=False):
    """Enhanced quiver plot."""
    x = ux.lon.values[::skip]; y = ux.lat.values[::skip]
    u = ux.values[::skip, ::skip]; v = uy.values[::skip, ::skip]
    
    q = ax.quiver(x, y, u, v, transform=PROJ, scale=scale, scale_units="inches",
                  width=width, color=color, alpha=alpha, minshaft=minshaft, 
                  zorder=2.5, headwidth=3, headlength=4)
    return q


# --------------------------------------------------------------------------- #
def load_composite():
    ds = xr.open_dataset(os.path.join(OUT, "diagnostics_native_1960_2024.nc"))
    # Composite slice follows the extended diagnostics domain (60-180E/-10-70N,
    # lat N->S descending); the calendar-day anomaly is per grid cell, so the
    # added cells get their own climatology.
    sub = ds[["Z500", "SLP", "IVT_mag", "IVT_x", "IVT_y", "VIMFC", "q850", "u850", "v850"]] \
            .sel(lat=slice(70, -10), lon=slice(60, 180)).load()
    ds.close()
    times = pd.to_datetime(sub.time.values)
    pf = pd.read_csv(os.path.join(OUT, "prediction_probability_full.csv"), parse_dates=["date"])
    pf = pf[pf.model == "combined"].set_index("date")
    yrecmp = (pf.reindex(times)["y_prob"] >= 0.5).astype(int).values
    pat = np.where(yrecmp == 1)[0]; non = np.where(yrecmp == 0)[0]
    print(f"[comp] pattern n={len(pat)} non n={len(non)}")

    def anom(da):
        rolled = da.rolling(time=31, center=True, min_periods=1).mean()
        clim = rolled.groupby("time.dayofyear").mean("time")
        return da.groupby("time.dayofyear") - clim

    out = {}
    for f in ["Z500", "SLP", "IVT_mag", "IVT_x", "IVT_y", "VIMFC", "q850", "u850", "v850"]:
        a = anom(sub[f])
        out[f] = (a.isel(time=pat).mean("time") - a.isel(time=non).mean("time")).load()
    out["SLP_hPa"]   = out["SLP"] / 100.0
    out["VIMFC_mmd"] = out["VIMFC"] * 86400.0
    out["q850_gkg"]  = out["q850"] * 1000.0
    return out


def load_typical_case(date="2016-07-19"):
    import torch
    from importlib import import_module
    m04 = import_module("04_train_cnn")
    m05 = import_module("05_explain_cnn")
    chans = m04.CHANNEL_SETS["combined"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = xr.open_dataset(os.path.join(OUT, "cnn_inputs_1deg_1960_2024.nc"))
    times = pd.to_datetime(ds.time.values)
    didx = int(np.where(times == pd.Timestamp(date))[0][0])
    Xd = np.stack([ds[c].values for c in chans], axis=1)[didx:didx + 1]
    ds.close()
    model = m04.YRCNN(len(chans), m04.HP).to(device)
    model.load_state_dict(torch.load(os.path.join(OUT, "model_combined.pt"), map_location=device))
    model.eval()
    with torch.no_grad():
        prob = float(torch.softmax(model(torch.from_numpy(Xd).float().to(device)), 1)[0, 1])
    lrp = m05.lrp_alpha1_beta0(model, torch.from_numpy(Xd).float().to(device), target=1)[0]
    lrp_tot = lrp.sum(0).detach().cpu().numpy()
    d = pd.Timestamp(date)
    raw = xr.open_dataset(os.path.join(CHM_DIR, f"CHM_PRE_V2_daily_{d.year}.nc"))["prec"]
    pday = raw.sel(time=d, method="nearest").sel(lat=slice(31, 46), lon=slice(100, 121))
    reg = xr.open_dataset(os.path.join(OUT, "precip_daily_region_1960_2024.nc"))["mask"]
    mk = (reg.reindex_like(pday, method="nearest") > 0)
    return pday.where(mk), lrp_tot, prob


# --------------------------------------------------------------------------- #
def main():
    comp = load_composite()
    rds = xr.open_dataset(os.path.join(OUT, "relevance_LRP_map_combined.nc"))
    rmap = rds["lrp_relevance_a1b0"]
    rch_lrp = rds["lrp_relevance_a1b0"].sum(("lat", "lon")).values
    rds.close()
    ig  = xr.open_dataset(os.path.join(OUT, "relevance_IG_combined.nc"))["relevance_per_channel_IG"].values
    sgr = xr.open_dataset(os.path.join(OUT, "relevance_SmoothGrad_combined.nc"))["relevance_per_channel_SmoothGrad"].values
    chans = ["Z500", "SLP", "IVT", "VIMFC", "PW"]
    rmap_total = rmap.sum("channel")

    pday, lrp_tot, prob = load_typical_case()

    fig = plt.figure(figsize=(7.48, 6.0))

    LM, RM = 0.052, 0.006
    MAP_W, CB_W, CB_PAD, GUT_X = 0.265, 0.0075, 0.010, 0.040
    _step = MAP_W + CB_PAD + CB_W + GUT_X
    col_x = [LM + k*_step for k in range(3)]
    cb_x  = [cx + MAP_W + CB_PAD for cx in col_x]
    GEO_ASP = (DISPLAY[1]-DISPLAY[0])/(DISPLAY[3]-DISPLAY[2])
    MAP_H = MAP_W*7.48/(GEO_ASP*6.0)
    # (i) colorbar spacing matches the other panels and the map is widened.
    # The effective aspect ratio is (MAP_W*7.48)/(MAP_H*6.0) = 1.5, so the (i)
    # window must use Δlon/Δlat = 1.5 (same 120°/80° ratio as the big (a)-(h)
    # window) to fill MAP_W x MAP_H without horizontal letterboxing.

    # Vertical layout: rows 1 and 2 span the same lon window, so they share the
    # x scale — lon labels appear once, under row 2 (_geo's lon_lbl=False hides
    # them on row 1). Map axes get exact-aspect boxes (height = MAP_H, no
    # cartopy letterboxing), so titles/labels hug the map edges and the
    # row-1->row-2 gutter only has to clear row-2's titles.
    TITLE_H  = 0.034   # title zone above a map
    GUT_12   = 0.001   # row-1 map bottom -> row-2 title top (breathing only)
    XLABEL_H = 0.025   # lon tick labels under a row-2 map
    GUT_23   = 0.007   # row-2 label bottom -> row-3 title top
    map1_bot = 0.925 - MAP_H                                    # row 1 (title above, nothing below)
    map2_bot = map1_bot - TITLE_H - GUT_12 - MAP_H              # row 2 (title above, lon labels below)
    map3_bot = map2_bot - XLABEL_H - GUT_23 - TITLE_H - MAP_H   # row 3 (panel h)

    row_map_y = {1: map1_bot, 2: map2_bot, 3: map3_bot}

    def _mapi(c, r):
        return fig.add_axes([col_x[c], row_map_y[r], MAP_W, MAP_H], projection=PROJ)

    lim_z500  = 35.0
    lim_slp   = 2.0
    lim_vimfc = 12.0
    lim_q850  = 2.0
    lim_ivt   = 100.0
    _ds = xr.open_dataset(os.path.join(OUT, "cnn_inputs_1deg_1960_2024.nc"))
    lat1, lon1 = _ds.lat.values, _ds.lon.values; _ds.close()

    f_raw = rmap_total
    i_raw = xr.DataArray(lrp_tot, dims=("lat", "lon"), coords=dict(lat=lat1, lon=lon1))
    # Symmetric saturation limits, re-derived on the retrained model
    # (full-window 82x122 domain). Same saturation design as the old 46x82 maps:
    # (f) mean-relevance map: fmax ~ 1/3 of max (old 0.04/0.124=0.32; new max 0.117
    #     -> 0.04 keeps ratio 0.34).
    # (h) 2016-07-19 single day: dominant side saturates ~5.7x (old lim 0.05 vs
    #     extreme -0.285; new extremes [-0.158,+0.350] -> 0.350/5.7 ~ 0.06).
    fmax = 0.04
    rel_lim = 0.06
    print(f"[lim] Z500±{lim_z500:.0f}m  SLP±{lim_slp:.1f}hPa  VIMFC±{lim_vimfc:.1f}mm/d  "
          f"q850±{lim_q850:.2f}g/kg  IVT±{lim_ivt:.0f}  fmax={fmax:.2e} rel±{rel_lim:.3f}")

    # ===== Row 1 — circulation =====
    ax_a = _mapi(0, 1)
    pcm_a = ax_a.contourf(_clip(comp["Z500"]).lon, _clip(comp["Z500"]).lat, 
                          _clip(comp["Z500"]).values, transform=PROJ,
                          cmap=CMAP_Z500, levels=np.linspace(-lim_z500, lim_z500, 61), 
                          extend='both', alpha=0.95, antialiased=True)
    _geo(ax_a, lon_lbl=False, lat_lbl=True); _basin(ax_a)
    ax_a.set_title("(a) Z500 anomaly", **TTL)

    ax_b = _mapi(1, 1)
    pcm_b = ax_b.contourf(_clip(comp["SLP_hPa"]).lon, _clip(comp["SLP_hPa"]).lat,
                          _clip(comp["SLP_hPa"]).values, transform=PROJ,
                          cmap=CMAP_SLP, levels=np.linspace(-lim_slp, lim_slp, 61),
                          extend='both', alpha=0.95, antialiased=True)
    _geo(ax_b, lon_lbl=False, lat_lbl=False); _basin(ax_b)
    ax_b.set_title("(b) SLP anomaly", **TTL)

    ax_c = _mapi(2, 1)
    pcm_c = ax_c.contourf(_clip(comp["IVT_mag"]).lon, _clip(comp["IVT_mag"]).lat,
                          _clip(comp["IVT_mag"]).values, transform=PROJ,
                          cmap=CMAP_IVT, levels=np.linspace(-lim_ivt, lim_ivt, 61),
                          extend='both', alpha=0.95, antialiased=True)
    _geo(ax_c, lon_lbl=False, lat_lbl=False); _basin(ax_c)
    ax_c.set_title("(c) IVT anomaly + vectors", **TTL)
    _quiver(ax_c, _clip(comp["IVT_x"]), _clip(comp["IVT_y"]), scale=1.4e3, skip=2)

    # ===== Row 2 — moisture + bridge =====
    ax_d = _mapi(0, 2)
    pcm_d = ax_d.contourf(_clip(comp["VIMFC_mmd"]).lon, _clip(comp["VIMFC_mmd"]).lat,
                          _clip(comp["VIMFC_mmd"]).values, transform=PROJ,
                          cmap=CMAP_VIMFC, levels=np.linspace(-lim_vimfc, lim_vimfc, 61),
                          extend='both', alpha=0.95, antialiased=True)
    _geo(ax_d, lon_lbl=True, lat_lbl=True); _basin(ax_d)
    ax_d.set_title("(d) VIMFC anomaly", **TTL)

    ax_e = _mapi(1, 2)
    pcm_e = ax_e.contourf(_clip(comp["q850_gkg"]).lon, _clip(comp["q850_gkg"]).lat,
                          _clip(comp["q850_gkg"]).values, transform=PROJ,
                          cmap=CMAP_Q850, levels=np.linspace(-lim_q850, lim_q850, 61),
                          extend='both', alpha=0.95, antialiased=True)
    _geo(ax_e, lon_lbl=True, lat_lbl=False); _basin(ax_e)
    ax_e.set_title("(e) 850-hPa q + wind anomaly", **TTL)
    _quiver(ax_e, _clip(comp["u850"]), _clip(comp["v850"]), scale=32, skip=2)

    ax_f = _mapi(2, 2)
    pcm_f = ax_f.contourf(f_raw.lon, f_raw.lat, f_raw.values, transform=PROJ,
                          cmap=CMAP_F, levels=np.linspace(0, fmax, 61), extend='max',
                          alpha=0.98, antialiased=True)
    _geo(ax_f, lon_lbl=True, lat_lbl=False); _basin(ax_f)
    ax_f.set_title("(f) LRP total relevance", **TTL)

    # ===== Row 3 — summary + case =====
    ax_g = fig.add_axes([col_x[0], map3_bot, MAP_W + CB_PAD + CB_W, MAP_H])
    order = np.argsort(rch_lrp)[::-1]
    xg = np.arange(len(chans)); wb = 0.26
    
    bars_lrp = ax_g.bar(xg - wb, rch_lrp[order], wb, color=OI["vermillion"], 
                        label="LRP α=1β=0", edgecolor="white", linewidth=0.4)
    bars_ig = ax_g.bar(xg, ig[order], wb, color=OI["blue"], 
                       label="Integrated Gradients", edgecolor="white", linewidth=0.4)
    bars_sg = ax_g.bar(xg + wb, sgr[order], wb, color=OI["green"], 
                       label="SmoothGrad", edgecolor="white", linewidth=0.4)
    
    ax_g.set_xticks(xg)
    ax_g.set_xticklabels([chans[i] for i in order], fontsize=7.8)
    ax_g.set_ylabel("per-channel relevance", fontsize=8.1)
    ax_g.set_title("(g) Channel relevance ranking", **TTL)
    ax_g.legend(fontsize=6.8, loc="upper left", framealpha=0.95, 
                borderpad=0.3, labelspacing=0.3, handlelength=1.5)
    ax_g.tick_params(labelsize=7.5, pad=1.2)
    for s in ("top", "right"):
        ax_g.spines[s].set_visible(False)
    ax_g.spines["bottom"].set_linewidth(0.7)
    ax_g.spines["left"].set_linewidth(0.7)
    # ylim follows the data: upper limit raised for headroom/legend.
    ax_g.set_ylim(0, 10)
    ax_g.set_yticks([0, 2, 4, 6, 8, 10])
    ax_g.yaxis.grid(True, color="#E0E0E0", linestyle="-", linewidth=0.5, alpha=0.7)
    ax_g.set_axisbelow(True)
    ax_g.margins(x=0.01)

    ax_h = _mapi(1, 3)
    pcm_h = ax_h.contourf(i_raw.lon, i_raw.lat, i_raw.values, transform=PROJ,
                          cmap=CMAP_PERDAY, levels=np.linspace(-rel_lim, rel_lim, 61), 
                          extend='both', alpha=0.95, antialiased=True)
    _geo(ax_h); _basin(ax_h)
    ax_h.set_title(f"(h) Per-day LRP  (P={prob:.3f})", **TTL)

    # (i) precipitation panel: basin observation range 105.2-119.3°E / 33.1-43.4°N;
    # window Δlon/Δlat = 16.5/11.0 = 1.5 = box inch aspect ratio -> fills the box
    # proportionally; colorbar lands at the same CB_PAD as the other panels; basin
    # margins 0.36°(lat) / 1.2°(lon); ticks 105/110/115°E, 35/40°N all in window.
    I_BOX = [104.0, 120.5, 32.75, 43.75]
    ax_i = fig.add_axes([col_x[2], map3_bot, MAP_W, MAP_H],
                        projection=PROJ)
    vmax = 150.0
    pcm_i = ax_i.contourf(pday.lon, pday.lat, pday.values, transform=PROJ,
                          cmap=CMAP_PRECIP, levels=np.linspace(0, max(vmax, 40), 61), 
                          extend='max', alpha=0.98, antialiased=True)
    try:
        cs = ax_i.contour(pday.lon, pday.lat, pday.values, levels=[25, 50, 100],
                          colors=["#c0392b", "#7d1a1a", "#4a0e0e"], 
                          linewidths=[0.9, 1.1, 1.3], transform=PROJ, zorder=4)
        ax_i.clabel(cs, inline=True, fontsize=6.2, fmt='%g', colors="#333333")
    except Exception:
        pass
    _geo(ax_i, extent=I_BOX, ticks=False); _basin(ax_i)
    ax_i.set_xticks([105, 110, 115], crs=PROJ)
    ax_i.set_yticks([35, 40], crs=PROJ)
    ax_i.xaxis.set_major_formatter(LongitudeFormatter())
    ax_i.yaxis.set_major_formatter(LatitudeFormatter())
    ax_i.tick_params(axis="x", labelsize=7.3, pad=1.2, direction="out", length=3)
    ax_i.tick_params(axis="y", labelsize=7.3, pad=1.2, direction="in", length=3)
    ax_i.set_title("(i) Precip, 2016-07-19", **TTL)

    # ===== per-panel colorbars =====
    _g = FuncFormatter(lambda x, p: f"{abs(x):.3f}"[1:])
    def _cb(pcm, c, r, label, vmin, vmax, dec=1, scaled=False, ticks=None, ticklabels=None,
            h_frac=MAP_H, tick_pad=1.8):
        cax = fig.add_axes([cb_x[c], row_map_y[r] + (MAP_H - h_frac)/2, CB_W, h_frac])
        cb = fig.colorbar(pcm, cax=cax)
        if ticks is None:
            ticks = np.round(np.linspace(vmin, vmax, 3), dec)
        cb.set_ticks(ticks)
        if ticklabels is not None:
            cb.set_ticklabels(ticklabels)
        elif scaled:
            cb.ax.yaxis.set_major_formatter(_g)
        cb.ax.tick_params(labelsize=8.1, length=2.5, pad=tick_pad)
        cb.ax.set_title(label, fontsize=8.1, pad=3)
        cb.outline.set_linewidth(0.5)
        return cb

    _cb(pcm_a, 0, 1, "m", -lim_z500, lim_z500, dec=0)
    _cb(pcm_b, 1, 1, "hPa", -lim_slp, lim_slp, dec=0)
    _cb(pcm_c, 2, 1, "kg m$^{-1}$s$^{-1}$", -lim_ivt, lim_ivt, dec=0)
    _cb(pcm_d, 0, 2, "mm d$^{-1}$", -lim_vimfc, lim_vimfc, dec=0)
    _cb(pcm_e, 1, 2, "g kg$^{-1}$", -lim_q850, lim_q850, dec=0)
    _cb(pcm_f, 2, 2, "relevance", 0.0, fmax, dec=6, 
        ticks=[0, fmax/2, fmax], ticklabels=["0", f"{fmax/2:g}", f"{fmax:g}"])
    _cb(pcm_h, 1, 3, "relevance", -rel_lim, rel_lim, dec=6, tick_pad=0.8,
        ticks=[-rel_lim, 0.0, rel_lim],
        ticklabels=[f"{-rel_lim:g}", "0", f"{rel_lim:g}"])
    _cb(pcm_i, 2, 3, "mm d$^{-1}$", 0.0, max(vmax, 40), dec=0)

    out = os.path.join(FIGDIR, "fig4.png")
    save_figure(fig, out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig4] saved {out}  (2016-07-19 CNN P={prob:.4f}, precip p99.5={vmax:.1f} mm/d)")


if __name__ == "__main__":
    main()
