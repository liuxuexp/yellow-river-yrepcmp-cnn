#!/usr/bin/env python3
"""09a_fig1.py — Fig 1 study region (4 panels, Elsevier double-column).

Layout (7.48 x 5.8 in):
  (a) redrawn study-region elevation map — WorldClim 2.5' DEM masked to basin +
      replicated panel-a ramp, Huang He main stem, basin boundary, key cities,
      lat/lon graticule.
  (b) location within China + CNN input region (green dashed) + nine-dash line
  (c) annual mean precipitation (CHM_PRE, 1960-2024)
  (d) extreme precipitation frequency (days yr-1 over local wet-day Q95)

Output: figures/fig1.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
from matplotlib.patches import Rectangle
from PIL import Image
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import geopandas as gpd
import xarray as xr
import config

FIGDIR = config.FIGDIR
OUT    = config.OUT

CNTRY   = config.CNTRY_SHP
NINE    = config.NINEDASH_SHP
PROV    = config.PROV_SHP
STUDY_JPG = os.path.join(os.path.dirname(str(config.ROOT)), "huanghe_middle_lower_map.jpg")
CLIM_NC   = os.path.join(OUT, "fig1_climatology.nc")
PRECIP_NC = os.path.join(OUT, "precip_daily_region_1960_2024.nc")

CNN_REGION = dict(lon=(70, 151), lat=(15, 60))
# Panel (a) redrawn elevation map
DEM         = config.DEM
EXT_A       = [105, 120, 32.5, 43.5]              # study-region extent (== c/d)
VLOW_A, VHIGH_A = -10, 2500                       # panel-a DEM ramp
# key cities within EXT_A
CITIES_A = dict(Xi_an=(108.9, 34.3), Baotou=(109.9, 40.6), Taiyuan=(112.5, 37.9),
                Zhengzhou=(113.6, 34.7), Jinan=(117.0, 36.7))
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7")
MID_C = OI["sky"]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 9.1,
                     "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"]})

NE_RIVERS_CACHE = {}


def load_ne_rivers(bbox):
    key = tuple(round(float(v), 2) for v in bbox)
    if key in NE_RIVERS_CACHE:
        return NE_RIVERS_CACHE[key]
    fname = shpreader.natural_earth(resolution="10m", category="physical",
                                    name="rivers_lake_centerlines")
    g = gpd.read_file(fname).to_crs(4326)
    W, S, E, N = bbox
    sub = g.cx[W:E, S:N]
    name_col = next((c for c in ("name", "NAME", "name_en") if c in sub.columns), None)
    huang, trib = [], []
    for _, row in sub.iterrows():
        nm = str(row.get(name_col, "")).lower() if name_col else ""
        (huang if ("huang" in nm or "yellow" in nm) else trib).append(row.geometry)
    NE_RIVERS_CACHE[key] = (huang, trib)
    return huang, trib


def draw_rivers(ax, huang_color="white", huang_lw=1.0,
                trib=False, trib_color="0.6", trib_lw=0.4):
    extent = ax.get_extent(crs=ccrs.PlateCarree())  # [W, E, S, N]
    bbox = (extent[0], extent[2], extent[1], extent[3])
    huang, tri = load_ne_rivers(bbox)
    if trib:
        ax.add_geometries(tri, crs=ccrs.PlateCarree(), facecolor="none",
                          edgecolor=trib_color, lw=trib_lw, zorder=4)
    ax.add_geometries(huang, crs=ccrs.PlateCarree(), facecolor="none",
                      edgecolor=huang_color, lw=huang_lw, zorder=5)


def _union():
    return gpd.read_file(config.BASIN_SHP).to_crs(4326)


def _geo_ax(fig, rect, extent, union=None, tick_step=10, sharey=False):
    ax = fig.add_axes(rect, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(lw=0.4, color="0.4")
    try:
        prov = gpd.read_file(PROV).to_crs(4326)
        ax.add_geometries(prov.geometry, crs=ccrs.PlateCarree(),
                          facecolor="none", edgecolor="0.82", lw=0.3, ls=":")
    except Exception:
        pass
    if union is not None:
        ax.add_geometries(union.geometry, crs=ccrs.PlateCarree(),
                          facecolor="none", edgecolor="k", lw=0.9)
    _xt = np.arange(np.floor(extent[0]/5)*5, extent[1]+1, tick_step)
    _yt = np.arange(np.floor(extent[2]/5)*5, extent[3]+1, tick_step)
    # drop any tick that falls OUTSIDE the extent: a tick beyond the map edge
    # makes matplotlib expand the axis limits to accommodate it, silently
    # changing the geographic range.
    _xt = _xt[(_xt >= extent[0]) & (_xt <= extent[1])]
    _yt = _yt[(_yt >= extent[2]) & (_yt <= extent[3])]
    ax.set_xticks(_xt, crs=ccrs.PlateCarree())
    ax.set_yticks(_yt, crs=ccrs.PlateCarree())
    if sharey:
        ax.tick_params(labelleft=False)   # y-axis shared with the left neighbour
    return ax


# --------------------------------------------------------------------------- #
# Panel (a) redrawn elevation map: WorldClim 2.5' DEM masked to the basin +
# panel-a replicated ramp + hillshade.
# --------------------------------------------------------------------------- #
def _panelA_cmap():
    """Replicated DEM ramp: the 48 strip pixels of the panel-a JPG colorbar
    (top=HIGH white -> bottom=LOW green), reversed to low->high for matplotlib."""
    im = np.asarray(Image.open(STUDY_JPG).convert("RGB")).astype(float)
    strip = im[540:588, 735:770].mean(axis=1)          # (48,3) top=HIGH -> bottom=LOW
    low2high = list(reversed(["#%02X%02X%02X" % (int(round(c[0])), int(round(c[1])), int(round(c[2])))
                              for c in strip]))
    cmap = mcolors.LinearSegmentedColormap.from_list("dem_panelA", low2high, N=512)
    cmap.set_bad(color="white", alpha=0)   # transparent outside basin -> coastline/province show
    return cmap


def _hillshade(elev, azimuth=315, altitude=45):
    dy, dx = np.gradient(elev)
    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect = np.arctan2(-dy, dx)
    ar, aa = np.radians(azimuth), np.radians(altitude)
    return np.clip(np.sin(aa) * np.cos(slope)
                   + np.cos(aa) * np.sin(slope) * np.cos(ar - aspect), 0, 1)


def _load_dem_masked(extent, basin_geom):
    """WorldClim DEM over extent, NaN outside the basin polygon."""
    with rasterio.open(DEM) as src:
        r0, c0 = src.index(extent[0], extent[3])
        r1, c1 = src.index(extent[1], extent[2])
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(src.height, r1), min(src.width, c1)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        data = src.read(1, window=win).astype(float)
        tr = src.window_transform(win)
    data[data < -1000] = np.nan
    inside = geometry_mask([basin_geom], out_shape=data.shape, transform=tr, invert=True)
    data[~inside] = np.nan
    lons = np.linspace(extent[0], extent[1], data.shape[1])
    lats = np.linspace(extent[3], extent[2], data.shape[0])
    return lons, lats, data


def _north_arrow(ax):
    ax.annotate("", xy=(0.95, 0.93), xytext=(0.95, 0.84), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))
    ax.text(0.95, 0.945, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.8, fontweight="bold")


def main():
    union = _union()
    clim  = xr.open_dataset(CLIM_NC)
    annual_mean = clim["annual_mean"].values      # mm yr-1, NaN outside basin
    ext_freq    = clim["extreme_freq_yr"].values  # days yr-1, NaN outside basin
    lat, lon = clim.lat.values, clim.lon.values
    clim.close()
    b = union.total_bounds

    fig = plt.figure(figsize=(7.48, 5.8))

    # ---- equal-size 2x2 grid ------------------------------------------------
    # All four panels share the same width W and height H (figure fraction).
    # Colorbars live in their OWN axes (cax=...), so they never resize a panel.
    ML = MR = 0.045                          # outer left/right margins
    MB = 0.045                               # bottom margin
    W, H = 0.400, 0.386                      # panel width / height
    PAD, CW = 0.010, 0.020                   # panel->colorbar pad / colorbar width
    HGAP, VGAP = 0.055, 0.086                # inter-column gap / inter-row gap
    xL = ML                                  # left column left edge
    xR = xL + W + HGAP                       # right column left edge
    yB = MB                                  # bottom row bottom edge
    yT = yB + H + VGAP                       # top row bottom edge
    cby, cbh = yB + 0.014, H - 0.028         # colorbar y / height (vertically inset)
    cbx_d = xR + W + PAD                     # right-margin colorbar x (c top + d bottom)
    TTL = dict(fontsize=9.8, fontweight="bold", loc="left", pad=2)

    # (a) redrawn study-region elevation map (DEM + rivers + cities + graticule).
    #     WorldClim DEM masked to basin + panel-a replicated ramp; extent/ticks
    #     mirror c/d so the three basin maps align.
    dem_cmap = _panelA_cmap()
    try:
        basin_geom = union.geometry.union_all()
    except AttributeError:
        basin_geom = union.geometry.unary_union
    lons_a, lats_a, elev_a = _load_dem_masked(EXT_A, basin_geom)
    hs_a = _hillshade(np.nan_to_num(elev_a, nan=0.0))
    hs_a = np.where(np.isnan(elev_a), np.nan, hs_a)
    _gray_a = plt.get_cmap("gray").copy(); _gray_a.set_bad(alpha=0)
    norm_a = mcolors.Normalize(vmin=VLOW_A, vmax=VHIGH_A)
    print("[fig1] panel-a DEM (in-basin): %.1f to %.1f m (nan %.1f%%)"
          % (np.nanmin(elev_a), np.nanmax(elev_a), 100 * np.isnan(elev_a).mean()))

    axa = _geo_ax(fig, [xL, yT, W, H], EXT_A, union=None, tick_step=5)
    axa.pcolormesh(lons_a, lats_a, np.clip(elev_a, VLOW_A, VHIGH_A), cmap=dem_cmap, norm=norm_a,
                   transform=ccrs.PlateCarree(), zorder=1)
    axa.pcolormesh(lons_a, lats_a, hs_a, cmap=_gray_a, vmin=0, vmax=1, alpha=0.18,
                   transform=ccrs.PlateCarree(), zorder=2)
    draw_rivers(axa, huang_color=OI["blue"], huang_lw=1.0, trib=True,
                trib_color="0.45", trib_lw=0.4)
    for name, (lo, la) in CITIES_A.items():
        axa.plot(lo, la, "o", ms=2.6, mfc="white", mec="k", mew=0.5,
                 transform=ccrs.PlateCarree(), zorder=8)
        axa.text(lo + 0.12, la + 0.12, name.replace("_", " "), fontsize=6.8,
                 color=OI["vermillion"], fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=1.4, foreground="white")],
                 transform=ccrs.PlateCarree(), zorder=9)
    axa.xaxis.set_major_formatter(LongitudeFormatter())
    axa.yaxis.set_major_formatter(LatitudeFormatter())
    _north_arrow(axa)
    axa.set_title("(a) Study region", **TTL)

    # elevation colorbar — compact INSET inside panel (a)'s bottom-right corner
    cax_a = axa.inset_axes([0.87, 0.020, 0.028, 0.10])
    cb_a = fig.colorbar(plt.cm.ScalarMappable(norm=norm_a, cmap=dem_cmap), cax=cax_a)
    cb_a.ax.yaxis.set_ticks_position("right")
    cb_a.set_ticks([0, 1000, 2500])
    cb_a.ax.tick_params(labelsize=5.9)
    cb_a.ax.set_title("Elevation (m)", fontsize=5.9, pad=4)
    cb_a.outline.set_linewidth(0.5)

    # (b) China inset: national border + nine-dash line + basin + CNN region box + Huang He
    cntry = gpd.read_file(CNTRY).to_crs(4326)
    nine  = gpd.read_file(NINE).to_crs(4326)
    axb = _geo_ax(fig, [xL, yB, W, H], [70, 140, 3, 54], union)
    axb.add_geometries(cntry.geometry, crs=ccrs.PlateCarree(), facecolor="0.92",
                       edgecolor="0.4", lw=0.5)
    axb.add_geometries(union.geometry, crs=ccrs.PlateCarree(), facecolor=MID_C,
                       edgecolor="k", alpha=0.6, lw=0.5)
    axb.add_geometries(nine.geometry, crs=ccrs.PlateCarree(), facecolor="none",
                       edgecolor="black", lw=0.7, ls="--")
    axb.add_patch(Rectangle((CNN_REGION["lon"][0], CNN_REGION["lat"][0]), 81, 45,
                  transform=ccrs.PlateCarree(), fill=False, edgecolor=OI["green"],
                  lw=0.8, ls="--"))
    draw_rivers(axb, huang_color=OI["vermillion"], huang_lw=0.9)
    axb.set_title("(b) Location in China", **TTL)
    axb.xaxis.set_major_formatter(LongitudeFormatter()); axb.yaxis.set_major_formatter(LatitudeFormatter())
    # latitude y-axis starts at 3°N — the generic 10° grid's first tick is 10°N,
    # leaving 3–10°N unlabeled, so add the edge tick.
    axb.set_yticks([3, 10, 20, 30, 40, 50], crs=ccrs.PlateCarree())

    # (c) annual mean precipitation
    axc = _geo_ax(fig, [xR, yT, W, H], [105, 120, 32.5, 43.5], union, tick_step=5)
    pcm = axc.pcolormesh(lon, lat, annual_mean, transform=ccrs.PlateCarree(),
                         cmap="YlGnBu", vmin=100, vmax=1000, shading="auto")
    cax_c = fig.add_axes([cbx_d, yT + 0.014, CW, cbh])
    fig.colorbar(pcm, cax=cax_c)
    cax_c.set_ylabel("Annual mean precipitation (mm yr$^{-1}$)", fontsize=7.8, labelpad=-0.2)
    cax_c.yaxis.set_label_position("right")
    cax_c.tick_params(labelsize=7.2, pad=1)
    axc.xaxis.set_major_formatter(LongitudeFormatter()); axc.yaxis.set_major_formatter(LatitudeFormatter())
    draw_rivers(axc, huang_color="white", huang_lw=1.0, trib=True, trib_color="0.7", trib_lw=0.35)
    axc.set_title("(c) Annual mean precipitation", **TTL)

    # (d) extreme precipitation frequency (days yr-1 over local Q95)
    axd = _geo_ax(fig, [xR, yB, W, H], [105, 120, 32.5, 43.5], union,
                  tick_step=5)
    pcm2 = axd.pcolormesh(lon, lat, ext_freq, transform=ccrs.PlateCarree(),
                          cmap="YlOrRd", vmin=0, vmax=6, shading="auto")
    cax_d = fig.add_axes([cbx_d, cby, CW, cbh])
    fig.colorbar(pcm2, cax=cax_d)
    cax_d.set_ylabel("Extreme precipitation frequency (days yr$^{-1}$)", fontsize=7.8)
    cax_d.tick_params(labelsize=7.2)
    axd.xaxis.set_major_formatter(LongitudeFormatter()); axd.yaxis.set_major_formatter(LatitudeFormatter())
    draw_rivers(axd, huang_color="white", huang_lw=1.0, trib=True, trib_color="0.7", trib_lw=0.35)
    axd.set_title("(d) Extreme precipitation frequency", **TTL)

    # c & d share the x-axis (stacked right column, identical basin extent).
    axc.sharex(axd)
    axc.tick_params(labelbottom=True)
    # top row (a)+(c) share the y-axis (both basin extent 32.5-43.5N).
    axc.sharey(axa)
    axc.tick_params(labelleft=True)
    _ce = axc.get_extent(crs=ccrs.PlateCarree())
    _de = axd.get_extent(crs=ccrs.PlateCarree())
    print(f"[fig1] c extent={_ce}  d extent={_de}  identical={np.allclose(_ce, _de)}")
    assert np.allclose(_ce, _de), f"c/d extent mismatch: {_ce} vs {_de}"

    out = os.path.join(FIGDIR, "fig1.png")
    save_figure(fig, out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig1] saved {out}")


if __name__ == "__main__":
    main()
