#!/usr/bin/env python3
"""09a_figS7.py — Fig S7: CNN error analysis via circulation composites.

Composites for the four confusion-matrix categories (hit / miss / false-alarm /
correct-negative), defined by CNN P>0.5 prediction vs the YREP label.

Panels:
  (a) Confusion matrix (2×2 with n per cell)
  (b) Z500 anomaly composite: HITS minus MISSES — tests whether misses share the
      pattern
  (c) Z500 anomaly composite: FALSE-ALARMS minus HITS — tests why false alarms
      don't rain
  (d) SLP anomaly composite: FALSE-ALARMS minus HITS — complementary circulation
      view

Uses the same map conventions as Fig 4 (Z500 anomaly RdBu_r, SLP anomaly).
Composites and confusion counts are from OUT (new-domain classifier), consistent
with Fig 4.

Reads: out/cnn_inputs_1deg_1960_2024.nc, out/precip_daily_region_1960_2024.nc,
       out/prediction_probability_full.csv
Output: figures/figS7.png
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import geopandas as gpd
from _style import save_figure
import config

FIGDIR = config.FIGDIR
OUT    = config.OUT

PROJ   = ccrs.PlateCarree()
DISPLAY = [60, 180, -10, 70]                      # large window (inputs cover 59-180E/-11-70N)
CNTRY   = config.CNTRY_SHP
NINE    = config.NINEDASH_SHP
BASIN   = config.BASIN_SHP

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777")

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 7.5,
                     "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6})
TTL = dict(fontsize=7.5, fontweight="bold", loc="left", pad=3)

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
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, color="0.45")
    for geom in _geoms(CNTRY):
        for ln in _lines(geom):
            ax.plot(ln.xy[0], ln.xy[1], transform=PROJ, color="#444", lw=0.6)
    for geom in _geoms(NINE):
        for ln in _lines(geom):
            ax.plot(ln.xy[0], ln.xy[1], transform=PROJ, color="#222", lw=1.0)   # nine-dash: solid+bold


def _basin(ax):
    g = _geoms(BASIN).union_all().boundary
    for ln in (list(g.geoms) if hasattr(g, "geoms") else [g]):
        ax.plot(ln.xy[0], ln.xy[1], transform=PROJ, color=OI["vermillion"], lw=0.9, alpha=0.9)


def _geo(ax, xtick_labels=True, ytick_labels=True):
    _borders(ax); _basin(ax)
    ax.set_extent(DISPLAY, crs=PROJ)
    ax.set_xticks(range(60, 181, 30), crs=PROJ)   # large window: 60,90,120,150,180°E
    ax.set_yticks(range(-10, 71, 20), crs=PROJ)   # large window: -10,10,30,50,70°N
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=7.5)
    if not xtick_labels:
        ax.set_xticklabels([])
    if not ytick_labels:
        ax.set_yticklabels([])
    ax.set_title("")


def _clip(da):
    m = (da.lat >= DISPLAY[2]) & (da.lat <= DISPLAY[3]) & \
        (da.lon >= DISPLAY[0]) & (da.lon <= DISPLAY[1])
    return da.where(m, other=np.nan)


def _scalar_map(fig, rect, cax_rect, da, title, vmax, cmap="RdBu_r", label="",
                xtick_labels=True, ytick_labels=True, cticks=None, labelpad=3.0):
    """Generic scalar map panel for circulation anomalies."""
    ax = fig.add_axes(rect, projection=PROJ)
    cax = fig.add_axes(cax_rect)
    da_c = _clip(da)
    pcm = ax.pcolormesh(da_c.lon, da_c.lat, da_c.values, transform=PROJ, cmap=cmap,
                        vmin=-vmax, vmax=vmax, shading="auto")
    _geo(ax, xtick_labels=xtick_labels, ytick_labels=ytick_labels)
    ax.set_title(title, **TTL)
    cb = fig.colorbar(pcm, cax=cax, extend="both")
    cb.set_ticks(cticks if cticks is not None else [-vmax, -vmax / 2, 0, vmax / 2, vmax])
    cb.set_label(label, fontsize=7.0, labelpad=labelpad)
    cb.ax.tick_params(labelsize=6.5)
    return ax


def main():
    # Load data
    ds = xr.open_dataset(os.path.join(OUT, "cnn_inputs_1deg_1960_2024.nc"))
    dates = pd.to_datetime(ds.time.values).normalize()
    cache = xr.open_dataset(os.path.join(OUT, "precip_daily_region_1960_2024.nc"))
    yrep = cache["YREP"].values.astype(int)
    cache.close()
    prob = pd.read_csv(os.path.join(OUT, "prediction_probability_full.csv"), parse_dates=["date"])
    prob = prob[prob.model == "combined"].set_index("date")["y_prob"]
    prob = prob.reindex(dates)
    pp = (prob.values >= 0.5)

    # Define confusion categories
    cat = np.where(pp & (yrep == 1), "hit",
          np.where(pp & (yrep == 0), "fa",
          np.where(~pp & (yrep == 1), "miss", "cn")))
    counts = {k: int((cat == k).sum()) for k in ["hit", "fa", "miss", "cn"]}

    print("[Fig S7] confusion counts:", counts)
    print(f"  YREP base rate: {yrep.mean():.3f}, CNN positive rate: {pp.mean():.3f}")

    # Compute composites per category
    comps = {}
    for k in ["hit", "fa", "miss", "cn"]:
        m = cat == k
        comps[k] = {
            "Z500": ds["Z500_anom"][m].mean(dim="time").load(),
            "SLP": ds["SLP_anom"][m].mean(dim="time").load(),
        }
    ds.close()

    # Difference maps
    hit_minus_miss = comps["hit"]["Z500"] - comps["miss"]["Z500"]
    fa_minus_hit = comps["fa"]["Z500"] - comps["hit"]["Z500"]
    fa_minus_hit_slp = comps["fa"]["SLP"] - comps["hit"]["SLP"]

    # --- layout: inch-anchored (W×H = 7.0×5.2). Map rects match the large display
    #     aspect 120/80 = 1.5 exactly, so cartopy letterboxes nothing and the
    #     b/c/d rows/columns align; (a) is 90% of the row-1 slot, top-aligned,
    #     right edge flush with (c)'s so the a–b and c–d column gaps match ---
    W, H = 7.0, 5.2
    GEO_ASP = (DISPLAY[1] - DISPLAY[0]) / (DISPLAY[3] - DISPLAY[2])
    MW, MH = 2.55, 2.55 / GEO_ASP               # map box (in): 2.55 × 1.70 (aspect 1.50)
    R1_BOT, R2_BOT = 2.87, 0.64                 # row-1 / row-2 map bottoms (in)
    X1, X2, CBX, CBW = 0.55, 3.48, 6.13, 0.11   # col-1 / col-2 left, colorbar strip (in)
    rect = lambda x, y, w, h: [x / W, y / H, w / W, h / H]

    fig = plt.figure(figsize=(W, H))

    ax_a = fig.add_axes(rect(X1 + 0.1 * MW, R1_BOT + MH - 0.9 * MH, 0.9 * MW, 0.9 * MH))
    M = np.array([[counts["hit"], counts["fa"]], [counts["miss"], counts["cn"]]])
    im = ax_a.imshow(M, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            pct = 100 * M[i, j] / M.sum()
            ax_a.text(j, i, f"{M[i,j]}\n({pct:.1f}%)", ha="center", va="center",
                      fontsize=9, color="white" if M[i, j] > M.max() / 2 else "black")
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(["YREP", "Non-YREP"], fontsize=7.5)
    ax_a.set_yticks([0, 1])
    ax_a.set_yticklabels(["P≥0.5", "P<0.5"], fontsize=7.5)
    ax_a.set_xlabel("Label", fontsize=7.5)
    ax_a.set_ylabel("CNN prediction", fontsize=7.5)
    total = M.sum()
    ax_a.set_title(f"(a) Confusion matrix\n(n = {total:,})", **TTL)

    _scalar_map(fig, rect(X2, R1_BOT, MW, MH), rect(CBX, R1_BOT, CBW, MH),
                hit_minus_miss, "(b) Z500 anomaly: Hit − Miss\n(do missed extremes share pattern?)",
                vmax=0.5, cmap="RdBu_r", label="Z500 anomaly (σ)",
                xtick_labels=True, ytick_labels=True)

    _scalar_map(fig, rect(X1, R2_BOT, MW, MH), rect(CBX, R2_BOT + MH / 2 + 0.03, CBW, MH / 2 - 0.03),
                fa_minus_hit, "(c) Z500 anomaly: False alarm − Hit\n(why pattern days don't rain extreme?)",
                vmax=0.15, cmap="RdBu_r", label="Z500 anomaly (σ)",
                xtick_labels=True, ytick_labels=True,
                cticks=[-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])

    _scalar_map(fig, rect(X2, R2_BOT, MW, MH), rect(CBX, R2_BOT, CBW, MH / 2 - 0.03),
                fa_minus_hit_slp, "(d) SLP anomaly: False alarm − Hit\n(complementary circulation view)",
                vmax=0.2, cmap="RdBu_r", label="SLP anomaly (σ)",
                xtick_labels=True, ytick_labels=True, labelpad=8.0)

    save_figure(fig, "figS7.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("[Fig S7] saved to figures/figS7.png")

    # Key findings
    print("[Fig S7] Key interpretation:")
    print("  - Hits vs misses: if hit-minus-miss ≈ 0, misses share the circulation pattern")
    print("    (extremes occurred via local moisture/trigger, not large-scale dynamics)")
    print("  - False alarms vs hits: positive Z500/SLP over study region = pattern days with")
    print("    stronger circulation but no extreme rain (local moisture insufficient)")
    print(f"  - Misses are rare ({counts['miss']}/{counts['hit']}) → CNN captures most dynamics")
    print(f"  - False alarms are common ({counts['fa']}) → pattern necessary but not sufficient")


if __name__ == "__main__":
    main()
