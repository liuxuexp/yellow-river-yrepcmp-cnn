#!/usr/bin/env python3
"""01_prepare_precip.py — Build daily precipitation inputs and YREP-day labels.

Reads CHM_PRE V2 daily precipitation, crops to the Yellow River middle+lower
region, computes per-grid wet-day percentiles, area-weighted daily mean,
extreme coverage fractions, and YREP-day labels.
"""
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely

import config

# --------------------------------------------------------------------------- #
CHMPRE_DIR = config.CHM_DIR
CHMPRE_VAR = "prec"  # mm day^-1

REGION_SHP = config.BASIN_SHP
_BASIN_DIR = os.path.dirname(config.BASIN_SHP)
MIDDLE_SHP = os.path.join(_BASIN_DIR, "huanghe_middle.shp")
LOWER_SHP = os.path.join(_BASIN_DIR, "huanghe_lower.shp")

BBOX = dict(lon=slice(100.0, 122.0), lat=slice(30.0, 44.0))

WET_DAY_MM        = 1.0
COVERAGE_MIN      = 0.10
LOCAL_INTENSITY_Q = 0.95
AREAMEAN_Q        = 0.90
SENS_Q = (0.90, 0.95, 0.99)

DEFAULT_YEARS = (1960, 2024)
OUT_DIR = config.OUT


# --------------------------------------------------------------------------- #
def load_chmpre(year0, year1, smoke=False):
    """Open + crop CHM_PRE daily for [year0, year1]. Returns a cropped, loaded DataArray."""
    all_years = list(range(year0, year1 + 1))
    years = all_years[-3:] if smoke else all_years
    files = [os.path.join(CHMPRE_DIR, f"CHM_PRE_V2_daily_{y}.nc") for y in years]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} CHM_PRE file(s) missing, e.g. {missing[0]}")
    ds0 = xr.open_dataset(files[0])
    var = CHMPRE_VAR if CHMPRE_VAR in ds0 else list(ds0.data_vars)[0]
    units = ds0[var].attrs.get("units", "(unspecified)")
    print(f"[load] dir={CHMPRE_DIR}")
    print(f"[load] variable='{var}' units={units}  file0 grid: lat[{float(ds0.lat.min()):.2f},"
          f"{float(ds0.lat.max()):.2f}] lon[{float(ds0.lon.min()):.2f},{float(ds0.lon.max()):.2f}]")
    ds0.close()
    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False, chunks={"time": 120})
    da = ds[var].sel(**BBOX).load()  # eager load — shared-FS dask groupby/quantile is slow
    ds.close()
    print(f"[load] cropped shape={da.shape}  years={int(da.time.dt.year.min())}-"
          f"{int(da.time.dt.year.max())}  n_days={da.sizes['time']}  loaded={da.data.nbytes/1e9:.2f}GB")
    return da, var


def build_basin_mask(da, region_shp=REGION_SHP):
    """Middle+lower Yellow River mask on da's grid, from the union polygon."""
    gdf = gpd.read_file(region_shp).to_crs(4326)
    geom = gdf.geometry.union_all()
    lon2d, lat2d = np.meshgrid(da.lon.values, da.lat.values)
    mask = shapely.contains_xy(geom, lon2d.ravel(), lat2d.ravel()).reshape(lon2d.shape)
    print(f"[mask] {os.path.basename(region_shp)}: grid cells in region = {int(mask.sum())} / "
          f"{mask.size} ({100 * mask.mean():.1f}%)")
    return mask


def build_reach_masks(da, middle_shp=MIDDLE_SHP, lower_shp=LOWER_SHP):
    """Per-reach grid masks {reach: bool (lat,lon)} for middle/lower. Used by script 08."""
    lon2d, lat2d = np.meshgrid(da.lon.values, da.lat.values)
    out = {}
    for reach, shp in (("middle", middle_shp), ("lower", lower_shp)):
        g = gpd.read_file(shp).to_crs(4326).geometry.union_all()
        out[reach] = shapely.contains_xy(g, lon2d.ravel(), lat2d.ravel()).reshape(lon2d.shape)
        print(f"[reach] {reach}: {int(out[reach].sum())} grid cells")
    return out


def run(year0, year1, smoke=False, outdir=OUT_DIR):
    os.makedirs(outdir, exist_ok=True)
    da, var = load_chmpre(year0, year1, smoke=smoke)
    mask = build_basin_mask(da)  # (lat, lon) bool

    # area weights (cosine latitude), masked
    wlat = np.cos(np.deg2rad(da.lat.values))
    W = np.broadcast_to(wlat[:, None], mask.shape).astype(float)
    Wm = W * mask
    Wm_sum = float(Wm.sum())

    P = da.where(mask)  # masked precip (time,lat,lon), NaN outside

    # ---- per-grid wet-day percentiles (on the masked cells) ----
    wet = P.where(P >= WET_DAY_MM)
    qgrid = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices outside mask
        for q in set(SENS_Q + (LOCAL_INTENSITY_Q,)):
            qgrid[q] = wet.quantile(q, dim="time").compute()
            print(f"[thresh] per-grid wet-day Q{int(q*100):02d} computed "
                  f"(grid-mean={float(qgrid[q].mean()):.2f} mm/d)")

    # ---- daily area-weighted mean Pbar(d) ----
    Pbar = (P * Wm).sum(dim=("lat", "lon")) / Wm_sum

    # ---- daily extreme coverage F_q(d) at each intensity percentile ----
    Fq = {}
    for q in SENS_Q:
        exceed = (P >= qgrid[q]) & mask  # (time,lat,lon) bool
        Fq[q] = ((exceed * Wm).sum(dim=("lat", "lon")) / Wm_sum).compute()
    Q95 = qgrid[LOCAL_INTENSITY_Q]
    F = Fq[LOCAL_INTENSITY_Q]  # primary coverage series

    # ---- area-mean wet-day P90 threshold ----
    Pbar_wet = Pbar.where(Pbar >= WET_DAY_MM).dropna("time")
    P90_area = float(Pbar_wet.quantile(AREAMEAN_Q).values)
    print(f"[thresh] area-mean wet-day P{int(AREAMEAN_Q*100):02d} = {P90_area:.2f} mm/d "
          f"(over {Pbar_wet.sizes['time']} wet region-days)")

    # ---- YREP label ----
    yrep = ((F >= COVERAGE_MIN) & (Pbar >= P90_area)).astype("int8")
    n_yrep = int(yrep.sum()); n = int(yrep.sizes["time"])
    print(f"[label] YREP days = {n_yrep} / {n} ({100*n_yrep/n:.2f}%)")

    # ---- save ----
    ds_out = xr.Dataset(
        data_vars=dict(
            Pbar=(("time",), Pbar.values, {"units": "mm day-1",
                  "long_name": "area-weighted mean precipitation over YR middle+lower"}),
            F=(("time",), F.values, {"long_name": "extreme coverage fraction (local Q95)",
                "note": "area-weighted fraction of grid cells with P >= local wet-day Q95"}),
            YREP=(("time",), yrep.values, {"long_name": "Yellow River Regional Extreme Precip day flag"}),
            mask=(("lat", "lon"), mask.astype("int8"), {"long_name": "study-region mask"}),
            **{f"Q{int(q*100):02d}": (("lat", "lon"), qgrid[q].values,
                {"units": "mm day-1", "long_name": f"per-grid wet-day {int(q*100)}th percentile"})
               for q in SENS_Q},
            **{f"F{int(q*100):02d}": (("time",), Fq[q].values,
                {"long_name": f"daily coverage fraction at local wet-day Q{int(q*100)}"})
               for q in SENS_Q},
        ),
        coords=dict(time=da.time.values, lat=da.lat.values, lon=da.lon.values),
        attrs=dict(source="CHM_PRE V2 (Hu & Miao 2025, ESSD 17, 3987)",
                   time_window=f"{year0}-{year1}",
                   region="Yellow River middle+lower",
                   wet_day_mm=WET_DAY_MM, coverage_min=COVERAGE_MIN,
                   local_intensity_q=LOCAL_INTENSITY_Q, areamean_q=AREAMEAN_Q,
                   areamean_p90_mm_d=P90_area, yrep_count=n_yrep, n_days=n),
    )
    nc_path = os.path.join(outdir, f"precip_daily_region_{year0}_{year1}.nc")
    ds_out.to_netcdf(nc_path)
    print(f"[save] {nc_path}")

    df = pd.DataFrame(dict(
        date=pd.to_datetime(da.time.values),
        Pbar_mm_d=Pbar.values, F=F.values,
        coverage_pct=100 * F.values, YREP=yrep.values,
    ))
    df["year"] = df.date.dt.year; df["month"] = df.date.dt.month
    csv_path = os.path.join(outdir, f"YREP_labels_{year0}_{year1}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")
    print("[done] YREP monthly distribution (YREP=1 days by month):")
    print(df[df.YREP == 1].month.value_counts().sort_index().to_string())
    return nc_path, csv_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=list(DEFAULT_YEARS))
    ap.add_argument("--smoke", action="store_true", help="run on the last 3 years only")
    ap.add_argument("--outdir", default=OUT_DIR)
    a = ap.parse_args()
    run(a.years[0], a.years[1], smoke=a.smoke, outdir=a.outdir)
