#!/usr/bin/env python3
"""01b_fig1_climatology.py — Two climatology fields for Fig 1 (c),(d).

  Fig 1c = annual mean precipitation (CHM_PRE 1960-2024 ×365)
  Fig 1d = per-grid days/year exceeding local wet-day Q95

Script 01 caches only the Q-thresholds + mask, not these two fields, so this
helper recomputes them from CHM_PRE and writes fig1_climatology.nc.

Inputs: precip_daily_region_1960_2024.nc (Q95 + mask from script 01), CHM_PRE daily
Output: out/fig1_climatology.nc
"""
import os, sys, warnings
import numpy as np
import xarray as xr
import geopandas as gpd
import shapely

import config

CHMPRE_DIR = config.CHM_DIR
CHMPRE_VAR = "prec"
REGION_SHP = config.BASIN_SHP
BBOX = dict(lon=slice(100.0, 122.0), lat=slice(30.0, 44.0))
WET_DAY_MM = 1.0
YEARS = (1960, 2024)
OUT = config.OUT
Q95_SRC = os.path.join(OUT, "precip_daily_region_1960_2024.nc")  # produced by 01


def main():
    files = [os.path.join(CHMPRE_DIR, f"CHM_PRE_V2_daily_{y}.nc")
             for y in range(YEARS[0], YEARS[1] + 1)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} CHM_PRE files missing, e.g. {missing[0]}")
    print(f"[load] {len(files)} CHM_PRE annual files ({YEARS[0]}–{YEARS[1]}) from {CHMPRE_DIR}")
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 120})
    da = ds[CHMPRE_VAR].sel(**BBOX).load()  # eager load — shared-FS dask groupby is slow
    ds.close()
    print(f"[load] shape={da.shape}  nbytes={da.data.nbytes/1e9:.1f} GB")

    # basin mask (same logic as 01_prepare_precip.build_basin_mask)
    gdf = gpd.read_file(REGION_SHP).to_crs(4326)
    geom = gdf.geometry.union_all()
    lon2d, lat2d = np.meshgrid(da.lon.values, da.lat.values)
    mask = shapely.contains_xy(geom, lon2d.ravel(), lat2d.ravel()).reshape(lon2d.shape)
    print(f"[mask] basin cells = {int(mask.sum())}")

    # read the locked Q95 threshold field (from 01's output)
    if not os.path.exists(Q95_SRC):
        raise FileNotFoundError(
            f"{Q95_SRC} not found — run 01_prepare_precip.py first (it caches Q95 on the same grid).")
    pre = xr.open_dataset(Q95_SRC)
    q95 = pre["Q95"].values  # (lat, lon) on the SAME grid (BBOX overlap)
    pre.close()
    assert q95.shape == mask.shape, f"grid mismatch {q95.shape} vs {mask.shape}"

    P = da.where(mask)  # masked precip (NaN outside basin)
    n_years = YEARS[1] - YEARS[0] + 1

    # (c) annual mean precipitation (mm yr-1) — mean daily × 365
    mean_daily = P.mean(dim="time", skipna=True).values  # mm d-1 per grid
    annual_mean = mean_daily * 365.0

    # (d) extreme-precip frequency — per-grid mean days/year with P >= local Q95
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        exceed = (P.values >= q95[None, :, :]) & mask[None, :, :]
    extreme_freq_yr = exceed.sum(axis=0) / n_years  # days yr-1 per grid
    extreme_freq_yr_masked = np.where(mask, extreme_freq_yr, np.nan)
    annual_mean_masked = np.where(mask, annual_mean, np.nan)

    out = xr.Dataset(
        data_vars=dict(
            mask=(("lat", "lon"), mask.astype("int8"),
                  {"long_name": "study-region mask"}),
            mean_daily=(("lat", "lon"), np.where(mask, mean_daily, np.nan),
                        {"units": "mm day-1", "long_name": "long-term mean daily precipitation"}),
            annual_mean=(("lat", "lon"), annual_mean_masked,
                         {"units": "mm yr-1", "long_name": "annual mean precipitation (= mean daily * 365)"}),
            extreme_freq_yr=(("lat", "lon"), extreme_freq_yr_masked,
                             {"units": "days yr-1",
                              "long_name": "mean days per year with P >= local wet-day Q95"}),
        ),
        coords=dict(lat=da.lat.values, lon=da.lon.values),
    )
    out_path = os.path.join(OUT, "fig1_climatology.nc")
    out.to_netcdf(out_path)
    print(f"[save] {out_path}")
    print(f"  annual_mean  (masked): min={np.nanmin(annual_mean_masked):.0f} "
          f"median={np.nanmedian(annual_mean_masked):.0f} max={np.nanmax(annual_mean_masked):.0f} mm/yr")
    print(f"  extreme_freq (masked): min={np.nanmin(extreme_freq_yr_masked):.2f} "
          f"median={np.nanmedian(extreme_freq_yr_masked):.2f} max={np.nanmax(extreme_freq_yr_masked):.2f} d/yr")


if __name__ == "__main__":
    main()
