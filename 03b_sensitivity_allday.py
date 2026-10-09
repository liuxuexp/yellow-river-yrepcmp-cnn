#!/usr/bin/env python3
"""03b_sensitivity_allday.py — All-day axis of the threshold sensitivity.

Implements two alternative extreme-day labels and reports the resulting rate so
the primary wet-day definition can be compared against alternatives:
  - Mouassom all-day: per-grid standardized anomaly δ, coverage of δ >= +2σ
  - Davenport: regional-mean P̄ >= p95 of all days / wet days

Re-reads the CHM_PRE masked grid (script 01 cached only wet-day thresholds, not
all-day stats), so it is heavier than 03 — run once after script 02 frees I/O.

Outputs:
  sensitivity_allday_<y0>_<y1>.csv          — extreme-day rate by definition
  sensitivity_allday_monthly_<y0>_<y1>.csv  — monthly distribution per label
"""
import argparse, os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely

import config

CHMPRE_DIR = config.CHM_DIR
REGION_SHP = config.BASIN_SHP
BBOX = dict(lon=slice(100.0, 122.0), lat=slice(30.0, 44.0))
WET_DAY_MM = 1.0
YEARS = (1960, 2024)
OUT_DIR = config.OUT


def load_grid_mask():
    files = [os.path.join(CHMPRE_DIR, f"CHM_PRE_V2_daily_{y}.nc")
             for y in range(YEARS[0], YEARS[1] + 1)]
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 120})
    da = ds["prec"].sel(**BBOX).load()
    ds.close()
    gdf = gpd.read_file(REGION_SHP).to_crs(4326)
    geom = gdf.geometry.union_all()
    lon2d, lat2d = np.meshgrid(da.lon.values, da.lat.values)
    mask = shapely.contains_xy(geom, lon2d.ravel(), lat2d.ravel()).reshape(lon2d.shape)
    print(f"[load] cropped shape={da.shape}  mask cells={int(mask.sum())}  "
          f"loaded={da.data.nbytes/1e9:.2f}GB")
    return da, mask


def run(outdir=OUT_DIR):
    os.makedirs(outdir, exist_ok=True)
    da, mask = load_grid_mask()
    P = da.where(mask).values.astype(np.float64)  # (time, lat, lon), NaN outside mask
    dates = pd.to_datetime(da.time.values)
    wlat = np.cos(np.deg2rad(da.lat.values))
    Wm = (np.broadcast_to(wlat[:, None], mask.shape) * mask)
    Wm_sum = float(Wm.sum())
    n = P.shape[0]

    # ---- regional mean P̄(d) (same definition as script 01) ----
    Pbar = np.nansum(P * Wm[None], axis=(1, 2)) / Wm_sum
    p95_all = float(np.quantile(Pbar, 0.95))
    p95_wet = float(np.quantile(Pbar[Pbar >= WET_DAY_MM], 0.95))
    print(f"[davenport] P̄ all-day p95 = {p95_all:.3f} mm/d  | wet-day p95 = {p95_wet:.3f} mm/d")

    # ---- per-grid ALL-DAY μ, σ (mask cells only) ----
    m2d = mask[None].astype(bool)
    Pmask = np.where(m2d, P, np.nan)
    mu = np.nanmean(Pmask, axis=0)  # (lat, lon)
    sd = np.nanstd(Pmask, axis=0)
    sd = np.where(sd < 1e-6, np.nan, sd)  # avoid div-by-zero on dry cells
    print(f"[mouassom] per-grid all-day μ grid-mean={np.nanmean(mu):.3f}  "
          f"σ grid-mean={np.nanmean(sd):.3f} mm/d")

    # ---- daily standardized anomaly δ and +2σ coverage ----
    delta = (P - mu[None]) / sd[None]  # NaN where sd is NaN (dry cells) -> excluded
    exceed2 = (delta >= 2.0)
    exceed2 = np.where(m2d, exceed2, False)
    F2 = (exceed2 * Wm[None]).sum(axis=(1, 2)) / Wm_sum  # daily coverage of δ>=+2

    # ---- assemble alternative-label rates ----
    rows = []
    def monthly(label):
        m = pd.Series(dates.month)[label == 1].value_counts().reindex(range(1, 13), fill_value=0)
        return m
    primary_csv = os.path.join(outdir, "YREP_day_table_1960_2024.csv")
    primary = pd.read_csv(primary_csv, parse_dates=["date"])["YREP_label"].values
    rows.append(dict(definition="primary (wet Q95+F>=10%+P90)",
                     n=int(primary.sum()), pct=round(100*primary.mean(), 3)))
    for cov in (0.05, 0.10, 0.15):
        lab = (F2 >= cov).astype(int)
        rows.append(dict(definition=f"Mouassom all-day +2σ, F>={int(cov*100)}%",
                         n=int(lab.sum()), pct=round(100*lab.mean(), 3)))
    for nm, thr in (("Davenport P̄>=p95(all)", p95_all), ("Davenport P̄>=p95(wet)", p95_wet)):
        lab = (Pbar >= thr).astype(int)
        rows.append(dict(definition=nm, n=int(lab.sum()), pct=round(100*lab.mean(), 3)))
    sens = pd.DataFrame(rows)
    out_csv = os.path.join(outdir, f"sensitivity_allday_{YEARS[0]}_{YEARS[1]}.csv")
    sens.to_csv(out_csv, index=False)
    print(f"\n[summary] extreme-day rate by label definition (n={n} days):")
    print(sens.to_string(index=False))
    print(f"\n[save] {out_csv}")

    # ---- monthly distribution of each label (expose any winter inflation of +2σ) ----
    print("\n[monthly] label-day counts by calendar month (rows=definition):")
    mo = pd.DataFrame({"month": dates.month})
    for r in rows:
        # recompute label vector quickly for the monthly breakdown
        d = r["definition"]
        if d.startswith("primary"):
            lab = primary
        elif d.startswith("Mouassom"):
            cov = float(d.split(">=")[1].rstrip("%").replace("F", "")) / 100
            lab = (F2 >= cov).astype(int)
        else:
            thr = p95_all if "all" in d else p95_wet
            lab = (Pbar >= thr).astype(int)
        mo[d] = lab
    mon = mo.groupby("month").sum().T
    print(mon.to_string())
    mon.to_csv(os.path.join(outdir, f"sensitivity_allday_monthly_{YEARS[0]}_{YEARS[1]}.csv"))
    jas_share = mon[[7, 8, 9]].sum(axis=1) / mon.sum(axis=1)
    print("\n[JAS share] (high = monsoon-concentrated like primary; low = winter-inflated):")
    print((100 * jas_share).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUT_DIR)
    run(ap.parse_args().outdir)
