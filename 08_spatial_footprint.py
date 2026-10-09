#!/usr/bin/env python3
"""08_spatial_footprint.py — Connected-component analysis of YREP-day extreme fields.

For each YREP day, on the native 0.1° CHM_PRE grid + reach masks:
1. Binary field: precip >= per-grid local wet-day Q95.
2. Connected-component labeling, 8-connectivity.
3. Per object: area (km²), mean/max intensity, event-total, centroid,
   middle_lower_connected flag.
4. Per YREP day: n objects, largest-object area, total affected area,
   middle|lower synchronized flag (both reaches' coverage >= 10%).
5. Yearly aggregates.

Inputs: precip_daily_region_1960_2024.nc, YREP_day_table_1960_2024.csv,
        prediction_probability_full.csv, huanghe_middle/lower.shp, CHM_PRE daily
Outputs: event_objects_1960_2024.csv, footprint_daily_1960_2024.csv, footprint_yearly_1960_2024.csv
"""
import argparse, os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
from scipy import ndimage

import config

OUT_DIR = config.OUT
CHMPRE_DIR = config.CHM_DIR
CHMPRE_VAR = "prec"
_BASIN_DIR = os.path.dirname(config.BASIN_SHP)
MIDDLE_SHP = os.path.join(_BASIN_DIR, "huanghe_middle.shp")
LOWER_SHP = os.path.join(_BASIN_DIR, "huanghe_lower.shp")
BBOX = dict(lon=slice(100.0, 122.0), lat=slice(30.0, 44.0))
Q_LEVEL = 0.95
SYNC_COV_MIN = 0.10
EARTH_R_KM = 6371.0
CELL_DEG = 0.1

da_lon = da_lat = None  # set in run()


def reach_masks_on_grid(da, middle_shp=MIDDLE_SHP, lower_shp=LOWER_SHP):
    lon2d, lat2d = np.meshgrid(da.lon.values, da.lat.values)
    out = {}
    for reach, shp in (("middle", middle_shp), ("lower", lower_shp)):
        g = gpd.read_file(shp).to_crs(4326).geometry.union_all()
        out[reach] = shapely.contains_xy(g, lon2d.ravel(), lat2d.ravel()).reshape(lon2d.shape)
    return out


def cell_area_km2(lat):
    dlon = np.deg2rad(CELL_DEG); dlat = np.deg2rad(CELL_DEG)
    return (EARTH_R_KM ** 2) * dlon * dlat * np.cos(np.deg2rad(lat))


def analyze_day(field, mask, qgrid, mid_mask, low_mask, area2d, date):
    extreme = (field >= qgrid) & mask
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labels, n_obj = ndimage.label(extreme, structure=structure)
    mid_cov = float((extreme & mid_mask).sum()) / max(int(mid_mask.sum()), 1)
    low_cov = float((extreme & low_mask).sum()) / max(int(low_mask.sum()), 1)
    synced = bool(mid_cov >= SYNC_COV_MIN and low_cov >= SYNC_COV_MIN)
    rows = []; total_ext_area = 0.0; largest_area = 0.0
    for oid in range(1, n_obj + 1):
        sel = labels == oid; inten = field[sel]
        cy, cx = ndimage.center_of_mass(np.ones_like(sel), labels, oid)
        in_mid = bool((sel & mid_mask).any()); in_low = bool((sel & low_mask).any())
        area = float(area2d[sel].sum())
        rows.append(dict(date=date, object_id=oid, area_km2=round(area, 1), n_cells=int(sel.sum()),
                         mean_intensity_mm_d=round(float(inten.mean()), 2),
                         max_intensity_mm_d=round(float(inten.max()), 2),
                         event_total_mm=round(float(inten.sum()), 1),
                         centroid_lat=round(float(da_lat[int(round(cy))]), 2),
                         centroid_lon=round(float(da_lon[int(round(cx))]), 2),
                         middle_lower_connected=bool(in_mid and in_low)))
        total_ext_area += area; largest_area = max(largest_area, area)
    day = dict(date=date, n_objects=int(n_obj), largest_object_km2=round(largest_area, 1),
               total_extreme_area_km2=round(total_ext_area, 1), mid_coverage=round(mid_cov, 4),
               low_coverage=round(low_cov, 4), synced=synced)
    return rows, day


def run(y0=1960, y1=2024, outdir=OUT_DIR):
    global da_lon, da_lat
    cache = os.path.join(outdir, f"precip_daily_region_{y0}_{y1}.nc")
    yrep_csv = os.path.join(outdir, f"YREP_day_table_{y0}_{y1}.csv")
    prob_csv = os.path.join(outdir, "prediction_probability_full.csv")
    ds = xr.open_dataset(cache)
    mask = ds["mask"].values.astype(bool); qgrid = ds["Q95"].values
    da_lon = ds.lon.values; da_lat = ds.lat.values
    area2d = cell_area_km2(da_lat)[:, None] * np.ones((len(da_lat), len(da_lon)))
    ds.close()
    yrep = pd.read_csv(yrep_csv, parse_dates=["date"])
    yrep_dates = yrep.loc[yrep.YREP_label == 1, "date"].dt.normalize().tolist()
    # combined-model YR-EPCMP probability per date
    prob = pd.read_csv(prob_csv, parse_dates=["date"])
    prob = prob[prob.model == "combined"][["date", "y_prob"]].copy()
    prob["date"] = prob["date"].dt.normalize()
    pmap = dict(zip(prob.date, prob.y_prob))
    print(f"[load] {len(yrep_dates)} YREP days; region cells={int(mask.sum())}")

    dummy = xr.Dataset(coords=dict(lat=da_lat, lon=da_lon))
    rmask = reach_masks_on_grid(dummy)
    print(f"[reach] middle cells={int(rmask['middle'].sum())}  lower cells={int(rmask['lower'].sum())}")

    by_year = {}
    for d in yrep_dates:
        by_year.setdefault(d.year, []).append(d)
    obj_rows, day_rows = [], []
    for yr, dates in sorted(by_year.items()):
        f = os.path.join(CHMPRE_DIR, f"CHM_PRE_V2_daily_{yr}.nc")
        dyr = xr.open_dataset(f)[CHMPRE_VAR].sel(**BBOX)
        for d in dates:
            field = dyr.sel(time=np.datetime64(d.strftime("%Y-%m-%d"))).values
            orow, drow = analyze_day(field, mask, qgrid, rmask["middle"], rmask["lower"], area2d,
                                     d.strftime("%Y-%m-%d"))
            drow["YR_EPCMP_probability"] = float(pmap.get(pd.Timestamp(d).normalize(), np.nan))
            obj_rows.extend(orow); day_rows.append(drow)
        dyr.close()
    objs = pd.DataFrame(obj_rows); daily = pd.DataFrame(day_rows)
    objs.to_csv(os.path.join(outdir, f"event_objects_{y0}_{y1}.csv"), index=False)
    daily.to_csv(os.path.join(outdir, f"footprint_daily_{y0}_{y1}.csv"), index=False)
    print(f"[save] event_objects + footprint_daily  (synced days = {int(daily['synced'].sum())}/"
          f"{len(daily)} = {100*daily['synced'].mean():.1f}%)")
    daily["year"] = pd.to_datetime(daily.date).dt.year
    yr_agg = daily.groupby("year").agg(
        n_yrep_days=("date", "size"), max_object_km2=("largest_object_km2", "max"),
        mean_largest_km2=("largest_object_km2", "mean"), synced_days=("synced", "sum"),
        mean_total_extreme_km2=("total_extreme_area_km2", "mean")).reset_index()
    yr_agg.to_csv(os.path.join(outdir, f"footprint_yearly_{y0}_{y1}.csv"), index=False)
    print(f"[save] footprint_yearly (rows={len(yr_agg)})")
    print("[done] yearly (first/last 5):")
    print(yr_agg[["year", "n_yrep_days", "synced_days", "max_object_km2"]].head(5).to_string(index=False))
    print("...\n" + yr_agg[["year", "n_yrep_days", "synced_days", "max_object_km2"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=[1960, 2024])
    a = ap.parse_args()
    run(a.years[0], a.years[1])
