#!/usr/bin/env python3
"""07_moisture_budget.py — P-E budget and boundary moisture flux analysis.

Two moisture-mechanism analyses:
  1. P-E budget: E (residual) = P - VIMFC + dW/dt. NCEP R1 has no daily
     evaporation product, so E is the residual. Reported as basin-area-mean
     P (CHM_PRE), VIMFC, PW tendency (NCEP, basin-masked 2.5°).
  2. Boundary moisture flux through box 105-120°E / 34-41°N: F_S, F_W, F_E,
     F_N, F_net. Units kg/s + mm/day equivalent.

Stratified YR-EPCMP vs non × early(1960-1991)/late(1992-2024).

Inputs: diagnostics_native_1960_2024.nc, precip_daily_region_1960_2024.nc,
        prediction_probability_full.csv
Outputs: moisture_budget_terms.csv, boundary_flux_timeseries.csv, moisture_budget_summary.csv
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely

import config

OUT = config.OUT
TAG = "1960_2024"
EARLY_END = 1991
THRESH = 0.5
BASIN_SHP = config.BASIN_SHP
BLON0, BLON1 = 105.0, 120.0  # boundary box
BLAT0, BLAT1 = 34.0, 41.0
A_EARTH = 6.371e6
MM_DAY = 86400.0


def basin_mask_on_grid(lat1d, lon1d, shp=BASIN_SHP):
    g = gpd.read_file(shp).to_crs(4326).geometry.union_all()
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    return shapely.contains_xy(g, lon2d.ravel(), lat2d.ravel()).reshape(lon2d.shape)


def w_mean(field, mask, lat1d):
    w = np.cos(np.deg2rad(lat1d))[:, None] * mask
    return (field * w).sum(axis=(-2, -1)) / w.sum()


def main():
    diag = xr.open_dataset(os.path.join(OUT, f"diagnostics_native_{TAG}.nc"))
    pre = xr.open_dataset(os.path.join(OUT, f"precip_daily_region_{TAG}.nc"))
    prob = pd.read_csv(os.path.join(OUT, "prediction_probability_full.csv"), parse_dates=["date"])
    prob = prob[prob.model == "combined"][["date", "y_prob"]].copy()
    prob["date"] = prob["date"].dt.normalize()
    dates = pd.to_datetime(diag.time.values).normalize()
    assert len(dates) == pre.sizes["time"] == 23742, (len(dates), pre.sizes["time"])

    df = pd.DataFrame({"date": dates}).merge(prob, on="date", how="left")
    df["yrecmp"] = (df.y_prob >= THRESH).astype(int)
    df["year"] = df.date.dt.year
    df["Pbar"] = pre["Pbar"].values

    nlat, nlon = diag.lat.values, diag.lon.values
    bmask = basin_mask_on_grid(nlat, nlon)
    print(f"[mask] basin cells on NCEP 2.5° grid: {int(bmask.sum())}/{bmask.size}")

    # ---- budget terms (basin area-mean) ----
    VIMFC_basin = w_mean(diag["VIMFC"].values, bmask, nlat) * MM_DAY  # mm/day
    PW_basin = w_mean(diag["PW"].values, bmask, nlat)  # kg/m²
    dWdt = np.gradient(PW_basin) / 86400.0 * MM_DAY  # mm/day (daily-sampled)
    df["VIMFC_mmday"] = VIMFC_basin
    df["neg_dWdt_mmday"] = -dWdt
    df["E_app_mmday"] = df.Pbar.values - VIMFC_basin + dWdt  # residual E

    # ---- boundary fluxes ----
    ilat_S = int(np.argmin(np.abs(nlat - BLAT0))); ilat_N = int(np.argmin(np.abs(nlat - BLAT1)))
    ilon_W = int(np.argmin(np.abs(nlon - BLON0))); ilon_E = int(np.argmin(np.abs(nlon - BLON1)))
    dlon_rad = np.deg2rad(np.median(np.diff(nlon))) if len(nlon) > 1 else np.deg2rad(2.5)
    dlat_rad = np.deg2rad(np.median(np.diff(nlat))) if len(nlat) > 1 else np.deg2rad(2.5)
    cosS, cosN = np.cos(np.deg2rad(nlat[ilat_S])), np.cos(np.deg2rad(nlat[ilat_N]))
    IVTx, IVTy = diag["IVT_x"].values, diag["IVT_y"].values
    lon_mask = (nlon >= BLON0) & (nlon <= BLON1)
    lat_mask = (nlat >= BLAT0) & (nlat <= BLAT1)
    F_S = (IVTy[:, ilat_S, :][:, lon_mask] * (A_EARTH * cosS * dlon_rad)).sum(1)
    F_N = -(IVTy[:, ilat_N, :][:, lon_mask] * (A_EARTH * cosN * dlon_rad)).sum(1)
    F_W = (IVTx[:, :, ilon_W][:, lat_mask] * (A_EARTH * dlat_rad)).sum(1)
    F_E = -(IVTx[:, :, ilon_E][:, lat_mask] * (A_EARTH * dlat_rad)).sum(1)
    F_net = F_S + F_N + F_W + F_E
    box_area_m2 = (np.deg2rad(BLON1 - BLON0)) * (np.sin(np.deg2rad(BLAT1)) - np.sin(np.deg2rad(BLAT0))) * A_EARTH ** 2
    F_net_mmday = F_net * MM_DAY / box_area_m2
    bnd = pd.DataFrame(dict(date=dates, F_S_kgs=F_S, F_N_kgs=F_N, F_W_kgs=F_W,
                            F_E_kgs=F_E, F_net_kgs=F_net, F_net_mmday=F_net_mmday))
    bnd["year"] = bnd.date.dt.year
    bnd.to_csv(os.path.join(OUT, "boundary_flux_timeseries.csv"), index=False)
    print(f"[save] boundary_flux_timeseries.csv ({len(bnd)} d, net mean={F_net.mean():.2e} kg/s)")
    df = df.merge(bnd[["date", "F_S_kgs", "F_N_kgs", "F_W_kgs", "F_E_kgs",
                       "F_net_kgs", "F_net_mmday"]], on="date")

    # ---- stratified summary (early/late × YR-EPCMP/non) ----
    df["period"] = np.where(df.year <= EARLY_END, "early", "late")
    df["cat"] = np.where(df.yrecmp == 1, "YREPCMP", "non")
    cols = ["Pbar", "VIMFC_mmday", "neg_dWdt_mmday", "E_app_mmday",
            "F_S_kgs", "F_W_kgs", "F_E_kgs", "F_N_kgs", "F_net_kgs", "F_net_mmday"]
    summ = df.groupby(["period", "cat"])[cols].mean().round(4).reset_index()
    summ.to_csv(os.path.join(OUT, "moisture_budget_summary.csv"), index=False)
    print("[save] moisture_budget_summary.csv"); print(summ.to_string(index=False))

    rows = []
    for (per, cat), g in df.groupby(["period", "cat"]):
        rows.append(dict(period=per, category=cat, n=len(g),
                         P=g.Pbar.mean(), E_app=g.E_app_mmday.mean(), VIMFC=g.VIMFC_mmday.mean(),
                         neg_dWdt=g.neg_dWdt_mmday.mean(),
                         # closure of P = E_app + VIMFC + neg_dWdt (neg_dWdt = -dW/dt);
                         # E_app is solved as the residual so this is ~0 by construction.
                         residual=g.Pbar.mean() - g.E_app_mmday.mean() - g.VIMFC_mmday.mean() - g.neg_dWdt_mmday.mean()))
    pd.DataFrame(rows).round(5).to_csv(os.path.join(OUT, "moisture_budget_terms.csv"), index=False)
    print("[save] moisture_budget_terms.csv")

    y = df[df.cat == "YREPCMP"]
    print("\n[headline] YR-EPCMP-day moisture-budget shift late−early:")
    for c in cols:
        d = y[y.period == "late"][c].mean() - y[y.period == "early"][c].mean()
        print(f"  Δ{c:16s} = {d:+.4f}")


if __name__ == "__main__":
    main()
