#!/usr/bin/env python3
"""02_prepare_reanalysis.py — Build CNN inputs and diagnostics from NCEP/NCAR R1.

Loads 6 daily variables (slp, hgt, shum, uwnd, vwnd, omega) for 1960-2024,
computes standardized anomalies (31-day circular calendar-day window),
IVT/VIMFC/PW, detrends Z500 area-mean, and regrids anomalies to 1° for the CNN.

Outputs:
  cnn_inputs_1deg_1960_2024.nc     — standardized anomalies on 1° CNN grid
  diagnostics_native_1960_2024.nc   — raw fields on native 2.5° grid
"""
import os, time
import numpy as np
import xarray as xr

import config

NCAR_DAILY = config.NCAR_DIR
OUT = config.OUT
YEARS = list(range(1960, 2025))
TAG = "1960_2024"

# CNN region: 59-180°E / -11-70°N (82×122 at 1°). Both dims ≡2 mod 4 so
# conv padding=0 leaves zero orphan pixels (see assert_no_orphan in 04_train_cnn).
LON0, LON1 = 59, 180
LAT0, LAT1 = -11, 70
# Broader diagnostics domain (native 2.5° cells bracketing every CNN target point).
DLON0, DLON1 = 57.5, 180
DLAT0, DLAT1 = -12.5, 70
# Anomaly/CNN-input box (same as the extended diagnostics domain).
ALON0, ALON1, ALAT0, ALAT1 = 57.5, 180, -12.5, 70

# Physics constants
G = 9.80665               # m s^-2
A_EARTH = 6.371e6         # m
LEVELS_Q = [1000, 925, 850, 700, 600, 500, 400, 300]   # shum integration layers
T0 = time.time()


def log(msg):
    print(f"[{(time.time()-T0)/60:6.1f} min] {msg}", flush=True)


def load_var(var, level_sel=None):
    """Load one NCEP R1 daily variable across YEARS, return a DataArray.
    If level_sel given, select those pressure levels (hPa)."""
    files = [os.path.join(NCAR_DAILY, f"{var}.{y}.nc") for y in YEARS]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        log(f"  WARNING: {len(missing)} files missing for {var}: e.g. {missing[0]}")
        files = [f for f in files if os.path.exists(f)]
    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False, chunks={})
    da = ds[var]
    if level_sel is not None and "level" in da.dims:
        da = da.sel(level=level_sel).squeeze(drop=True)  # drop singleton if one level
    return da


def subset_region(da, lon0, lon1, lat0, lat1):
    """Select lon/lat box and eager-load (shared-FS dask groupby is slow)."""
    return da.sel(lon=slice(lon0, lon1), lat=slice(lat1, lat0)).load()  # NCEP lat N→S


def standardized_anomaly(da):
    """Standardized anomaly with 31-day circular calendar-day window.
    da: DataArray with time dim. Returns anomaly (mean 0, std 1 per cell per cal day)."""
    doy = da.time.dt.dayofyear.values
    doy365 = np.where(doy > 365, 365, doy)  # collapse leap day into 365-day cal
    da2 = da.assign_coords(doy365=("time", doy365))
    clim_g = da2.groupby("doy365")
    mean_raw = clim_g.mean("time")
    std_raw = clim_g.std("time")
    pad = 15
    mean_p = xr.concat([mean_raw.isel(doy365=slice(-pad, None)),
                        mean_raw,
                        mean_raw.isel(doy365=slice(0, pad))], dim="doy365")
    std_p = xr.concat([std_raw.isel(doy365=slice(-pad, None)),
                       std_raw,
                       std_raw.isel(doy365=slice(0, pad))], dim="doy365")
    mean_sm = mean_p.rolling(doy365=31, center=True, min_periods=1).mean().isel(
        doy365=slice(pad, pad + 365))
    std_sm = std_p.rolling(doy365=31, center=True, min_periods=1).mean().isel(
        doy365=slice(pad, pad + 365))
    mean_t = mean_sm.sel(doy365=xr.DataArray(doy365, dims="time", coords={"time": da.time}))
    std_t = std_sm.sel(doy365=xr.DataArray(doy365, dims="time", coords={"time": da.time}))
    anom = (da2 - mean_t) / std_t
    return anom.drop_vars("doy365", errors="ignore")


def detrend_area_mean(anom, lat_name="lat", lon_name="lon"):
    """Remove the area-mean (bulk-thickening) linear trend from a
    standardized-anomaly field, leaving only spatial-pattern anomalies."""
    w = np.cos(np.deg2rad(anom[lat_name]))
    am = anom.weighted(w).mean(dim=(lat_name, lon_name))  # time series
    t = np.arange(anom.time.size, dtype=float)
    sl = np.polyfit(t, am.values, 1)
    trend = sl[0] * t + sl[1]
    return anom - xr.DataArray(trend, dims="time", coords={"time": anom.time})


def compute_ivt(shum, uwnd, vwnd):
    """IVT_x, IVT_y by vertical integration (1/g)∫q·v dp over 1000-300 hPa.
    NCEP shum is NaN below ground → those layers drop out. Returns kg m^-1 s^-1."""
    p = shum["level"].values * 100.0  # Pa
    dp = -np.gradient(p)  # Pa, positive layer thickness (levels descend 1000->300)
    dp_da = xr.DataArray(dp, coords={"level": shum["level"]}, dims="level")
    qx = shum * uwnd
    qy = shum * vwnd
    IVT_x = (qx * dp_da).fillna(0).sum("level") / G
    IVT_y = (qy * dp_da).fillna(0).sum("level") / G
    return IVT_x, IVT_y


def compute_pw(shum):
    """PW = (1/g) ∫ q dp over the 8 shum levels (1000-300 hPa). kg m^-2 (= mm).
    shum is NaN below ground → those layers drop out. Captures >97% of column water."""
    p = shum["level"].values * 100.0  # Pa
    dp = -np.gradient(p)  # positive thickness
    dp_da = xr.DataArray(dp, coords={"level": shum["level"]}, dims="level")
    return (shum * dp_da).fillna(0).sum("level") / G


def compute_vimfc(IVT_x, IVT_y):
    """VIMFC = -∇·IVT, full spherical divergence. Positive = convergence.
    Units: kg m^-2 s^-1."""
    lat = IVT_x["lat"].values
    cosphi = np.cos(np.deg2rad(lat))
    cosphi_da = xr.DataArray(cosphi, coords={"lat": IVT_x["lat"]}, dims="lat")
    # xarray .differentiate('lon') treats lon (degrees) as numeric → ∂/∂λ_deg;
    # convert to per-radian via chain rule ×(180/π)
    DEG2RAD_INV = 180.0 / np.pi
    dIVTx_dlon = IVT_x.differentiate("lon") * DEG2RAD_INV
    dIVTy_cos_dlat = (IVT_y * cosphi_da).differentiate("lat") * DEG2RAD_INV
    div = (dIVTx_dlon + dIVTy_cos_dlat) / (A_EARTH * cosphi_da)
    return -div


def regrid_to_1deg(da):
    """Regrid to 1° over CNN region (59-180°E/-11-70°N → 82×122)."""
    newlat = np.arange(LAT0, LAT1 + 1, 1.0)
    newlon = np.arange(LON0, LON1 + 1, 1.0)
    return da.interp(lat=newlat, lon=newlon)


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---------- load + region-subset raw fields ----------
    log(f"Loading NCEP R1 daily fields {YEARS[0]}-{YEARS[-1]} ...")
    slp = subset_region(load_var("slp"),                    DLON0, DLON1, DLAT0, DLAT1)
    z500 = subset_region(load_var("hgt", level_sel=500),    DLON0, DLON1, DLAT0, DLAT1).rename("Z500")
    om500 = subset_region(load_var("omega", level_sel=500), DLON0, DLON1, DLAT0, DLAT1).rename("omega500")
    shum = subset_region(load_var("shum"),                  DLON0, DLON1, DLAT0, DLAT1)   # 8 levels
    uwnd = subset_region(load_var("uwnd", level_sel=LEVELS_Q), DLON0, DLON1, DLAT0, DLAT1)  # 8 levels
    vwnd = subset_region(load_var("vwnd", level_sel=LEVELS_Q), DLON0, DLON1, DLAT0, DLAT1)  # 8 levels
    shum_levels = [int(l) for l in shum["level"].values]
    log(f"  Fields loaded. shum/uwnd/vwnd levels (hPa): {shum_levels}")

    # ---------- IVT + VIMFC + PW (on native 2.5° diagnostics domain) ----------
    log("Computing IVT (1000-300 hPa), VIMFC (spherical div), PW (shum integral) ...")
    IVT_x_raw, IVT_y_raw = compute_ivt(shum, uwnd, vwnd)
    IVT_mag_raw = np.sqrt(IVT_x_raw**2 + IVT_y_raw**2).rename("IVT_mag")
    IVT_x_raw = IVT_x_raw.rename("IVT_x"); IVT_y_raw = IVT_y_raw.rename("IVT_y")
    VIMFC_raw = compute_vimfc(IVT_x_raw, IVT_y_raw).rename("VIMFC")
    PW_raw = compute_pw(shum).rename("PW")
    log(f"  IVT_mag range: [{float(IVT_mag_raw.min()):.1f}, {float(IVT_mag_raw.max()):.1f}] kg m^-1 s^-1 "
        f"(expect ~0-1000)")
    log(f"  VIMFC range:   [{float(VIMFC_raw.min()):.3e}, {float(VIMFC_raw.max()):.3e}] kg m^-2 s^-1")
    log(f"  PW range:      [{float(PW_raw.min()):.2f}, {float(PW_raw.max()):.2f}] kg m^-2 (=mm; expect ~0-70)")

    # low-level diagnostics for composites — drop singleton level coord for merge
    q850 = shum.sel(level=850).drop_vars("level", errors="ignore").rename("q850")
    q700 = shum.sel(level=700).drop_vars("level", errors="ignore").rename("q700")
    u850 = uwnd.sel(level=850).drop_vars("level", errors="ignore").rename("u850")
    v850 = vwnd.sel(level=850).drop_vars("level", errors="ignore").rename("v850")

    # ---------- write diagnostics (native 2.5°, RAW fields) ----------
    log(f"Writing diagnostics_native_{TAG}.nc (raw fields for composites/budget) ...")
    diag = xr.Dataset({
        "SLP": slp, "Z500": z500, "omega500": om500,
        "IVT_x": IVT_x_raw, "IVT_y": IVT_y_raw, "IVT_mag": IVT_mag_raw, "VIMFC": VIMFC_raw,
        "PW": PW_raw, "q850": q850, "q700": q700, "u850": u850, "v850": v850,
    }).assign_attrs(
        source="NCEP/NCAR R1 daily (NOAA PSL)",
        note="RAW daily fields (not anomalies) on native 2.5° grid; for composites/figures/moisture budget",
        PW_source="computed from shum vertical integral (1/g)∫q dp; pr_wtr.eatm not in databank",
    )
    diag.to_netcdf(os.path.join(OUT, f"diagnostics_native_{TAG}.nc"))
    log(f"  diagnostics written. dims={ {k: v for k, v in diag.sizes.items() if k in ('time','lat','lon')} }")
    del slp, z500, om500, q850, q700, u850, v850

    # ---------- standardized anomalies ----------
    # Re-read RAW fields from the diagnostics file so every anomaly shares one
    # load path and matches what was stored. Subsetting to the anomaly box
    # reproduces the CNN-input computation so cnn_inputs_1deg is bit-stable.
    log("Computing standardized anomalies (31-day circular calendar-day window) ...")
    diag_ds = xr.open_dataset(os.path.join(OUT, f"diagnostics_native_{TAG}.nc")).sel(
        lon=slice(ALON0, ALON1), lat=slice(ALAT1, ALAT0))
    slp_a = standardized_anomaly(diag_ds["SLP"]).rename("SLP_anom")
    z500_a = standardized_anomaly(diag_ds["Z500"]).rename("Z500_anom")
    z500_a = detrend_area_mean(z500_a).rename("Z500_anom")  # remove bulk-thickening trend
    IVTmag_a = standardized_anomaly(diag_ds["IVT_mag"]).rename("IVT_mag_anom")
    IVTx_a = standardized_anomaly(diag_ds["IVT_x"]).rename("IVT_x_anom")
    IVTy_a = standardized_anomaly(diag_ds["IVT_y"]).rename("IVT_y_anom")
    VIMFC_a = standardized_anomaly(diag_ds["VIMFC"]).rename("VIMFC_anom")
    PW_a = standardized_anomaly(diag_ds["PW"]).rename("PW_anom")
    cnn_anom = {
        "SLP_anom": slp_a, "Z500_anom": z500_a,
        "IVT_mag_anom": IVTmag_a, "IVT_x_anom": IVTx_a, "IVT_y_anom": IVTy_a,
        "VIMFC_anom": VIMFC_a, "PW_anom": PW_a,
    }
    log(f"  Z500_anom after detrend: area-mean should be ~0. sample mean={float(z500_a.mean()):.3e}")

    # Regrid anomalies to the 1° CNN grid. Do NOT pre-subset to [LON0,LON1]×[LAT0,LAT1]:
    # the anomaly fields live on the full diagnostics domain whose native cells
    # bracket every CNN target point, so interp() never touches an edge.
    log("Regridding standardized anomalies to 1° CNN grid (59-180°E/-11-70°N → 82×122) ...")
    cnn_1deg = {}
    for name, da in cnn_anom.items():
        cnn_1deg[name] = regrid_to_1deg(da)

    # ---------- write CNN inputs ----------
    log(f"Writing cnn_inputs_1deg_{TAG}.nc ...")
    cnn_ds = xr.Dataset(cnn_1deg).assign_attrs(
        source="NCEP/NCAR R1 daily standardized anomalies",
        region="59-180°E, -11-70°N, 1° regrid; effective resolution 2.5° (bilinear upsample)",
        climatology="31-day circular calendar-day window",
        Z500_note="area-mean thickening trend removed (Horton/Davenport)",
        channels_combined="Z500_anom, SLP_anom, IVT_mag_anom, VIMFC_anom, PW_anom (main model)",
        channels_circulation="SLP_anom, Z500_anom (Model 1)",
        channels_moisture="VIMFC_anom (sensitivity: IVT_x/IVT_y/VIMFC) (Model 2)",
        UTC_day_note="NCEP dailyavg = UTC day; ~half-day offset vs CHM_PRE Beijing day",
    )
    cnn_ds.to_netcdf(os.path.join(OUT, f"cnn_inputs_1deg_{TAG}.nc"))

    n_time = cnn_ds.sizes["time"]
    log(f"DONE. CNN inputs: {cnn_ds.sizes['lat']}×{cnn_ds.sizes['lon']}×{len(cnn_ds.data_vars)} "
        f"channels, {n_time} days.")
    expected = 23742  # 1960-2024 inclusive, 17 leap days
    log(f"n_time={n_time} expected~{expected} — "
        f"{'MATCH' if n_time == expected else 'CHECK (investigate if far off)'}")
    log(f"Outputs: out/cnn_inputs_1deg_{TAG}.nc, out/diagnostics_native_{TAG}.nc")


if __name__ == "__main__":
    main()
